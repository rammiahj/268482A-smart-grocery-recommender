"""candidate generation and feature engineering.

For each user we predict which previously bought items will appear in a *target* order.
Two feature sets are built so the model never sees the future:

    target = "val"   past orders = history          -> used to TRAIN models (label = item in val order)
    target = "test"  past orders = history + val    -> used to EVALUATE   (label = item in test order)

Candidates: every product the user bought in their past orders.

Features (all computed from past orders only):
  User (u_*)     number of orders, mean basket size, mean days between orders, habit (reorder) ratio, distinct products
  Product (p_*)  popularity, number of buyers, reorder rate, mean cart position (across all users)
  Aisle/dept     reorder rate of the item's aisle and department
  User x product (up_*)
                 times bought, purchase rate since first purchase, orders and days since last purchase,
                 mean / CV of repurchase gap, due ratio (days since last / mean gap),
                 bought in last order, mean cart position
  Target context (t_*)
                 days since prior order, day of week, hour of the target order.
                 Assumption: the time of the next shopping trip is known when the list is generated
                 (the user plans when they shop). Document this in the report.

Outputs (data/processed/):
    features_val.parquet, features_test.parquet   one row per candidate (user_id, product_id) with label
    truth_val.parquet,    truth_test.parquet      every item actually bought in the target order (incl. new items)

Usage:
    python -m src.features.build_features
"""
import duckdb

from src.config import PROCESSED_DIR

TARGETS = {"val": ("history",), "test": ("history", "val")}


def _p(name: str) -> str:
    return (PROCESSED_DIR / f"{name}.parquet").as_posix()


def build(target: str, con: duckdb.DuckDBPyConnection) -> None:
    past = ", ".join(f"'{s}'" for s in TARGETS[target])
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW past_op AS SELECT * FROM op WHERE split IN ({past});
        CREATE OR REPLACE TEMP VIEW past_orders AS SELECT * FROM orders WHERE split IN ({past});
        CREATE OR REPLACE TEMP VIEW tgt AS
            SELECT user_id, order_number AS t_order_number, day_index AS t_day,
                   days_since_prior_order AS t_days_since_prior, days_capped AS t_capped,
                   order_dow AS t_dow, order_hour_of_day AS t_hour
            FROM orders WHERE split = '{target}';
    """)

    # User features
    con.execute("""
        CREATE OR REPLACE TEMP TABLE u AS
        SELECT o.user_id,
               COUNT(*)                                                    AS u_orders,
               MAX(o.order_number)                                         AS u_last_order_number,
               AVG(CASE WHEN o.order_number > 1 THEN o.days_since_prior_order END) AS u_mean_order_gap,
               ANY_VALUE(i.u_total_items) / COUNT(*)                       AS u_avg_basket,
               ANY_VALUE(i.u_distinct_products)                            AS u_distinct_products,
               ANY_VALUE(i.u_reorder_ratio)                                AS u_reorder_ratio
        FROM past_orders o
        JOIN (SELECT user_id, COUNT(*) AS u_total_items,
                     COUNT(DISTINCT product_id) AS u_distinct_products,
                     AVG(CASE WHEN order_number > 1 THEN reordered END) AS u_reorder_ratio
              FROM past_op GROUP BY user_id) i USING (user_id)
        GROUP BY o.user_id
    """)

    # Product, aisle and department features (from all users' past orders)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE p AS
        SELECT product_id,
               ANY_VALUE(aisle_id) AS aisle_id, ANY_VALUE(department_id) AS department_id,
               COUNT(*) AS p_purchases, COUNT(DISTINCT user_id) AS p_users,
               AVG(reordered) AS p_reorder_rate, AVG(add_to_cart_order) AS p_mean_cart_pos
        FROM past_op GROUP BY product_id;
        CREATE OR REPLACE TEMP TABLE a AS
        SELECT aisle_id, AVG(reordered) AS a_reorder_rate FROM past_op GROUP BY aisle_id;
        CREATE OR REPLACE TEMP TABLE d AS
        SELECT department_id, AVG(reordered) AS d_reorder_rate FROM past_op GROUP BY department_id;
    """)

    # User x product features
    con.execute("""
        CREATE OR REPLACE TEMP TABLE up AS
        WITH g AS (
            SELECT user_id, product_id, order_number, day_index, add_to_cart_order,
                   day_index - LAG(day_index) OVER (
                       PARTITION BY user_id, product_id ORDER BY order_number) AS gap
            FROM past_op)
        SELECT user_id, product_id,
               COUNT(*) AS up_count, MIN(order_number) AS up_first_order, MAX(order_number) AS up_last_order,
               MAX(day_index) AS up_last_day, AVG(add_to_cart_order) AS up_mean_cart_pos,
               AVG(gap) AS up_mean_gap, STDDEV_SAMP(gap) AS up_sd_gap
        FROM g GROUP BY user_id, product_id
    """)

    con.execute(f"""
        COPY (
            SELECT up.user_id, up.product_id, p.aisle_id, p.department_id,
                   -- user
                   u.u_orders, u.u_avg_basket, u.u_mean_order_gap, u.u_reorder_ratio, u.u_distinct_products,
                   -- product / aisle / department
                   p.p_purchases, p.p_users, p.p_reorder_rate, p.p_mean_cart_pos,
                   a.a_reorder_rate, d.d_reorder_rate,
                   -- user x product
                   up.up_count,
                   up.up_count / u.u_orders                                        AS up_share,
                   up.up_count / (u.u_last_order_number - up.up_first_order + 1)   AS up_rate,
                   t.t_order_number - up.up_last_order                             AS up_orders_since_last,
                   t.t_day - up.up_last_day                                        AS up_days_since_last,
                   up.up_mean_gap,
                   up.up_sd_gap / NULLIF(up.up_mean_gap, 0)                        AS up_gap_cv,
                   (t.t_day - up.up_last_day)
                       / NULLIF(COALESCE(up.up_mean_gap, u.u_mean_order_gap), 0)  AS up_due_ratio,
                   (up.up_last_order = u.u_last_order_number)::INT                 AS up_in_last_order,
                   up.up_mean_cart_pos,
                   -- target order context
                   t.t_days_since_prior, t.t_capped::INT AS t_capped, t.t_dow, t.t_hour,
                   -- label
                   (y.product_id IS NOT NULL)::INT                                 AS label
            FROM up
            JOIN u   USING (user_id)
            JOIN tgt t USING (user_id)
            JOIN p   USING (product_id)
            LEFT JOIN a USING (aisle_id)
            LEFT JOIN d USING (department_id)
            LEFT JOIN (SELECT DISTINCT user_id, product_id FROM op WHERE split = '{target}') y
                   USING (user_id, product_id)
        ) TO '{_p(f"features_{target}")}' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (SELECT DISTINCT user_id, product_id FROM op WHERE split = '{target}')
        TO '{_p(f"truth_{target}")}' (FORMAT PARQUET)
    """)

    s = con.execute(f"""
        SELECT COUNT(*) AS candidates, COUNT(DISTINCT user_id) AS users,
               AVG(label) AS positive_rate, COUNT(*) / COUNT(DISTINCT user_id) AS candidates_per_user
        FROM '{_p(f"features_{target}")}'
    """).df()
    print(f"\n[{target}] features written")
    print(s.to_string(index=False))


def main() -> None:
    tmp = PROCESSED_DIR / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute(f"CREATE VIEW orders AS SELECT * FROM '{_p('orders_clean')}'")
    con.execute(f"CREATE VIEW op AS SELECT * FROM '{_p('order_products_clean')}'")
    for target in TARGETS:
        build(target, con)
    con.close()


if __name__ == "__main__":
    main()
