"""Step 2: clean the data, rebuild each user's timeline, and create the splits.

Outputs (data/processed/):
    orders_clean.parquet          one row per order, with day_index and split
    order_products_clean.parquet  one row per (order, product), with user_id, split and day_index

Design decisions (see CLAUDE.md):
- Kaggle "test" orders have no products, so they are dropped.
- Prior and train order-products are combined into one purchase history.
- Relative timeline: day_index = cumulative sum of days_since_prior_order per user
  (the first order is day 0). Values of 30 mean "30 or more days" (capped in the source).
- Split per user by order_number:
    last order        -> "test"
    second-last order -> "val"
    all earlier       -> "history"
  Validation features must use "history" only; test features may use "history" + "val".
  This keeps future information out of training.

Usage:
    python -m src.data.preprocess
"""
import duckdb

from src.config import DAYS_CAP, PROCESSED_DIR


def build() -> None:
    p = lambda name: (PROCESSED_DIR / f"{name}.parquet").as_posix()  # noqa: E731
    con = duckdb.connect()

    con.execute(f"""
        CREATE TABLE orders AS
        SELECT
            order_id, user_id, order_number, order_dow, order_hour_of_day,
            COALESCE(days_since_prior_order, 0)              AS days_since_prior_order,
            COALESCE(days_since_prior_order >= {DAYS_CAP}, FALSE) AS days_capped,
            SUM(COALESCE(days_since_prior_order, 0)) OVER (
                PARTITION BY user_id ORDER BY order_number
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )                                                AS day_index,
            MAX(order_number) OVER (PARTITION BY user_id)    AS max_order_number
        FROM '{p("orders")}'
        WHERE eval_set IN ('prior', 'train')
    """)

    con.execute("""
        ALTER TABLE orders ADD COLUMN split VARCHAR;
        UPDATE orders SET split = CASE
            WHEN order_number = max_order_number     THEN 'test'
            WHEN order_number = max_order_number - 1 THEN 'val'
            ELSE 'history' END;
    """)

    con.execute(f"COPY (SELECT * EXCLUDE (max_order_number) FROM orders) "
                f"TO '{p('orders_clean')}' (FORMAT PARQUET)")

    con.execute(f"""
        COPY (
            SELECT op.order_id, o.user_id, o.order_number, o.day_index, o.split,
                   op.product_id, pr.aisle_id, pr.department_id,
                   op.add_to_cart_order, op.reordered
            FROM (
                SELECT * FROM '{p("order_products_prior")}'
                UNION ALL
                SELECT * FROM '{p("order_products_train")}'
            ) op
            JOIN orders o USING (order_id)
            JOIN '{p("products")}' pr USING (product_id)
        ) TO '{p('order_products_clean')}' (FORMAT PARQUET)
    """)

    summary = con.execute(f"""
        SELECT split, COUNT(DISTINCT order_id) AS orders, COUNT(*) AS items,
               COUNT(DISTINCT user_id) AS users
        FROM '{p('order_products_clean')}' GROUP BY split ORDER BY split
    """).df()
    print(summary.to_string(index=False))
    con.close()


if __name__ == "__main__":
    build()
