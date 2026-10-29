"""Smart Grocery Recommender: demo dashboard (Streamlit).

Pick a user and see the predicted shopping list for their next trip, with a reason for each
item, next to what they actually bought. Also shows overall model performance.

Requires: python -m src.models.train  and  python -m src.models.score
Run:      streamlit run src/app/dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, PROCESSED_DIR, ROOT

st.set_page_config(page_title="Smart Grocery Recommender", page_icon="🛒", layout="wide")

P = lambda n: (PROCESSED_DIR / f"{n}.parquet").as_posix()  # noqa: E731


@st.cache_resource
def connect():
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW pred AS SELECT * FROM '{P("predictions_test")}';
        CREATE VIEW truth AS SELECT * FROM '{P("truth_test")}';
        CREATE VIEW op AS SELECT * FROM '{P("order_products_clean")}';
        CREATE TABLE prod AS
            SELECT p.product_id, p.product_name, a.aisle, d.department
            FROM '{P("products")}' p
            JOIN '{P("aisles")}' a USING (aisle_id)
            JOIN '{P("departments")}' d USING (department_id);
    """)
    return con


def q(sql: str, params=None) -> pd.DataFrame:
    return connect().cursor().execute(sql, params or []).df()


@st.cache_data
def user_ids() -> pd.Series:
    return q("SELECT DISTINCT user_id FROM truth ORDER BY user_id")["user_id"]


def ranked_bars(series: pd.Series, title: str):
    d = series.dropna().rename("value").reset_index().rename(columns={series.index.name or "index": "method"})
    d["type"] = d["method"].map(lambda m: "ML model" if m in ("lightgbm", "logistic_regression") else "Baseline")
    return alt.Chart(d, title=title).mark_bar().encode(
        x=alt.X("value:Q", title=None),
        y=alt.Y("method:N", sort="-x", title=None),
        color=alt.Color("type:N", scale=alt.Scale(domain=["ML model", "Baseline"], range=["#2E7D32", "#90A4AE"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["method", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=alt.Step(30))


def reason(r: pd.Series) -> str:
    parts = []
    gap, days, due = r.up_mean_gap, r.up_days_since_last, r.up_due_ratio
    if pd.notna(gap) and gap > 0:
        status = "Due now" if 0.8 <= due <= 1.5 else "Overdue" if due > 1.5 else "Not due yet"
        parts.append(f"{status}: every ~{gap:.0f}d, last {days:.0f}d ago")
    parts.append(f"in {int(r.up_count)}/{int(r.u_orders)} orders")
    if r.up_in_last_order == 1:
        parts.append("in last order")
    return " · ".join(parts)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛒 Smart Grocery")
ids = user_ids()
if "uid" not in st.session_state:
    st.session_state.uid = int(ids.iloc[0])
if st.sidebar.button("🎲 Random user", width="stretch"):
    st.session_state.uid = int(ids.sample(1).iloc[0])
uid = int(st.sidebar.number_input("User ID", min_value=int(ids.min()), max_value=int(ids.max()),
                                  value=st.session_state.uid, step=1))
st.session_state.uid = uid

model = st.sidebar.radio("Ranking method", ["LightGBM (ML model)", "Most frequent (baseline)"])
score_col = "p_lgbm" if model.startswith("LightGBM") else "s_most_frequent"

cand = q(f"""
    SELECT c.*, p.product_name, p.aisle, p.department
    FROM pred c JOIN prod p USING (product_id)
    WHERE c.user_id = ? ORDER BY {score_col} DESC, product_id""", [uid])

if cand.empty:
    st.warning("No candidates for this user. Try another user ID.")
    st.stop()

u = cand.iloc[0]
k = st.sidebar.slider("List length (K)", 3, 30, int(max(3, min(30, round(u.u_avg_basket)))),
                      help="Defaults to this user's average basket size")
st.sidebar.caption("Data: Instacart 2017 · predicting each user's last (held-out) order.")

# ---------------------------------------------------------------- header
st.title("Smart Grocery Shopping Recommender")
st.caption("Personalised replenishment prediction: which previously bought items will this user need on their next trip?")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Past orders", int(u.u_orders))
c2.metric("Avg basket size", f"{u.u_avg_basket:.1f} items")
c3.metric("Avg days between orders", f"{u.u_mean_order_gap:.1f}")
c4.metric("Habit ratio", f"{u.u_reorder_ratio:.0%}", help="Share of past items that were reorders")
c5.metric("Next trip", f"in {u.t_days_since_prior:.0f} days")

tab_list, tab_history, tab_perf = st.tabs(["🧾 Shopping list", "📊 Purchase history", "📈 Model performance"])

# ---------------------------------------------------------------- shopping list
with tab_list:
    actual = q("""
        SELECT t.product_id, p.product_name, p.department
        FROM truth t JOIN prod p USING (product_id) WHERE t.user_id = ?""", [uid])
    actual_ids = set(actual.product_id)
    cand_ids = set(cand.product_id)

    top = cand.head(k).copy()
    top["hit"] = top.product_id.isin(actual_ids)
    hits = int(top.hit.sum())
    repeats = actual.product_id.isin(cand_ids).sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Correct suggestions", f"{hits} / {k}")
    m2.metric("Precision", f"{hits / k:.0%}")
    m3.metric("Recall (whole order)", f"{hits / max(len(actual), 1):.0%}")
    m4.metric("Recall (repeat items)", f"{hits / max(repeats, 1):.0%}",
              help="Of the items in the actual order that the user had bought before")

    row_h = lambda n: 35 * (n + 1) + 3  # noqa: E731  (show every row, no inner scroll)

    st.subheader(f"Predicted shopping list (top {k})")
    show = pd.DataFrame({
        "#": range(1, len(top) + 1),
        "Bought?": top.hit.map({True: "✅", False: "–"}),
        "Product": top.product_name,
        "Department": top.department,
        "Score": top[score_col].round(3) if score_col == "p_lgbm" else top.up_count.astype(int),
        "Why it's on the list": top.apply(reason, axis=1),
    })
    st.dataframe(show, hide_index=True, width="stretch", height=row_h(len(show)),
                 column_config={
                     "#": st.column_config.NumberColumn(width="small"),
                     "Bought?": st.column_config.TextColumn(width="small"),
                     "Product": st.column_config.TextColumn(width="medium"),
                     "Department": st.column_config.TextColumn(width="small"),
                     "Score": st.column_config.NumberColumn(
                         width="small", help="LightGBM probability" if score_col == "p_lgbm" else "Times bought"),
                     "Why it's on the list": st.column_config.TextColumn(width="large"),
                 })

    st.subheader(f"Actual next order ({len(actual)} items)")
    actual["Type"] = actual.product_id.map(lambda p: "🔁 Repeat" if p in cand_ids else "🆕 New item")
    actual["Predicted?"] = actual.product_id.isin(set(top.product_id)).map({True: "✅", False: "–"})
    actual = actual.sort_values(["Predicted?", "Type"], ascending=[False, True])
    n_new = len(actual) - repeats
    st.caption(f"{repeats} repeat items could be predicted from history · "
               f"{n_new} new items need the recommendation component (planned for the final submission).")
    st.dataframe(actual[["Predicted?", "product_name", "department", "Type"]]
                 .rename(columns={"product_name": "Product", "department": "Department"}),
                 hide_index=True, width="stretch", height=row_h(min(len(actual), 25)))

# ---------------------------------------------------------------- history
with tab_history:
    hist = q("""
        SELECT p.department, COUNT(*) AS purchases
        FROM op o JOIN prod p USING (product_id)
        WHERE o.user_id = ? AND o.split IN ('history', 'val')
        GROUP BY 1 ORDER BY 2 DESC""", [uid])
    a, b = st.columns(2)
    with a:
        st.subheader("Purchases by department")
        st.altair_chart(alt.Chart(hist).mark_bar(color="#2E7D32").encode(
            x=alt.X("purchases:Q", title="items bought"), y=alt.Y("department:N", sort="-x", title=None),
            tooltip=["department", "purchases"]).properties(height=alt.Step(22)), width="stretch")
    with b:
        st.subheader("Most bought items")
        fav = cand.sort_values("up_count", ascending=False).head(10)
        st.dataframe(pd.DataFrame({"Product": fav.product_name, "Times bought": fav.up_count.astype(int),
                                   "Usual gap (days)": fav.up_mean_gap.round(1)}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- performance
with tab_perf:
    comp_path = ROOT / "reports" / "comparison_test.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path, index_col="method")
        cols = ["f1@basket", "precision@basket", "recall@basket", "hit_rate@basket", "recall@20", "pr_auc", "roc_auc"]
        st.subheader("Baselines vs ML models (test set: each user's last order, all 206K users)")
        st.dataframe(comp[[c for c in cols if c in comp.columns]].round(3), width="stretch")
        a, b = st.columns(2)
        a.altair_chart(ranked_bars(comp["f1@basket"], "F1@basket (shopping-list quality)"), width="stretch")
        if "pr_auc" in comp:
            b.altair_chart(ranked_bars(comp["pr_auc"], "PR-AUC (candidate ranking quality)"), width="stretch")
    abl_path = ROOT / "reports" / "ablation_test.csv"
    a, b = st.columns(2)
    if (FIGURES_DIR / "11_feature_importance.png").exists():
        a.subheader("Feature importance")
        a.image(str(FIGURES_DIR / "11_feature_importance.png"))
    if abl_path.exists():
        b.subheader("Feature-group ablation")
        b.dataframe(pd.read_csv(abl_path, index_col="run")[["pr_auc", "f1@basket", "pr_auc_change", "f1@basket_change"]]
                    .round(4), width="stretch")
        b.caption("Negative change = removing that feature group hurt the model.")
