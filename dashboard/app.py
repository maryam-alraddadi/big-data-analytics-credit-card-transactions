"""
Bank Fraud Risk Intelligence Dashboard
POV: Fraud & Risk Operations team at a card-issuing bank
"""

from __future__ import annotations

import warnings
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

warnings.filterwarnings("ignore")

#  PATHS 
ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = ROOT / "data" / "processed" / "features_parquet"

#  DATA LOADING 
print("Loading features parquet …")
df = pd.read_parquet(FEATURES_PARQUET)
if "amount" not in df.columns and "amount_log1p" in df.columns:
    df["amount"] = np.expm1(df["amount_log1p"])
df["transaction_ts"] = pd.to_datetime(df["transaction_ts"])
df["tx_hour"]        = df["transaction_ts"].dt.hour
df["tx_month"]       = df["transaction_ts"].dt.month
df["tx_dow"]         = df["transaction_ts"].dt.dayofweek   # 0 = Monday
df["year_month"]     = df["transaction_ts"].dt.to_period("M").astype(str)
print(f"Loaded {len(df):,} rows · {len(df.columns)} columns")

#  PORTFOLIO KPIs 
TOTAL_TX          = len(df)
FRAUD_TX          = int(df["is_fraud"].sum())
FRAUD_RATE        = FRAUD_TX / TOTAL_TX * 100
UNIQUE_CARDS      = df["card_id"].nunique()
UNIQUE_MERCH      = df["merchant"].nunique()
AVG_AMOUNT        = df["amount"].mean()
DATE_MIN          = df["transaction_ts"].min().strftime("%b %Y")
DATE_MAX          = df["transaction_ts"].max().strftime("%b %Y")
NUM_STATES        = df["state"].nunique()
NUM_CATS          = df["category"].nunique()
TOTAL_SPEND       = df["amount"].sum()

# Financial fraud exposure
FRAUD_AMT_TOTAL   = df[df["is_fraud"] == 1]["amount"].sum()
FRAUD_AMT_AVG     = df[df["is_fraud"] == 1]["amount"].mean()
LEGIT_AMT_AVG     = df[df["is_fraud"] == 0]["amount"].mean()
CARDS_WITH_FRAUD  = df[df["is_fraud"] == 1]["card_id"].nunique()
FRAUD_LOSS_PCT    = FRAUD_AMT_TOTAL / TOTAL_SPEND * 100   # % of portfolio $ lost to fraud

# GBT model estimates (from notebook 04 outputs)
# Recall ≈ 0.987 derived from confusion matrix; precision ≈ 0.912
GBT_RECALL        = 0.987
GBT_PRECISION     = 0.912
ESTIMATED_CAUGHT  = FRAUD_AMT_TOTAL * GBT_RECALL
ESTIMATED_MISSED  = FRAUD_AMT_TOTAL * (1 - GBT_RECALL)
# False positives: legitimate tx wrongly blocked by the model
ESTIMATED_FP      = int(FRAUD_TX * GBT_RECALL * (1 - GBT_PRECISION) / GBT_PRECISION)

#  AGGREGATIONS 
cat_agg = (
    df.groupby("category", as_index=False)
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"),
         avg_amount=("amount", "mean"), fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
)

hour_agg = (
    df.groupby("tx_hour", as_index=False)
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"),
         fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
)

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_agg = (
    df.groupby("tx_dow", as_index=False)
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"))
    .assign(
        fraud_rate=lambda x: x["fraud"] / x["total"] * 100,
        day_name=lambda x: x["tx_dow"].map(lambda d: DOW_LABELS[d]),
    )
)

monthly_agg = (
    df.groupby("year_month", as_index=False)
    .agg(
        tx_count=("is_fraud", "count"),
        total_spend=("amount", "sum"),
        avg_amount=("amount", "mean"),
        fraud_count=("is_fraud", "sum"),
        fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()),
    )
    .assign(fraud_rate=lambda x: x["fraud_count"] / x["tx_count"] * 100)
    .sort_values("year_month")
)
_t = np.arange(len(monthly_agg))
monthly_agg["spend_trend"] = np.polyval(np.polyfit(_t, monthly_agg["total_spend"], 1), _t)

state_agg = (
    df.groupby("state", as_index=False)
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"),
         avg_amount=("amount", "mean"),
         fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
    .query("total >= 100")
    .sort_values("fraud_rate", ascending=False)
)

_amt_bins   = [0, 10, 50, 100, 250, 500, float("inf")]
_amt_labels = ["< $10", "$10-50", "$50-100", "$100-250", "$250-500", "> $500"]
df["amount_bin_c"] = pd.cut(df["amount"], bins=_amt_bins, labels=_amt_labels, right=False)
amtbin_agg = (
    df.groupby("amount_bin_c", observed=True, as_index=False)
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"),
         fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
)

AGE_COL = "cardholder_age" if "cardholder_age" in df.columns else None
if AGE_COL:
    _age_bins   = [0, 25, 35, 45, 55, 65, 200]
    _age_labels = ["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_group"] = pd.cut(df[AGE_COL], bins=_age_bins, labels=_age_labels, right=False)
    age_agg = (
        df.groupby("age_group", observed=True, as_index=False)
        .agg(tx_count=("is_fraud", "count"), avg_amount=("amount", "mean"),
             fraud=("is_fraud", "sum"),
             fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
        .assign(fraud_rate=lambda x: x["fraud"] / x["tx_count"] * 100)
    )
else:
    age_agg = pd.DataFrame()

gender_agg = (
    df.groupby("gender", as_index=False)
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"), avg_amount=("amount", "mean"),
         fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
)

DIST_COL = "merch_distance_km" if "merch_distance_km" in df.columns else None
if DIST_COL:
    _dist_bins   = [0, 10, 50, 100, 300, float("inf")]
    _dist_labels = ["< 10 km", "10-50 km", "50-100 km", "100-300 km", "> 300 km"]
    df["dist_bin"] = pd.cut(df[DIST_COL], bins=_dist_bins, labels=_dist_labels, right=False)
    dist_agg = (
        df.groupby("dist_bin", observed=True, as_index=False)
        .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"), avg_amount=("amount", "mean"),
             fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
        .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
    )
else:
    dist_agg = pd.DataFrame()

TOD_COL = "tx_time_of_day" if "tx_time_of_day" in df.columns else None
if TOD_COL:
    tod_agg = (
        df.groupby(TOD_COL, as_index=False)
        .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"), avg_amount=("amount", "mean"))
        .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
    )
else:
    tod_agg = pd.DataFrame()

merch_agg = (
    df.groupby("merchant", as_index=False)
    .agg(tx_count=("is_fraud", "count"), fraud=("is_fraud", "sum"),
         fraud_amount=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()))
    .assign(fraud_rate=lambda x: x["fraud"] / x["tx_count"] * 100)
    .query("tx_count >= 20")
    .sort_values("fraud_rate", ascending=False)
)

card_agg = (
    df.groupby("card_id", as_index=False)
    .agg(
        tx_count=("is_fraud", "count"),
        total_spend=("amount", "sum"),
        avg_amount=("amount", "mean"),
        fraud_count=("is_fraud", "sum"),
    )
    .assign(fraud_rate=lambda x: x["fraud_count"] / x["tx_count"] * 100)
)

# Fraud heatmap: day-of-week × hour
heatmap_pivot = (
    df.groupby(["tx_dow", "tx_hour"])["is_fraud"]
    .mean()
    .mul(100)
    .reset_index()
    .pivot(index="tx_dow", columns="tx_hour", values="is_fraud")
    .fillna(0)
)

VEL_COL = "card_tx_count_1h" if "card_tx_count_1h" in df.columns else None

#  HARDCODED MODEL RESULTS (notebook 04 outputs) 
MODEL_METRICS = pd.DataFrame({
    "Model":    ["Logistic Regression", "Random Forest", "GBT"],
    "AUC-ROC":  [0.8908, 0.9930, 0.9966],
    "AUC-PR":   [0.2880, 0.8305, 0.8683],
    "F1 Score": [0.9719, 0.9871, 0.9853],
})

FRAUD_FI = pd.Series({
    "card_tx_count_1h":   0.280,
    "merch_dist_km":      0.215,
    "amount":             0.138,
    "card_amount_sum_1h": 0.082,
    "amount_log1p":       0.089,
    "age":                0.062,
    "category_idx":       0.050,
    "city_pop":           0.033,
    "tx_hour":            0.022,
    "tx_month":           0.012,
    "tx_dow":             0.010,
    "tx_is_weekend":      0.005,
    "gender_bin":         0.002,
}, name="Fraud Detection RF").sort_values(ascending=True)

CAT_FI = pd.Series({
    "amount":        0.347,
    "amount_log1p":  0.218,
    "merch_dist_km": 0.152,
    "city_pop":      0.101,
    "age":           0.074,
    "tx_hour":       0.054,
    "tx_dow":        0.030,
    "tx_is_weekend": 0.016,
    "gender_bin":    0.008,
}, name="Category Classification RF").sort_values(ascending=True)

SILHOUETTE_SCORES = pd.DataFrame({
    "K":          [2,      3,      4,      5,      6,      7],
    "Silhouette": [0.2841, 0.3012, 0.3198, 0.3525, 0.3311, 0.3187],
})

CLUSTER_PROFILES = pd.DataFrame({
    "Cluster":        [0, 1, 2, 3, 4],
    "Label":          ["Low-Spend\nCasual", "High-Spend\nFrequent", "Elderly\nConservative",
                       "Mid-Spend\nRegular", "Young\nOnline"],
    "card_count":     [212, 198, 201, 186, 186],
    "tx_count":       [1201, 1456, 983, 1312, 1647],
    "avg_amount":     [58.3, 145.2, 72.8, 98.4, 89.6],
    "total_spend":    [70004, 211371, 71552, 129137, 147600],
    "fraud_rate_pct": [0.41, 0.73, 0.28, 0.58, 0.65],
    "age":            [52, 38, 61, 44, 31],
    "weekend_ratio":  [0.28, 0.35, 0.22, 0.31, 0.42],
})
CLUSTER_PROFILES["fraud_exposure"] = (
    CLUSTER_PROFILES["total_spend"] * CLUSTER_PROFILES["fraud_rate_pct"] / 100
)

ANOMALY_DF = pd.DataFrame({
    "Method":  ["IQR Outliers\n(>$193)", "IQR → Fraud", "Z-Score Outliers\n(|z|>3)", "Z-Score → Fraud"],
    "Count":   [67335, 5705, 12738, 3597],
    "Type":    ["Outlier", "Fraud", "Outlier", "Fraud"],
})

PARTITION_BENCH = pd.DataFrame({
    "Partitions": ["16", "64", "200", "400"],
    "Time (s)":   [4.21, 2.87, 3.94, 5.12],
    "Best":       [False, True, False, False],
})
CACHE_BENCH = pd.DataFrame({
    "Strategy": ["No Cache", "MEMORY_AND_DISK", "DISK_ONLY"],
    "Time (s)": [8.43, 3.12, 4.78],
    "Speedup":  [1.0, 2.7, 1.76],
})
JOIN_BENCH = pd.DataFrame({
    "Strategy": ["Sort-Merge Join", "Broadcast Join\n(explicit hint)"],
    "Time (s)": [7.21, 2.34],
})
OPTIM_SUMMARY = pd.DataFrame({
    "Optimisation":  ["Partition\nTuning", "Cache\n(mem)", "Broadcast\nJoin",
                      "AQE\nEnabled", "approx_count\n_distinct", "Column\nPruning"],
    "Baseline (s)":  [3.94, 8.43, 7.21, 6.87, 5.43, 4.21],
    "Optimised (s)": [2.87, 3.12, 2.34, 4.23, 0.87, 1.98],
})
OPTIM_SUMMARY["Speedup"] = OPTIM_SUMMARY["Baseline (s)"] / OPTIM_SUMMARY["Optimised (s)"]

#  COLOUR PALETTE (bank-grade: navy + alert red + safe green) 
NAVY    = "#003087"
CRIMSON = "#C41E3A"
FOREST  = "#00875A"
AMBER   = "#E07B00"
STEEL   = "#4A90D9"
SLATE   = "#6B7C93"
CLUSTER_COLORS = [NAVY, CRIMSON, FOREST, AMBER, STEEL]

#  LAYOUT HELPERS 
def _card(header: str, graph_id: str, height: int = 320) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.Strong(header, className="small")),
        dbc.CardBody(dcc.Graph(id=graph_id, config={"displayModeBar": False},
                               style={"height": f"{height}px"})),
    ], className="shadow-sm h-100")


def kpi_card(title: str, value: str, sub: str = "", color: str = "primary") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H4(value, className="fw-bold mb-0 text-center"),
                html.Small(title, className="text-muted d-block text-center"),
                html.Small(sub, className="text-muted d-block text-center fst-italic") if sub else None,
            ], className="py-2"),
            className=f"border-top border-{color} border-3 shadow-sm h-100",
        ),
        xs=12, sm=6, md=3, lg=True,
    )


#  ALERT BANNER 
FRAUD_ALERT_THRESHOLD = 1.0   # % alert if fraud rate exceeds this
alert_banner = dbc.Alert(
    [
        html.Strong("FRAUD RATE ELEVATED "),
        f"Portfolio fraud rate is {FRAUD_RATE:.3f}%, above the {FRAUD_ALERT_THRESHOLD:.1f}% alert threshold. "
        f"Estimated exposure: ${FRAUD_AMT_TOTAL:,.0f}. Immediate review recommended.",
    ],
    color="danger", className="mb-3 py-2 small",
    is_open=(FRAUD_RATE > FRAUD_ALERT_THRESHOLD),
    dismissable=True,
)

#  KPI ROW 
kpi_row = dbc.Row([
    kpi_card("Total Transactions",       f"{TOTAL_TX:,}",              color="primary"),
    kpi_card("Fraud Transactions",       f"{FRAUD_TX:,}",              color="danger"),
    kpi_card("Portfolio Fraud Rate",     f"{FRAUD_RATE:.3f}%",         color="warning"),
    kpi_card("Fraud Exposure ($)",       f"${FRAUD_AMT_TOTAL:,.0f}",   color="danger"),
    kpi_card("% Portfolio $ Lost",       f"{FRAUD_LOSS_PCT:.3f}%",     color="warning"),
    kpi_card("Cardholders Affected",     f"{CARDS_WITH_FRAUD:,}",      color="warning"),
    kpi_card("Avg Fraud Amount",         f"${FRAUD_AMT_AVG:.2f}",      color="secondary"),
    kpi_card("Coverage Period",          f"{DATE_MIN} – {DATE_MAX}",   color="secondary"),
], className="g-2 mb-4")


#  TAB 1 EXECUTIVE RISK OVERVIEW 
tab_overview = dbc.Tab(label="Executive Overview", tab_id="tab-overview", children=[
    dbc.Row([
        dbc.Col(_card("Portfolio: Fraud vs Legitimate Transactions",       "pie-fraud",       280), md=3),
        dbc.Col(_card("Fraud Exposure ($) & Rate by Spending Category",    "bar-cat-volume",  280), md=9),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Fraud Rate & Exposure by Transaction Amount Tier",  "bar-amount-bin",  300), md=6),
        dbc.Col(_card("Transaction Amount Distribution Fraud vs Legit",  "hist-amount",     300), md=6),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Monthly Fraud Exposure ($) vs Transaction Volume",  "bar-monthly",     320), md=12),
    ], className="g-3"),
])

#  TAB 2 FRAUD PATTERNS 
tab_temporal = dbc.Tab(label="Fraud Patterns", tab_id="tab-temporal", children=[
    dbc.Row([
        dbc.Col(_card("Fraud Rate Heatmap Hour of Day × Day of Week",   "heatmap-fraud",   340), md=8),
        dbc.Col(_card("Fraud Rate by Cardholder–Merchant Distance",        "bar-dist-fraud",  340), md=4),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Fraud Rate by Hour of Day",                         "line-hour-fraud", 280), md=6),
        dbc.Col(_card("Transaction Volume & Fraud Rate by Day of Week",    "bar-dow",         280), md=6),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("High-Risk States Fraud Rate (min 100 tx)",        "bar-state-fraud", 300), md=6),
        dbc.Col(_card("Fraud Rate & Exposure by Gender",                   "bar-gender",      300), md=6),
    ], className="g-3"),
])

#  TAB 3 CUSTOMER PORTFOLIO RISK 
tab_portfolio = dbc.Tab(label="Customer Portfolio Risk", tab_id="tab-portfolio", children=[
    dbc.Row([
        dbc.Col(_card("K-Means Customer Segments Silhouette (Best K=5)", "line-silhouette",     280), md=4),
        dbc.Col(_card("Segment Risk Profile: Avg Spend vs Fraud Rate",     "bar-cluster-profiles",280), md=8),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Fraud Exposure ($) by Customer Segment",            "bar-cluster-exposure",280), md=6),
        dbc.Col(_card("Segment Size & Average Age",                        "bar-cluster-size",    280), md=6),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Fraud Rate by Age Group",                           "bar-age-fraud",       280), md=4),
        dbc.Col(_card("Avg Transaction Amount by Age Group",               "bar-age-amount",      280), md=4),
        dbc.Col(_card("Cardholder Transaction Velocity Distribution",      "hist-card-vel",       280), md=4),
    ], className="g-3"),
])

#  TAB 4 DETECTION & ROI 
_roi_card = dbc.Card([
    dbc.CardHeader(html.Strong("GBT Model Financial Impact", className="small")),
    dbc.CardBody([
        html.H5("Gradient Boosted Trees", className="fw-bold mb-1", style={"color": FOREST}),
        html.P("50 iterations · max depth 5 · class-weighted", className="text-muted small mb-2"),
        dbc.ListGroup([
            dbc.ListGroupItem([html.Span("AUC-ROC",    className="text-muted small me-2"), html.Strong("0.9966")], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("AUC-PR",     className="text-muted small me-2"), html.Strong("0.8683")], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("F1 Score",   className="text-muted small me-2"), html.Strong("0.9853")], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("Est. Recall",className="text-muted small me-2"), html.Strong(f"{GBT_RECALL:.1%}")], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("Est. Precision",className="text-muted small me-2"),html.Strong(f"{GBT_PRECISION:.1%}")],className="d-flex justify-content-between py-1"),
        ], flush=True, className="mb-3 small"),
        html.Hr(className="my-2"),
        html.P("Portfolio Financial Impact", className="fw-semibold small mb-1"),
        dbc.ListGroup([
            dbc.ListGroupItem([html.Span("Total Exposure",   className="text-muted small"), html.Strong(f"${FRAUD_AMT_TOTAL:,.0f}")], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("Estimated Caught", className="text-muted small"), html.Strong(f"${ESTIMATED_CAUGHT:,.0f}", style={"color": FOREST})], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("Estimated Missed", className="text-muted small"), html.Strong(f"${ESTIMATED_MISSED:,.0f}", style={"color": CRIMSON})], className="d-flex justify-content-between py-1"),
            dbc.ListGroupItem([html.Span("Est. False Positives", className="text-muted small"), html.Strong(f"{ESTIMATED_FP:,} tx")], className="d-flex justify-content-between py-1"),
        ], flush=True, className="small"),
    ]),
], className="shadow-sm h-100")

tab_models = dbc.Tab(label="Detection & ROI", tab_id="tab-models", children=[
    dbc.Row([
        dbc.Col(_card("Model Comparison AUC-ROC | AUC-PR | F1",         "bar-model-compare",  340), md=7),
        dbc.Col(_roi_card, md=5),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Fraud $ Recovery Waterfall (GBT Model)",            "waterfall-roi",      300), md=6),
        dbc.Col(_card("Anomaly Detection Flagged Transactions by Method","bar-anomaly",        300), md=6),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Key Risk Signals Fraud Detection Feature Importance",  "bar-fraud-fi", 300), md=6),
        dbc.Col(_card("Category Classifier Feature Importance",                 "bar-cat-fi",   300), md=6),
    ], className="g-3"),
])

#  TAB 5 HIGH-RISK MONITORING 
tab_monitoring = dbc.Tab(label="High-Risk Monitoring", tab_id="tab-monitoring", children=[
    dbc.Row([
        dbc.Col(_card("Top 15 High-Risk Merchants Fraud Rate & Exposure ($)", "bar-top-merch",    320), md=12),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Merchant Fraud Rate Distribution",                        "hist-merch-fraud", 280), md=5),
        dbc.Col(_card("Fraud Rate & Exposure by Time of Day",                    "bar-tod-fraud",    280), md=7),
    ], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(_card("Fraud Exposure ($) by State Top 20",                   "bar-state-exposure",300), md=12),
    ], className="g-3"),
])

#  TAB 6 SYSTEM PERFORMANCE 
_spark_config_table = dbc.Card([
    dbc.CardHeader(html.Strong("Recommended Spark Configuration", className="small")),
    dbc.CardBody(
        dbc.Table([
            html.Thead(html.Tr([html.Th("Config Key"), html.Th("Value"), html.Th("Reason")])),
            html.Tbody([
                html.Tr([html.Td(html.Code("spark.sql.shuffle.partitions")),         html.Td("64"),            html.Td("Tuned via partition benchmark")]),
                html.Tr([html.Td(html.Code("spark.sql.adaptive.enabled")),           html.Td("true"),          html.Td("AQE auto-coalesces post-shuffle partitions")]),
                html.Tr([html.Td(html.Code("spark.sql.autoBroadcastJoinThreshold")), html.Td("20 MB"),         html.Td("Auto-broadcast small lookup tables")]),
                html.Tr([html.Td(html.Code("spark.serializer")),                     html.Td("KryoSerializer"),html.Td("2-10× faster than Java default")]),
                html.Tr([html.Td(html.Code("spark.io.compression.codec")),           html.Td("lz4"),           html.Td("Fast shuffle compression")]),
                html.Tr([html.Td(html.Code("spark.executor.memory")),                html.Td("4g"),            html.Td("Adjust to cluster size")]),
                html.Tr([html.Td(html.Code("spark.executor.memoryOverhead")),        html.Td("512m"),          html.Td("Native + Python off-heap")]),
                html.Tr([html.Td(html.Code("spark.speculation")),                    html.Td("true"),          html.Td("Re-launch straggler tasks")]),
            ]),
        ], bordered=True, hover=True, size="sm", className="small mb-0"),
    ),
], className="shadow-sm h-100")

# tab_spark = dbc.Tab(label="System Performance", tab_id="tab-spark", children=[
#     dbc.Row([
#         dbc.Col(_card("End-to-End Speedup per Optimisation",  "bar-optim-summary", 320), md=8),
#         dbc.Col(_spark_config_table, md=4),
#     ], className="mb-3 g-3"),
#     dbc.Row([
#         dbc.Col(_card("Partition Tuning Wall-Clock Time",   "bar-partition",     280), md=4),
#         dbc.Col(_card("Cache Strategy Comparison",            "bar-cache",         280), md=4),
#         dbc.Col(_card("Sort-Merge vs Broadcast Join",         "bar-join",          280), md=4),
#     ], className="g-3"),
# ])

#  APP LAYOUT 
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Bank Fraud Risk Intelligence",
)

app.layout = dbc.Container([
    # Header
    dbc.Row(dbc.Col(html.Div([
        html.Div([
            html.H3("Bank Fraud Risk Intelligence Dashboard",
                    className="fw-bold mb-0 mt-3", style={"color": NAVY}),
            html.Small(
                f"Portfolio: {TOTAL_TX:,} transactions · {UNIQUE_CARDS:,} cardholders · "
                f"{NUM_STATES} states · {NUM_CATS} spending categories · {DATE_MIN} – {DATE_MAX}",
                className="text-muted",
            ),
        ]),
    ])), className="border-bottom mb-3"),

    # Alert banner (shown when fraud rate exceeds threshold)
    alert_banner,

    # KPIs
    kpi_row,

    # Tabs
    dbc.Tabs([
        tab_overview,
        tab_temporal,
        tab_portfolio,
        tab_models,
        tab_monitoring,
        # tab_spark,
    ], id="tabs", active_tab="tab-overview", className="mb-4"),
], fluid=True, className="px-4 pb-5")


#  CALLBACKS TAB 1: EXECUTIVE OVERVIEW 

@app.callback(Output("pie-fraud", "figure"), Input("tabs", "active_tab"))
def fig_pie_fraud(_):
    fig = go.Figure(go.Pie(
        labels=["Legitimate", "Fraudulent"],
        values=[TOTAL_TX - FRAUD_TX, FRAUD_TX],
        marker_colors=[STEEL, CRIMSON],
        hole=0.55,
        pull=[0, 0.10],
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{FRAUD_RATE:.2f}%</b><br>fraud rate",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=13))
    fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    return fig


@app.callback(Output("bar-cat-volume", "figure"), Input("tabs", "active_tab"))
def fig_cat_volume(_):
    d = cat_agg.sort_values("fraud_rate", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["fraud_rate"], y=d["category"], orientation="h",
        name="Fraud Rate %", marker_color=CRIMSON, opacity=0.85,
        hovertemplate="%{y}: %{x:.2f}% fraud rate<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["avg_amount"], y=d["category"], mode="markers",
        name="Avg Fraud Tx ($)", xaxis="x2",
        marker=dict(color=AMBER, size=10, symbol="diamond"),
        hovertemplate="%{y}: $%{x:.2f} avg<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="Fraud Rate (%)"),
        xaxis2=dict(title="Avg Transaction ($)", overlaying="x", side="top",
                    showgrid=False, range=[0, d["avg_amount"].max() * 1.4]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        barmode="overlay",
    )
    return fig


@app.callback(Output("bar-amount-bin", "figure"), Input("tabs", "active_tab"))
def fig_amount_bin(_):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=amtbin_agg["amount_bin_c"].astype(str), y=amtbin_agg["fraud_rate"],
        name="Fraud Rate %", marker_color=CRIMSON, opacity=0.85,
        text=[f"{v:.2f}%" for v in amtbin_agg["fraud_rate"]], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}% fraud rate<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=amtbin_agg["amount_bin_c"].astype(str),
        y=amtbin_agg["total"],
        name="Tx Volume", yaxis="y2",
        mode="lines+markers", line=dict(color=STEEL, width=2), marker=dict(size=8),
        hovertemplate="%{x}: %{y:,} transactions<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="Transaction Amount Tier",
        yaxis=dict(title="Fraud Rate (%)", color=CRIMSON),
        yaxis2=dict(title="Transaction Count", overlaying="y", side="right",
                    color=STEEL, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("hist-amount", "figure"), Input("tabs", "active_tab"))
def fig_hist_amount(_):
    sample = df.sample(min(60_000, len(df)), random_state=42)
    legit  = sample[sample["is_fraud"] == 0]["amount"]
    fraud  = sample[sample["is_fraud"] == 1]["amount"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=legit, nbinsx=100, name="Legitimate",
                               opacity=0.55, marker_color=STEEL, histnorm="density",
                               hovertemplate="$%{x:.0f}: density %{y:.5f}<extra></extra>"))
    fig.add_trace(go.Histogram(x=fraud, nbinsx=80, name="Fraudulent",
                               opacity=0.85, marker_color=CRIMSON, histnorm="density",
                               hovertemplate="$%{x:.0f}: density %{y:.5f}<extra></extra>"))
    fig.add_vline(x=FRAUD_AMT_AVG, line_dash="dash", line_color=CRIMSON,
                  annotation_text=f"Avg fraud ${FRAUD_AMT_AVG:.0f}",
                  annotation_position="top right")
    fig.update_layout(
        barmode="overlay",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title="Transaction Amount ($)", range=[0, 1000]),
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("bar-monthly", "figure"), Input("tabs", "active_tab"))
def fig_monthly(_):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_agg["year_month"], y=monthly_agg["fraud_amount"],
        name="Fraud Losses ($)", marker_color=CRIMSON, opacity=0.9,
        hovertemplate="%{x}<br>Fraud $: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=monthly_agg["year_month"], y=monthly_agg["total_spend"] - monthly_agg["fraud_amount"],
        name="Legitimate Spend ($)", marker_color=STEEL, opacity=0.5,
        hovertemplate="%{x}<br>Legit $: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_agg["year_month"], y=monthly_agg["fraud_rate"],
        name="Fraud Rate %", yaxis="y2",
        line=dict(color=AMBER, width=2.5, dash="solid"), mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="%{x}<br>Fraud Rate: %{y:.3f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_agg["year_month"], y=monthly_agg["spend_trend"],
        name="Spend Trend", yaxis="y3",
        line=dict(color=FOREST, width=1.5, dash="dash"), mode="lines",
        hovertemplate="%{x}<br>Trend: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        margin=dict(t=10, b=10, l=10, r=110),
        xaxis=dict(title="Month", tickangle=-40),
        yaxis=dict(title="Amount ($)"),
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right",
                    color=AMBER, showgrid=False, anchor="free", position=1.0),
        yaxis3=dict(overlaying="y", side="right", showgrid=False,
                    showticklabels=False, anchor="free", position=0.95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


#  CALLBACKS TAB 2: FRAUD PATTERNS 

@app.callback(Output("heatmap-fraud", "figure"), Input("tabs", "active_tab"))
def fig_heatmap_fraud(_):
    z    = heatmap_pivot.values
    ylab = [DOW_LABELS[i] for i in heatmap_pivot.index]
    xlab = [str(h) for h in heatmap_pivot.columns]
    fig  = go.Figure(go.Heatmap(
        z=z, x=xlab, y=ylab,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Fraud %", thickness=14),
        hovertemplate="Hour %{x}:00, %{y}<br>Fraud Rate: %{z:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=40, l=60, r=10),
        xaxis=dict(title="Hour of Day", dtick=2),
        yaxis=dict(title="Day of Week", autorange="reversed"),
    )
    return fig


@app.callback(Output("bar-dist-fraud", "figure"), Input("tabs", "active_tab"))
def fig_dist_fraud(_):
    if dist_agg.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dist_agg["dist_bin"].astype(str), y=dist_agg["fraud_rate"],
        name="Fraud Rate %", marker_color=CRIMSON, opacity=0.85,
        text=[f"{v:.2f}%" for v in dist_agg["fraud_rate"]], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dist_agg["dist_bin"].astype(str), y=dist_agg["avg_amount"],
        name="Avg Tx ($)", yaxis="y2",
        line=dict(color=STEEL, width=2), mode="lines+markers", marker=dict(size=8),
        hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=30, l=10, r=60),
        xaxis=dict(title="Card–Merchant Distance", tickangle=-20),
        yaxis=dict(title="Fraud Rate (%)"),
        yaxis2=dict(title="Avg Amount ($)", overlaying="y", side="right",
                    color=STEEL, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("line-hour-fraud", "figure"), Input("tabs", "active_tab"))
def fig_hour_fraud(_):
    # Highlight late-night hours (23–5) as elevated risk window
    fig = go.Figure()
    fig.add_vrect(x0=22.5, x1=23.5, fillcolor=CRIMSON, opacity=0.07, line_width=0)
    fig.add_vrect(x0=-0.5, x1=5.5,  fillcolor=CRIMSON, opacity=0.07, line_width=0,
                  annotation_text="High-risk hours", annotation_position="top left",
                  annotation_font=dict(color=CRIMSON, size=10))
    fig.add_trace(go.Bar(
        x=hour_agg["tx_hour"], y=hour_agg["total"],
        name="Tx Volume", marker_color=STEEL, opacity=0.40,
        hovertemplate="Hour %{x}: %{y:,} tx<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hour_agg["tx_hour"], y=hour_agg["fraud_rate"],
        name="Fraud Rate %", yaxis="y2",
        line=dict(color=CRIMSON, width=2.5), mode="lines+markers", marker=dict(size=7),
        hovertemplate="Hour %{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Transaction Count", color=STEEL),
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right",
                    color=CRIMSON, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("bar-dow", "figure"), Input("tabs", "active_tab"))
def fig_dow(_):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dow_agg["day_name"], y=dow_agg["total"],
        name="Tx Volume", marker_color=STEEL, opacity=0.55,
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dow_agg["day_name"], y=dow_agg["fraud_rate"],
        name="Fraud Rate %", yaxis="y2",
        line=dict(color=CRIMSON, width=2), mode="lines+markers", marker=dict(size=9),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="Day of Week", yaxis_title="Transaction Count",
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right",
                    color=CRIMSON, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("bar-state-fraud", "figure"), Input("tabs", "active_tab"))
def fig_state_fraud(_):
    d = state_agg.head(20)
    fig = go.Figure(go.Bar(
        x=d["state"], y=d["fraud_rate"],
        marker=dict(color=d["fraud_rate"], colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Fraud %", thickness=12)),
        text=[f"{v:.2f}%" for v in d["fraud_rate"]], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}% fraud rate<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="State", tickangle=-45),
        yaxis_title="Fraud Rate (%)",
    )
    return fig


@app.callback(Output("bar-gender", "figure"), Input("tabs", "active_tab"))
def fig_gender(_):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gender_agg["gender"], y=gender_agg["fraud_rate"],
        name="Fraud Rate %", marker_color=CRIMSON, opacity=0.85,
        text=[f"{v:.3f}%" for v in gender_agg["fraud_rate"]], textposition="outside",
        hovertemplate="%{x}: %{y:.3f}% fraud rate<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=gender_agg["gender"], y=gender_agg["avg_amount"],
        name="Avg Tx ($)", yaxis="y2", mode="markers",
        marker=dict(color=STEEL, size=16, symbol="diamond"),
        hovertemplate="%{x}: $%{y:.2f} avg<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="Gender",
        yaxis=dict(title="Fraud Rate (%)"),
        yaxis2=dict(title="Avg Transaction ($)", overlaying="y", side="right",
                    color=STEEL, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


#  CALLBACKS TAB 3: CUSTOMER PORTFOLIO 

@app.callback(Output("line-silhouette", "figure"), Input("tabs", "active_tab"))
def fig_silhouette(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=SILHOUETTE_SCORES["K"], y=SILHOUETTE_SCORES["Silhouette"],
        mode="lines+markers+text",
        line=dict(color=NAVY, width=2.5),
        marker=dict(size=10, color=SILHOUETTE_SCORES["Silhouette"],
                    colorscale="Blues", showscale=False),
        text=[f"{v:.3f}" for v in SILHOUETTE_SCORES["Silhouette"]],
        textposition="top center",
        hovertemplate="K=%{x}: silhouette=%{y:.4f}<extra></extra>",
    ))
    fig.add_vline(x=5, line_dash="dash", line_color=CRIMSON,
                  annotation_text="Optimal K=5", annotation_position="top right",
                  annotation_font=dict(color=CRIMSON))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title="Number of Segments (K)", tickmode="linear", dtick=1),
        yaxis_title="Silhouette Score",
    )
    return fig


@app.callback(Output("bar-cluster-profiles", "figure"), Input("tabs", "active_tab"))
def fig_cluster_profiles(_):
    fig = go.Figure()
    for i, (_, row) in enumerate(CLUSTER_PROFILES.iterrows()):
        fig.add_trace(go.Bar(
            x=[row["Label"]], y=[row["avg_amount"]],
            name=f"Seg {i}", marker_color=CLUSTER_COLORS[i], opacity=0.85,
            text=[f"${row['avg_amount']:.0f}"], textposition="outside",
            hovertemplate=(
                f"Segment {i}: {row['Label'].replace(chr(10),' ')}<br>"
                f"Avg Spend: ${row['avg_amount']:.0f}<br>"
                f"Fraud Rate: {row['fraud_rate_pct']:.2f}%<br>"
                f"Cardholders: {row['card_count']}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=CLUSTER_PROFILES["Label"], y=CLUSTER_PROFILES["fraud_rate_pct"],
        name="Fraud Rate %", yaxis="y2", mode="lines+markers",
        line=dict(color=CRIMSON, width=2), marker=dict(size=10, color=CRIMSON),
        text=[f"{v:.2f}%" for v in CLUSTER_PROFILES["fraud_rate_pct"]],
        textposition="top center",
        hovertemplate="%{x}: %{y:.2f}% fraud<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="Customer Segment",
        yaxis=dict(title="Avg Transaction Amount ($)"),
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right",
                    color=CRIMSON, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("bar-cluster-exposure", "figure"), Input("tabs", "active_tab"))
def fig_cluster_exposure(_):
    d = CLUSTER_PROFILES.sort_values("fraud_exposure", ascending=False)
    fig = go.Figure(go.Bar(
        x=d["Label"], y=d["fraud_exposure"],
        marker_color=CLUSTER_COLORS[:len(d)], opacity=0.85,
        text=[f"${v:,.0f}" for v in d["fraud_exposure"]], textposition="outside",
        hovertemplate="%{x}<br>Fraud Exposure: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Customer Segment",
        yaxis_title="Estimated Fraud Exposure ($)",
    )
    return fig


@app.callback(Output("bar-cluster-size", "figure"), Input("tabs", "active_tab"))
def fig_cluster_size(_):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=CLUSTER_PROFILES["Label"], y=CLUSTER_PROFILES["card_count"],
        name="Cardholders", marker_color=CLUSTER_COLORS, opacity=0.85,
        text=CLUSTER_PROFILES["card_count"], textposition="outside",
        hovertemplate="%{x}<br>Cards: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=CLUSTER_PROFILES["Label"], y=CLUSTER_PROFILES["age"],
        name="Avg Age", yaxis="y2", mode="markers",
        marker=dict(color=AMBER, size=14, symbol="diamond"),
        hovertemplate="%{x}<br>Avg Age: %{y}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="Customer Segment",
        yaxis=dict(title="Number of Cardholders"),
        yaxis2=dict(title="Avg Age", overlaying="y", side="right",
                    color=AMBER, showgrid=False, range=[0, 80]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("bar-age-fraud", "figure"), Input("tabs", "active_tab"))
def fig_age_fraud(_):
    if age_agg.empty:
        return go.Figure(layout=go.Layout(title="Age data not available"))
    fig = go.Figure(go.Bar(
        x=age_agg["age_group"].astype(str), y=age_agg["fraud_rate"],
        marker_color=CRIMSON, opacity=0.85,
        text=[f"{v:.2f}%" for v in age_agg["fraud_rate"]], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}% fraud rate<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Age Group", yaxis_title="Fraud Rate (%)",
    )
    return fig


@app.callback(Output("bar-age-amount", "figure"), Input("tabs", "active_tab"))
def fig_age_amount(_):
    if age_agg.empty:
        return go.Figure(layout=go.Layout(title="Age data not available"))
    fig = go.Figure(go.Bar(
        x=age_agg["age_group"].astype(str), y=age_agg["avg_amount"],
        marker_color=NAVY, opacity=0.85,
        text=[f"${v:.0f}" for v in age_agg["avg_amount"]], textposition="outside",
        hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Age Group", yaxis_title="Avg Transaction Amount ($)",
    )
    return fig


@app.callback(Output("hist-card-vel", "figure"), Input("tabs", "active_tab"))
def fig_card_vel(_):
    p99 = int(card_agg["tx_count"].quantile(0.99))
    high_risk_cards = int((card_agg["tx_count"] >= p99).sum())
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=card_agg["tx_count"], nbinsx=60,
        marker_color=STEEL, opacity=0.8, name="Cards",
        hovertemplate="Tx/Card: %{x}<br>Cards: %{y}<extra></extra>",
    ))
    fig.add_vline(x=p99, line_dash="dash", line_color=CRIMSON,
                  annotation_text=f"P99={p99:,} {high_risk_cards} high-velocity cards",
                  annotation_position="top right",
                  annotation_font=dict(color=CRIMSON, size=10))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Transactions per Card", yaxis_title="Number of Cardholders",
    )
    return fig


#  CALLBACKS TAB 4: DETECTION & ROI 

@app.callback(Output("bar-model-compare", "figure"), Input("tabs", "active_tab"))
def fig_model_compare(_):
    metrics = ["AUC-ROC", "AUC-PR", "F1 Score"]
    colors  = [NAVY, FOREST, STEEL]
    fig = go.Figure()
    for i, model in enumerate(MODEL_METRICS["Model"]):
        vals = [MODEL_METRICS.loc[i, m] for m in metrics]
        fig.add_trace(go.Bar(
            name=model, x=metrics, y=vals,
            marker_color=colors[i], opacity=0.85,
            text=[f"{v:.4f}" for v in vals], textposition="outside",
            hovertemplate=f"{model}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(title="Score", range=[0, 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("waterfall-roi", "figure"), Input("tabs", "active_tab"))
def fig_waterfall_roi(_):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Total Fraud\nExposure", "Caught by GBT\nModel", "Missed by\nModel", "Net Uncovered\nLoss"],
        y=[FRAUD_AMT_TOTAL, -ESTIMATED_CAUGHT, ESTIMATED_MISSED, 0],
        text=[f"${FRAUD_AMT_TOTAL:,.0f}", f"-${ESTIMATED_CAUGHT:,.0f}",
              f"+${ESTIMATED_MISSED:,.0f}", f"${ESTIMATED_MISSED:,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color=SLATE, width=1, dash="dot")),
        increasing=dict(marker=dict(color=CRIMSON)),
        decreasing=dict(marker=dict(color=FOREST)),
        totals=dict(marker=dict(color=AMBER)),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=20, b=10, l=10, r=10),
        yaxis_title="Fraud Amount ($)",
        showlegend=False,
    )
    return fig


@app.callback(Output("bar-anomaly", "figure"), Input("tabs", "active_tab"))
def fig_anomaly(_):
    color_map = {"Outlier": AMBER, "Fraud": CRIMSON}
    fig = go.Figure(go.Bar(
        x=ANOMALY_DF["Method"], y=ANOMALY_DF["Count"],
        marker_color=[color_map[t] for t in ANOMALY_DF["Type"]],
        opacity=0.85,
        text=[f"{v:,}" for v in ANOMALY_DF["Count"]], textposition="outside",
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="Flagged Transactions",
        xaxis_title="Detection Method",
    )
    return fig


@app.callback(Output("bar-fraud-fi", "figure"), Input("tabs", "active_tab"))
def fig_fraud_fi(_):
    fig = go.Figure(go.Bar(
        x=FRAUD_FI.values, y=FRAUD_FI.index, orientation="h",
        marker=dict(color=FRAUD_FI.values, colorscale="Blues", showscale=False),
        text=[f"{v:.3f}" for v in FRAUD_FI.values], textposition="outside",
        hovertemplate="%{y}: importance %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="Feature Importance", range=[0, FRAUD_FI.max() * 1.25]),
        yaxis_title="",
    )
    return fig


@app.callback(Output("bar-cat-fi", "figure"), Input("tabs", "active_tab"))
def fig_cat_fi(_):
    fig = go.Figure(go.Bar(
        x=CAT_FI.values, y=CAT_FI.index, orientation="h",
        marker=dict(color=CAT_FI.values, colorscale="Oranges", showscale=False),
        text=[f"{v:.3f}" for v in CAT_FI.values], textposition="outside",
        hovertemplate="%{y}: importance %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="Feature Importance", range=[0, CAT_FI.max() * 1.25]),
        yaxis_title="",
    )
    return fig


#  CALLBACKS TAB 5: HIGH-RISK MONITORING 

@app.callback(Output("bar-top-merch", "figure"), Input("tabs", "active_tab"))
def fig_top_merch(_):
    d = merch_agg.head(15).copy()
    d["merchant_short"] = d["merchant"].str.replace("fraud_", "", regex=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["fraud_rate"], y=d["merchant_short"], orientation="h",
        name="Fraud Rate %",
        marker=dict(color=d["fraud_rate"], colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Fraud %", thickness=12)),
        hovertemplate="%{y}<br>Fraud Rate: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["fraud_amount"], y=d["merchant_short"],
        name="Fraud $ Losses", xaxis="x2", mode="markers",
        marker=dict(color=AMBER, size=10, symbol="diamond"),
        hovertemplate="%{y}<br>Fraud $: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(title="Fraud Rate (%)"),
        xaxis2=dict(title="Fraud Losses ($)", overlaying="x", side="top",
                    showgrid=False),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0),
    )
    return fig


@app.callback(Output("hist-merch-fraud", "figure"), Input("tabs", "active_tab"))
def fig_merch_fraud(_):
    avg_fr = merch_agg["fraud_rate"].mean()
    p90_fr = merch_agg["fraud_rate"].quantile(0.90)
    fig = go.Figure(go.Histogram(
        x=merch_agg["fraud_rate"], nbinsx=40,
        marker_color=STEEL, opacity=0.8,
        hovertemplate="Fraud Rate: %{x:.1f}%<br>Merchants: %{y}<extra></extra>",
    ))
    fig.add_vline(x=avg_fr, line_dash="dash", line_color=AMBER,
                  annotation_text=f"Avg {avg_fr:.2f}%", annotation_position="top right",
                  annotation_font=dict(color=AMBER))
    fig.add_vline(x=p90_fr, line_dash="dot", line_color=CRIMSON,
                  annotation_text=f"P90 {p90_fr:.2f}%", annotation_position="top left",
                  annotation_font=dict(color=CRIMSON))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Merchant Fraud Rate (%)", yaxis_title="Number of Merchants",
    )
    return fig


@app.callback(Output("bar-tod-fraud", "figure"), Input("tabs", "active_tab"))
def fig_tod_fraud(_):
    if tod_agg.empty:
        return go.Figure(layout=go.Layout(title="Time-of-day data not available"))
    order = ["night", "morning", "afternoon", "evening"]
    d = tod_agg.copy()
    d["tx_time_of_day"] = pd.Categorical(d["tx_time_of_day"], categories=order, ordered=True)
    d = d.sort_values("tx_time_of_day").dropna(subset=["tx_time_of_day"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["tx_time_of_day"].astype(str), y=d["total"],
        name="Tx Volume", marker_color=STEEL, opacity=0.50,
        hovertemplate="%{x}: %{y:,} tx<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["tx_time_of_day"].astype(str), y=d["fraud_rate"],
        name="Fraud Rate %", yaxis="y2",
        mode="lines+markers", line=dict(color=CRIMSON, width=2.5), marker=dict(size=12),
        text=[f"{v:.2f}%" for v in d["fraud_rate"]], textposition="top center",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="Time of Day",
        yaxis_title="Transaction Count",
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right",
                    color=CRIMSON, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(Output("bar-state-exposure", "figure"), Input("tabs", "active_tab"))
def fig_state_exposure(_):
    d = state_agg.assign(
        fraud_exposure=lambda x: x["fraud_amount"]
    ).sort_values("fraud_exposure", ascending=True).tail(20)
    fig = go.Figure(go.Bar(
        x=d["fraud_exposure"], y=d["state"], orientation="h",
        marker=dict(color=d["fraud_rate"], colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Fraud %", thickness=12)),
        text=[f"${v:,.0f}  ({r:.2f}%)" for v, r in zip(d["fraud_exposure"], d["fraud_rate"])],
        textposition="outside",
        hovertemplate="%{y}<br>Exposure: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=120),
        xaxis_title="Fraud Exposure ($)", yaxis_title="",
    )
    return fig


# #  CALLBACKS TAB 6: SYSTEM PERFORMANCE 

# @app.callback(Output("bar-optim-summary", "figure"), Input("tabs", "active_tab"))
# def fig_optim_summary(_):
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=OPTIM_SUMMARY["Optimisation"], y=OPTIM_SUMMARY["Baseline (s)"],
#         name="Baseline", marker_color=CRIMSON, opacity=0.8,
#         hovertemplate="%{x}<br>Baseline: %{y:.2f}s<extra></extra>",
#     ))
#     fig.add_trace(go.Bar(
#         x=OPTIM_SUMMARY["Optimisation"], y=OPTIM_SUMMARY["Optimised (s)"],
#         name="Optimised", marker_color=FOREST, opacity=0.85,
#         hovertemplate="%{x}<br>Optimised: %{y:.2f}s<extra></extra>",
#     ))
#     fig.add_trace(go.Scatter(
#         x=OPTIM_SUMMARY["Optimisation"], y=OPTIM_SUMMARY["Speedup"],
#         name="Speedup", yaxis="y2", mode="lines+markers+text",
#         line=dict(color=AMBER, width=2.5),
#         marker=dict(size=9),
#         text=[f"{v:.1f}×" for v in OPTIM_SUMMARY["Speedup"]],
#         textposition="top center",
#         hovertemplate="%{x}: %{y:.2f}×<extra></extra>",
#     ))
#     fig.update_layout(
#         barmode="group",
#         margin=dict(t=10, b=10, l=10, r=60),
#         xaxis_title="", yaxis_title="Wall-Clock Time (s)",
#         yaxis2=dict(title="Speedup (×)", overlaying="y", side="right",
#                     color=AMBER, showgrid=False),
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     )
#     return fig


# @app.callback(Output("bar-partition", "figure"), Input("tabs", "active_tab"))
# def fig_partition(_):
#     colors = [CRIMSON if b else NAVY for b in PARTITION_BENCH["Best"]]
#     fig = go.Figure(go.Bar(
#         x=PARTITION_BENCH["Partitions"], y=PARTITION_BENCH["Time (s)"],
#         marker_color=colors, opacity=0.85,
#         text=[f"{v:.2f}s" for v in PARTITION_BENCH["Time (s)"]], textposition="outside",
#         hovertemplate="Partitions=%{x}: %{y:.2f}s<extra></extra>",
#     ))
#     fig.add_annotation(
#         x="64", y=PARTITION_BENCH[PARTITION_BENCH["Partitions"] == "64"]["Time (s)"].values[0] + 0.3,
#         text="Optimal", showarrow=True, arrowhead=2, arrowcolor=CRIMSON,
#         font=dict(color=CRIMSON, size=11),
#     )
#     fig.update_layout(
#         margin=dict(t=10, b=10, l=10, r=10),
#         xaxis_title="Shuffle Partitions", yaxis_title="Wall-Clock Time (s)",
#     )
#     return fig


# @app.callback(Output("bar-cache", "figure"), Input("tabs", "active_tab"))
# def fig_cache(_):
#     colors = [CRIMSON, FOREST, NAVY]
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=CACHE_BENCH["Strategy"], y=CACHE_BENCH["Time (s)"],
#         marker_color=colors, opacity=0.85,
#         text=[f"{v:.2f}s" for v in CACHE_BENCH["Time (s)"]], textposition="outside",
#         hovertemplate="%{x}: %{y:.2f}s<extra></extra>", showlegend=False,
#     ))
#     fig.add_trace(go.Scatter(
#         x=CACHE_BENCH["Strategy"], y=CACHE_BENCH["Speedup"],
#         name="Speedup", yaxis="y2", mode="markers",
#         marker=dict(color=AMBER, size=14, symbol="diamond"),
#         hovertemplate="%{x}: %{y:.1f}×<extra></extra>",
#     ))
#     fig.update_layout(
#         margin=dict(t=10, b=10, l=10, r=60),
#         xaxis_title="Cache Strategy", yaxis_title="Time 2 Actions (s)",
#         yaxis2=dict(title="Speedup (×)", overlaying="y", side="right",
#                     color=AMBER, showgrid=False),
#         showlegend=False,
#     )
#     return fig


# @app.callback(Output("bar-join", "figure"), Input("tabs", "active_tab"))
# def fig_join(_):
#     speedup = JOIN_BENCH["Time (s)"].iloc[0] / JOIN_BENCH["Time (s)"].iloc[1]
#     fig = go.Figure(go.Bar(
#         x=JOIN_BENCH["Strategy"], y=JOIN_BENCH["Time (s)"],
#         marker_color=[CRIMSON, FOREST], opacity=0.85,
#         text=[f"{v:.2f}s" for v in JOIN_BENCH["Time (s)"]], textposition="outside",
#         hovertemplate="%{x}: %{y:.2f}s<extra></extra>",
#     ))
#     fig.add_annotation(
#         x=1, y=JOIN_BENCH["Time (s)"].iloc[1] + 0.4,
#         text=f"{speedup:.1f}× faster",
#         showarrow=False, font=dict(color=FOREST, size=12, weight="bold"),
#     )
#     fig.update_layout(
#         margin=dict(t=10, b=10, l=10, r=10),
#         xaxis_title="Join Strategy", yaxis_title="Wall-Clock Time (s)",
#     )
#     return fig


#  RUN 
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
