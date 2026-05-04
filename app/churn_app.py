import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jonalights Customer Churn Analyser",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.block-container { padding-top: 1.8rem; }

/* KPI cards */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    flex: 1;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 4px solid;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    color: #f1f5f9;
}
.kpi-card .label { font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; }
.kpi-card .value { font-size: 2rem; font-weight: 800; margin-top: 0.2rem; }
.kpi-card .delta { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; font-family: 'DM Mono', monospace; }
.accent-orange { border-left-color: #f97316; }
.accent-blue   { border-left-color: #3b82f6; }
.accent-green  { border-left-color: #22c55e; }
.accent-purple { border-left-color: #a855f7; }

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.5rem 0 0.7rem;
    padding-left: 0.5rem;
    border-left: 3px solid #f97316;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    st.markdown("---")
    model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Both"])
    test_size    = st.slider("Test Split %", 10, 40, 20)
    n_estimators = st.slider("RF Trees (if used)", 50, 300, 100, step=50)
    st.markdown("---")
    st.markdown("**Jonathan Shoyemi Showunmi**  \n*Customer Churn Analyser v1.0*")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(src):
    return pd.read_csv(src)

if uploaded:
    df = load_data(uploaded)
else:
    try:
        df = load_data("customer_churn_dataset.csv")
    except FileNotFoundError:
        st.error("Please upload the dataset using the sidebar.")
        st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.4rem; font-weight:800; margin-bottom:0.2rem;'>
  📡 Customer Churn Analyser
</h1>
<p style='color:#64748b; font-family:DM Mono,monospace; font-size:0.9rem; margin-bottom:1.5rem;'>
  Predict & Understand Customer Attrition · 64,374 Records · 12 Features
</p>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 EDA", "🤖 Models", "🎯 Predict"])

NUM_COLS = ["Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Total Spend", "Last Interaction"]
CAT_COLS = ["Gender", "Subscription Type", "Contract Length"]

# ════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════
with tab1:
    churn_rate = df["Churn"].mean()
    churned    = df["Churn"].sum()
    retained   = len(df) - churned

    st.markdown("""<div class='kpi-grid'>
      <div class='kpi-card accent-orange'>
        <div class='label'>Total Customers</div>
        <div class='value'>{:,}</div>
        <div class='delta'>Full dataset</div>
      </div>
      <div class='kpi-card accent-blue'>
        <div class='label'>Churned</div>
        <div class='value'>{:,}</div>
        <div class='delta'>{:.1%} of total</div>
      </div>
      <div class='kpi-card accent-green'>
        <div class='label'>Retained</div>
        <div class='value'>{:,}</div>
        <div class='delta'>{:.1%} of total</div>
      </div>
      <div class='kpi-card accent-purple'>
        <div class='label'>Features</div>
        <div class='value'>11</div>
        <div class='delta'>0 missing values</div>
      </div>
    </div>""".format(len(df), churned, churn_rate, retained, 1 - churn_rate),
    unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Dataset Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=320)
    with col2:
        st.markdown("<div class='section-title'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df[NUM_COLS].describe().round(2), use_container_width=True, height=320)

# ════════════════════════════════════════════════════════════════════════
# TAB 2: EDA
# ════════════════════════════════════════════════════════════════════════
with tab2:
    palette = ["#3b82f6", "#f97316"]
    sns.set_theme(style="darkgrid")

    # Churn distribution
    st.markdown("<div class='section-title'>Churn Distribution</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0f172a")
    for ax in axes: ax.set_facecolor("#1e293b")
    counts = df["Churn"].value_counts()
    axes[0].pie(counts, labels=["No Churn", "Churned"], autopct="%1.1f%%",
                colors=palette, startangle=90, textprops={"color": "white"})
    axes[0].set_title("Split", color="white", fontweight="bold")
    sns.countplot(x="Churn", data=df, ax=axes[1], palette=palette)
    axes[1].set_xticklabels(["No Churn", "Churned"], color="white")
    axes[1].tick_params(colors="white")
    axes[1].set_title("Counts", color="white", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar in axes[1].patches:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                     f"{int(bar.get_height()):,}", ha="center", color="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Numeric feature histograms
    st.markdown("<div class='section-title'>Feature Distributions by Churn</div>", unsafe_allow_html=True)
    selected_feat = st.selectbox("Select feature", NUM_COLS)
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    for val, color, lbl in [(0, "#3b82f6", "No Churn"), (1, "#f97316", "Churned")]:
        ax.hist(df[df["Churn"] == val][selected_feat], bins=35,
                alpha=0.65, color=color, label=lbl, density=True)
    ax.set_title(f"{selected_feat} Distribution", color="white", fontweight="bold")
    ax.legend(facecolor="#1e293b", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Categorical churn rates
    st.markdown("<div class='section-title'>Churn Rate by Category</div>", unsafe_allow_html=True)
    cat_choice = st.selectbox("Category", CAT_COLS)
    rate = df.groupby(cat_choice)["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    bars = ax.bar(rate.index, rate.values, color="#f97316", edgecolor="#1e293b")
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", color="white", fontsize=10)
    ax.set_ylabel("Churn Rate", color="white")
    ax.set_title(f"Churn Rate by {cat_choice}", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Correlation heatmap
    st.markdown("<div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    corr = df[NUM_COLS + ["Churn"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"color": "white", "size": 9})
    ax.tick_params(colors="white")
    ax.set_title("Pearson Correlation", color="white", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ════════════════════════════════════════════════════════════════════════
# TAB 3: MODELS
# ════════════════════════════════════════════════════════════════════════
with tab3:
    @st.cache_resource
    def train_models(test_sz, n_est):
        df_m = df.drop(columns=["CustomerID"]).copy()
        le_enc = LabelEncoder()
        for col in CAT_COLS:
            df_m[col] = le_enc.fit_transform(df_m[col])

        X = df_m.drop(columns=["Churn"])
        y = df_m["Churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_sz / 100, random_state=42, stratify=y
        )
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc  = sc.transform(X_te)

        lr  = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_sc, y_tr)
        rf  = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)

        return lr, rf, sc, le_enc, X_tr_sc, X_te_sc, X_tr, X_te, y_tr, y_te, X.columns.tolist()

    with st.spinner("Training models…"):
        lr, rf, scaler, le_enc, X_tr_sc, X_te_sc, X_tr, X_te, y_tr, y_te, feat_names = \
            train_models(test_size, n_estimators)

    lr_pred  = lr.predict(X_te_sc);  lr_proba = lr.predict_proba(X_te_sc)[:, 1]
    rf_pred  = rf.predict(X_te);     rf_proba = rf.predict_proba(X_te)[:, 1]
    lr_auc   = roc_auc_score(y_te, lr_proba)
    rf_auc   = roc_auc_score(y_te, rf_proba)
    lr_acc   = (lr_pred == y_te).mean()
    rf_acc   = (rf_pred == y_te).mean()

    st.markdown(f"""<div class='kpi-grid'>
      <div class='kpi-card accent-blue'>
        <div class='label'>LR Accuracy</div>
        <div class='value'>{lr_acc:.1%}</div>
        <div class='delta'>ROC-AUC: {lr_auc:.4f}</div>
      </div>
      <div class='kpi-card accent-orange'>
        <div class='label'>RF Accuracy</div>
        <div class='value'>{rf_acc:.1%}</div>
        <div class='delta'>ROC-AUC: {rf_auc:.4f}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Confusion matrices
    with col1:
        st.markdown("<div class='section-title'>Confusion Matrices</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0f172a")
        for ax, preds, title in [(axes[0], lr_pred, "Logistic Reg"), (axes[1], rf_pred, "Random Forest")]:
            ax.set_facecolor("#1e293b")
            ConfusionMatrixDisplay(confusion_matrix(y_te, preds),
                                   display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(title, color="white", fontweight="bold")
            ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ROC curves
    with col2:
        st.markdown("<div class='section-title'>ROC Curves</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        for proba, lbl, color in [
            (lr_proba, f"LR  AUC={lr_auc:.3f}", "#3b82f6"),
            (rf_proba, f"RF  AUC={rf_auc:.3f}", "#f97316")
        ]:
            fpr, tpr, _ = roc_curve(y_te, proba)
            ax.plot(fpr, tpr, label=lbl, lw=2, color=color)
        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
        ax.set_xlabel("FPR", color="white"); ax.set_ylabel("TPR", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
        ax.set_title("ROC Curve Comparison", color="white", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Feature importance
    st.markdown("<div class='section-title'>Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    colors = ["#f97316" if v == imp.max() else "#3b82f6" for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_xlabel("Importance", color="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Feature Importances", color="white", fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Classification report
    with st.expander("📋 Full Classification Report — Random Forest"):
        report = classification_report(y_te, rf_pred,
                                        target_names=["No Churn", "Churned"],
                                        output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 4: SINGLE CUSTOMER PREDICTOR
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🎯 Predict Churn for a Single Customer")
    st.markdown("Adjust the sliders and dropdowns to profile a customer, then click **Predict**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age           = st.slider("Age", 18, 65, 35)
        tenure        = st.slider("Tenure (months)", 1, 60, 24)
        usage_freq    = st.slider("Usage Frequency", 1, 30, 15)
        gender        = st.selectbox("Gender", ["Female", "Male"])
    with c2:
        support_calls = st.slider("Support Calls", 0, 10, 3)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 10)
        total_spend   = st.slider("Total Spend ($)", 100, 1000, 400)
    with c3:
        last_interact = st.slider("Last Interaction (days ago)", 1, 30, 10)
        sub_type      = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_len  = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        gender_enc   = 0 if gender == "Female" else 1
        sub_enc      = {"Basic": 0, "Premium": 1, "Standard": 2}[sub_type]
        contract_enc = {"Annual": 0, "Monthly": 1, "Quarterly": 2}[contract_len]

        sample = np.array([[age, gender_enc, tenure, usage_freq,
                            support_calls, payment_delay, sub_enc,
                            contract_enc, total_spend, last_interact]])

        sample_sc = scaler.transform(sample)
        lr_prob  = lr.predict_proba(sample_sc)[0, 1]
        rf_prob  = rf.predict_proba(sample)[0, 1]
        avg_prob = (lr_prob + rf_prob) / 2

        risk_label = "🔴 HIGH RISK" if avg_prob > 0.6 else ("🟡 MEDIUM RISK" if avg_prob > 0.4 else "🟢 LOW RISK")

        st.markdown(f"""
        <div style='background:#0f172a; border:1px solid #1e293b;
                    border-left:5px solid {"#ef4444" if avg_prob > 0.6 else "#eab308" if avg_prob > 0.4 else "#22c55e"};
                    border-radius:12px; padding:1.5rem; margin-top:1rem;'>
          <div style='font-size:1.5rem; font-weight:800; color:white;'>{risk_label}</div>
          <div style='font-size:0.85rem; color:#94a3b8; font-family:DM Mono,monospace; margin-top:0.5rem;'>
            Logistic Regression: {lr_prob:.1%} &nbsp;|&nbsp;
            Random Forest: {rf_prob:.1%} &nbsp;|&nbsp;
            <b>Ensemble: {avg_prob:.1%}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 1.2), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.barh(["Churn Probability"], [avg_prob], color="#ef4444" if avg_prob > 0.6 else "#eab308" if avg_prob > 0.4 else "#22c55e")
        ax.barh(["Churn Probability"], [1 - avg_prob], left=[avg_prob], color="#1e293b")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="white", lw=1, ls="--", alpha=0.5)
        ax.tick_params(colors="white")
        ax.set_title(f"Ensemble Score: {avg_prob:.1%}", color="white", fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
