import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import os
from sklearn.metrics import roc_curve

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="FNB DataQuest 2026 - Credit Risk",
    layout="wide"
)

st.title("FNB DataQuest 2026")
st.subheader("Interpretable Credit Modelling -- EDA Tool")

# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("loan_book.csv")
    return df

df = load_data()

numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'default_flag']

# ══════════════════════════════════════════════
# LOAD MODEL OUTPUTS (once, at top level)
# ══════════════════════════════════════════════
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path   = os.path.join(script_dir, "model_outputs.pkl")

model_outputs_loaded = False
y_test_loaded        = None
y_pred_proba_loaded  = None

if os.path.exists(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            outputs = pickle.load(f)
        y_test_loaded        = outputs["y_test"]
        y_pred_proba_loaded  = outputs["y_pred_proba"]
        model_outputs_loaded = True
    except Exception as e:
        st.warning(f"Could not load model_outputs.pkl: {e}")

# ══════════════════════════════════════════════
# WoE / IV HELPER
# ══════════════════════════════════════════════
def calculate_woe_iv(df, feature, target='default_flag', bins=10):
    temp = df[[feature, target]].dropna()
    try:
        temp['bin'] = pd.qcut(temp[feature], q=bins, duplicates='drop')
    except Exception:
        return None, None
    grouped = temp.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['bad', 'total']
    grouped['good'] = grouped['total'] - grouped['bad']
    grouped['pct_good'] = grouped['good'] / grouped['good'].sum()
    grouped['pct_bad']  = grouped['bad']  / grouped['bad'].sum()
    grouped['woe'] = np.log(
        (grouped['pct_good'] + 0.0001) /
        (grouped['pct_bad']  + 0.0001)
    )
    grouped['iv'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']
    return grouped, grouped['iv'].sum()

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dataset Overview",
    "Univariate Explorer",
    "Bivariate Explorer",
    "WoE & IV Analysis",
    "Research",
    "Model Evaluation",
    "Business Dashboard",
])

# ══════════════════════════════════════════════
# TAB 1: DATASET OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Applicants", f"{len(df):,}")
    with col2:
        defaults = int(df['default_flag'].sum())
        st.metric("Total Defaults", f"{defaults:,}")
    with col3:
        rate = df['default_flag'].mean() * 100
        st.metric("Default Rate", f"{rate:.1f}%")

    if st.checkbox("Show raw data"):
        st.dataframe(df.head(100))

    st.header("Data Quality Report")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig_missing = px.bar(
            x=missing.index,
            y=missing.values,
            labels={'x': 'Feature', 'y': 'Missing Count'},
            title="Missing Values by Feature",
            color=missing.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found in the dataset.")

# ══════════════════════════════════════════════
# TAB 2: UNIVARIATE EXPLORER
# ══════════════════════════════════════════════
with tab2:
    st.header("Univariate Explorer")

    selected_col = st.selectbox(
        "Select a feature to explore:", numeric_cols, key="uni"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overall Distribution")
        fig1 = px.histogram(
            df, x=selected_col, nbins=50,
            title=f"Distribution of {selected_col}"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Distribution by Default Status")
        fig2 = px.histogram(
            df, x=selected_col, color="default_flag",
            nbins=50, barmode="overlay",
            title=f"{selected_col} by Default Status"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: BIVARIATE EXPLORER
# ══════════════════════════════════════════════
with tab3:
    st.header("Bivariate Explorer")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X axis:", numeric_cols, key="x")
    with col2:
        y_col = st.selectbox("Select Y axis:", numeric_cols, key="y")

    fig3 = px.scatter(
        df.sample(2000, random_state=42),
        x=x_col, y=y_col,
        color="default_flag",
        title=f"{x_col} vs {y_col}",
        opacity=0.5
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr().round(2)
    fig4 = px.imshow(
        corr,
        text_auto=True,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4: WoE & IV ANALYSIS
# ══════════════════════════════════════════════
with tab4:
    st.header("Weight of Evidence & Information Value")

    st.markdown("""
**Weight of Evidence (WoE)** measures how much a feature bin separates
good customers (repaid) from bad customers (defaulted).
- **Positive WoE** = bin has more good customers = lower risk
- **Negative WoE** = bin has more bad customers = higher risk

**Information Value (IV)** summarises the total predictive power of a feature.
    """)

    st.subheader("WoE Explorer")
    selected_woe = st.selectbox(
        "Select a feature to see its WoE:",
        numeric_cols,
        key="woe_feature"
    )

    woe_df, iv = calculate_woe_iv(df, selected_woe)

    if woe_df is not None:
        if iv < 0.02:
            label = "Useless predictor"
        elif iv < 0.1:
            label = "Weak predictor"
        elif iv < 0.3:
            label = "Medium predictor"
        elif iv < 0.5:
            label = "Strong predictor"
        else:
            label = "Very strong -- check for leakage!"

        st.metric(
            f"Information Value (IV) for {selected_woe}",
            f"{iv:.4f}", label
        )

        fig_woe = px.bar(
            woe_df.reset_index().assign(
                bin=lambda x: x['bin'].astype(str)
            ),
            x='bin', y='woe',
            title=f"WoE by bin -- {selected_woe}",
            color='woe',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_woe, use_container_width=True)

        st.dataframe(
            woe_df[['good', 'bad', 'total',
                    'pct_good', 'pct_bad', 'woe', 'iv']].round(4)
        )

    st.subheader("IV Rankings -- All Features")
    st.write("This ranks every feature by predictive power:")

    iv_results = []
    for col in numeric_cols:
        _, iv_val = calculate_woe_iv(df, col)
        if iv_val is not None:
            iv_results.append({'Feature': col, 'IV': round(iv_val, 4)})

    iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)

    fig_iv = px.bar(
        iv_df, x='IV', y='Feature',
        orientation='h',
        title="Information Value -- Feature Ranking",
        color='IV',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_iv, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5: RESEARCH
# ══════════════════════════════════════════════
with tab5:
    st.header("Research -- Credit Modelling Concepts")

    st.subheader("1. Generalised Linear Models vs Non-Linear Models")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Logistic Regression (GLM)")
        st.success("""
        The model learns a **single linear decision boundary**:

        **eta = B0 + B1*x1 + B2*x2 + ... + Bn*xn**

        **P(default) = 1 / (1 + e^(-eta))**

        Every prediction traces back to **one equation**
        with fixed, interpretable coefficients.
        The relationship between each feature and the
        outcome is monotonic and additive.
        """)

    with col2:
        st.markdown("### Random Forest / LightGBM")
        st.error("""
        Builds hundreds of decision trees and aggregates:

        **P(default) = (1/B) * sum of Tree_b(x)**

        Each tree recursively partitions feature space
        into regions using nested conditional rules.

        There is **no single equation** -- the prediction
        emerges from hundreds of trees with millions
        of node splits.

        **Why banks cannot use it:**
        - No individual coefficient exists to inspect
        - Cannot explain a decline to a customer
        - Cannot be audited by SARB or NCR
        - "437 out of 500 trees voted default" is not a
          legally defensible reason for credit denial
        """)

    st.markdown("### Direct Comparison")
    comparison_data = {
        "Property": [
            "Decision boundary", "Feature relationships",
            "Parameters", "Explainability",
            "Regulatory compliance", "Performance (this dataset)"
        ],
        "Logistic Regression": [
            "Single linear hyperplane", "Monotonic, additive",
            "p+1 coefficients", "One equation -- fully transparent",
            "Satisfies NCA, Basel III", "AUC = 0.7916"
        ],
        "Random Forest / LightGBM": [
            "Arbitrary non-linear partitions", "Any shape, any interaction",
            "Millions of node splits", "Hundreds of trees -- black box",
            "Cannot satisfy NCA", "AUC = 0.8200 (ceiling)"
        ]
    }
    st.dataframe(pd.DataFrame(comparison_data),
                 use_container_width=True, hide_index=True)

    st.info("""
    **The Cost of Interpretability:**
    Our logistic regression achieves AUC = 0.7916 vs LightGBM's 0.82.
    The gap of 0.0284 is the price of regulatory compliance and customer
    fairness -- and it is worth paying.
    """)

    tradeoff_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree",
                  "Random Forest", "XGBoost / LightGBM", "Neural Network"],
        "Interpretability": [10, 7, 3, 2, 1],
        "Typical Performance": [7, 5, 8, 9, 9],
    })
    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(
        x=tradeoff_data["Interpretability"],
        y=tradeoff_data["Typical Performance"],
        mode='markers+text',
        text=tradeoff_data["Model"],
        textposition="top center",
        marker=dict(size=15, color='teal'),
    ))
    fig_tradeoff.update_layout(
        xaxis_title="Interpretability (10 = fully transparent)",
        yaxis_title="Typical Performance",
        title="The Interpretability-Complexity Trade-off",
        height=400
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    st.divider()

    st.subheader("2. Weight of Evidence & Information Value -- Theory")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Weight of Evidence (WoE)")
        st.markdown("""
        For each bin of a feature, WoE measures how much that bin
        separates good customers from bad customers:

        **WoE = ln(pct_good / pct_bad)**

        | WoE Value | Meaning | Risk Level |
        |---|---|---|
        | > +1.0 | Dominated by good customers | Very low risk |
        | 0 to +1 | More good than bad | Low risk |
        | ~0 | Balanced | Neutral |
        | -0.5 to 0 | More bad than good | Elevated risk |
        | < -1.0 | Dominated by bad customers | Very high risk |

        **From our data (interest_rate, bin 15.87-27.72%):**
        - pct_good = 7.72% of all good customers
        - pct_bad = 22.38% of all bad customers
        - WoE = ln(0.0772 / 0.2238) = **-1.063**
        """)

    with col2:
        st.markdown("### Information Value (IV)")
        st.markdown("""
        IV summarises the **total predictive power** of a feature
        across all its bins:

        **IV = sum of (pct_good - pct_bad) x WoE**

        | IV Score | Predictive Power |
        |---|---|
        | < 0.02 | Useless -- exclude |
        | 0.02 to 0.1 | Weak predictor |
        | 0.1 to 0.3 | Medium predictor |
        | 0.3 to 0.5 | Strong predictor |
        | > 0.5 | Suspiciously strong -- check leakage |

        **From our model:**
        - interest_rate IV = **0.41** (Strong)
        - annual_income IV = **0.41** (Strong)
        - age IV = **0.25** (Medium-Strong)
        - branch_code_id IV = **~0.01** (Excluded)
        """)

    st.markdown("### Why WoE/IV is Valuable in Credit")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("""
        **Handles Non-Linearity**
        Captures non-linear relationships within
        a linear model -- each bin gets its own
        risk weight rather than assuming a
        straight-line relationship.
        """)
    with col2:
        st.success("""
        **Handles Missing Values**
        Missing values become their own bin
        with their own WoE score -- no imputation
        needed. months_since_last_delinquency
        missing = never delinquent = own WoE.
        """)
    with col3:
        st.success("""
        **Aligns with Model Mathematics**
        Negative WoE (high risk bin) + negative
        coefficient = increased eta = increased
        P(default). Encoding and model
        direction align perfectly.
        """)

    st.divider()

    st.subheader("3. Key Metrics in Credit Risk")
    st.warning("""
    **The Accuracy Problem at 15.4% Default Rate**

    A model that predicts "no default" for every single applicant achieves:

    **Accuracy = 102,289 / 120,960 = 84.6%**

    This model has learned nothing and catches zero defaulters.
    Accuracy is not only uninformative -- it is actively misleading
    when classes are imbalanced. A model that approves everyone
    achieves 84.6% accuracy while catching zero defaults.
    """)

    metrics_data = {
        "Metric": ["Accuracy", "AUC", "Gini Coefficient",
                   "KS Statistic", "Precision", "Recall", "F1 Score"],
        "Formula": [
            "(TP + TN) / Total", "P(score(bad) > score(good))",
            "2 x AUC - 1", "max(TPR - FPR)",
            "TP / (TP + FP)", "TP / (TP + FN)", "2 x (P x R) / (P + R)"
        ],
        "Credit Meaning": [
            "MISLEADING at 15.4% default rate",
            "Rank-ordering ability -- gold standard in credit",
            "Industry standard -- our model: 0.5832",
            "Max separation between good/bad distributions",
            "Of approved loans, % that were truly good",
            "Of all defaulters, % correctly identified",
            "Balance between precision and recall"
        ],
        "Our Model": [
            "~86% (misleading)", "0.7916", "0.5832",
            "Calculated from ROC",
            "Threshold dependent", "Threshold dependent", "Threshold dependent"
        ]
    }
    st.dataframe(pd.DataFrame(metrics_data),
                 use_container_width=True, hide_index=True)

    st.markdown("""
    **Why AUC is the correct metric for credit:**

    AUC = P(randomly chosen defaulter scores higher than a randomly
    chosen non-defaulter). Completely **threshold-independent** and
    **class-imbalance-immune.**

    **Gini = 2 x AUC - 1**
    Our model: Gini = 2 x 0.7916 - 1 = **0.5832**
    Industry benchmark: Gini > 0.50 = strong model
    """)

    st.divider()

    st.subheader("4. Regulatory Considerations")
    st.error("""
    Despite the dataset being simulated, the following features
    would attract regulatory scrutiny under South African law
    (National Credit Act, POPIA, FSCA guidelines):
    """)

    reg_data = {
        "Feature": ["age", "region", "branch_code_id"],
        "Risk Type": [
            "Direct discrimination",
            "Proxy discrimination -- Redlining",
            "Proxy discrimination -- Geography"
        ],
        "Regulatory Concern": [
            "Protected characteristic under SA equality law. "
            "Must be justified on actuarial grounds.",
            "In SA, geography correlates with race due to "
            "apartheid-era spatial planning. Using region "
            "proxies race -- constituting redlining, illegal under the NCA.",
            "Branch codes may correlate with township vs suburban "
            "locations -- geographic proxy for race. NCR would scrutinise carefully."
        ],
        "Decision": [
            "Retained with actuarial justification -- IV=0.25 with strong predictive validity",
            "Excluded -- regulatory risk outweighs predictive benefit",
            "Excluded -- IV near zero AND regulatory risk"
        ]
    }
    st.dataframe(pd.DataFrame(reg_data),
                 use_container_width=True, hide_index=True)

    st.info("""
    **The SA Context:**
    South Africa's apartheid history makes location-based features
    particularly sensitive. The NCA and POPIA both require that
    credit decisions cannot be based on characteristics that serve
    as proxies for race, gender, or other protected attributes --
    even indirectly.
    """)

    st.divider()

    st.subheader("5. Why We Choose Interpretability")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **Customer Rights**
        Under the National Credit Act, a declined
        customer has the right to know why.
        Logistic regression provides this.
        Random forest cannot.
        """)
    with col2:
        st.info("""
        **Regulatory Audit**
        SARB and NCR can request a full model
        audit. Every coefficient is inspectable
        and defensible. Black-box models
        cannot be fully audited.
        """)
    with col3:
        st.info("""
        **Model Risk Management**
        Banks must validate and stress-test
        models. A single equation can be
        stress-tested analytically. Black-box
        models require expensive simulation.
        """)

    st.divider()

    st.subheader("How Research Informed Our Model")
    st.success("""
    Every research concept above directly shaped modelling decisions:

    - **GLMs chosen** because NCA requires explainable credit decisions
    - **WoE encoding applied** to all continuous features -- handles
      non-linearity, missing values, and aligns with logistic regression
    - **AUC used not accuracy** -- 15.4% default rate makes accuracy misleading
    - **age retained** with actuarial justification despite regulatory sensitivity
    - **region and branch_code_id excluded** -- regulatory risk outweighs
      predictive benefit

    Result: AUC improved from 0.68 to 0.7916 -- closing 80% of the gap to
    LightGBM while satisfying every regulatory and interpretability constraint.
    """)

# ══════════════════════════════════════════════
# TAB 6: MODEL EVALUATION
# ══════════════════════════════════════════════
with tab6:
    st.header("Model Evaluation")

    if not model_outputs_loaded:
        st.warning(
            "model_outputs.pkl not found. "
            "Run model.py first to enable the interactive ROC curve "
            "and threshold slider. Static results are shown below."
        )

    st.subheader("Final Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test AUC", "0.7916", "+0.1116 vs baseline")
    with col2:
        st.metric("Test Gini", "0.5832", "Strong -- above 0.50")
    with col3:
        st.metric("Baseline AUC", "0.6800", "Starting point")
    with col4:
        st.metric("Gap to LightGBM", "0.0284", "80% closed")

    st.divider()

    st.subheader("Performance Journey")
    journey_data = {
        "Version": [
            "Baseline", "V1 -- WoE Encoding", "V2 -- Calibration",
            "V3 -- New Features", "V4 -- Inquiry WoE", "Final -- Clean Model"
        ],
        "AUC": [0.6800, 0.7872, 0.7876, 0.7883, 0.7914, 0.7916],
        "Improvement": ["--", "+0.1072", "+0.0004",
                        "+0.0007", "+0.0031", "+0.0002"],
        "Key Change": [
            "Starting point",
            "WoE encoding + 8 engineered features",
            "Isotonic calibration for PD accuracy",
            "Hard inquiries + absolute utilisation added",
            "Binary flag upgraded to continuous WoE",
            "Removed 2 noise features -- cleaner model"
        ]
    }
    journey_df = pd.DataFrame(journey_data)
    st.dataframe(journey_df, use_container_width=True, hide_index=True)

    fig_journey = px.line(
        journey_df, x="Version", y="AUC", markers=True,
        title="AUC Improvement Journey",
        color_discrete_sequence=["teal"]
    )
    fig_journey.add_hline(y=0.82, line_dash="dash", line_color="red",
                          annotation_text="LightGBM ceiling (0.82)")
    fig_journey.add_hline(y=0.68, line_dash="dash", line_color="grey",
                          annotation_text="Baseline (0.68)")
    fig_journey.update_layout(height=400)
    st.plotly_chart(fig_journey, use_container_width=True)

    st.divider()

    st.subheader("Final Model Equation")

    st.latex(r"""
        \hat{\eta} = \beta_0 + \sum_{i=1}^{16} \beta_i x_i \qquad
        \hat{P}(\text{default}) = \frac{1}{1 + e^{-\hat{\eta}}}
    """)

    st.markdown("**Intercept:** β₀ = −1.7654")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Behavioural**")
        st.latex(r"""
        \begin{aligned}
        &-1.1152 \cdot x_{\text{hard\_inq}} \\
        &-0.8739 \cdot x_{\text{delinq\_recency}} \\
        &+0.5197 \cdot x_{\text{utilisation}} \\
        &+0.2195 \cdot x_{\text{delinq\_2yr}} \\
        &-0.0996 \cdot x_{\text{ever\_delinq}}
        \end{aligned}
        """)

    with col2:
        st.markdown("**Stability**")
        st.latex(r"""
        \begin{aligned}
        &-0.8400 \cdot x_{\text{age}} \\
        &-0.5024 \cdot x_{\text{employment}} \\
        &-0.4522 \cdot x_{\text{income}} \\
        &-0.4522 \cdot x_{\text{log\_income}} \\
        &-0.1796 \cdot x_{\text{emp\_age\_ratio}}
        \end{aligned}
        """)

    with col3:
        st.markdown("**Debt Structure**")
        st.latex(r"""
        \begin{aligned}
        &-0.2543 \cdot x_{\text{pct\_current}} \\
        &-0.2262 \cdot x_{\text{dti\_rate}} \\
        &-0.1929 \cdot x_{\text{delinq\_severity}} \\
        &+0.1482 \cdot x_{\text{high\_dti\_rate}} \\
        &-0.1331 \cdot x_{\text{interest\_rate}} \\
        &-0.1242 \cdot x_{\text{payment\_burden}}
        \end{aligned}
        """)

    eq_data = {
        "Act": [
            "Behavioural", "Behavioural", "Behavioural", "Behavioural", "Behavioural",
            "Stability", "Stability", "Stability", "Stability", "Stability",
            "Debt Structure", "Debt Structure", "Debt Structure",
            "Debt Structure", "Debt Structure", "Debt Structure"
        ],
        "Feature": [
            "woe_num_hard_inquiries_6mo_raw", "woe_months_since_last_delinquency",
            "woe_absolute_utilisation", "any_delinquency_2yr", "ever_delinquent",
            "woe_age_custom_bin", "woe_employment_length_years",
            "woe_annual_income", "woe_log_income", "woe_employment_age_ratio",
            "woe_pct_accounts_current", "woe_dti_rate_interaction",
            "woe_num_delinquencies_2yr", "high_dti_high_rate",
            "woe_interest_rate", "woe_monthly_payment_burden"
        ],
        "Coefficient": [
            -1.1152, -0.8739, 0.5197, 0.2195, -0.0996,
            -0.8400, -0.5024, -0.4522, -0.4522, -0.1796,
            -0.2543, -0.2262, -0.1929, 0.1482, -0.1331, -0.1242
        ],
        "Business Meaning": [
            "Desperately seeking credit -- strongest signal",
            "Recent payment failures -- recency matters most",
            "Already overextended on credit",
            "Currently struggling with payments",
            "Historical delinquency -- recovered borrower context",
            "Youth = inexperience with debt management",
            "Job stability = resilience to financial shocks",
            "Income = capacity to repay",
            "Normalised income -- handles right skew",
            "Career maturity contextualised by life stage",
            "Proportion of accounts in good standing",
            "Debt burden amplified by interest rate",
            "Severity of recent delinquency behaviour",
            "Danger zone -- high DTI AND high rate",
            "Borrowing cost signal",
            "Monthly payment as % of monthly income"
        ]
    }
    eq_df = pd.DataFrame(eq_data)
    st.dataframe(eq_df, use_container_width=True, hide_index=True)

    st.caption(
        "Model Governance Note: woe_annual_income and woe_log_income share "
        "identical coefficients (-0.4522), suggesting partial redundancy. "
        "The log transformation was retained as income distributions in credit "
        "data are typically right-skewed -- the log transformation dampens "
        "high-income outliers more effectively than the raw feature. "
        "Removing it is noted as a potential simplification for future iterations."
    )

    st.divider()
    # ══════════════════════════════════════════
    # COEFFICIENT CHART
    # WoE-encoded vs Direct flag distinction --
    # coefficient sign alone does not determine
    # risk direction for WoE-encoded features.
    # ══════════════════════════════════════════
    st.subheader("Feature Importance -- Coefficient Magnitude")

    coef_df = pd.DataFrame({
        "Feature":     eq_data["Feature"],
        "Coefficient": eq_data["Coefficient"],
        "Absolute":    [abs(c) for c in eq_data["Coefficient"]],
        "Type":        [
            "WoE-encoded" if f.startswith("woe_") else "Direct flag"
            for f in eq_data["Feature"]
        ]
    }).sort_values("Absolute", ascending=True)

    fig_coef = px.bar(
        coef_df, x="Coefficient", y="Feature", orientation="h",
        color="Type",
        color_discrete_map={
            "WoE-encoded": "#1B6B3A",
            "Direct flag": "#B71C1C"
        },
        title="Model Coefficients -- Sorted by Magnitude"
    )
    fig_coef.update_layout(height=600)
    st.plotly_chart(fig_coef, use_container_width=True)

    st.caption("""
    How to read this chart: For WoE-encoded features (green), the coefficient sign alone
    does not indicate risk direction. WoE values are already signed — negative WoE bins
    represent high-risk segments. The combined effect of (coefficient x WoE value)
    determines the contribution to P(default). For direct binary flags (red), a positive
    coefficient directly increases predicted default probability.
    """)

    st.divider()

    st.subheader("ROC Curve")
    if model_outputs_loaded:
        fpr, tpr, _ = roc_curve(y_test_loaded, y_pred_proba_loaded)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name="Final Model (AUC = 0.7916)",
            line=dict(color="teal", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random baseline (AUC = 0.50)",
            line=dict(color="red", dash="dash")
        ))
        fig_roc.add_hline(y=0.80, line_dash="dot", line_color="grey",
                          annotation_text="80% TPR reference",
                          annotation_position="right")
        fig_roc.add_vline(x=0.20, line_dash="dot", line_color="grey",
                          annotation_text="20% FPR reference",
                          annotation_position="top")
        fig_roc.update_layout(
            title="ROC Curve -- Final Calibrated Credit Model | AUC=0.7916 | Gini=0.5832 | +0.1116 over baseline",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.02]),
            height=500,
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info(
            "**Operating Point:** At FPR=0.20 the model achieves TPR~0.80 -- "
            "correctly identifying 80% of all defaulters while only incorrectly "
            "declining 20% of good customers. This is the model's optimal discrimination "
            "point — the recommended business threshold is PD < 0.24 as discussed "
            "in the Business Dashboard."
        )
    else:
        try:
            st.image("roc_curve_final.png",
                     caption="ROC Curve -- AUC=0.7916 | Gini=0.5832 | +0.1116 over baseline",
                     use_container_width=True)
        except Exception:
            st.error("ROC curve unavailable. Run model.py to generate model_outputs.pkl.")

    st.divider()

    st.subheader("Interactive Threshold Analysis")
    st.markdown(
        "Adjust the decision threshold to see how approval volume, "
        "default capture rate, and false positives change in real time."
    )

    if model_outputs_loaded:
        threshold = st.slider(
            "Decision threshold -- approve if predicted PD is below:",
            min_value=0.05, max_value=0.50, value=0.24, step=0.01,
            help="Lower = stricter policy. Higher = more lenient.",
            key="tab6_slider"
        )

        total      = len(y_test_loaded)
        total_def  = int(y_test_loaded.sum())
        total_good = total - total_def
        approved        = int((y_pred_proba_loaded < threshold).sum())
        declined        = total - approved
        defaults_caught = int(((y_pred_proba_loaded >= threshold) & (y_test_loaded == 1)).sum())
        false_positives = int(((y_pred_proba_loaded >= threshold) & (y_test_loaded == 0)).sum())
        defaults_missed = total_def - defaults_caught
        recall_pct      = defaults_caught / total_def * 100
        precision_val   = (approved - defaults_missed) / approved * 100 if approved > 0 else 0.0
        fp_rate         = false_positives / total_good * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Approved", f"{approved:,}", f"{approved/total*100:.1f}% of applicants")
        col2.metric("Declined", f"{declined:,}", f"{declined/total*100:.1f}% of applicants")
        col3.metric("Defaults Caught", f"{defaults_caught:,}", f"{recall_pct:.1f}% of all defaults")
        col4.metric("Good Customers Declined", f"{false_positives:,}", f"{fp_rate:.1f}% of all good customers")

        col1b, col2b, col3b = st.columns(3)
        col1b.metric("Precision", f"{precision_val:.1f}%", "% approved who will not default")
        col2b.metric("Recall", f"{recall_pct:.1f}%", "% of defaulters correctly declined")
        col3b.metric("Defaults Missed", f"{defaults_missed:,}", "Approved but will default")

        exp_loss = defaults_missed * 0.60 * 100_000
        st.warning(
            f"**Expected Portfolio Loss at threshold {threshold:.2f}:** "
            f"R{exp_loss:,.0f} (assuming avg loan = R100,000, LGD = 60%)"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Precision** = Of all approved loans, what % will repay? "
                    f"At this threshold: **{precision_val:.1f}%** of your approved "
                    f"portfolio are creditworthy. Higher precision = healthier loan book.")
        with col_b:
            st.warning(f"**Recall** = Of all actual defaulters, what % did we catch? "
                       f"At this threshold: **{recall_pct:.1f}%** of defaulters are "
                       f"correctly declined. Lower recall = more defaults slip through.")
    else:
        st.info("Run model.py to generate model_outputs.pkl and enable the threshold slider.")

    st.divider()

    st.subheader("Overfitting Assessment")
    overfit_data = {
        "Dataset":    ["Training", "Test", "Gap"],
        "AUC":        ["0.7901", "0.7916", "0.0015"],
        "Gini":       ["0.5802", "0.5832", "0.0030"],
        "Assessment": [
            "Slightly lower -- calibration deflates train estimates",
            "Final evaluation metric",
            "Negligible -- well within acceptable range (<0.01)"
        ]
    }
    st.dataframe(pd.DataFrame(overfit_data), use_container_width=True, hide_index=True)

    st.info("""
    **Note on Test AUC > Train AUC:**
    The isotonic calibration uses internal cross-validation which slightly deflates
    training probability estimates. This is not overfitting -- it is the calibration
    working correctly. The gap of 0.0015 is negligible.
    """)

    st.divider()

    st.subheader("The Three Submission Claims")
    st.success("""
    **Claim 1 -- Performance:**
    Feature engineering and WoE encoding lifted logistic regression AUC from 0.68
    to 0.7916 -- closing 80% of the gap to LightGBM while preserving full interpretability.
    """)
    st.success("""
    **Claim 2 -- Discovery:**
    The strongest predictor was not income or age, but hard credit inquiries in the
    past 6 months -- a real-time liquidity desperation signal that dominated all
    static demographic features.
    """)
    st.success("""
    **Claim 3 -- Business Value:**
    The calibrated model produces true probabilities suitable for Basel III capital
    provisioning and IFRS 9 expected loss calculation -- not merely rankings, but
    actionable risk estimates.
    """)

# ══════════════════════════════════════════════
# TAB 7: BUSINESS DASHBOARD
# ══════════════════════════════════════════════
with tab7:
    st.header("Business Value Dashboard")
    st.markdown(
        "This dashboard helps business users understand how the model supports "
        "lending decisions. Adjust the policy assumptions to explore volume, "
        "risk, and profitability trade-offs."
    )

    if not model_outputs_loaded:
        st.warning("Run model.py first to enable the Business Dashboard.")
        st.stop()

    total_biz      = len(y_test_loaded)
    total_def_biz  = int(y_test_loaded.sum())
    total_good_biz = total_biz - total_def_biz

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 1: POLICY ASSUMPTIONS
    # ══════════════════════════════════════════
    st.subheader("Step 1 -- Set Your Business Assumptions")
    st.markdown("These assumptions drive all calculations below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_loan = st.number_input(
            "Average loan amount (R)",
            min_value=10_000, max_value=1_000_000,
            value=100_000, step=10_000
        )
    with col2:
        interest_rate_pct = st.number_input(
            "Annual interest rate (%)",
            min_value=5.0, max_value=30.0,
            value=12.0, step=0.5
        )
    with col3:
        lgd_pct = st.number_input(
            "Loss Given Default -- LGD (%)",
            min_value=10, max_value=100,
            value=60, step=5
        )

    lgd = lgd_pct / 100
    annual_revenue_per_loan = avg_loan * (interest_rate_pct / 100)

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 2: APPROVAL STRATEGY SELECTOR
    # ══════════════════════════════════════════
    st.subheader("Step 2 -- Choose Your Approval Strategy")

    strategy = st.radio(
        "Select a lending policy objective:",
        options=[
            "Conservative -- Minimise defaults",
            "Balanced -- Optimise profit",
            "Growth -- Maximise approvals",
            "Custom -- Set my own threshold"
        ],
        horizontal=True
    )

    if strategy == "Conservative -- Minimise defaults":
        threshold_biz = 0.10
        st.info("**Conservative policy:** Approves only the lowest-risk applicants. "
                "Maximises portfolio quality but sacrifices volume.")
    elif strategy == "Balanced -- Optimise profit":
        threshold_biz = 0.24
        st.info("**Balanced policy:** Approves ~80% of applicants while maintaining "
                "91% portfolio quality. Recommended threshold balancing volume, "
                "risk, and long-term customer relationship value.")
    elif strategy == "Growth -- Maximise approvals":
        threshold_biz = 0.35
        st.info("**Growth policy:** Maximises loan volume but accepts higher default risk. "
                "Suitable when market share is the primary objective.")
    else:
        threshold_biz = st.slider(
            "Custom threshold -- approve if PD is below:",
            min_value=0.05, max_value=0.50,
            value=0.18, step=0.01,
            key="biz_custom_slider",
            help=(
                "Default set to 0.18 — the profit-maximising threshold "
                "(Net Revenue: R192.5M). Slide right to increase approval "
                "volume at the cost of higher expected losses. Slide left "
                "to tighten risk acceptance at the cost of lost revenue."
            )
        )

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 3: PORTFOLIO OUTCOMES
    # ══════════════════════════════════════════
    st.subheader("Step 3 -- Portfolio Outcomes at This Policy")

    approved_biz        = int((y_pred_proba_loaded < threshold_biz).sum())
    declined_biz        = total_biz - approved_biz
    defaults_caught_biz = int(((y_pred_proba_loaded >= threshold_biz) & (y_test_loaded == 1)).sum())
    false_pos_biz       = int(((y_pred_proba_loaded >= threshold_biz) & (y_test_loaded == 0)).sum())
    defaults_missed_biz = total_def_biz - defaults_caught_biz
    recall_biz          = defaults_caught_biz / total_def_biz * 100
    precision_biz       = (
        (approved_biz - defaults_missed_biz) / approved_biz * 100
        if approved_biz > 0 else 0.0
    )

    gross_revenue = approved_biz * annual_revenue_per_loan
    expected_loss = defaults_missed_biz * lgd * avg_loan
    net_revenue   = gross_revenue - expected_loss
    loss_rate     = (
        expected_loss / (approved_biz * avg_loan) * 100
        if approved_biz > 0 else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Loans Approved", f"{approved_biz:,}",
                f"{approved_biz/total_biz*100:.1f}% of applicants")
    col2.metric("Defaults Prevented", f"{defaults_caught_biz:,}",
                f"{recall_biz:.1f}% of all defaults")
    col3.metric("Portfolio Quality", f"{precision_biz:.1f}%",
                "% approved who will repay")
    col4.metric("Missed Opportunities", f"{false_pos_biz:,}",
                "Good customers incorrectly declined")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Interest Revenue", f"R{gross_revenue:,.0f}",
                f"R{annual_revenue_per_loan:,.0f} per loan")
    col2.metric("Expected Credit Losses", f"R{expected_loss:,.0f}",
                f"Loss rate: {loss_rate:.2f}%")
    col3.metric(
        "Net Revenue After Losses",
        f"R{net_revenue:,.0f}",
        f"+R{net_revenue/approved_biz:,.0f} per loan" if approved_biz > 0 else "--"
    )

    if net_revenue > 0:
        st.success(
            f"**This policy is profitable.** Net revenue of R{net_revenue:,.0f} "
            f"after absorbing R{expected_loss:,.0f} in expected credit losses."
        )
    else:
        st.error(
            "**This policy is loss-making.** Expected losses exceed interest revenue. "
            "Consider tightening the threshold."
        )

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 4: VOLUME VS RISK TRADE-OFF CURVE
    # ══════════════════════════════════════════
    st.subheader("Volume vs Risk Trade-off -- Full Threshold Sweep")
    st.markdown(
        "This chart shows how approval volume, default rate, and net revenue "
        "change across every possible threshold. Use it to find your optimal policy."
    )

    thresholds = np.arange(0.05, 0.51, 0.01)
    sweep_rows = []

    for t in thresholds:
        app    = int((y_pred_proba_loaded < t).sum())
        d_miss = int(((y_pred_proba_loaded < t) & (y_test_loaded == 1)).sum())
        d_catch = total_def_biz - d_miss
        prec   = (app - d_miss) / app * 100 if app > 0 else 0.0
        rec    = d_catch / total_def_biz * 100
        g_rev  = app * annual_revenue_per_loan
        e_loss = d_miss * lgd * avg_loan
        net    = g_rev - e_loss
        def_rate = d_miss / app * 100 if app > 0 else 0.0

        sweep_rows.append({
            "Threshold":                     round(t, 2),
            "Approved":                      app,
            "Approval Rate (%)":             round(app / total_biz * 100, 1),
            "Default Rate in Portfolio (%)": round(def_rate, 2),
            "Precision (%)":                 round(prec, 1),
            "Recall (%)":                    round(rec, 1),
            "Net Revenue (R)":               round(net, 0)
        })

    sweep_df = pd.DataFrame(sweep_rows)

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["Threshold"],
        y=sweep_df["Approval Rate (%)"],
        mode="lines", name="Approval Rate (%)",
        line=dict(color="teal", width=2.5)
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["Threshold"],
        y=sweep_df["Default Rate in Portfolio (%)"],
        mode="lines", name="Default Rate in Portfolio (%)",
        line=dict(color="red", width=2.5),
        yaxis="y2"
    ))
    fig_sweep.add_vline(
        x=threshold_biz, line_dash="dot", line_color="yellow",
        annotation_text=f"Current policy ({threshold_biz:.2f})",
        annotation_position="top"
    )
    fig_sweep.update_layout(
        title="Volume vs Risk -- Approval Rate and Portfolio Default Rate",
        xaxis_title="Decision Threshold",
        yaxis=dict(title="Approval Rate (%)", color="teal"),
        yaxis2=dict(title="Portfolio Default Rate (%)",
                    overlaying="y", side="right", color="red"),
        height=450,
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    fig_revenue = px.line(
        sweep_df, x="Threshold", y="Net Revenue (R)",
        title="Net Revenue After Expected Losses -- Across All Thresholds",
        color_discrete_sequence=["#2ecc71"]
    )
    fig_revenue.add_vline(
        x=threshold_biz, line_dash="dot", line_color="yellow",
        annotation_text=f"Current policy ({threshold_biz:.2f})",
        annotation_position="top"
    )
    fig_revenue.add_hline(
        y=0, line_dash="dash", line_color="red",
        annotation_text="Break-even", annotation_position="right"
    )
    fig_revenue.update_layout(height=400)
    st.plotly_chart(fig_revenue, use_container_width=True)

    profitable = sweep_df[sweep_df["Net Revenue (R)"] > 0]
    if len(profitable) > 0:
        breakeven_threshold = profitable["Threshold"].min()
        st.success(
            f"**Break-even threshold: {breakeven_threshold:.2f}** -- "
            f"Any threshold above {breakeven_threshold:.2f} generates positive net revenue "
            f"under the current assumptions. Below this, expected losses exceed interest income."
        )

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 5: PRECISION VS RECALL TRADE-OFF
    # ══════════════════════════════════════════
    st.subheader("Precision vs Recall -- The Business Trade-off")

    col1, col2 = st.columns(2)
    with col1:
        st.error("""
        **The cost of low Precision (approving bad loans):**
        - Every approved defaulter costs: LGD x Loan Amount
        - At R100,000 loan and 60% LGD = **R60,000 per bad loan**
        - These losses come directly off the bottom line
        - A lenient policy risks portfolio deterioration
        """)
    with col2:
        st.warning("""
        **The cost of low Recall (missing good customers):**
        - Every declined good customer = lost interest revenue
        - At R100,000 loan and 12% rate = **R12,000 lost per year**
        - Overly strict policy sacrifices market share
        - Good customers may go to competitors
        """)

    st.markdown(f"""
    **At the current threshold of {threshold_biz:.2f}:**
    - Cost of each bad loan approved: **R{lgd * avg_loan:,.0f}**
    - Revenue from each good loan approved: **R{annual_revenue_per_loan:,.0f}**
    - Break-even ratio: you need **{lgd * avg_loan / annual_revenue_per_loan:.1f} good loans**
      to offset every 1 bad loan
    """)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=sweep_df["Threshold"],
        y=sweep_df["Precision (%)"],
        mode="lines", name="Precision -- Portfolio Quality (%)",
        line=dict(color="teal", width=2.5)
    ))
    fig_pr.add_trace(go.Scatter(
        x=sweep_df["Threshold"],
        y=sweep_df["Recall (%)"],
        mode="lines", name="Recall -- Defaults Caught (%)",
        line=dict(color="orange", width=2.5)
    ))
    fig_pr.add_vline(
        x=threshold_biz, line_dash="dot", line_color="yellow",
        annotation_text=f"Current policy ({threshold_biz:.2f})",
        annotation_position="top"
    )
    fig_pr.update_layout(
        title="Precision vs Recall Across All Thresholds",
        xaxis_title="Decision Threshold",
        yaxis_title="Percentage (%)",
        height=400,
        legend=dict(x=0.01, y=0.01)
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 6: STRATEGY COMPARISON TABLE
    # ══════════════════════════════════════════
    st.subheader("Strategy Comparison -- Side by Side")

    strategies = {
        "Conservative (0.10)": 0.10,
        "Balanced (0.24)":     0.24,
        "Growth (0.35)":       0.35,
    }

    compare_rows = []
    for name, t in strategies.items():
        app    = int((y_pred_proba_loaded < t).sum())
        d_miss = int(((y_pred_proba_loaded < t) & (y_test_loaded == 1)).sum())
        d_catch = total_def_biz - d_miss
        prec   = (app - d_miss) / app * 100 if app > 0 else 0.0
        rec    = d_catch / total_def_biz * 100
        net    = (app * annual_revenue_per_loan) - (d_miss * lgd * avg_loan)
        compare_rows.append({
            "Strategy":          name,
            "Approved":          f"{app:,}",
            "Approval Rate":     f"{app/total_biz*100:.1f}%",
            "Defaults Caught":   f"{rec:.1f}%",
            "Portfolio Quality": f"{prec:.1f}%",
            "Net Revenue":       f"R{net:,.0f}",
            "Recommended":       "YES" if t == 0.24 else "--"
        })

    st.dataframe(
        pd.DataFrame(compare_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ══════════════════════════════════════════
    # SECTION 7: BUSINESS RECOMMENDATIONS
    # ══════════════════════════════════════════
    st.subheader("Business Recommendations")

    st.success("""
**Recommendation 1 — Approval Threshold: Use PD < 0.24**

At a threshold of 0.24, the model approves 79.3% of applicants while catching 53.7% of all
defaulters, maintaining 91% portfolio quality and generating R189.9M in net revenue after
expected credit losses.

While a stricter threshold of 0.18 produces marginally higher net revenue (R192.5M) and catches
67.1% of defaulters, it approves 3,545 fewer customers — all of whom are genuinely creditworthy.
The difference in net revenue between the two thresholds is R2.5M, meaning the implied cost of
serving those additional customers is approximately R705 each.

For a retail lender, R705 is a defensible customer acquisition cost. A single loan generates
R12,000 in annual interest revenue, and customers who repay successfully typically return for
subsequent lending products. The long-term relationship value of a creditworthy customer
substantially exceeds the marginal R705 cost of approving them at 0.24 rather than 0.18.

We therefore recommend PD < 0.24 as the primary approval threshold, with applicants scoring
between 0.24 and 0.30 referred for manual review rather than automatic decline — a credit analyst
can assess employment stability, tenure, and contextual factors the model cannot capture.
    """)

    st.success("""
**Recommendation 2 — Early Warning System**

The strongest predictor in the model is hard credit inquiries in the past 6 months
(coefficient: -1.1152) — customers seeking credit from multiple lenders simultaneously
are exhibiting active financial distress.

FNB should implement a monitoring trigger for existing customers: when a significant spike
in hard inquiries is detected within any 6-month window, flag the account for proactive
outreach. Intervening before a missed payment costs a fraction of the R60,000 expected loss
on a defaulted R100,000 loan at 60% LGD.
    """)

    st.success("""
**Recommendation 3 — Risk-Based Pricing**

The calibrated model produces true probability estimates per applicant, making individualised
risk-based pricing directly actionable:

Expected Loss = PD × LGD × EAD

The correct risk premium follows directly: Required Premium = PD × LGD, expressed as an
additional annual rate. At PD = 0.15 and LGD = 60%, this implies a 9% premium above the
base rate — not the commonly cited 150-200bp, which significantly underprices this risk level.

The dashboard already computes the inputs needed to operationalise this per applicant at scale.
    """)

    st.info("""
**Note on Calibration:** These recommendations depend on the model producing accurate probability
estimates — not just rankings. The isotonic calibration applied during training ensures the PD
outputs are suitable for Expected Loss calculations, Basel III capital provisioning, and IFRS 9
day-one provisioning.
    """)