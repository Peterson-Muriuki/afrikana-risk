"""
afrikana ecosystem — Streamlit portfolio dashboard
===================================================
Run:
    pip install afrikana-analytics afrikana-risk streamlit plotly
    streamlit run streamlit_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="afrikana · risk & analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
      background: #f8f9fa; border-radius: 12px; padding: 20px;
      border-left: 4px solid #1a5276; margin-bottom: 10px;
  }
  .metric-label { font-size: 13px; color: #666; margin-bottom: 4px; }
  .metric-value { font-size: 28px; font-weight: 600; color: #1a5276; }
  .metric-delta { font-size: 13px; color: #27ae60; }
  .section-header {
      font-size: 18px; font-weight: 600; color: #1a5276;
      border-bottom: 2px solid #1a5276; padding-bottom: 6px; margin: 20px 0 12px;
  }
  .alert-red    { background: #fdecea; border-left: 4px solid #e74c3c; padding: 12px; border-radius: 6px; }
  .alert-amber  { background: #fef9e7; border-left: 4px solid #f39c12; padding: 12px; border-radius: 6px; }
  .alert-green  { background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Data generation ──────────────────────────────────────────────────────────

@st.cache_data
def generate_portfolio(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure   = rng.integers(1, 120, n)
    income   = rng.lognormal(np.log(50_000), 0.6, n)
    dti      = rng.beta(2, 5, n)
    ltv      = rng.beta(3, 4, n)
    age      = rng.integers(21, 70, n)
    util     = rng.beta(2, 3, n)
    deliq    = rng.poisson(0.3, n)
    country  = rng.choice(["Kenya", "Nigeria", "Rwanda", "Uganda", "Ghana"], n,
                           p=[0.40, 0.25, 0.15, 0.12, 0.08])

    logit_p = (-3.0 + 0.5 * dti + 0.8 * ltv - 0.02 * tenure / 12
               - 0.3 * np.log(income / 50_000) + 0.6 * util + 0.4 * deliq)
    pd_true  = 1 / (1 + np.exp(-logit_p))
    default_ = (rng.random(n) < pd_true).astype(int)
    recovery = np.where(default_ == 1, rng.beta(5, 2, n), np.nan)

    return pd.DataFrame({
        "customer_id":        [f"KE{i:05d}" for i in range(n)],
        "country":            country,
        "age":                age,
        "tenure_months":      tenure,
        "annual_income":      income.round(2),
        "dti":                dti.round(4),
        "ltv":                ltv.round(4),
        "utilisation":        util.round(4),
        "past_delinquencies": deliq,
        "outstanding":        (income * ltv * 0.5).round(2),
        "limit":              (income * 0.6).round(2),
        "drawn":              (income * ltv * 0.3).round(2),
        "remaining_months":   rng.integers(12, 60, n),
        "recovery_rate":      recovery.round(4),
        "default":            default_,
    })


@st.cache_data
def generate_transactions(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    secs = rng.integers(0, 90 * 86400, n)
    ts   = [base + pd.Timedelta(seconds=int(s)) for s in secs]
    amt  = rng.lognormal(np.log(3_000), 1.0, n).clip(10, 2_000_000)
    fraud = (rng.random(n) < 0.012).astype(int)
    amt   = np.where(fraud, amt * 5, amt)
    return pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "customer_id":    rng.choice([f"KE{i:05d}" for i in range(500)], n),
        "amount":         amt.round(2),
        "timestamp":      ts,
        "country":        rng.choice(["Kenya", "Nigeria", "Rwanda", "Uganda"], n,
                                      p=[0.6, 0.2, 0.1, 0.1]),
        "fraud":          fraud,
    })
@st.cache_data
def run_credit_model(portfolio: pd.DataFrame):
    from afrikana_risk.credit import CreditScorer
    scorer = CreditScorer()
    train  = portfolio.iloc[:1600]
    scorer.fit(train)
    scored = scorer.score(portfolio)
    metrics = scorer.summary()
    importances = scorer.feature_importances()
    return scored, metrics, importances, scorer


@st.cache_data
def run_ecl(_scored: pd.DataFrame):
    from afrikana_risk.risk import ECLEngine, MacroScenario
    scenarios = [
        MacroScenario("base",    probability=0.60, pd_multiplier=1.00, lgd_multiplier=1.00),
        MacroScenario("upside",  probability=0.20, pd_multiplier=0.75, lgd_multiplier=0.85),
        MacroScenario("adverse", probability=0.20, pd_multiplier=1.50, lgd_multiplier=1.20),
    ]
    engine  = ECLEngine(eir=0.15, scenarios=scenarios)
    results = engine.compute(_scored)
    return results, engine


@st.cache_data
def run_stress(_scored: pd.DataFrame):
    from afrikana_risk.risk import StressTestor, STANDARD_SCENARIOS
    st = StressTestor(
        pd_macro_sensitivity={"gdp_growth_shock": -0.12, "unemployment_shock": 0.10},
        lgd_macro_sensitivity={"gdp_growth_shock": -0.06, "fx_shock": 0.04},
    )
    results = st.scenario_stress(_scored, STANDARD_SCENARIOS)
    comp    = st.scenario_comparison(results)
    var     = st.credit_var(_scored, n_simulations=5_000)
    return results, comp, var


@st.cache_data
def run_fraud(transactions: pd.DataFrame):
    from afrikana_risk.fraud import FraudDetector
    det    = FraudDetector()
    train  = transactions.iloc[:4000]
    det.fit(train)
    scored = det.score(transactions.iloc[4000:])
    alerts = det.alerts(scored, threshold=0.70)
    return scored, alerts


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown(
    '<span style="background:#1a5276;color:white;padding:4px 12px;'
    'border-radius:12px;font-size:12px;font-weight:600;">afrikana-risk v1.0.0</span>',
    unsafe_allow_html=True
)
st.sidebar.markdown("## 🌍 afrikana ecosystem")
st.sidebar.markdown("*Risk · Analytics · Intelligence*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 Credit Scoring",
     "🏦 IFRS 9 ECL",
     "⚡ Stress Testing",
     "🔍 Fraud Detection",
     "📡 Model Monitoring",
     "🤖 AI Assistant"],
    label_visibility="collapsed",
)

st.sidebar.divider()
with st.sidebar.expander("⚙️ Portfolio settings"):
    n_accounts = st.slider("Accounts", 500, 3000, 2000, 500)
    seed       = st.number_input("Random seed", value=42)

portfolio    = generate_portfolio(n_accounts, seed)
transactions = generate_transactions(5000, seed)

st.sidebar.markdown(f"**{n_accounts:,}** accounts · **{portfolio['default'].sum()}** defaults")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🌍 afrikana — risk & analytics platform")
    st.markdown(
        "Production-grade quantitative risk analytics built in Africa. "
        "Applicable to any bank, fintech, telco, or subscription business."
    )

    col1, col2, col3, col4 = st.columns(4)
    dr = portfolio["default"].mean()
    col1.metric("Portfolio size", f"{n_accounts:,}", "accounts")
    col2.metric("Default rate", f"{dr:.1%}", f"{portfolio['default'].sum()} defaults")
    col3.metric("Avg income", f"KES {portfolio['annual_income'].mean():,.0f}")
    col4.metric("Countries", portfolio["country"].nunique())

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Country distribution</div>', unsafe_allow_html=True)
        country_counts = portfolio["country"].value_counts().reset_index()
        fig = px.bar(country_counts, x="country", y="count",
                     color="country", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Income distribution by default status</div>', unsafe_allow_html=True)
        fig = px.histogram(portfolio, x="annual_income", color="default",
                           nbins=40, barmode="overlay", opacity=0.7,
                           color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                           labels={"annual_income": "Annual income (KES)", "default": "Default"})
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Ecosystem modules</div>', unsafe_allow_html=True)
    m = {
        "CreditScorer":       ("PD · LGD · EAD · Basel III capital",          "📊"),
        "ScorecardBuilder":   ("WoE binning · IV selection · scorecard points","📋"),
        "ECLEngine":          ("IFRS 9 Stage 1/2/3 · macro scenarios",         "🏦"),
        "StressTestor":       ("Scenario testing · credit VaR · NPL trajectory","⚡"),
        "FraudDetector":      ("Isolation Forest · supervised · rule engine",  "🔍"),
        "ModelMonitor":       ("PSI · Gini/KS drift · feature drift",          "📡"),
        "ChampionChallenger": ("DeLong AUC test · audit trail · rollback",     "🏆"),
    }
    cols = st.columns(4)
    for i, (name, (desc, icon)) in enumerate(m.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{icon} {name}</div>
              <div style="font-size:12px;color:#555;margin-top:6px">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Credit Scoring
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Credit Scoring":
    st.title("📊 Credit Scoring — PD · LGD · EAD")

    with st.spinner("Training credit model..."):
        scored, metrics, importances, scorer = run_credit_model(portfolio)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC",        f"{metrics['auc']:.4f}")
    col2.metric("Gini",       f"{metrics['gini']:.4f}")
    col3.metric("KS stat",    f"{metrics['ks']:.4f}")
    col4.metric("Brier score",f"{metrics['brier']:.4f}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">PD distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(scored, x="pd", color="default", nbins=50,
                           barmode="overlay", opacity=0.75,
                           color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                           labels={"pd": "Probability of default", "default": "Actual default"})
        fig.update_layout(height=320, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Portfolio grade distribution</div>', unsafe_allow_html=True)
        grade_summary = scorer.portfolio_summary(scored)
        fig = px.bar(grade_summary, x="pd_grade", y="count",
                     color="el_rate", color_continuous_scale="RdYlGn_r",
                     labels={"pd_grade": "PD grade", "count": "Accounts", "el_rate": "EL rate"})
        fig.update_layout(height=320, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">IFRS 9 stage breakdown</div>', unsafe_allow_html=True)
    stage_counts = scored["ifrs9_stage"].value_counts().sort_index()
    stage_ead    = scored.groupby("ifrs9_stage")["ead"].sum()
    stage_el     = scored.groupby("ifrs9_stage")["el"].sum()

    c1, c2, c3 = st.columns(3)
    for stage, col in zip([1, 2, 3], [c1, c2, c3]):
        cnt = int(stage_counts.get(stage, 0))
        ead = float(stage_ead.get(stage, 0))
        el  = float(stage_el.get(stage, 0))
        colour = {"1": "#27ae60", "2": "#f39c12", "3": "#e74c3c"}[str(stage)]
        col.markdown(f"""
        <div class="metric-card" style="border-left-color:{colour}">
          <div class="metric-label">Stage {stage}</div>
          <div class="metric-value">{cnt:,}</div>
          <div class="metric-delta">EAD: {ead:,.0f} · EL: {el:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    if not importances.empty:
        st.markdown('<div class="section-header">Feature importances</div>', unsafe_allow_html=True)
        fig = px.bar(importances.head(8), x="importance", y="feature",
                     orientation="h", color="importance",
                     color_continuous_scale="Blues")
        fig.update_layout(height=280, margin=dict(t=20,b=20), showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IFRS 9 ECL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏦 IFRS 9 ECL":
    st.title("🏦 IFRS 9 Expected Credit Loss")

    with st.spinner("Computing ECL..."):
        scored, metrics, _, scorer = run_credit_model(portfolio)
        ecl_df, engine = run_ecl(scored)

    portfolio_ecl = engine.portfolio_ecl(ecl_df)
    total_ecl     = ecl_df["ecl_pw"].sum()
    total_ead     = ecl_df["ead"].sum()
    coverage      = total_ecl / total_ead

    col1, col2, col3 = st.columns(3)
    col1.metric("Total EAD",      f"KES {total_ead:,.0f}")
    col2.metric("Total ECL (PW)", f"KES {total_ecl:,.0f}")
    col3.metric("ECL coverage",   f"{coverage:.2%}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">ECL by IFRS 9 stage</div>', unsafe_allow_html=True)
        fig = px.bar(portfolio_ecl, x="ifrs9_stage", y="total_ecl",
                     color="ifrs9_stage",
                     color_discrete_map={1: "#27ae60", 2: "#f39c12", 3: "#e74c3c"},
                     labels={"ifrs9_stage": "Stage", "total_ecl": "Total ECL (KES)"},
                     text_auto=True)
        fig.update_layout(height=320, margin=dict(t=20,b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Coverage ratio by stage</div>', unsafe_allow_html=True)
        fig = px.bar(portfolio_ecl, x="ifrs9_stage", y="coverage_ratio",
                     color="ifrs9_stage",
                     color_discrete_map={1: "#27ae60", 2: "#f39c12", 3: "#e74c3c"},
                     labels={"ifrs9_stage": "Stage", "coverage_ratio": "ECL / EAD"},
                     text_auto=".2%")
        fig.update_layout(height=320, margin=dict(t=20,b=20), showlegend=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">ECL scenario comparison</div>', unsafe_allow_html=True)
    sc_cols = [c for c in ecl_df.columns if c.startswith("ecl_scenario_")]
    if sc_cols:
        sc_totals = {c.replace("ecl_scenario_", ""): ecl_df[c].sum() for c in sc_cols}
        sc_df = pd.DataFrame(sc_totals.items(), columns=["Scenario", "Total ECL"])
        fig = px.bar(sc_df, x="Scenario", y="Total ECL",
                     color="Scenario", color_discrete_sequence=["#2ecc71", "#3498db", "#e74c3c"],
                     text_auto=True)
        fig.update_layout(height=300, margin=dict(t=20,b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full ECL table by stage"):
        st.dataframe(portfolio_ecl.style.format({
            "total_ead": "{:,.0f}", "total_ecl": "{:,.0f}",
            "avg_pd": "{:.3%}", "coverage_ratio": "{:.2%}",
            "pct_portfolio": "{:.1%}",
        }))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Stress Testing
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚡ Stress Testing":
    st.title("⚡ Stress Testing · Credit VaR")

    with st.spinner("Running stress scenarios..."):
        scored, _, _, _ = run_credit_model(portfolio)
        stress_results, comparison, var_result = run_stress(scored)

    col1, col2, col3 = st.columns(3)
    col1.metric("EL (base)",     f"KES {var_result['el']:,.0f}")
    col2.metric("VaR 99.9%",     f"KES {var_result.get('var_0.999', var_result.get('var_0.990', 0)):,.0f}")
    col3.metric("Economic capital", f"KES {var_result['economic_capital']:,.0f}")

    st.divider()
    st.markdown('<div class="section-header">Scenario comparison</div>', unsafe_allow_html=True)

    comp_df = comparison.reset_index()
    cols_to_show = [c for c in ["scenario", "total_ecl", "avg_stressed_pd",
                                 "ecl_coverage", "npl_ratio"] if c in comp_df.columns]
    st.dataframe(comp_df[cols_to_show].style.format({
        "total_ecl": "{:,.0f}", "avg_stressed_pd": "{:.3%}",
        "ecl_coverage": "{:.2%}", "npl_ratio": "{:.2%}",
    }))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">ECL under each scenario</div>', unsafe_allow_html=True)
        fig = px.bar(comp_df, x="scenario", y="total_ecl",
                     color="scenario",
                     color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                     text_auto=True, labels={"total_ecl": "Total ECL (KES)", "scenario": "Scenario"})
        fig.update_layout(height=300, margin=dict(t=20,b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Loss distribution (credit VaR)</div>', unsafe_allow_html=True)
        losses = var_result["loss_dist"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=losses, nbinsx=60, name="Simulated losses",
                                   marker_color="#3498db", opacity=0.75))
        for conf, col in [(0.99, "orange"), (0.999, "red")]:
            key = f"var_{conf:.3f}"
            if key in var_result:
                fig.add_vline(x=var_result[key], line_dash="dash", line_color=col,
                              annotation_text=f"VaR {conf:.0%}")
        fig.update_layout(height=300, margin=dict(t=20,b=20),
                          xaxis_title="Portfolio loss (KES)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Fraud Detection
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Fraud Detection":
    st.title("🔍 Fraud Detection")

    with st.spinner("Running fraud models..."):
        scored_txns, alerts = run_fraud(transactions)

    band_counts = scored_txns["risk_band"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LOW",      int(band_counts.get("LOW", 0)))
    c2.metric("MEDIUM",   int(band_counts.get("MEDIUM", 0)))
    c3.metric("HIGH",     int(band_counts.get("HIGH", 0)), delta_color="inverse")
    c4.metric("CRITICAL", int(band_counts.get("CRITICAL", 0)), delta_color="inverse")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Risk band distribution</div>', unsafe_allow_html=True)
        fig = px.pie(band_counts.reset_index(), values="count", names="risk_band",
                     color="risk_band",
                     color_discrete_map={"LOW": "#27ae60", "MEDIUM": "#f39c12",
                                         "HIGH": "#e67e22", "CRITICAL": "#e74c3c"})
        fig.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Score distribution by actual fraud</div>', unsafe_allow_html=True)
        fig = px.histogram(scored_txns, x="ensemble_score", color="fraud",
                           nbins=40, barmode="overlay", opacity=0.75,
                           color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                           labels={"ensemble_score": "Ensemble score", "fraud": "Actual fraud"})
        fig.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Top alerts</div>', unsafe_allow_html=True)
    alert_cols = [c for c in ["customer_id", "amount", "ensemble_score",
                               "risk_band", "rule_flags", "fraud"] if c in alerts.columns]
    st.dataframe(alerts[alert_cols].head(20).style.format({
        "amount": "{:,.0f}", "ensemble_score": "{:.3f}"
    }))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📡 Model Monitoring":
    st.title("📡 Model Monitoring · Stability")

    with st.spinner("Running monitoring..."):
        scored, metrics, _, _ = run_credit_model(portfolio)
        from afrikana_risk.monitoring import ModelMonitor
        monitor = ModelMonitor()
        monitor.set_reference(scored, score_col="pd")

    st.markdown('<div class="section-header">Simulate population shift</div>', unsafe_allow_html=True)
    pd_multiplier = st.slider("PD inflation multiplier (simulates macro deterioration)",
                               0.5, 3.0, 1.4, 0.1)

    shifted = scored.copy()
    shifted["pd"] = (shifted["pd"] * pd_multiplier).clip(0, 1)
    report = monitor.monitor_period(shifted, period="Current", score_col="pd", target_col="default")

    psi = report["psi_score"]
    status = report["psi_status"]

    # Normalise status to one of three known CSS classes
    _status_map = {
        "STABLE": "STABLE", "WARNING": "WARNING", "ALERT": "ALERT",
        "significant_shift": "ALERT", "moderate_shift": "WARNING",
        "no_shift": "STABLE",
    }
    status_key = _status_map.get(status, "WARNING")

    col1, col2, col3 = st.columns(3)
    col1.metric("PSI", f"{psi:.4f}", help="< 0.10 stable · 0.10–0.25 warning · > 0.25 alert")
    col2.metric("Status", status)
    if "gini" in report:
        col3.metric("Gini", f"{report['gini']:.4f}")

    psi_colour = {"STABLE": "alert-green", "WARNING": "alert-amber", "ALERT": "alert-red"}[status_key]
    st.markdown(
        f'<div class="{psi_colour}">💡 {monitor.recommend_action(report)}</div>',
        unsafe_allow_html=True
    )

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">PD distribution shift</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scored["pd"], name="Reference",
                                   marker_color="#3498db", opacity=0.6, nbinsx=40))
        fig.add_trace(go.Histogram(x=shifted["pd"], name="Current",
                                   marker_color="#e74c3c", opacity=0.6, nbinsx=40))
        fig.update_layout(barmode="overlay", height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Decile default rate</div>', unsafe_allow_html=True)
        decile_tbl = report.get("decile_table", pd.DataFrame())
        if isinstance(decile_tbl, pd.DataFrame) and not decile_tbl.empty and "default_rate" in decile_tbl.columns:
            fig = px.bar(decile_tbl, x="decile", y="default_rate",
                         labels={"decile": "Score decile", "default_rate": "Default rate"})
            fig.update_yaxes(tickformat=".1%")
            fig.update_layout(height=300, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Decile table unavailable for this period.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI Assistant
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 AI Assistant":
    st.title("🤖 AI Risk Assistant — powered by Mistral (local)")
    st.markdown(
        "Ask questions about your portfolio in plain English using your local Ollama Mistral model. "
        "Generate regulatory reports. Explain individual credit decisions."
    )

    st.info("🦙 Running on **Ollama mistral** (local) — no API key required.", icon="✅")

    # Load context once
    with st.spinner("Loading portfolio context..."):
        scored, _, _, _ = run_credit_model(portfolio)
        ecl_df, engine  = run_ecl(scored)
        _, comp, var    = run_stress(scored)
        sc_txns, alerts = run_fraud(transactions)

        from afrikana_risk.monitoring import ModelMonitor
        monitor = ModelMonitor()
        monitor.set_reference(scored, score_col="pd")
        m_report = monitor.monitor_period(
            scored.copy(), period="current",
            score_col="pd", target_col="default"
        )

    # Build context summary once
    def _build_context(scored, ecl_df, m_report, sc_txns):
        lines = []
        lines.append(f"Portfolio: {len(scored):,} accounts, "
                     f"default rate {scored['default'].mean():.1%}, "
                     f"avg PD {scored['pd'].mean():.3%}")
        stage_dist = scored["ifrs9_stage"].value_counts().sort_index()
        lines.append(f"IFRS9 stages: {stage_dist.to_dict()}")
        lines.append(f"Total EAD: {scored['ead'].sum():,.0f}  "
                     f"Total EL: {scored['el'].sum():,.0f}  "
                     f"Coverage: {scored['el'].sum()/scored['ead'].sum():.2%}")
        lines.append(f"ECL prob-weighted: {ecl_df['ecl_pw'].sum():,.0f}")
        lines.append(f"Model PSI: {m_report['psi_score']}  "
                     f"Status: {m_report['psi_status']}")
        if "risk_band" in sc_txns.columns:
            lines.append(f"Fraud risk bands: {sc_txns['risk_band'].value_counts().to_dict()}")
        return "\n".join(lines)

    portfolio_context = _build_context(scored, ecl_df, m_report, sc_txns)

    SYSTEM_PROMPT = (
        "You are a senior quantitative risk analyst embedded in the afrikana-risk platform. "
        "You have expertise in credit risk, IFRS 9, Basel III, fraud analytics, and model governance. "
        "Answer questions using the portfolio data below. Be precise and cite numbers.\n\n"
        f"--- PORTFOLIO CONTEXT ---\n{portfolio_context}\n--- END CONTEXT ---"
    )

    def ask_ollama(messages: list[dict]) -> str:
        """Call local Ollama mistral API."""
        import urllib.request, json
        payload = json.dumps({
            "model": "mistral",
            "messages": messages,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["message"]["content"]
        except Exception as e:
            return f"❌ Ollama error: {e}\n\nMake sure Ollama is running: `ollama serve`"

    def stream_ollama(messages: list[dict]):
        """Stream tokens from local Ollama mistral."""
        import urllib.request, json
        payload = json.dumps({
            "model": "mistral",
            "messages": messages,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"❌ Ollama error: {e}\n\nMake sure Ollama is running: `ollama serve`"

    REPORT_PROMPTS = {
        "executive_summary": (
            "Write a concise executive summary (400-500 words) of the portfolio covering: "
            "1) Portfolio overview and default rate, 2) Credit quality and IFRS9 stage breakdown, "
            "3) ECL coverage ratio, 4) Key risks, 5) Top 3 recommended actions."
        ),
        "ifrs9_disclosure": (
            "Draft an IFRS 9 credit risk disclosure note covering: staging criteria, "
            "ECL methodology, stage distribution table, key assumptions, "
            "macro overlay scenarios, and sensitivity analysis."
        ),
        "model_validation": (
            "Write a model validation summary covering: model description, "
            "discriminatory power (AUC/Gini/KS), calibration, PSI stability, "
            "limitations, and overall validation conclusion."
        ),
        "fraud_risk_summary": (
            "Write a fraud risk summary covering: transaction overview, "
            "alert counts by risk band, top rule flags, high-risk segments, "
            "and recommendations for threshold adjustments."
        ),
    }

    report_col, chat_col = st.columns([1, 2])

    with report_col:
        st.markdown("**📄 Generate report**")
        report_type = st.selectbox("Report type", list(REPORT_PROMPTS.keys()))
        if st.button("Generate", type="primary"):
            with st.spinner("Mistral is thinking..."):
                messages = [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": REPORT_PROMPTS[report_type]},
                ]
                result = ask_ollama(messages)
            st.markdown(result)

    with chat_col:
        st.markdown("**💬 Chat with your portfolio**")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your portfolio..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                ollama_messages = (
                    [{"role": "system", "content": SYSTEM_PROMPT}]
                    + st.session_state.messages[-10:]
                )
                for token in stream_ollama(ollama_messages):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(
    "**Peterson Mutegi** · Nairobi, Kenya  \n"
    "[github.com/Peterson-Muriuki](https://github.com/Peterson-Muriuki)  \n"
    "pitmuriuki@gmail.com"
)
st.sidebar.caption("afrikana-risk v1.0.0 · afrikana-analytics v2.0")