"""
afrikana-risk end-to-end demo
=============================
Demonstrates the full risk analytics pipeline on synthetic African
fintech / banking data:

    Credit scoring → ECL (IFRS9) → Stress testing → Fraud detection →
    Model monitoring → Champion-challenger governance

Run:
    pip install afrikana-risk
    python examples/risk_demo.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# 1. Synthetic portfolio data
# ============================================================

N = 2_000

def make_portfolio(n: int, default_rate: float = 0.08) -> pd.DataFrame:
    tenure   = np.random.exponential(18, n).clip(1, 120).astype(int)
    income   = np.random.lognormal(np.log(50_000), 0.6, n)
    dti      = np.random.beta(2, 5, n)            # debt-to-income ratio
    ltv      = np.random.beta(3, 4, n)            # loan-to-value
    age      = np.random.randint(21, 70, n)
    util     = np.random.beta(2, 3, n)            # credit utilisation
    deliq    = np.random.poisson(0.3, n)          # past delinquencies

    # Logit-based default generation (correlated with risk factors)
    logit_p = (
        -3.0
        + 0.5 * dti
        + 0.8 * ltv
        - 0.02 * tenure
        - 0.3 * np.log(income / 50_000)
        + 0.6 * util
        + 0.4 * deliq
    )
    pd_true = 1 / (1 + np.exp(-logit_p))
    default = (np.random.rand(n) < pd_true).astype(int)

    recovery = np.where(default == 1,
                        np.random.beta(5, 2, n),   # high recovery
                        np.nan)

    return pd.DataFrame({
        "customer_id":     [f"KE{i:05d}" for i in range(n)],
        "tenure_months":   tenure,
        "annual_income":   income.round(2),
        "dti":             dti.round(4),
        "ltv":             ltv.round(4),
        "age":             age,
        "utilisation":     util.round(4),
        "past_delinquencies": deliq,
        "outstanding":     (income * ltv * 0.5).round(2),
        "limit":           (income * 0.6).round(2),
        "drawn":           (income * ltv * 0.3).round(2),
        "remaining_months": np.random.randint(12, 60, n),
        "recovery_rate":   recovery.round(4),
        "default":         default,
    })


def make_transactions(n: int = 5_000) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-01-01")
    seconds = np.random.randint(0, 90 * 86400, n)
    ts      = [base_ts + pd.Timedelta(seconds=int(s)) for s in seconds]
    amount  = np.random.lognormal(np.log(3_000), 1.0, n).clip(10, 2_000_000)
    fraud   = (np.random.rand(n) < 0.012).astype(int)
    # Fraud transactions: higher amounts, odd hours
    amount  = np.where(fraud, amount * 5, amount)

    return pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "customer_id":    np.random.choice([f"KE{i:05d}" for i in range(500)], n),
        "amount":         amount.round(2),
        "timestamp":      ts,
        "merchant_category": np.random.choice(["retail", "fuel", "online", "atm"], n),
        "country":        np.random.choice(["KEN", "NGA", "RWA", "UGA"], n,
                                           p=[0.6, 0.2, 0.1, 0.1]),
        "fraud":          fraud,
    })


portfolio   = make_portfolio(N)
transactions = make_transactions()

train_df, test_df = portfolio.iloc[:1600], portfolio.iloc[1600:]

print(f"\n{'='*60}")
print(f"  afrikana-risk demo — portfolio: {N} accounts, "
      f"{portfolio['default'].sum()} defaults ({portfolio['default'].mean():.1%})")
print(f"{'='*60}\n")


# ============================================================
# 2. Credit Scoring — PD / LGD / EAD
# ============================================================

from afrikana_risk.credit import CreditScorer, CreditScorerConfig, ScorecardBuilder

print("── CreditScorer (PD / LGD / EAD) ─────────────────────────")
config = CreditScorerConfig(pd_model="logistic", calibrate_pd=True, verbose=True)
scorer = CreditScorer(config)
scorer.fit(train_df)

scored = scorer.score(test_df)
print("\nPortfolio grade distribution:")
print(scorer.portfolio_summary(scored).to_string(index=False))

print("\nFeature importances:")
print(scorer.feature_importances().head(5).to_string(index=False))

cv = scorer.cross_validate(portfolio)
print(f"\nCross-validation (5-fold):")
print(f"  AUC  {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
print(f"  Gini {cv['gini_mean']:.4f}   KS {cv['ks_mean']:.4f}")


# ============================================================
# 3. ScorecardBuilder — WoE / IV / scorecard points
# ============================================================

print("\n── ScorecardBuilder (WoE · scorecard points) ──────────────")
builder = ScorecardBuilder()
builder.fit(train_df, target="default")
print("\nIV summary:")
print(builder.iv_summary().head(8).to_string(index=False))
card = builder.scorecard_table()
print(f"\nScorecard table ({len(card)} rows):")
print(card.head(10).to_string(index=False))
scores = builder.score(test_df)
pds    = builder.score_to_pd(scores)
print(f"\nScore range: {scores.min()} – {scores.max()}, "
      f"Avg PD from score: {pds.mean():.3%}")
print(f"Builder summary: {builder.summary()}")


# ============================================================
# 4. IFRS 9 ECL Engine
# ============================================================

from afrikana_risk.risk import ECLEngine, MacroScenario, StageThresholds

print("\n── ECLEngine (IFRS 9 Stage 1 / 2 / 3) ─────────────────────")
scenarios = [
    MacroScenario("base",    probability=0.60, pd_multiplier=1.0,  lgd_multiplier=1.0),
    MacroScenario("upside",  probability=0.20, pd_multiplier=0.75, lgd_multiplier=0.85),
    MacroScenario("adverse", probability=0.20, pd_multiplier=1.50, lgd_multiplier=1.20),
]
ecl_engine = ECLEngine(eir=0.15, scenarios=scenarios)
ecl_results = ecl_engine.compute(scored)
print(ecl_engine.portfolio_ecl(ecl_results).to_string(index=False))
print(f"\nTotal ECL (prob-weighted): KES {ecl_results['ecl_pw'].sum():,.0f}")
print(f"Coverage ratio: {ecl_results['ecl_pw'].sum() / ecl_results['ead'].sum():.2%}")


# ============================================================
# 5. Regulatory Capital (Basel III IRB)
# ============================================================

cap = scorer.regulatory_capital(scored)
print(f"\n── Regulatory Capital (Basel III IRB) ──────────────────────")
print(f"  Total EAD:      KES {cap['ead'].sum():>15,.0f}")
print(f"  Total RWA:      KES {cap['rwa'].sum():>15,.0f}")
print(f"  Capital charge: KES {cap['capital_charge'].sum():>15,.0f}")
print(f"  Avg asset corr: {cap['asset_correlation'].mean():.4f}")


# ============================================================
# 6. Stress Testing
# ============================================================

from afrikana_risk.risk import StressTestor, STANDARD_SCENARIOS

print("\n── StressTestor (macro scenario analysis) ──────────────────")
st = StressTestor(
    pd_macro_sensitivity={"gdp_growth_shock": -0.12, "unemployment_shock": 0.10},
    lgd_macro_sensitivity={"gdp_growth_shock": -0.06, "fx_shock": 0.04},
)
stress_results = st.scenario_stress(scored, scenarios=STANDARD_SCENARIOS)
print(st.scenario_comparison(stress_results).to_string())

print("\nCredit VaR (Monte Carlo, n=5,000):")
var = st.credit_var(scored, n_simulations=5_000)
print(f"  EL:      KES {var['el']:>12,.0f}")
print(f"  VaR 99%: KES {var['var_0.990']:>12,.0f}")
print(f"  ES  99%: KES {var['es_0.990']:>12,.0f}")
print(f"  Econ. capital: KES {var['economic_capital']:>12,.0f}")


# ============================================================
# 7. Fraud Detection
# ============================================================

from afrikana_risk.fraud import FraudDetector, FraudDetectorConfig

print("\n── FraudDetector (multi-layer anomaly detection) ───────────")
fraud_config = FraudDetectorConfig(
    contamination=0.01,
    supervised=True,
    high_risk_countries=["SYR", "PRK"],
)
detector = FraudDetector(fraud_config)
train_txns = transactions.iloc[:4000]
test_txns  = transactions.iloc[4000:]

detector.fit(train_txns)
scored_txns = detector.score(test_txns)
alerts = detector.alerts(scored_txns, threshold=0.70)

print(f"  Transactions scored: {len(scored_txns)}")
print(f"  Alerts generated:    {len(alerts)}")
print(f"  Risk band distribution:")
print(scored_txns["risk_band"].value_counts().to_string())
print(f"\nTop 5 alerts:")
cols = ["transaction_id", "customer_id", "amount", "ensemble_score", "risk_band", "rule_flags"]
print(alerts[cols].head().to_string(index=False))


# ============================================================
# 8. Model Monitoring (PSI / drift)
# ============================================================

from afrikana_risk.monitoring import ModelMonitor

print("\n── ModelMonitor (stability & drift) ────────────────────────")
monitor = ModelMonitor()
monitor.set_reference(scored, score_col="pd")

# Simulate a shifted current population (e.g. macro downturn)
shifted = scored.copy()
shifted["pd"] = (shifted["pd"] * 1.4).clip(0, 1)   # PD inflated by 40%
report = monitor.monitor_period(shifted, period="2026-Q1",
                                score_col="pd", target_col="default")
print(f"  PSI: {report['psi_score']} → {report['psi_status']}")
if "gini" in report:
    print(f"  Gini: {report['gini']}")
print(f"  Recommendation: {monitor.recommend_action(report)}")

feat_drift = monitor.feature_drift(shifted)
print(f"\nFeature drift (top 5):")
print(feat_drift.head(5).to_string(index=False))


# ============================================================
# 9. Champion-Challenger Governance
# ============================================================

from afrikana_risk.monitoring import ChampionChallenger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n── ChampionChallenger (model governance) ───────────────────")
feat_cols = ["tenure_months", "dti", "ltv", "utilisation", "past_delinquencies"]
X_tr = train_df[feat_cols].fillna(0)
y_tr = train_df["default"]

champ_model = Pipeline([
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
])
champ_model.fit(X_tr, y_tr)

# Challenger: slightly different regularisation
chall_model = Pipeline([
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")),
])
chall_model.fit(X_tr, y_tr)

cc = ChampionChallenger(challenger_traffic=0.10, significance_level=0.05)
cc.register_champion(champ_model,  name="Logistic v1", version="1.0")
cc.register_challenger(chall_model, name="Logistic v2", version="2.0")

eval_result = cc.evaluate(test_df[feat_cols + ["default"]], target_col="default")
print(f"  Champion AUC:   {eval_result['champion_auc']}")
print(f"  Challenger AUC: {eval_result['challenger_auc']}")
print(f"  Delta:          {eval_result['auc_delta']:+.4f}")
print(f"  p-value:        {eval_result['p_value']:.4f}")
print(f"  Recommendation: {eval_result['recommendation']}")

if cc.should_promote(eval_result):
    cc.promote_challenger()
    print("  → Challenger promoted to champion.")
else:
    print("  → Champion retained.")


print(f"\n{'='*60}")
print("  afrikana-risk demo complete.")
print(f"{'='*60}\n")