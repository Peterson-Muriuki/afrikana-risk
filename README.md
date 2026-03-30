# afrikana-risk

[![PyPI](https://img.shields.io/pypi/v/afrikana-risk?color=blue)](https://pypi.org/project/afrikana-risk/)
[![Python](https://img.shields.io/pypi/pyversions/afrikana-risk?color=green)](https://pypi.org/project/afrikana-risk/)
[![License](https://img.shields.io/pypi/l/afrikana-risk?color=orange)](https://pypi.org/project/afrikana-risk/)
[![](https://static.pepy.tech/badge/afrikana-risk)](https://pepy.tech/project/afrikana-risk)
[![Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Peterson-Muriuki/afrikana-risk)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen)
[![CI](https://img.shields.io/github/actions/workflow/status/Peterson-Muriuki/afrikana-risk/ci.yml?label=CI)](https://github.com/Peterson-Muriuki/afrikana-risk)
[![PyPI](https://img.shields.io/pypi/v/afrikana-risk)](https://pypi.org/project/afrikana-risk/)
[![Python](https://img.shields.io/pypi/pyversions/afrikana-risk)](https://pypi.org/project/afrikana-risk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Production-ready quantitative risk toolkit for credit scoring, IFRS 9 ECL, fraud detection, and model governance.**

Built from real analytical work across African financial markets. Applicable to any bank, fintech, microfinance institution, telco, or subscription business that needs rigorous, regulatory-grade risk analytics.

A sibling package to [`afrikana-analytics`](https://github.com/Peterson-Muriuki/afrikana-analytics) - which covers customer retention, LTV, demand forecasting, and infrastructure deployment.

---

## What it does

| Module | What it solves |
|---|---|
| `CreditScorer` | PD / LGD / EAD models · Basel III IRB capital · TtC PD conversion |
| `ScorecardBuilder` | WoE binning · Information Value selection · points-based scorecard |
| `ECLEngine` | IFRS 9 Stage 1/2/3 ECL · probability-weighted macro scenarios |
| `StressTestor` | Scenario stress testing · Vasicek credit VaR · NPL trajectory |
| `FraudDetector` | Isolation Forest + supervised fraud scoring · rule layer · behavioural profiles |
| `ModelMonitor` | PSI · Gini/KS drift · decile stability · feature drift |
| `ChampionChallenger` | DeLong AUC test · auto-promotion · regulatory audit trail |

---

## Installation

```bash
pip install afrikana-risk
```

With optional extras:

```bash
pip install "afrikana-risk[ai]"    # Mistral AI integration
pip install "afrikana-risk[viz]"   # matplotlib / plotly visualisations
pip install "afrikana-risk[all]"   # everything
```

---

## Quick start

```python
from afrikana_risk.credit      import CreditScorer, ScorecardBuilder
from afrikana_risk.risk        import ECLEngine, StressTestor, MacroScenario
from afrikana_risk.fraud       import FraudDetector
from afrikana_risk.monitoring  import ModelMonitor, ChampionChallenger

# ── 1. Credit scoring ──────────────────────────────────────────────
scorer = CreditScorer()
scorer.fit(train_df)                       # trains PD, LGD, EAD sub-models
scored = scorer.score(portfolio_df)        # adds pd, lgd, ead, el, pd_grade, ifrs9_stage
print(scorer.summary())                    # auc, gini, ks, brier
print(scorer.portfolio_summary(scored))    # grade-level EL distribution

cap = scorer.regulatory_capital(scored, confidence=0.999)
print(f"Total RWA: {cap['rwa'].sum():,.0f}")

# ── 2. Traditional scorecard ───────────────────────────────────────
builder = ScorecardBuilder()
builder.fit(train_df, target="default")
print(builder.iv_summary())                # Information Value per variable
print(builder.scorecard_table())           # WoE bins → point allocations
scores = builder.score(portfolio_df)       # integer credit scores
pds    = builder.score_to_pd(scores)       # convert scores back to PD

# ── 3. IFRS 9 ECL ──────────────────────────────────────────────────
scenarios = [
    MacroScenario("base",    probability=0.60, pd_multiplier=1.00, lgd_multiplier=1.00),
    MacroScenario("upside",  probability=0.20, pd_multiplier=0.75, lgd_multiplier=0.85),
    MacroScenario("adverse", probability=0.20, pd_multiplier=1.50, lgd_multiplier=1.20),
]
engine  = ECLEngine(eir=0.15, scenarios=scenarios)
results = engine.compute(scored)           # adds ifrs9_stage, ecl, ecl_pw, provision
print(engine.portfolio_ecl(results))       # stage-level ECL aggregation

# ── 4. Stress testing ──────────────────────────────────────────────
st = StressTestor(
    pd_macro_sensitivity={"gdp_growth_shock": -0.12, "unemployment_shock": 0.10},
)
stress = st.scenario_stress(scored)
print(st.scenario_comparison(stress))      # base / adverse / severe comparison

var = st.credit_var(scored, n_simulations=10_000)
print(f"VaR 99.9%: {var['var_0.999']:,.0f}  Economic capital: {var['economic_capital']:,.0f}")

# ── 5. Fraud detection ─────────────────────────────────────────────
detector = FraudDetector()
detector.fit(train_transactions)
scored_txns = detector.score(new_transactions)
alerts = detector.alerts(scored_txns, threshold=0.70)

# ── 6. Model monitoring ────────────────────────────────────────────
monitor = ModelMonitor()
monitor.set_reference(scored, score_col="pd")
report = monitor.monitor_period(current_scored, period="2026-Q1", score_col="pd")
print(monitor.recommend_action(report))

# ── 7. Champion-challenger governance ──────────────────────────────
cc = ChampionChallenger(challenger_traffic=0.10)
cc.register_champion(champion_model,   name="Scorecard v3", version="3.0")
cc.register_challenger(challenger_model, name="LightGBM v1", version="1.0")

result = cc.evaluate(test_df, target_col="default")
if cc.should_promote(result):
    cc.promote_challenger()
```

---

## Modules

### `CreditScorer`

Produces calibrated PD, LGD, and EAD estimates for Basel III and IFRS 9 reporting.

- **PD model** - logistic regression on raw or WoE features, Platt-scaled for calibration. Optional LightGBM backend.
- **LGD model** - beta regression (logit-transform + Ridge) on observed recovery data.
- **EAD** - credit conversion factor (CCF) approach for undrawn lines; outstanding balance for drawn.
- **Regulatory capital** - Vasicek ASRF formula: K = LGD × N(N⁻¹(PD)/√(1−R) + √(R/(1−R)) × N⁻¹(0.999)) - PD×LGD with maturity adjustment.
- **TtC PD** - converts point-in-time PD to through-the-cycle PD for capital reporting.

```
Required columns: default (0/1), recovery_rate, outstanding, limit, drawn
+ any numeric feature columns
```

---

### `ScorecardBuilder`

Classic points-based credit scorecard - the industry standard for regulatory transparency.

- Equal-frequency WoE binning with small-bin merging
- Information Value (IV) feature selection (auto-rejects IV < 0.02 and IV > 0.5)
- Logistic regression on WoE-encoded features
- Scorecard scaling: `score = offset + factor × ln(odds)`, PDO = 20

**IV interpretation:** < 0.02 useless · 0.02–0.1 weak · 0.1-0.3 medium · 0.3-0.5 strong · > 0.5 suspicious

---

### `ECLEngine`

IFRS 9 Expected Credit Loss with full stage lifecycle.

- **Stage 1** - performing: 12-month ECL
- **Stage 2** - SICR (significant increase in credit risk): lifetime ECL
- **Stage 3** - credit-impaired: lifetime ECL
- Monthly discounting at effective interest rate (EIR)
- Probability-weighted multi-scenario overlay (base / upside / adverse)
- Stage migration matrix for period-over-period reporting

---

### `StressTestor`

Macro scenario stress testing and portfolio VaR.

- PD and LGD sensitivity to GDP, unemployment, interest rate, and FX shocks
- Standard scenarios: base / adverse / severe (configurable)
- **Credit VaR** - Gaussian copula Monte Carlo using Vasicek single-factor model; reports P10/P50/P90, VaR, Expected Shortfall, and economic capital
- NPL trajectory projection over multi-month horizon
- Sensitivity sweep for regulatory ICAAP tables

---

### `FraudDetector`

Multi-layer real-time transaction fraud detection.

| Layer | Method | When active |
|---|---|---|
| Rule engine | Velocity · amount caps · geography · odd hours | Always |
| Isolation Forest | Unsupervised anomaly scoring | Always |
| LightGBM | Supervised fraud classifier | When `fraud` labels present |
| Behavioural | Per-customer z-score deviation | When fit on historical data |

Ensemble score = weighted combination. Output includes `anomaly_score`, `fraud_prob`, `behavioural_score`, `rule_flags`, `ensemble_score`, `risk_band` (LOW / MEDIUM / HIGH / CRITICAL).

---

### `ModelMonitor`

Production model stability tracking.

- **PSI** (Population Stability Index): < 0.10 stable · 0.10-0.25 warning · > 0.25 alert
- Time-series of AUC, Gini, KS, Brier across monitoring periods
- Decile table: score distribution and default rate by score decile
- Feature-level PSI for input drift detection
- Plain-English recommendations for model risk teams

---

### `ChampionChallenger`

Regulatory-grade A/B model governance.

- Route configurable traffic fraction to challenger
- **DeLong's AUC test** for statistically rigorous comparison
- Automatic promotion when challenger wins with p < α
- Immutable audit log for model risk and regulator review
- Rollback support

---

## Running the demo

```bash
pip install afrikana-risk
python examples/risk_demo.py
```

The demo generates a synthetic 2,000-account African fintech portfolio and runs the complete pipeline from credit scoring through to champion-challenger governance.

---

## Running tests

```bash
pip install "afrikana-risk[dev]"
pytest tests/ -v --cov=afrikana_risk
```

---

## Target markets

Built for and validated against data patterns from:

| Country | Cities | Currency |
|---|---|---|
| Kenya | Nairobi, Mombasa, Kisumu, Nakuru | KES |
| Nigeria | Lagos, Abuja, Kano, Port Harcourt | NGN |
| Rwanda | Kigali, Butare, Gisenyi | RWF |
| Uganda | Kampala, Entebbe, Jinja | UGX |
| Ghana | Accra, Kumasi, Tamale | GHS |
| Ethiopia | Addis Ababa, Dire Dawa | ETB |

Regulatory references: CBK (Kenya), CBN (Nigeria), BNR (Rwanda), BOU (Uganda) — aligned with IFRS 9 and Basel III/IV frameworks.

---

## Ecosystem

`afrikana-risk` is one of two sibling packages:

```
afrikana-analytics  →  customer · LTV · churn · forecasting · infrastructure
afrikana-risk       →  credit · fraud · ECL · stress testing · governance
```

Both packages share a common Mistral AI integration layer for natural language querying and narrative report generation.

---

## Author

**Peterson Mutegi** - Data Analyst · AI Engineer · Financial Engineer  
Nairobi, Kenya · [pitmuriuki@gmail.com](mailto:pitmuriuki@gmail.com)  
[GitHub](https://github.com/Peterson-Muriuki) · [LinkedIn](https://linkedin.com/in/peterson-mutegi)

Built on top of real analytical work in African financial services.

---

## License

MIT — see [LICENSE](LICENSE) for details.
