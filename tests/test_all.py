"""
Tests for afrikana-risk
Run: pytest tests/ -v --cov=afrikana_risk
"""

import numpy as np
import pandas as pd
import pytest

np.random.seed(0)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _portfolio(n=400, default_rate=0.12):
    logit = np.random.randn(n) * 0.8 - 1.5
    pd_   = 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "tenure_months":      np.random.randint(1, 60, n),
        "annual_income":      np.random.lognormal(11, 0.5, n),
        "dti":                np.random.beta(2, 5, n),
        "ltv":                np.random.beta(3, 4, n),
        "utilisation":        np.random.beta(2, 3, n),
        "past_delinquencies": np.random.poisson(0.3, n),
        "outstanding":        np.random.uniform(5_000, 500_000, n),
        "limit":              np.random.uniform(50_000, 1_000_000, n),
        "drawn":              np.random.uniform(1_000, 400_000, n),
        "remaining_months":   np.random.randint(12, 60, n),
        "recovery_rate":      np.where(
            np.random.rand(n) < default_rate,
            np.random.beta(5, 2, n), np.nan
        ),
        "default":            (np.random.rand(n) < pd_).astype(int),
    })
    # Ensure at least some defaults
    df.loc[:10, "default"] = 1
    df.loc[:10, "recovery_rate"] = 0.6
    return df


def _transactions(n=500):
    ts = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(n)],
        "customer_id":    np.random.choice(["C001", "C002", "C003"], n),
        "amount":         np.random.lognormal(8, 1, n),
        "timestamp":      ts,
        "fraud":          (np.random.rand(n) < 0.02).astype(int),
    })


@pytest.fixture
def portfolio():
    return _portfolio()


@pytest.fixture
def transactions():
    return _transactions()


# ─── CreditScorer ────────────────────────────────────────────────────────────

class TestCreditScorer:
    def test_fit_and_score(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        scorer.fit(portfolio)
        out = scorer.score(portfolio)
        assert "pd" in out.columns
        assert "lgd" in out.columns
        assert "ead" in out.columns
        assert "el" in out.columns
        assert "pd_grade" in out.columns
        assert "ifrs9_stage" in out.columns
        assert out["pd"].between(0, 1).all()
        assert out["lgd"].between(0, 1).all()

    def test_summary_keys(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        scorer.fit(portfolio)
        s = scorer.summary()
        assert "auc" in s
        assert "gini" in s
        assert s["auc"] > 0.5

    def test_regulatory_capital(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        scorer.fit(portfolio)
        scored = scorer.score(portfolio)
        cap = scorer.regulatory_capital(scored)
        assert "rwa" in cap.columns
        assert (cap["rwa"] >= 0).all()

    def test_cross_validate(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        scorer.fit(portfolio)
        cv = scorer.cross_validate(portfolio)
        assert cv["auc_mean"] > 0.5
        assert cv["n_folds"] == 5

    def test_ttc_pd(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        scorer.fit(portfolio)
        scored = scorer.score(portfolio)
        ttc = scorer.ttc_pd(scored)
        assert len(ttc) == len(scored)
        assert ttc.between(0, 1).all()

    def test_not_fitted_raises(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        with pytest.raises(RuntimeError):
            scorer.score(portfolio)

    def test_missing_target_raises(self, portfolio):
        from afrikana_risk.credit import CreditScorer
        scorer = CreditScorer()
        bad_df = portfolio.drop(columns=["default"])
        with pytest.raises(ValueError):
            scorer.fit(bad_df)


# ─── ScorecardBuilder ────────────────────────────────────────────────────────

class TestScorecardBuilder:
    def test_fit_and_score(self, portfolio):
        from afrikana_risk.credit import ScorecardBuilder
        builder = ScorecardBuilder()
        builder.fit(portfolio, target="default")
        scores = builder.score(portfolio)
        assert len(scores) == len(portfolio)
        assert scores.dtype in [np.int32, np.int64, int]

    def test_scorecard_table(self, portfolio):
        from afrikana_risk.credit import ScorecardBuilder
        builder = ScorecardBuilder()
        builder.fit(portfolio, target="default")
        card = builder.scorecard_table()
        assert "variable" in card.columns
        assert "woe" in card.columns
        assert "points" in card.columns

    def test_score_to_pd(self, portfolio):
        from afrikana_risk.credit import ScorecardBuilder
        builder = ScorecardBuilder()
        builder.fit(portfolio, target="default")
        scores = builder.score(portfolio)
        pds    = builder.score_to_pd(scores)
        assert pds.between(0, 1).all()

    def test_iv_summary(self, portfolio):
        from afrikana_risk.credit import ScorecardBuilder
        builder = ScorecardBuilder()
        builder.fit(portfolio, target="default")
        iv = builder.iv_summary()
        assert "iv" in iv.columns
        assert "category" in iv.columns
        assert "selected" in iv.columns


# ─── ECLEngine ───────────────────────────────────────────────────────────────

class TestECLEngine:
    def _scored(self, n=200):
        from afrikana_risk.credit import CreditScorer
        port = _portfolio(n)
        scorer = CreditScorer()
        scorer.fit(port)
        return scorer.score(port)

    def test_compute_columns(self):
        from afrikana_risk.risk import ECLEngine
        scored = self._scored()
        engine = ECLEngine(eir=0.15)
        out = engine.compute(scored)
        for col in ["ifrs9_stage", "ecl_12m", "ecl_lifetime", "ecl", "ecl_pw"]:
            assert col in out.columns

    def test_stage_values(self):
        from afrikana_risk.risk import ECLEngine
        scored = self._scored()
        engine = ECLEngine()
        out = engine.compute(scored)
        assert out["ifrs9_stage"].isin([1, 2, 3]).all()

    def test_ecl_positive(self):
        from afrikana_risk.risk import ECLEngine
        scored = self._scored()
        engine = ECLEngine()
        out = engine.compute(scored)
        assert (out["ecl"] >= 0).all()

    def test_scenario_probabilities_must_sum_to_one(self):
        from afrikana_risk.risk import ECLEngine, MacroScenario
        with pytest.raises(ValueError):
            ECLEngine(scenarios=[
                MacroScenario("a", probability=0.6),
                MacroScenario("b", probability=0.6),
            ])

    def test_portfolio_ecl(self):
        from afrikana_risk.risk import ECLEngine
        scored = self._scored()
        engine = ECLEngine()
        out = engine.compute(scored)
        summary = engine.portfolio_ecl(out)
        assert set(summary["ifrs9_stage"]).issubset({1, 2, 3})


# ─── StressTestor ────────────────────────────────────────────────────────────

class TestStressTestor:
    def _scored(self):
        from afrikana_risk.credit import CreditScorer
        port = _portfolio(300)
        scorer = CreditScorer()
        scorer.fit(port)
        return scorer.score(port)

    def test_scenario_stress(self):
        from afrikana_risk.risk import StressTestor, STANDARD_SCENARIOS
        st = StressTestor()
        scored = self._scored()
        results = st.scenario_stress(scored, STANDARD_SCENARIOS)
        assert "base" in results
        assert "adverse" in results
        assert "stressed_pd" in results["base"].columns

    def test_scenario_comparison(self):
        from afrikana_risk.risk import StressTestor, STANDARD_SCENARIOS
        st = StressTestor()
        scored = self._scored()
        results = st.scenario_stress(scored, STANDARD_SCENARIOS)
        comp = st.scenario_comparison(results)
        assert "total_ecl" in comp.columns

    def test_credit_var(self):
        from afrikana_risk.risk import StressTestor
        st = StressTestor(confidence_levels=[0.95, 0.99])
        scored = self._scored()
        var = st.credit_var(scored, n_simulations=1_000)
        assert "el" in var
        assert "var_0.990" in var
        assert var["var_0.990"] >= var["el"]

    def test_sensitivity_analysis(self):
        from afrikana_risk.risk import StressTestor
        st = StressTestor()
        scored = self._scored()
        sa = st.sensitivity_analysis(scored, variable="gdp_growth_shock",
                                     shock_range=(-3, 3), n_points=7)
        assert len(sa) == 7
        assert "total_ecl" in sa.columns


# ─── FraudDetector ───────────────────────────────────────────────────────────

class TestFraudDetector:
    def test_fit_and_score(self, transactions):
        from afrikana_risk.fraud import FraudDetector
        d = FraudDetector()
        d.fit(transactions)
        out = d.score(transactions)
        for col in ["anomaly_score", "ensemble_score", "risk_band"]:
            assert col in out.columns
        assert out["ensemble_score"].between(0, 1).all()

    def test_alerts(self, transactions):
        from afrikana_risk.fraud import FraudDetector
        d = FraudDetector()
        d.fit(transactions)
        scored = d.score(transactions)
        alerts = d.alerts(scored, threshold=0.5)
        assert (alerts["ensemble_score"] >= 0.5).all()

    def test_missing_columns_raises(self):
        from afrikana_risk.fraud import FraudDetector
        d = FraudDetector()
        with pytest.raises(ValueError):
            d.fit(pd.DataFrame({"amount": [100]}))


# ─── ModelMonitor ────────────────────────────────────────────────────────────

class TestModelMonitor:
    def _scored(self):
        from afrikana_risk.credit import CreditScorer
        port = _portfolio(400)
        sc = CreditScorer()
        sc.fit(port)
        return sc.score(port)

    def test_monitor_period(self):
        from afrikana_risk.monitoring import ModelMonitor
        scored = self._scored()
        m = ModelMonitor()
        m.set_reference(scored, score_col="pd")
        shifted = scored.copy()
        shifted["pd"] = (shifted["pd"] * 1.3).clip(0, 1)
        report = m.monitor_period(shifted, period="T1", score_col="pd", target_col="default")
        assert "psi_score" in report
        assert report["psi_status"] in ["STABLE", "WARNING", "ALERT"]

    def test_feature_drift(self):
        from afrikana_risk.monitoring import ModelMonitor
        scored = self._scored()
        m = ModelMonitor()
        m.set_reference(scored)
        fd = m.feature_drift(scored)
        assert "psi" in fd.columns

    def test_drift_report(self):
        from afrikana_risk.monitoring import ModelMonitor
        scored = self._scored()
        m = ModelMonitor()
        m.set_reference(scored, score_col="pd")
        m.monitor_period(scored, period="T1", score_col="pd")
        dr = m.drift_report()
        assert len(dr) == 1


# ─── ChampionChallenger ──────────────────────────────────────────────────────

class TestChampionChallenger:
    def _models_and_data(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        port  = _portfolio(400)
        feats = ["dti", "ltv", "utilisation", "past_delinquencies"]
        X, y  = port[feats].fillna(0), port["default"]
        m1 = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=200))])
        m2 = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=200, C=0.5))])
        m1.fit(X, y); m2.fit(X, y)
        return m1, m2, port[feats + ["default"]]

    def test_register_and_route(self):
        from afrikana_risk.monitoring import ChampionChallenger
        m1, m2, _ = self._models_and_data()
        cc = ChampionChallenger(challenger_traffic=0.5)
        cc.register_champion(m1,  name="v1", version="1.0")
        cc.register_challenger(m2, name="v2", version="2.0")
        model_obj, model_id = cc.route()
        assert model_obj is not None

    def test_evaluate(self):
        from afrikana_risk.monitoring import ChampionChallenger
        m1, m2, test_df = self._models_and_data()
        cc = ChampionChallenger()
        cc.register_champion(m1,  name="v1", version="1.0")
        cc.register_challenger(m2, name="v2", version="2.0")
        result = cc.evaluate(test_df, target_col="default")
        assert "champion_auc" in result
        assert "p_value" in result
        assert "recommendation" in result

    def test_promote(self):
        from afrikana_risk.monitoring import ChampionChallenger
        m1, m2, test_df = self._models_and_data()
        cc = ChampionChallenger()
        cc.register_champion(m1,  name="v1", version="1.0")
        cc.register_challenger(m2, name="v2", version="2.0")
        cc.promote_challenger()
        assert cc._challenger is None
        assert cc._champion.name == "v2"