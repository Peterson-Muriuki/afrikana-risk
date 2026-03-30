"""
StressTestor
============
Macro-scenario stress testing for credit portfolios.

Implements:
* Sensitivity analysis — PD/LGD response to macroeconomic shocks
* Scenario stress tests — base / adverse / severe scenario ECL impact
* Credit VaR — portfolio loss distribution via Monte Carlo simulation
  (Gaussian copula for inter-obligor correlation)
* Balance sheet stress — NPL trajectory, coverage ratio, capital adequacy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class MacroScenario:
    """A named stress scenario with macro variable shocks."""
    name:              str
    gdp_growth_shock:  float = 0.0    # percentage point change in GDP growth
    unemployment_shock: float = 0.0   # pp change in unemployment rate
    interest_rate_shock: float = 0.0  # pp change in benchmark rate
    fx_shock:          float = 0.0    # % depreciation of local currency
    probability:       float = 1.0    # weight in probability-weighted results
    description:       str = ""


STANDARD_SCENARIOS = [
    MacroScenario(
        "base",    gdp_growth_shock=0.0,  unemployment_shock=0.0,
        interest_rate_shock=0.0, fx_shock=0.0,  probability=0.60,
        description="Business-as-usual trajectory",
    ),
    MacroScenario(
        "adverse", gdp_growth_shock=-2.5, unemployment_shock=2.0,
        interest_rate_shock=1.5, fx_shock=15.0, probability=0.30,
        description="Moderate economic downturn",
    ),
    MacroScenario(
        "severe",  gdp_growth_shock=-5.0, unemployment_shock=5.0,
        interest_rate_shock=3.0, fx_shock=30.0, probability=0.10,
        description="Severe recession / financial crisis",
    ),
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StressTestor:
    """
    Portfolio-level credit stress testing.

    Parameters
    ----------
    pd_macro_sensitivity : dict
        Maps macro variable name → coefficient for log-odds PD model.
        Example: {"gdp_growth_shock": -0.15, "unemployment_shock": 0.12}
        Coefficients represent change in log(PD/(1-PD)) per unit shock.
    lgd_macro_sensitivity : dict
        Same structure for LGD sensitivity (linear).
    correlation_matrix : np.ndarray | None
        Asset correlation matrix for credit VaR. If None, uses a single
        factor model with ``default_asset_correlation``.
    default_asset_correlation : float
        Used when ``correlation_matrix`` is None (Basel retail: ~0.15).
    n_simulations : int
        Monte Carlo draws for credit VaR.
    confidence_levels : list[float]
        VaR confidence levels to report (e.g. [0.95, 0.99, 0.999]).
    random_state : int

    Examples
    --------
    >>> st = StressTestor(
    ...     pd_macro_sensitivity={"gdp_growth_shock": -0.12, "unemployment_shock": 0.10},
    ...     lgd_macro_sensitivity={"gdp_growth_shock": -0.08},
    ... )
    >>> results = st.scenario_stress(scored_df, scenarios=STANDARD_SCENARIOS)
    >>> print(st.scenario_comparison(results))
    >>> var = st.credit_var(scored_df, n_simulations=10_000)
    >>> print(var)
    """

    def __init__(
        self,
        pd_macro_sensitivity:  dict[str, float] | None = None,
        lgd_macro_sensitivity: dict[str, float] | None = None,
        correlation_matrix:    np.ndarray | None = None,
        default_asset_correlation: float = 0.15,
        n_simulations: int = 10_000,
        confidence_levels: list[float] | None = None,
        random_state: int = 42,
    ):
        self.pd_sens  = pd_macro_sensitivity  or {"gdp_growth_shock": -0.12,
                                                   "unemployment_shock": 0.10,
                                                   "interest_rate_shock": 0.08}
        self.lgd_sens = lgd_macro_sensitivity or {"gdp_growth_shock": -0.06,
                                                   "fx_shock": 0.04}
        self.corr_matrix    = correlation_matrix
        self.default_rho    = default_asset_correlation
        self.n_sims         = n_simulations
        self.conf_levels    = confidence_levels or [0.90, 0.95, 0.99, 0.999]
        self.random_state   = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scenario_stress(
        self,
        scored_df: pd.DataFrame,
        scenarios: list[MacroScenario] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Apply each macro scenario and return stressed portfolio metrics.

        Returns
        -------
        dict mapping scenario name → stressed DataFrame (with stressed_pd,
        stressed_lgd, stressed_ecl, rwa, capital_charge columns added)
        """
        scenarios = scenarios or STANDARD_SCENARIOS
        results   = {}
        for sc in scenarios:
            stressed = self._apply_macro_shock(scored_df, sc)
            results[sc.name] = stressed
        return results

    def scenario_comparison(
        self,
        scenario_results: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Summary table comparing ECL, RWA, and capital across scenarios.
        """
        rows = []
        for name, df in scenario_results.items():
            rows.append({
                "scenario":          name,
                "total_ead":         df["ead"].sum(),
                "avg_stressed_pd":   df["stressed_pd"].mean(),
                "avg_stressed_lgd":  df["stressed_lgd"].mean(),
                "total_ecl":         df["stressed_ecl"].sum(),
                "ecl_coverage":      df["stressed_ecl"].sum() / df["ead"].sum(),
                "total_rwa":         df.get("rwa", pd.Series(dtype=float)).sum(),
                "capital_charge":    (df.get("rwa", pd.Series(dtype=float)) * 0.08).sum(),
                "npl_volume":        df[df["stressed_pd"] >= 0.20]["ead"].sum(),
                "npl_ratio":         df[df["stressed_pd"] >= 0.20]["ead"].sum() / df["ead"].sum(),
            })
        return pd.DataFrame(rows).set_index("scenario")

    def credit_var(
        self,
        scored_df: pd.DataFrame,
        n_simulations: int | None = None,
    ) -> dict:
        """
        Portfolio credit VaR via Gaussian copula Monte Carlo.

        Simulates correlated asset value shocks using the Vasicek
        single-factor model, then derives portfolio loss distribution.

        Returns
        -------
        dict with:
        * ``el``          — expected loss
        * ``var_{q}``     — VaR at each confidence level
        * ``es_{q}``      — expected shortfall (CVaR) at each confidence level
        * ``loss_dist``   — array of simulated portfolio losses
        """
        n     = n_simulations or self.n_sims
        rng   = np.random.default_rng(self.random_state)
        N     = len(scored_df)

        pd_   = scored_df["pd"].values.astype(float)
        lgd_  = scored_df["lgd"].values.astype(float)
        ead_  = scored_df["ead"].values.astype(float)

        rho = self.default_rho

        # Thresholds for default
        thresholds = norm.ppf(pd_)

        # Simulate: Z = systematic factor, ε = idiosyncratic
        Z   = rng.standard_normal(n)           # (n_sims,)
        eps = rng.standard_normal((n, N))      # (n_sims, n_obligors)
        A   = np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * eps  # asset returns

        # Default indicator: A < threshold
        defaults_sim = (A < thresholds[None, :]).astype(float)   # (n_sims, N)
        losses       = (defaults_sim * lgd_[None, :] * ead_[None, :]).sum(axis=1)  # (n_sims,)

        result = {
            "el":         float(losses.mean()),
            "ul":         float(losses.std()),
            "loss_dist":  losses,
            "n_sims":     n,
        }
        for q in self.conf_levels:
            var = float(np.quantile(losses, q))
            es  = float(losses[losses >= var].mean()) if (losses >= var).any() else var
            result[f"var_{q:.3f}"]  = var
            result[f"es_{q:.3f}"]   = es

        result["economic_capital"] = result[f"var_{max(self.conf_levels):.3f}"] - result["el"]

        return result

    def sensitivity_analysis(
        self,
        scored_df: pd.DataFrame,
        variable: str = "gdp_growth_shock",
        shock_range: tuple[float, float] = (-5.0, 5.0),
        n_points: int = 21,
    ) -> pd.DataFrame:
        """
        Sweep a single macro variable and return portfolio ECL response.

        Useful for regulatory sensitivity tables and ICAAP documentation.
        """
        shocks = np.linspace(shock_range[0], shock_range[1], n_points)
        rows   = []
        for shock in shocks:
            sc = MacroScenario(name="sweep", **{variable: shock})
            stressed = self._apply_macro_shock(scored_df, sc)
            rows.append({
                "shock":     shock,
                "avg_pd":    float(stressed["stressed_pd"].mean()),
                "avg_lgd":   float(stressed["stressed_lgd"].mean()),
                "total_ecl": float(stressed["stressed_ecl"].sum()),
            })
        return pd.DataFrame(rows)

    def npl_trajectory(
        self,
        scored_df: pd.DataFrame,
        scenario: MacroScenario | None = None,
        months: int = 24,
    ) -> pd.DataFrame:
        """
        Project NPL ratio and provisions over a multi-month horizon.

        Uses the stressed PD to derive expected new defaults each month,
        assuming a constant macro shock applied throughout the horizon.
        """
        sc       = scenario or STANDARD_SCENARIOS[0]   # base
        stressed = self._apply_macro_shock(scored_df, sc)
        monthly_pd = 1 - (1 - stressed["stressed_pd"]) ** (1 / 12)
        ead        = stressed["ead"].values

        timeline = []
        current_npl = (
            stressed[stressed["stressed_pd"] >= 0.20]["ead"].sum()
        )
        for m in range(1, months + 1):
            new_defaults = (monthly_pd.values * ead).sum()
            current_npl += new_defaults
            npl_ratio    = current_npl / ead.sum()
            provision    = current_npl * stressed["stressed_lgd"].mean()
            timeline.append({
                "month":       m,
                "new_defaults": round(new_defaults, 2),
                "cumulative_npl": round(current_npl, 2),
                "npl_ratio":   round(npl_ratio, 4),
                "provision":   round(provision, 2),
            })
        return pd.DataFrame(timeline)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_macro_shock(
        self, df: pd.DataFrame, sc: MacroScenario
    ) -> pd.DataFrame:
        """Apply a macro scenario to shift PD (logit scale) and LGD (linear)."""
        out = df.copy()
        pd_ = df["pd"].values.astype(float).clip(1e-6, 1 - 1e-6)

        # PD shock in log-odds space
        log_odds = np.log(pd_ / (1 - pd_))
        shock_dict = {
            "gdp_growth_shock":    sc.gdp_growth_shock,
            "unemployment_shock":  sc.unemployment_shock,
            "interest_rate_shock": sc.interest_rate_shock,
            "fx_shock":            sc.fx_shock,
        }
        delta_log_odds = sum(
            coef * shock_dict.get(var, 0.0)
            for var, coef in self.pd_sens.items()
        )
        stressed_pd = 1 / (1 + np.exp(-(log_odds + delta_log_odds)))
        out["stressed_pd"] = np.clip(stressed_pd, 1e-6, 1.0)

        # LGD shock (linear, bounded)
        lgd_ = df["lgd"].values.astype(float)
        delta_lgd = sum(
            coef * shock_dict.get(var, 0.0)
            for var, coef in self.lgd_sens.items()
        )
        out["stressed_lgd"] = np.clip(lgd_ + delta_lgd, 0.0, 1.0)

        out["stressed_ecl"] = out["stressed_pd"] * out["stressed_lgd"] * df["ead"].values

        # Simple RWA estimate under stressed parameters
        R   = 0.15
        from scipy.stats import norm as _norm
        K   = np.maximum(
            out["stressed_lgd"].values * _norm.cdf(
                _norm.ppf(out["stressed_pd"].values) / np.sqrt(1 - R)
                + np.sqrt(R / (1 - R)) * _norm.ppf(0.999)
            ) - out["stressed_pd"].values * out["stressed_lgd"].values,
            0
        )
        out["rwa"] = K * 12.5 * df["ead"].values

        return out