"""
ECLEngine
=========
IFRS 9 Expected Credit Loss (ECL) computation across all three stages.

Stage definitions (bank-configurable thresholds)
-------------------------------------------------
Stage 1 — performing : 12-month ECL
Stage 2 — SICR       : lifetime ECL (significant increase in credit risk)
Stage 3 — impaired   : lifetime ECL on credit-impaired assets

ECL formula
-----------
    ECL = PD × LGD × EAD × DF

where DF is a discount factor applying the effective interest rate (EIR)
over the expected lifetime of the instrument.

Forward-looking adjustments
---------------------------
Macro overlay is applied via a probability-weighted multi-scenario approach
(IFRS 9 §5.5.17): base, upside, and downside scenarios with user-supplied
probability weights and macro multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StageThresholds:
    """PD thresholds separating IFRS 9 stages."""

    stage_2_pd: float = 0.03
    stage_3_pd: float = 0.20


@dataclass
class MacroScenario:
    """Single macro scenario for forward-looking ECL overlay."""

    name: str
    probability: float
    pd_multiplier: float = 1.0
    lgd_multiplier: float = 1.0


class ECLEngine:
    """
    Compute IFRS 9 Expected Credit Loss at instrument level.
    """

    def __init__(
        self,
        eir: float = 0.15,
        max_lifetime_months: int = 60,
        stage_thresholds: StageThresholds | None = None,
        scenarios: list[MacroScenario] | None = None,
    ):
        self.eir = eir
        self.max_lifetime_months = max_lifetime_months
        self.thresholds = stage_thresholds or StageThresholds()
        self.scenarios = scenarios or []
        self._validate_scenarios()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ECL for each instrument.
        """
        self._validate_input(df)
        out = df.copy()

        pd_ = df["pd"].values.astype(float)
        lgd_ = df["lgd"].values.astype(float)
        ead_ = df["ead"].values.astype(float)
        eir_ = (
            df["eir"].values.astype(float)
            if "eir" in df.columns
            else np.full(len(df), self.eir)
        )
        rem_ = (
            df["remaining_months"].values.astype(float)
            if "remaining_months" in df.columns
            else np.full(len(df), self.max_lifetime_months)
        )

        out["ifrs9_stage"] = np.where(
            pd_ >= self.thresholds.stage_3_pd,
            3,
            np.where(pd_ >= self.thresholds.stage_2_pd, 2, 1),
        )

        out["ecl_12m"] = self._ecl_nmonths(pd_, lgd_, ead_, eir_, months=12)
        out["ecl_lifetime"] = self._ecl_lifetime(pd_, lgd_, ead_, eir_, rem_)
        out["ecl"] = np.where(
            out["ifrs9_stage"] == 1,
            out["ecl_12m"],
            out["ecl_lifetime"],
        )

        if self.scenarios:
            out = self._apply_scenarios(out, pd_, lgd_, ead_, eir_, rem_)
            scenario_cols = [f"ecl_scenario_{s.name}" for s in self.scenarios]
            weights = np.array([s.probability for s in self.scenarios])
            pwecl = np.zeros(len(out))
            for col, w in zip(scenario_cols, weights):
                pwecl += out[col].values * w
            out["ecl_pw"] = pwecl
        else:
            out["ecl_pw"] = out["ecl"]

        if "existing_allowance" in df.columns:
            out["provision"] = np.maximum(out["ecl_pw"] - df["existing_allowance"], 0)
        else:
            out["provision"] = out["ecl_pw"]

        return out

    def portfolio_ecl(self, ecl_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ECL by IFRS 9 stage."""
        return (
            ecl_df.groupby("ifrs9_stage")
            .agg(
                count=("ecl", "count"),
                total_ead=("ead", "sum"),
                total_ecl_12m=("ecl_12m", "sum"),
                total_ecl_lifetime=("ecl_lifetime", "sum"),
                total_ecl=("ecl_pw", "sum"),
                avg_pd=("pd", "mean"),
                avg_lgd=("lgd", "mean"),
            )
            .assign(
                coverage_ratio=lambda d: d["total_ecl"] / d["total_ead"],
                pct_portfolio=lambda d: d["count"] / d["count"].sum(),
            )
            .reset_index()
        )

    def stage_migration(self, current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a stage migration matrix comparing two periods.
        """
        mig = pd.merge(
            prior[["ifrs9_stage"]].rename(columns={"ifrs9_stage": "prior_stage"}),
            current[["ifrs9_stage"]].rename(columns={"ifrs9_stage": "current_stage"}),
            left_index=True,
            right_index=True,
        )
        return pd.crosstab(
            mig["prior_stage"],
            mig["current_stage"],
            rownames=["Prior stage"],
            colnames=["Current stage"],
            margins=True,
        )

    def _ecl_nmonths(
        self,
        pd_: np.ndarray,
        lgd_: np.ndarray,
        ead_: np.ndarray,
        eir_: np.ndarray,
        months: int,
    ) -> np.ndarray:
        """Compute ECL over a fixed horizon with monthly discounting."""
        monthly_eir = eir_ / 12
        monthly_pd = 1 - (1 - pd_) ** (1 / 12)
        ecl = np.zeros(len(pd_))

        for t in range(1, months + 1):
            survival = (1 - monthly_pd) ** (t - 1)
            df_factor = 1 / (1 + monthly_eir) ** t
            ecl += survival * monthly_pd * lgd_ * ead_ * df_factor

        return ecl

    def _ecl_lifetime(
        self,
        pd_: np.ndarray,
        lgd_: np.ndarray,
        ead_: np.ndarray,
        eir_: np.ndarray,
        remaining_months: np.ndarray,
    ) -> np.ndarray:
        """Compute lifetime ECL with instrument-level remaining term."""
        max_m = (
            int(remaining_months.max())
            if len(remaining_months) > 0
            else self.max_lifetime_months
        )
        monthly_eir = eir_ / 12
        monthly_pd = 1 - (1 - pd_) ** (1 / 12)
        ecl = np.zeros(len(pd_))

        for t in range(1, max_m + 1):
            active = (remaining_months >= t).astype(float)
            survival = (1 - monthly_pd) ** (t - 1)
            df_factor = 1 / (1 + monthly_eir) ** t
            ecl += active * survival * monthly_pd * lgd_ * ead_ * df_factor

        return ecl

    def _apply_scenarios(
        self,
        out: pd.DataFrame,
        pd_: np.ndarray,
        lgd_: np.ndarray,
        ead_: np.ndarray,
        eir_: np.ndarray,
        rem_: np.ndarray,
    ) -> pd.DataFrame:
        for scenario in self.scenarios:
            s_pd = np.minimum(pd_ * scenario.pd_multiplier, 1.0)
            s_lgd = np.minimum(lgd_ * scenario.lgd_multiplier, 1.0)
            stage = out["ifrs9_stage"].values
            ecl_s = np.where(
                stage == 1,
                self._ecl_nmonths(s_pd, s_lgd, ead_, eir_, months=12),
                self._ecl_lifetime(s_pd, s_lgd, ead_, eir_, rem_),
            )
            out[f"ecl_scenario_{scenario.name}"] = ecl_s
        return out

    def _validate_scenarios(self):
        if self.scenarios:
            total = sum(s.probability for s in self.scenarios)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Scenario probabilities must sum to 1.0 (got {total:.4f})."
                )

    def _validate_input(self, df: pd.DataFrame):
        required = {"pd", "lgd", "ead"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")