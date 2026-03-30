"""
ModelMonitor
============
Model stability and drift monitoring.

Implements:
* Population Stability Index (PSI)
* Gini / KS monitoring
* Simple feature drift summary
* Action recommendations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class ModelMonitor:
    def __init__(self):
        self.reference_df: pd.DataFrame | None = None
        self.reference_score_col: str | None = None

    def set_reference(self, df: pd.DataFrame, score_col: str = "score") -> None:
        if score_col not in df.columns:
            raise ValueError(f"'{score_col}' not found in reference data.")
        self.reference_df = df.copy()
        self.reference_score_col = score_col

    def monitor_period(
        self,
        current_df: pd.DataFrame,
        period: str,
        score_col: str = "score",
        target_col: str | None = None,
    ) -> dict:
        if self.reference_df is None or self.reference_score_col is None:
            raise ValueError("Reference population not set. Run set_reference() first.")

        psi_value = self._psi(
            self.reference_df[self.reference_score_col].astype(float),
            current_df[score_col].astype(float),
        )

        result = {
            "period": period,
            "psi_score": round(float(psi_value), 4),
            "psi_status": self._psi_status(psi_value),
        }

        if target_col and target_col in current_df.columns:
            y_true = current_df[target_col].astype(int)
            y_score = current_df[score_col].astype(float)

            if y_true.nunique() > 1:
                auc = roc_auc_score(y_true, y_score)
                gini = 2 * auc - 1
                ks = self._ks_statistic(y_true, y_score)
                result["auc"] = round(float(auc), 4)
                result["gini"] = round(float(gini), 4)
                result["ks"] = round(float(ks), 4)

        return result

    def recommend_action(self, report: dict) -> str:
        psi = report.get("psi_score", 0)

        if psi < 0.10:
            return "No action needed."
        if psi < 0.25:
            return "Monitor closely; mild population shift detected."
        return "Investigate urgently; consider recalibration or model refresh."

    def feature_drift(self, current_df: pd.DataFrame) -> pd.DataFrame:
        if self.reference_df is None:
            raise ValueError("Reference population not set. Run set_reference() first.")

        numeric_cols = [
            c for c in self.reference_df.select_dtypes(include=np.number).columns
            if c in current_df.columns
        ]

        rows = []
        for col in numeric_cols:
            ref_mean = float(self.reference_df[col].mean())
            cur_mean = float(current_df[col].mean())
            ref_std = float(self.reference_df[col].std(ddof=0)) + 1e-9
            z_shift = (cur_mean - ref_mean) / ref_std

            rows.append(
                {
                    "feature": col,
                    "reference_mean": round(ref_mean, 6),
                    "current_mean": round(cur_mean, 6),
                    "mean_shift": round(cur_mean - ref_mean, 6),
                    "z_shift": round(z_shift, 4),
                    "drift_flag": abs(z_shift) >= 0.5,
                }
            )

        return pd.DataFrame(rows).sort_values("z_shift", key=lambda s: s.abs(), ascending=False)

    def _psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
        actual = actual.replace([np.inf, -np.inf], np.nan).dropna()

        breakpoints = np.unique(
            np.quantile(expected, np.linspace(0, 1, bins + 1))
        )

        if len(breakpoints) < 3:
            return 0.0

        exp_counts, _ = np.histogram(expected, bins=breakpoints)
        act_counts, _ = np.histogram(actual, bins=breakpoints)

        exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, None)
        act_pct = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, None)

        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def _psi_status(self, psi: float) -> str:
        if psi < 0.10:
            return "stable"
        if psi < 0.25:
            return "moderate_shift"
        return "significant_shift"

    def _ks_statistic(self, y_true: pd.Series, y_score: pd.Series) -> float:
        df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score")
        bad = (df["y"] == 1).astype(int)
        good = (df["y"] == 0).astype(int)

        cum_bad = bad.cumsum() / max(bad.sum(), 1)
        cum_good = good.cumsum() / max(good.sum(), 1)

        return float(np.max(np.abs(cum_bad - cum_good)))