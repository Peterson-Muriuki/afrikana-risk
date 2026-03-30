"""
ScorecardBuilder
================
Traditional credit scorecard development pipeline:

    Raw features → WoE binning → IV feature selection →
    Logistic regression → Point-score scaling → Scorecard table

The output is a classic points-based scorecard (like FICO) where each
variable's bin maps to an integer score contribution. The total score
predicts creditworthiness and maps to a PD via the scaling equation:

    score = offset + factor × ln(odds)
    PD    = 1 / (1 + exp(offset/factor - score/factor))

References
----------
* Siddiqi, N. (2006). Credit Risk Scorecards. Wiley.
* Anderson, R. (2007). The Credit Scoring Toolkit. OUP.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScorecardConfig:
    """
    Parameters
    ----------
    target_score   : int   — desired score at ``target_odds``
    target_odds    : float — odds (good:bad) at ``target_score``
    pdo            : int   — points-to-double-odds (industry standard: 20)
    min_iv         : float — minimum Information Value to include a variable
    max_bins       : int   — maximum WoE bins per variable
    min_bin_size   : float — minimum fraction of observations per bin
    test_size      : float — held-out proportion for evaluation
    random_state   : int
    """
    target_score:  int   = 600
    target_odds:   float = 50.0     # 50:1 good:bad at score 600
    pdo:           int   = 20
    min_iv:        float = 0.02     # weak predictors < 0.02
    max_bins:      int   = 10
    min_bin_size:  float = 0.05
    test_size:     float = 0.20
    random_state:  int   = 42


# ---------------------------------------------------------------------------
# WoE / IV utilities
# ---------------------------------------------------------------------------

def _woe_iv(df: pd.DataFrame, var: str, target: str, n_bins: int = 10,
             min_size: float = 0.05) -> tuple[pd.DataFrame, float]:
    """Compute WoE and IV for a single variable using equal-frequency binning."""
    eps = 1e-8
    col = df[var].copy()
    y   = df[target].astype(int)

    total_events    = y.sum()
    total_nonevents = len(y) - total_events

    if total_events == 0 or total_nonevents == 0:
        return pd.DataFrame(), 0.0

    # Bin continuous variables; leave categoricals as-is
    if pd.api.types.is_numeric_dtype(col):
        try:
            col_binned = pd.qcut(col, q=n_bins, duplicates="drop")
        except Exception:
            col_binned = pd.cut(col, bins=n_bins, duplicates="drop")
    else:
        col_binned = col.astype(str)

    tmp = pd.DataFrame({"bin": col_binned, "target": y})
    agg = tmp.groupby("bin", observed=True)["target"].agg(
        events="sum", total="count"
    ).reset_index()
    agg["nonevents"] = agg["total"] - agg["events"]

    # Merge small bins iteratively
    while True:
        small = agg["total"] / len(df) < min_size
        if not small.any() or len(agg) <= 2:
            break
        idx = agg[small].index[0]
        merge_to = idx + 1 if idx < len(agg) - 1 else idx - 1
        agg.loc[merge_to, "events"]    += agg.loc[idx, "events"]
        agg.loc[merge_to, "nonevents"] += agg.loc[idx, "nonevents"]
        agg.loc[merge_to, "total"]     += agg.loc[idx, "total"]
        agg = agg.drop(idx).reset_index(drop=True)

    agg["dist_events"]    = agg["events"] / (total_events + eps)
    agg["dist_nonevents"] = agg["nonevents"] / (total_nonevents + eps)
    agg["woe"] = np.log(
        (agg["dist_events"] + eps) / (agg["dist_nonevents"] + eps)
    )
    agg["iv_component"] = (agg["dist_events"] - agg["dist_nonevents"]) * agg["woe"]
    agg["variable"]     = var
    iv = float(agg["iv_component"].sum())

    return agg[["variable", "bin", "events", "nonevents", "total",
                "dist_events", "dist_nonevents", "woe", "iv_component"]], iv


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ScorecardBuilder:
    """
    Build a traditional points-based credit scorecard.

    Workflow
    --------
    1. ``fit(df, target)``       — WoE binning, IV filter, logistic regression
    2. ``scorecard_table()``     — human-readable scorecard table
    3. ``transform(df)``         — apply WoE encoding
    4. ``score(df)``             — compute individual credit scores
    5. ``score_to_pd(scores)``   — convert scores to PD via scaling equation
    6. ``iv_summary()``          — variable importance via Information Value

    Information Value (IV) interpretation
    --------------------------------------
    < 0.02  — useless
    0.02–0.1 — weak
    0.1–0.3  — medium
    0.3–0.5  — strong
    > 0.5   — suspicious (likely data leakage)

    Examples
    --------
    >>> builder = ScorecardBuilder()
    >>> builder.fit(train_df, target="default")
    >>> card = builder.scorecard_table()
    >>> scores = builder.score(portfolio_df)
    >>> pds    = builder.score_to_pd(scores)
    """

    def __init__(self, config: ScorecardConfig | None = None):
        self.config = config or ScorecardConfig()
        self._woe_tables: dict[str, pd.DataFrame] = {}
        self._iv: dict[str, float] = {}
        self._selected_vars: list[str] = []
        self._lr_coefs: dict[str, float] = {}
        self._lr_intercept: float = 0.0
        self._offset: float = 0.0
        self._factor: float = 0.0
        self._metrics: dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target: str = "default",
            feature_cols: list[str] | None = None) -> "ScorecardBuilder":
        """
        Parameters
        ----------
        df          : training DataFrame
        target      : binary target column (1 = default / bad)
        feature_cols: columns to consider; auto-detected if None
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")

        cols = feature_cols or [
            c for c in df.select_dtypes(include=["number", "category", "object"]).columns
            if c != target
        ]

        # Step 1 — WoE / IV per variable
        for var in cols:
            try:
                woe_tbl, iv = _woe_iv(
                    df, var, target,
                    n_bins=self.config.max_bins,
                    min_size=self.config.min_bin_size,
                )
                self._woe_tables[var] = woe_tbl
                self._iv[var] = iv
            except Exception as e:
                warnings.warn(f"Skipping '{var}': {e}", UserWarning, stacklevel=2)

        # Step 2 — IV-based feature selection
        self._selected_vars = [
            v for v, iv in self._iv.items()
            if self.config.min_iv <= iv <= 0.5     # >0.5 = suspect leakage
        ]
        if not self._selected_vars:
            raise RuntimeError(
                "No variables passed the IV filter. "
                "Try lowering config.min_iv or check for data leakage."
            )

        # Step 3 — WoE-encode and fit logistic regression
        X_woe = self._woe_encode(df)
        y     = df[target].astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_woe, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state,
        )
        lr = LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state,
            class_weight="balanced",
        )
        lr.fit(X_tr, y_tr)

        self._lr_coefs     = dict(zip(self._selected_vars, lr.coef_[0]))
        self._lr_intercept = float(lr.intercept_[0])
        self._metrics      = self._evaluate(lr, X_te, y_te)

        # Step 4 — Scorecard scaling constants
        # score = offset + factor × ln(odds)
        cfg = self.config
        self._factor = cfg.pdo / np.log(2)
        self._offset = cfg.target_score - self._factor * np.log(cfg.target_odds)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform / Score
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace raw features with WoE values for the selected variables."""
        self._check_fitted()
        return self._woe_encode(df)

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Compute individual credit scores (higher = lower risk)."""
        self._check_fitted()
        X_woe = self._woe_encode(df)
        log_odds = X_woe.values @ np.array(
            [self._lr_coefs[v] for v in self._selected_vars]
        ) + self._lr_intercept
        scores = self._offset + self._factor * log_odds
        return pd.Series(scores.round().astype(int), index=df.index, name="credit_score")

    def score_to_pd(self, scores: pd.Series) -> pd.Series:
        """Convert credit scores to PD via inverse of the scaling equation."""
        self._check_fitted()
        log_odds = (scores - self._offset) / self._factor
        pd_vals  = 1 / (1 + np.exp(log_odds))
        return pd.Series(pd_vals, index=scores.index, name="pd_from_score")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def scorecard_table(self) -> pd.DataFrame:
        """
        Return the full scorecard table with point allocations per bin.

        Each row is a (variable, bin) pair with its WoE and point score.
        The point score is: -factor × coefficient × WoE + offset/n_vars.
        """
        self._check_fitted()
        n_vars = len(self._selected_vars)
        rows   = []
        for var in self._selected_vars:
            tbl  = self._woe_tables[var].copy()
            coef = self._lr_coefs.get(var, 0.0)
            # Points = -factor × β × WoE + (offset / n_vars)
            tbl["points"] = (
                -self._factor * coef * tbl["woe"] + self._offset / n_vars
            ).round().astype(int)
            tbl["iv"]   = self._iv.get(var, 0.0)
            tbl["coef"] = coef
            rows.append(tbl)

        return pd.concat(rows, ignore_index=True)[
            ["variable", "bin", "woe", "points", "iv", "coef",
             "events", "nonevents", "total"]
        ]

    def iv_summary(self) -> pd.DataFrame:
        """Return all variables with their IV and predictive category."""
        def category(iv):
            if iv < 0.02:   return "useless"
            if iv < 0.10:   return "weak"
            if iv < 0.30:   return "medium"
            if iv < 0.50:   return "strong"
            return "suspicious"

        return (
            pd.DataFrame({"variable": list(self._iv.keys()),
                          "iv":       list(self._iv.values())})
            .assign(
                category=lambda d: d["iv"].apply(category),
                selected=lambda d: d["variable"].isin(self._selected_vars),
            )
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def summary(self) -> dict:
        """Return training evaluation metrics."""
        self._check_fitted()
        return {
            **self._metrics,
            "selected_variables": len(self._selected_vars),
            "total_variables":    len(self._iv),
            "target_score":       self.config.target_score,
            "pdo":                self.config.pdo,
            "factor":             round(self._factor, 4),
            "offset":             round(self._offset, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _woe_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw values to WoE for each selected variable."""
        result = {}
        for var in self._selected_vars:
            tbl  = self._woe_tables[var]
            col  = df[var].copy()

            if pd.api.types.is_numeric_dtype(col):
                # Determine bin edges from training table
                bins = self._extract_bin_edges(tbl)
                if bins is not None:
                    col_binned = pd.cut(col, bins=bins, include_lowest=True)
                else:
                    col_binned = col.astype(str)
            else:
                col_binned = col.astype(str)

            woe_map    = dict(zip(tbl["bin"].astype(str), tbl["woe"]))
            col_str    = col_binned.astype(str)
            # Default unmatched to global mean WoE (zero)
            woe_vals   = col_str.map(woe_map).fillna(0.0)
            result[var] = woe_vals.values

        return pd.DataFrame(result, index=df.index)

    @staticmethod
    def _extract_bin_edges(woe_tbl: pd.DataFrame) -> np.ndarray | None:
        """Extract numeric bin edges from interval-labelled WoE table."""
        try:
            intervals = pd.IntervalIndex(
                [pd.Interval(b.left, b.right) for b in
                 pd.cut([], bins=1).cat.categories.__class__([]) ]
            )
        except Exception:
            pass

        bins = woe_tbl["bin"]
        if not hasattr(bins.iloc[0], "left"):
            return None
        edges = sorted(set(
            [b.left for b in bins] + [bins.iloc[-1].right]
        ))
        edges[0]  = -np.inf
        edges[-1] =  np.inf
        return np.array(edges)

    def _evaluate(self, model, X_te: pd.DataFrame, y_te: pd.Series) -> dict:
        proba = model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        return {
            "auc":  round(auc, 4),
            "gini": round(2 * auc - 1, 4),
        }

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before using this method.")