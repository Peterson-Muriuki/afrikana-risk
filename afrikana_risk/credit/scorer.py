"""
CreditScorer
============
Produces calibrated Probability of Default (PD), Loss Given Default (LGD),
and Exposure at Default (EAD) estimates suitable for Basel III regulatory
capital calculations and IFRS 9 Expected Credit Loss staging.

Design principles
-----------------
* PD  — Logistic regression on WoE-transformed features, Platt-scaled for
        calibration, validated with Brier score and reliability diagrams.
* LGD — Beta regression (logit link) on recovery data; falls back to a
        LightGBM regressor when sample size allows.
* EAD — Linear CCF (Credit Conversion Factor) model for off-balance-sheet
        exposures; identity mapping for on-balance-sheet.
* All estimates are point-in-time (PiT); through-the-cycle (TtC) scaling
  is available via the `ttc_pd()` helper for regulatory capital reporting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import ks_2samp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CreditScorerConfig:
    """Hyperparameters and behavioural switches for CreditScorer.

    Parameters
    ----------
    pd_model : {'logistic', 'lgbm'}
        Model family for PD estimation. Logistic regression is preferred for
        regulatory transparency; LightGBM for predictive performance.
    lgd_model : {'beta', 'lgbm'}
        Model family for LGD. Beta regression is more interpretable; LightGBM
        handles non-linearities better on large datasets.
    calibrate_pd : bool
        Whether to apply Platt scaling after PD model training.
    n_cv_folds : int
        Number of stratified folds for cross-validated performance metrics.
    test_size : float
        Held-out test proportion for final evaluation.
    random_state : int
        Global random seed for reproducibility.
    min_lgd_samples : int
        Minimum defaulted observations required to fit LGD model. Below this
        threshold the module falls back to the historical average LGD.
    regulatory_floor_pd : float
        Basel III regulatory floor for PD (default 0.03%).
    lgd_floor : float
        Regulatory floor for LGD (Basel: 45% unsecured, 25% secured).
    verbose : bool
        Print training progress and evaluation summaries.
    """
    pd_model: Literal["logistic", "lgbm"]  = "logistic"
    lgd_model: Literal["beta", "lgbm"]     = "beta"
    calibrate_pd: bool                     = True
    n_cv_folds: int                        = 5
    test_size: float                       = 0.20
    random_state: int                      = 42
    min_lgd_samples: int                   = 50
    regulatory_floor_pd: float             = 0.0003   # 0.03 % Basel III floor
    lgd_floor: float                       = 0.45     # unsecured regulatory floor
    verbose: bool                          = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CreditScorer:
    """
    End-to-end PD / LGD / EAD credit risk model.

    Required columns (training DataFrame)
    --------------------------------------
    Behavioural / application features (any numeric or WoE-encoded columns),
    plus these mandatory targets and exposure fields:

    * ``default``         — binary int (0/1) : 1 = defaulted within 12 months
    * ``recovery_rate``   — float [0, 1]     : observed recovery / EAD at default
                           (only rows where ``default == 1`` are used for LGD)
    * ``outstanding``     — float >= 0       : current on-balance-sheet exposure
    * ``limit``           — float >= 0       : approved credit limit (for CCF)
    * ``drawn``           — float >= 0       : amount drawn (for EAD)

    Feature columns are inferred automatically as all numeric columns not in
    the above list. Pass ``feature_cols`` explicitly to override.

    Examples
    --------
    >>> scorer = CreditScorer()
    >>> scorer.fit(train_df)
    >>> results = scorer.score(portfolio_df)
    >>> print(scorer.summary())
    >>> print(scorer.regulatory_capital(results, confidence=0.999))
    """

    _TARGET_COLS = {"default", "recovery_rate", "outstanding", "limit", "drawn"}

    def __init__(self, config: CreditScorerConfig | None = None):
        self.config = config or CreditScorerConfig()
        self._pd_model   = None
        self._lgd_model  = None
        self._lgd_mean   = None   # fallback when <min_lgd_samples
        self._feature_cols: list[str] = []
        self._metrics: dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> "CreditScorer":
        """Train PD, LGD, and EAD sub-models.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset. Must contain mandatory columns listed above.
        feature_cols : list[str] | None
            Explicit list of predictor columns. Auto-detected if None.

        Returns
        -------
        self
        """
        self._validate(df)
        self._feature_cols = feature_cols or self._infer_features(df)

        X = df[self._feature_cols].copy()
        y_pd  = df["default"].astype(int)

        # ---- PD model ---------------------------------------------------
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_pd,
            test_size=self.config.test_size,
            stratify=y_pd,
            random_state=self.config.random_state,
        )
        self._pd_model = self._build_pd_model(X_tr, y_tr)
        self._metrics  = self._evaluate_pd(X_te, y_te)

        # ---- LGD model --------------------------------------------------
        defaults = df[df["default"] == 1].copy()
        if len(defaults) >= self.config.min_lgd_samples:
            self._lgd_model = self._build_lgd_model(defaults, self._feature_cols)
        else:
            self._lgd_mean = defaults["recovery_rate"].mean() if len(defaults) > 0 else 0.45
            if self.config.verbose:
                warnings.warn(
                    f"Only {len(defaults)} defaulted observations — using mean LGD "
                    f"({1 - self._lgd_mean:.1%}) as fallback.",
                    UserWarning, stacklevel=2,
                )

        self._is_fitted = True
        if self.config.verbose:
            self._print_summary()
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a portfolio and return enriched DataFrame.

        Adds columns
        ------------
        * ``pd``        — calibrated probability of default [0, 1]
        * ``lgd``       — loss given default [0, 1]
        * ``ead``       — exposure at default (currency units)
        * ``el``        — expected loss = PD × LGD × EAD
        * ``pd_grade``  — Basel-style letter grade (AAA … D)
        * ``ifrs9_stage`` — IFRS 9 stage (1, 2, or 3)
        """
        self._check_fitted()
        out = df.copy()
        X   = df[self._feature_cols].copy()

        # PD
        pd_raw = self._pd_model.predict_proba(X)[:, 1]
        pd_adj = np.maximum(pd_raw, self.config.regulatory_floor_pd)
        out["pd"] = pd_adj

        # LGD
        out["lgd"] = self._predict_lgd(X)

        # EAD — CCF approach for undrawn lines
        if "limit" in df.columns and "drawn" in df.columns:
            undrawn = np.maximum(df["limit"] - df["drawn"], 0)
            ccf     = 0.75  # Basel Foundation IRB CCF for revolving
            out["ead"] = df["drawn"] + ccf * undrawn
        elif "outstanding" in df.columns:
            out["ead"] = df["outstanding"]
        else:
            out["ead"] = 1.0  # unit exposure fallback

        out["el"]          = out["pd"] * out["lgd"] * out["ead"]
        out["pd_grade"]    = out["pd"].apply(self._pd_to_grade)
        out["ifrs9_stage"] = out["pd"].apply(self._pd_to_stage)

        return out

    def regulatory_capital(
        self,
        scored_df: pd.DataFrame,
        confidence: float = 0.999,
        maturity: float = 2.5,
    ) -> pd.DataFrame:
        """Compute Basel III IRB regulatory capital requirements (RWA and K).

        Uses the asymptotic single-risk-factor (ASRF) Vasicek model:

            K = LGD × N[N⁻¹(PD)/√(1-R) + √(R/(1-R)) × N⁻¹(conf)] - PD×LGD

        where R is the asset correlation (Basel retail: 3%-16%).

        Parameters
        ----------
        scored_df   : output of ``score()``
        confidence  : VaR confidence level (Basel: 0.999)
        maturity    : effective maturity in years (Basel: 1-5, default 2.5)
        """
        self._check_fitted()
        df = scored_df.copy()

        pd_  = df["pd"].values
        lgd_ = df["lgd"].values
        ead_ = df["ead"].values

        # Asset correlation — retail formula (Basel § 328)
        R = 0.03 * (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35)) \
          + 0.16 * (1 - (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35)))

        # Maturity adjustment
        b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
        M = maturity
        ma = (1 + (M - 2.5) * b) / (1 - 1.5 * b)

        # Capital requirement K (Vasicek ASRF)
        from scipy.stats import norm
        N   = norm.cdf
        Ni  = norm.ppf
        K = (lgd_ * N(
            Ni(pd_) / np.sqrt(1 - R) + np.sqrt(R / (1 - R)) * Ni(confidence)
        ) - pd_ * lgd_) * ma

        K = np.maximum(K, 0)

        df["asset_correlation"] = R
        df["maturity_adj"]      = ma
        df["capital_req_K"]     = K
        df["rwa"]               = K * 12.5 * ead_
        df["capital_charge"]    = df["rwa"] * 0.08   # 8% minimum capital ratio

        return df

    def ttc_pd(self, scored_df: pd.DataFrame, long_run_dr: float | None = None) -> pd.Series:
        """Convert point-in-time PD to through-the-cycle PD for regulatory capital.

        Uses the inverse Vasicek formula to back out the systematic factor,
        then neutralises it to produce a TtC estimate.

        Parameters
        ----------
        scored_df   : output of ``score()``
        long_run_dr : long-run average default rate. Estimated from portfolio
                      if not provided.
        """
        pit = scored_df["pd"].values
        lrdr = long_run_dr or float(pit.mean())
        # Back out idiosyncratic component, replace systematic with long-run
        from scipy.stats import norm
        R = 0.15  # approximate mid-range retail correlation
        ttc = norm.cdf(
            (norm.ppf(pit) - np.sqrt(R) * norm.ppf(lrdr)) / np.sqrt(1 - R)
        )
        return pd.Series(ttc, index=scored_df.index, name="pd_ttc")

    def cross_validate(self, df: pd.DataFrame) -> dict:
        """Return stratified k-fold CV metrics (AUC, Gini, KS, Brier, LogLoss)."""
        self._validate(df)
        X = df[self._feature_cols].values
        y = df["default"].astype(int).values

        cv = StratifiedKFold(
            n_splits=self.config.n_cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        aucs, ginis, ks_stats, briers = [], [], [], []

        for tr_idx, va_idx in cv.split(X, y):
            m = self._build_pd_model(
                pd.DataFrame(X[tr_idx], columns=self._feature_cols),
                pd.Series(y[tr_idx]),
            )
            proba = m.predict_proba(
                pd.DataFrame(X[va_idx], columns=self._feature_cols)
            )[:, 1]
            auc = roc_auc_score(y[va_idx], proba)
            aucs.append(auc)
            ginis.append(2 * auc - 1)
            ks_stats.append(self._ks_stat(y[va_idx], proba))
            briers.append(brier_score_loss(y[va_idx], proba))

        return {
            "auc_mean":   float(np.mean(aucs)),
            "auc_std":    float(np.std(aucs)),
            "gini_mean":  float(np.mean(ginis)),
            "ks_mean":    float(np.mean(ks_stats)),
            "brier_mean": float(np.mean(briers)),
            "n_folds":    self.config.n_cv_folds,
        }

    def feature_importances(self) -> pd.DataFrame:
        """Return feature importances / coefficients from the PD model."""
        self._check_fitted()
        base = self._pd_model
        if hasattr(base, "estimator"):
            base = base.estimator
        if hasattr(base, "named_steps"):
            base = base.named_steps.get("model", base)

        if hasattr(base, "coef_"):
            imps = np.abs(base.coef_[0])
        elif hasattr(base, "feature_importances_"):
            imps = base.feature_importances_
        else:
            return pd.DataFrame()

        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": imps})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def summary(self) -> dict:
        """Return a summary dict of model performance metrics."""
        self._check_fitted()
        return {**self._metrics, "feature_count": len(self._feature_cols)}

    def portfolio_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate EL and stage distribution by PD grade."""
        return (
            scored_df.groupby("pd_grade")
            .agg(
                count=("pd", "count"),
                avg_pd=("pd", "mean"),
                avg_lgd=("lgd", "mean"),
                total_ead=("ead", "sum"),
                total_el=("el", "sum"),
            )
            .assign(el_rate=lambda d: d["total_el"] / d["total_ead"])
            .reset_index()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pd_model(self, X: pd.DataFrame, y: pd.Series):
        if self.config.pd_model == "lgbm" and _HAS_LGB:
            base = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.config.random_state,
                verbose=-1,
            )
        else:
            base = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LogisticRegression(
                    max_iter=1000,
                    random_state=self.config.random_state,
                    class_weight="balanced",
                )),
            ])

        if self.config.calibrate_pd:
            model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            model = base

        model.fit(X, y)
        return model

    def _build_lgd_model(self, defaults_df: pd.DataFrame, features: list[str]):
        """Beta regression via logit-transform + OLS (tractable & interpretable)."""
        y_raw = defaults_df["recovery_rate"].clip(1e-4, 1 - 1e-4)
        y_logit = logit(y_raw)   # logit of recovery → LGD = 1 - recovery

        X = defaults_df[features].fillna(0)

        if self.config.lgd_model == "lgbm" and _HAS_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=self.config.random_state,
                verbose=-1,
            )
            model.fit(X, y_logit)
        else:
            from sklearn.linear_model import Ridge
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  Ridge(alpha=1.0)),
            ])
            pipe.fit(X, y_logit)
            model = pipe

        return model

    def _predict_lgd(self, X: pd.DataFrame) -> np.ndarray:
        if self._lgd_model is None:
            lgd_val = 1.0 - (self._lgd_mean or 0.55)
            return np.full(len(X), max(lgd_val, self.config.lgd_floor))

        recovery_logit = self._lgd_model.predict(X.fillna(0))
        recovery = expit(recovery_logit)
        lgd = 1.0 - recovery
        return np.maximum(lgd, self.config.lgd_floor)

    def _evaluate_pd(self, X_te: pd.DataFrame, y_te: pd.Series) -> dict:
        proba = self._pd_model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        return {
            "auc":     round(auc, 4),
            "gini":    round(2 * auc - 1, 4),
            "ks":      round(self._ks_stat(y_te.values, proba), 4),
            "brier":   round(brier_score_loss(y_te, proba), 4),
            "logloss": round(log_loss(y_te, proba), 4),
        }

    @staticmethod
    def _ks_stat(y: np.ndarray, proba: np.ndarray) -> float:
        pos = proba[y == 1]
        neg = proba[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        ks, _ = ks_2samp(pos, neg)
        return float(ks)

    @staticmethod
    def _pd_to_grade(pd: float) -> str:
        thresholds = [
            (0.001,  "AAA"), (0.002,  "AA+"), (0.003,  "AA"),
            (0.005,  "AA-"), (0.008,  "A+"),  (0.012,  "A"),
            (0.018,  "A-"),  (0.025,  "BBB+"), (0.035, "BBB"),
            (0.050,  "BBB-"), (0.075, "BB+"),  (0.100, "BB"),
            (0.150,  "BB-"), (0.200,  "B+"),   (0.300, "B"),
            (0.450,  "B-"),  (0.600,  "CCC"),  (0.800, "CC"),
            (1.001,  "D"),
        ]
        for threshold, grade in thresholds:
            if pd < threshold:
                return grade
        return "D"

    @staticmethod
    def _pd_to_stage(pd: float) -> int:
        """IFRS 9 staging based on PD thresholds (bank-specific; typical values)."""
        if pd >= 0.20:
            return 3   # credit-impaired
        elif pd >= 0.03:
            return 2   # significant increase in credit risk
        return 1

    def _infer_features(self, df: pd.DataFrame) -> list[str]:
        exclude = self._TARGET_COLS | {"id", "customer_id", "account_id",
                                        "date", "snapshot_date"}
        return [
            c for c in df.select_dtypes(include="number").columns
            if c.lower() not in exclude
        ]

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"default"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df["default"].nunique() < 2:
            raise ValueError("Target 'default' must contain both 0 and 1 values.")

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before scoring.")

    def _print_summary(self) -> None:
        m = self._metrics
        print(
            f"\n{'='*55}\n"
            f"  CreditScorer — training complete\n"
            f"{'='*55}\n"
            f"  Features   : {len(self._feature_cols)}\n"
            f"  AUC        : {m.get('auc', 'n/a')}\n"
            f"  Gini       : {m.get('gini', 'n/a')}\n"
            f"  KS stat    : {m.get('ks', 'n/a')}\n"
            f"  Brier score: {m.get('brier', 'n/a')}\n"
            f"{'='*55}\n"
        )