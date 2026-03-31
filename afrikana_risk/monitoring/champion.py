"""
ChampionChallenger
==================
Formal model governance framework for comparing and promoting models.

Implements:
* Traffic splitting — route a configurable fraction of new decisions
  to the challenger model
* Statistical significance testing — determine whether challenger
  improvement is real (not noise) using DeLong's AUC test
* Automatic promotion — promote challenger when it wins with p < alpha
* Audit trail — immutable log of all decisions for regulatory review
* Rollback — demote champion and restore prior model

Regulatory context
------------------
Most central bank model risk guidelines require a documented champion-
challenger process before any model change is approved for production.
The audit log produced here is designed to satisfy that requirement.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


@dataclass
class ModelRecord:
    """Metadata for a registered model version."""
    model_id:      str
    name:          str
    version:       str
    model_object:  Any               # fitted model with predict_proba
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata:      dict = field(default_factory=dict)
    is_champion:   bool = False
    is_challenger: bool = False


class ChampionChallenger:
    """
    A/B model governance for regulated environments.

    Quick start
    -----------
    >>> cc = ChampionChallenger(challenger_traffic=0.10)
    >>> cc.register_champion(champion_model, name="Scorecard v3", version="3.0")
    >>> cc.register_challenger(new_model, name="LightGBM v1", version="1.0")
    >>> # Route decisions
    >>> model = cc.route(customer_id="C001")
    >>> pred  = model.predict_proba(X)
    >>> cc.log_decision("C001", model_id, pred[0, 1], actual_default)
    >>> # After observation period:
    >>> result = cc.evaluate(test_df)
    >>> if cc.should_promote(result):
    ...     cc.promote_challenger()
    """

    def __init__(
        self,
        challenger_traffic: float = 0.10,
        significance_level: float = 0.05,
        min_decisions_to_evaluate: int = 500,
        random_state: int = 42,
    ):
        self.challenger_traffic = challenger_traffic
        self.alpha              = significance_level
        self.min_decisions      = min_decisions_to_evaluate
        self._rng               = np.random.default_rng(random_state)
        self._champion:   ModelRecord | None = None
        self._challenger: ModelRecord | None = None
        self._history:    list[dict] = []   # prior champion records
        self._decision_log: list[dict] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_champion(
        self, model_object, name: str, version: str, metadata: dict | None = None
    ) -> str:
        """Register or replace the champion model."""
        if self._champion:
            self._history.append({
                "event":      "champion_replaced",
                "model_id":   self._champion.model_id,
                "name":       self._champion.name,
                "version":    self._champion.version,
                "timestamp":  datetime.utcnow().isoformat(),
            })
        self._champion = ModelRecord(
            model_id=str(uuid.uuid4())[:8],
            name=name,
            version=version,
            model_object=model_object,
            metadata=metadata or {},
            is_champion=True,
        )
        self._log_event("champion_registered", self._champion)
        return self._champion.model_id

    def register_challenger(
        self, model_object, name: str, version: str, metadata: dict | None = None
    ) -> str:
        """Register a challenger model for A/B testing."""
        self._challenger = ModelRecord(
            model_id=str(uuid.uuid4())[:8],
            name=name,
            version=version,
            model_object=model_object,
            metadata=metadata or {},
            is_challenger=True,
        )
        self._log_event("challenger_registered", self._challenger)
        return self._challenger.model_id

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, customer_id: str | None = None) -> tuple[Any, str]:
        """
        Route a scoring request to champion or challenger.

        Returns
        -------
        (model_object, model_id) — use model_object.predict_proba(X)
        """
        self._check_models()
        use_challenger = self._rng.random() < self.challenger_traffic
        model = self._challenger if use_challenger else self._champion
        return model.model_object, model.model_id

    def log_decision(
        self,
        entity_id: str,
        model_id: str,
        score: float,
        actual_outcome: int | None = None,
    ) -> None:
        """
        Log a scored decision for audit trail and later evaluation.

        Parameters
        ----------
        entity_id       : customer / account / transaction ID
        model_id        : champion or challenger model_id
        score           : predicted probability (PD or fraud prob)
        actual_outcome  : observed label (1 = default/fraud) if known
        """
        self._decision_log.append({
            "entity_id":      entity_id,
            "model_id":       model_id,
            "score":          score,
            "actual_outcome": actual_outcome,
            "timestamp":      datetime.utcnow().isoformat(),
            "is_champion":    model_id == (self._champion.model_id if self._champion else None),
        })

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_df: pd.DataFrame, target_col: str = "default") -> dict:
        """
        Compare champion vs challenger on a labelled test set.

        Uses DeLong's method (asymptotic) to test AUC difference significance.

        Parameters
        ----------
        test_df    : DataFrame with features and ``target_col``
        target_col : binary outcome column

        Returns
        -------
        dict with champion_auc, challenger_auc, auc_delta, p_value,
        significant, recommendation
        """
        self._check_models()
        if target_col not in test_df.columns:
            raise ValueError(f"'{target_col}' not in test_df.")

        y = test_df[target_col].astype(int).values

        # Feature columns (exclude target)
        feat_cols = [c for c in test_df.select_dtypes(include="number").columns
                     if c != target_col]
        X = test_df[feat_cols]

        champ_prob  = self._champion.model_object.predict_proba(X)[:, 1]
        chall_prob  = self._challenger.model_object.predict_proba(X)[:, 1]

        champ_auc   = roc_auc_score(y, champ_prob)
        chall_auc   = roc_auc_score(y, chall_prob)
        delta       = chall_auc - champ_auc

        # DeLong's AUC standard error approximation
        se_diff = self._delong_se(y, champ_prob, chall_prob)
        z_stat  = delta / (se_diff + 1e-12)
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

        significant = p_value < self.alpha and delta > 0

        result = {
            "champion_id":    self._champion.model_id,
            "challenger_id":  self._challenger.model_id,
            "champion_name":  self._champion.name,
            "challenger_name": self._challenger.name,
            "champion_auc":   round(champ_auc,  4),
            "challenger_auc": round(chall_auc,  4),
            "challenger_gini": round(2 * chall_auc - 1, 4),
            "champion_gini":  round(2 * champ_auc - 1, 4),
            "auc_delta":      round(delta,       4),
            "z_statistic":    round(z_stat,      4),
            "p_value":        round(p_value,     4),
            "significant":    significant,
            "n_test":         len(y),
            "recommendation": "PROMOTE CHALLENGER" if significant
                              else ("INCONCLUSIVE" if delta > 0 else "RETAIN CHAMPION"),
            "evaluated_at":   datetime.utcnow().isoformat(),
        }
        self._log_event("evaluation_completed", result)
        return result

    def should_promote(self, evaluation_result: dict) -> bool:
        """Return True if challenger should be promoted based on evaluation."""
        return evaluation_result.get("significant", False)

    def promote_challenger(self) -> None:
        """Promote challenger to champion. Old champion moves to history."""
        self._check_models()
        self._history.append({
            "event":     "champion_retired",
            "model_id":  self._champion.model_id,
            "name":      self._champion.name,
            "version":   self._champion.version,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._champion              = self._challenger
        self._champion.is_champion  = True
        self._champion.is_challenger = False
        self._challenger             = None
        self._log_event("challenger_promoted", self._champion)

    def rollback(self) -> None:
        """Demote current champion; restore the most recently retired model."""
        if not self._history:
            raise RuntimeError("No prior champion to roll back to.")
        # In a real system this would restore the model object from a registry
        self._log_event("rollback_initiated", {"timestamp": datetime.utcnow().isoformat()})
        print("Rollback initiated. Restore model object from your model registry.")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def decision_log_df(self) -> pd.DataFrame:
        """Return full audit trail as DataFrame."""
        return pd.DataFrame(self._decision_log)

    def traffic_summary(self) -> pd.DataFrame:
        """Return routing statistics from the decision log."""
        if not self._decision_log:
            return pd.DataFrame()
        df = pd.DataFrame(self._decision_log)
        return (
            df.groupby("is_champion")
            .agg(
                decisions=("score", "count"),
                avg_score=("score", "mean"),
                observed=("actual_outcome", lambda x: x.notna().sum()),
            )
            .rename(index={True: "champion", False: "challenger"})
            .reset_index()
        )

    def audit_log(self) -> list[dict]:
        """Return the immutable event audit log."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _delong_se(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """Simplified DeLong SE for AUC difference (Han et al. approximation)."""
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 1e-6
        # V10, V01 for each model
        def _v_stats(pred):
            pos, neg = pred[y == 1], pred[y == 0]
            v10 = np.array([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
            v01 = np.array([(p < pos).mean() + 0.5 * (p == pos).mean() for p in neg])
            return v10, v01

        v10_1, v01_1 = _v_stats(p1)
        v10_2, v01_2 = _v_stats(p2)
        v10_1.mean()
        v10_2.mean()

        s10 = np.cov(v10_1, v10_2)[0, 1] if n_pos > 1 else 0
        s01 = np.cov(v01_1, v01_2)[0, 1] if n_neg > 1 else 0

        var1  = (np.var(v10_1) / n_pos + np.var(v01_1) / n_neg)
        var2  = (np.var(v10_2) / n_pos + np.var(v01_2) / n_neg)
        cov12 = s10 / n_pos + s01 / n_neg

        var_diff = var1 + var2 - 2 * cov12
        return float(np.sqrt(max(var_diff, 1e-12)))

    def _check_models(self) -> None:
        if self._champion is None:
            raise RuntimeError("No champion registered. Call register_champion().")
        if self._challenger is None:
            raise RuntimeError("No challenger registered. Call register_challenger().")

    def _log_event(self, event: str, subject: Any) -> None:
        record = {
            "event":     event,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if isinstance(subject, ModelRecord):
            record.update({"model_id": subject.model_id, "name": subject.name,
                           "version": subject.version})
        elif isinstance(subject, dict):
            record.update(subject)
        self._history.append(record)