"""
FraudDetector
=============
Multi-layer fraud detection for transactions.

Implements:
* Rule-based flags
* Unsupervised anomaly detection via Isolation Forest
* Optional supervised fraud model when labels exist
* Ensemble fraud score and alert generation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier


@dataclass
class FraudDetectorConfig:
    contamination: float = 0.01
    supervised: bool = True
    high_risk_countries: list[str] = field(default_factory=list)
    random_state: int = 42


class FraudDetector:
    def __init__(self, config: FraudDetectorConfig | None = None):
        self.config = config or FraudDetectorConfig()
        self.iforest = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self.supervised_model = None
        self.feature_cols: list[str] = []

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["hour"] = out["timestamp"].dt.hour.fillna(12)
        out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
        out["is_high_risk_country"] = out["country"].isin(
            self.config.high_risk_countries
        ).astype(int)

        out["merchant_category"] = out["merchant_category"].fillna("unknown")
        merchant_dummies = pd.get_dummies(out["merchant_category"], prefix="mcc")

        features = pd.DataFrame(
            {
                "amount": pd.to_numeric(out["amount"], errors="coerce").fillna(0.0),
                "hour": pd.to_numeric(out["hour"], errors="coerce").fillna(12),
                "is_night": out["is_night"].astype(int),
                "is_high_risk_country": out["is_high_risk_country"].astype(int),
            }
        )

        features = pd.concat([features, merchant_dummies], axis=1)
        return features

    def fit(self, df: pd.DataFrame) -> "FraudDetector":
        X = self._prepare_features(df)
        self.feature_cols = list(X.columns)

        self.iforest.fit(X)

        if self.config.supervised and "fraud" in df.columns:
            y = df["fraud"].fillna(0).astype(int)
            if y.nunique() > 1:
                self.supervised_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    random_state=self.config.random_state,
                    class_weight="balanced",
                )
                self.supervised_model.fit(X, y)

        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_features(df)

        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols]

        out = df.copy()

        iso_raw = -self.iforest.decision_function(X)
        iso_score = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

        if self.supervised_model is not None:
            sup_score = self.supervised_model.predict_proba(X)[:, 1]
        else:
            sup_score = np.zeros(len(X))

        rule_flags = self._rule_flags(out)
        rule_score = np.clip(rule_flags.apply(len) / 3.0, 0, 1)

        ensemble = 0.50 * iso_score + 0.30 * sup_score + 0.20 * rule_score

        out["iso_score"] = iso_score
        out["supervised_score"] = sup_score
        out["rule_score"] = rule_score
        out["ensemble_score"] = np.clip(ensemble, 0, 1)
        out["rule_flags"] = rule_flags
        out["risk_band"] = pd.cut(
            out["ensemble_score"],
            bins=[-0.001, 0.30, 0.70, 1.0],
            labels=["low", "medium", "high"],
        )

        return out

    def alerts(self, scored_df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
        if "ensemble_score" not in scored_df.columns:
            raise ValueError("Run score() before alerts().")
        return scored_df.loc[scored_df["ensemble_score"] >= threshold].copy()

    def _rule_flags(self, df: pd.DataFrame) -> pd.Series:
        flags = []

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        hours = ts.dt.hour.fillna(12)

        for _, row in df.iterrows():
            row_flags = []

            amount = float(row.get("amount", 0) or 0)
            country = row.get("country", "")
            hour = hours.loc[_]

            if amount >= 200_000:
                row_flags.append("high_amount")
            if country in self.config.high_risk_countries:
                row_flags.append("high_risk_country")
            if hour in [0, 1, 2, 3, 4, 5]:
                row_flags.append("odd_hour")

            flags.append(row_flags)

        return pd.Series(flags, index=df.index)