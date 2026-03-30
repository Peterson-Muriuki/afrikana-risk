"""
afrikana-risk
=============
Production-ready quantitative risk toolkit for credit scoring, IFRS9 ECL,
fraud detection, and model governance. Built in Africa. Applicable anywhere.
"""

__version__ = "1.0.0"
__author__  = "Peterson Mutegi"
__email__   = "pitmuriuki@gmail.com"

from afrikana_risk.credit.scorer    import CreditScorer
from afrikana_risk.credit.scorecard import ScorecardBuilder
from afrikana_risk.risk.ecl         import ECLEngine
from afrikana_risk.risk.stress      import StressTestor
from afrikana_risk.fraud.detector   import FraudDetector
from afrikana_risk.monitoring.monitor   import ModelMonitor
from afrikana_risk.monitoring.champion  import ChampionChallenger

__all__ = [
    "CreditScorer", "ScorecardBuilder",
    "ECLEngine", "StressTestor",
    "FraudDetector",
    "ModelMonitor", "ChampionChallenger",
]