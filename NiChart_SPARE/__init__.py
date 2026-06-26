"""
NiChart_SPARE — SPARE scores from brain ROI volumes and white matter lesion volumes.

Workflow:
  1. prep_data.prep_data()  — task-specific preprocessing → standardized CSV
  2. train.train_model()    — SVM training → .joblib model
  3. inference.infer_model() — apply model → predictions CSV
"""

__version__ = "0.1.0"
__author__  = "Kyunglok Baik"
__email__   = "software@cbica.upenn.edu"
__url__     = "https://github.com/CBICA/NiChart_SPARE"
