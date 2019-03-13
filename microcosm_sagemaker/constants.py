"""
SageMaker global constants.

"""
from pathlib import Path


SAGEMAKER_PREFIX = Path("/opt/ml/")
CONFIGURATION_TAG_KEY = "experiment_tag"


class SagemakerPath:
    INPUT = SAGEMAKER_PREFIX / "input/data"
    MODEL = SAGEMAKER_PREFIX / "model"
    OUTPUT = SAGEMAKER_PREFIX / "output"
    HYPERPARAMETERS = SAGEMAKER_PREFIX / "input/config/hyperparameters.json"
