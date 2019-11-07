import collections

from boto3 import client
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError
from microcosm_logging.decorators import logger

from microcosm_sagemaker.decorators import metrics_observer, training_initializer
from microcosm_sagemaker.metrics.sagemaker.models import MetricUnit


# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key="", sep="__"):
    """
    Flattens a nested dictionary into `sep` seperated key/vals.

    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def normalize_hyperparam_value(v):
    """
    Normalize hyperparameter values to pass botocore's parameter validation

    """
    if v == "":
        # TODO: what is the right solution?
        return "EMPTY_STRING"
    else:
        return str(v)


@training_initializer()
@metrics_observer()
@logger
class SageMakerMetrics:
    def __init__(self, graph):
        self.graph = graph
        self.testing = graph.metadata.testing
        self.model_name = graph.metadata.name

    def init(self):
        pass

    def log_static(self, **kwargs):

        hyperparameters = flatten(self.graph.config)

        # Metric dimensions allow us to analyze metric performance against the
        # hyperparameters of our model
        dimensions = [
            {
                "Name": key,
                "Value": normalize_hyperparam_value(value),
            }
            for key, value in hyperparameters.items()
        ]

        metric_data = [
            {
                "MetricName": metric_name,
                "Dimensions": dimensions,
                "Value": metric_value,
                "Unit": MetricUnit.NONE.name,
                "StorageResolution": 1
            }
            for metric_name, metric_value in kwargs.items()
        ]

        try:
            cloudwatch = client("cloudwatch")
            response = cloudwatch.put_metric_data(
                Namespace="/aws/sagemaker/" + self.model_name,
                MetricData=metric_data,
            )
        except (ClientError, NoCredentialsError, NoRegionError):
            self.logger.warning("CloudWatch publishing disabled", extra=dict(metric_data=metric_data))
            response = None

        return response
