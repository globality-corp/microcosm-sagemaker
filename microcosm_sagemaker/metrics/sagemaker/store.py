import datetime

from boto3 import client
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError
from microcosm_logging.decorators import logger

from microcosm_sagemaker.decorators import metrics_observer, training_initializer
from microcosm_sagemaker.hyperparameters import get_graph_hyperparams
from microcosm_sagemaker.metrics.sagemaker.models import LogMode, MetricUnit


# NOTE: We can only use up to 10 dimensions:
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudwatch.html
MAX_DIMENSIONS = 10


@training_initializer()
@metrics_observer()
@logger
class SageMakerMetrics:
    def __init__(self, graph):
        self.graph = graph
        self.testing = graph.metadata.testing
        self.model_name = graph.metadata.name
        self.dimensions = self._get_dimensions()

    def init(self):
        self.logger.info("`cloudwatch` was registered as a metric observer.")

    def _get_dimensions(self):
        hyperparameters = get_graph_hyperparams()
        if len(hyperparameters) > MAX_DIMENSIONS:
            self.logger.warning(
                f"The number of hyperparameters ({len(hyperparameters)}) is more than the maximum dimensions "
                f"allowed by `cloudwatch` ({MAX_DIMENSIONS})."
            )

        # Metric dimensions allow us to analyze metric performance against the
        # hyperparameters of our model
        dimensions = [
            {
                "Name": f"{bundle_name}__{parameter}",
                "Value": str(getattr(getattr(self.graph.config, bundle_name), parameter)),
            }
            for (bundle_name, parameter) in hyperparameters[:MAX_DIMENSIONS]
        ]

        return dimensions

    def _metric_data(self, metric_name, metric_value, timestamp):
        metric_data = dict(
            MetricName=metric_name,
            Dimensions=self.dimensions,
            Value=metric_value,
            Unit=MetricUnit.NONE.name,
            StorageResolution=1,
        )

        if timestamp:
            metric_data.update(Timestamp=timestamp)

        return metric_data

    def _log_metric(self, log_mode: LogMode, **kwargs):

        if log_mode == LogMode.TIMESERIES:
            timestamp = datetime.datetime.now()
        else:
            timestamp = None

        metric_data = [
            self._metric_data(metric_name, metric_value, timestamp)
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

    def log_static(self, **kwargs):
        self._log_metric(LogMode.STATIC, **kwargs)

    def log_timeseries(self, **kwargs):
        self._log_metric(LogMode.TIMESERIES, **kwargs)
