import datetime
from unittest.mock import patch

from hamcrest import (
    assert_that,
    contains_inanyorder,
    equal_to,
    has_entries,
    instance_of,
    is_,
)
from microcosm.api import create_object_graph

from microcosm_sagemaker.metrics.sagemaker.models import MetricUnit


class TestCloudwatch():
    def setup(self):
        self.graph = create_object_graph(
            name="test-project",
            testing=False,
        )
        self.graph.use("training_initializers", "cloudwatch", "bundle_with_metric")
        self.graph.lock()

        self.expected_namespace = "/aws/sagemaker/" + "test-project"
        self.expected_dimensions = [
            {
                "Name": "bundle_with_metric__hyperparam",
                "Value": str(2),
            }
        ]

    def test_log_static_metric(self):
        with patch("boto3.client") as boto3_client:
            self.graph.training_initializers.init()
            self.graph.bundle_with_metric.log_static_metric()

            boto3_client().put_metric_data.assert_called_with(
                Namespace=self.expected_namespace,
                MetricData=[
                    dict(
                        MetricName="static_metric",
                        Dimensions=self.expected_dimensions,
                        Value=3,
                        Unit=MetricUnit.NONE.name,
                        StorageResolution=1,
                    )
                ],
            )

    def test_log_timeseries_metric(self):
        with patch("boto3.client") as boto3_client:
            self.graph.training_initializers.init()
            self.graph.bundle_with_metric.log_timeseries_metric()

            _, kwargs = boto3_client().put_metric_data.call_args

            assert_that(
                kwargs["Namespace"],
                is_(equal_to(self.expected_namespace))
            )

            assert_that(
                kwargs["MetricData"],
                contains_inanyorder(
                    has_entries(
                        MetricName="timeseries_metric",
                        Dimensions=self.expected_dimensions,
                        Value=1,
                        Unit=MetricUnit.NONE.name,
                        StorageResolution=1,
                        Timestamp=instance_of(datetime.datetime),
                    ),
                    has_entries(
                        MetricName="step",
                        Dimensions=self.expected_dimensions,
                        Value=0,
                        Unit=MetricUnit.NONE.name,
                        StorageResolution=1,
                        Timestamp=instance_of(datetime.datetime),
                    )
                )
            )
