from microcosm.api import binding, defaults

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.input_data import InputData


@binding("simple_bundle_with_metric")
@defaults(
    param1=1,
    param2=2,
)
class SimpleBundleWithMetric():
    def __init__(self, graph):
        self.graph = graph
        self.param1 = graph.config.simple_bundle_with_metric.param1
        self.param2 = graph.config.simple_bundle_with_metric.param2

        self.metrics = graph.experiment_metrics

    @property
    def dependencies(self):
        return []

    def save(self, output_artifact: OutputArtifact) -> None:
        pass

    def load(self, input_artifact: InputArtifact) -> None:
        pass

    def fit(self, input_data: InputData) -> None:
        pass

    def log_static_metric(self):
        self.metrics.log_static(static_metric=3)

    def log_timeseries_metric(self):
        self.metrics.log_timeseries(step=0, timeseries_metric=1)


@binding("compound_bundle_with_metric")
@defaults(
    param1=3,
    param2=4,
)
class CompoundBundleWithMetric():
    def __init__(self, graph):
        self.graph = graph
        self.param1 = graph.config.compound_bundle_with_metric.param1
        self.param2 = graph.config.compound_bundle_with_metric.param2

    @property
    def dependencies(self):
        return [
            self.graph.simple_bundle_with_metric
        ]

    def save(self, output_artifact: OutputArtifact) -> None:
        pass

    def load(self, input_artifact: InputArtifact) -> None:
        pass

    def fit(self, input_data: InputData) -> None:
        pass

    def log_static_metric(self):
        self.metrics.log_static(static_metric=3)

    def log_timeseries_metric(self):
        self.metrics.log_timeseries(step=0, timeseries_metric=1)
