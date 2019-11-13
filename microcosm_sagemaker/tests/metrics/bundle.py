from microcosm.api import binding, defaults

from microcosm_sagemaker.hyperparameters import hyperparameter


@binding("bundle_with_metric")
@defaults(
    param=1,
    hyperparam=hyperparameter(2),
)
class BundleWithMetirc():
    def __init__(self, graph):
        self.graph = graph
        self.param = graph.config.bundle_with_metric.param
        self.hyperparam = graph.config.bundle_with_metric.hyperparam

        self.metrics = graph.experiment_metrics

    def log_static_metric(self):
        self.metrics.log_static(static_metric=3)

    def log_timeseries_metric(self):
        self.metrics.log_timeseries(step=0, timeseries_metric=1)
