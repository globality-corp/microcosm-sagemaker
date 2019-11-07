class MetricLoggers:
    """
    This registry is a place to register functions that log metrics.
    This registry calls the `init` method of the registered components
    to initialize them.
    The `log_metric` and `run_summary` methods are used to store multiple-value
    and single-value metrics, respectively.

    """
    def __init__(self, graph):
        self.graph = graph
        self.metric_observers = []

    def register(self, observer):
        self.metric_observers.append(observer)

    def log_metric(self, *args, **kwargs):
        for metric_observer in self.metric_observers:
            metric_observer.log_metric(*args, **kwargs)

    def run_summary(self, *args, **kwargs):
        for metric_observer in self.metric_observers:
            metric_observer.run_summary(*args, **kwargs)
