from microcosm_logging.decorators import logger


@logger
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
        self.testing = graph.metadata.testing
        self.metric_observers = []

    def register(self, observer):
        if not self.testing:
            self.metric_observers.append(observer)

    def log_time_series(self, *args, **kwargs):
        if not self.testing:
            for metric_observer in self.metric_observers:
                response = metric_observer.log_time_series(*args, **kwargs)
                if response:
                    self.logger.info(response)

    def log_static(self, *args, **kwargs):
        if not self.testing:
            for metric_observer in self.metric_observers:
                response = metric_observer.log_static(*args, **kwargs)
                if response:
                    self.logger.info(response)
