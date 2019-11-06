class MetricLoggers:
    """
    The training initializer registry is a place to register functions that need to be
    called during training initialization.  A typical example of this is to
    seed random number generators.

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
