class Metrics:
    """
    The training initializer registry is a place to register functions that need to be
    called during training initialization.  A typical example of this is to
    seed random number generators.

    """
    def __init__(self, graph):
        self.graph = graph
        self.metrics_observers = []

    def register(self, observer):
        self.metrics_observers.append(observer)

    def log_metric(...):
        for metric_observer in metric_observers:
            metric_observer.log_metric(...)

graph.metrics.log_metric(...)
