from random import seed

from microcosm.api import defaults


@defaults(
    seed=42,
)
class Random:
    def __init__(self, graph):
        self.seed = graph.config.random.seed

    def init(self):
        seed(self.seed)
