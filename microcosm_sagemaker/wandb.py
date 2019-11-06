from random import seed

from microcosm.api import defaults

from microcosm_sagemaker.decorators import training_initializer


@training_initializer()
@metrics_observer()
class WeightsAndBiases:
    def __init__(self, graph):
        pass

    def init(self):
        wandb.init()

    def log_metric(...):
        wandb.log_metric(...)

graph.use(
    ...
    "wandb",
)
