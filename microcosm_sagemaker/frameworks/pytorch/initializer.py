from microcosm.api import defaults
from torch import manual_seed

from microcosm_sagemaker.decorators import training_initializer


@defaults(
    seed=42,
)
@training_initializer()
class PyTorchInitializer:
    def __init__(self, graph):
        self.seed = graph.config.pytorch_initializer.seed

    def init(self):
        manual_seed(self.seed)
