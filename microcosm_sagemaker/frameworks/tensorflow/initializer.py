from microcosm.api import defaults
from tf.random import set_random_seed

from microcosm_sagemaker.decorators import training_initializer


@defaults(
    seed=42,
)
@training_initializer()
class TensorFlowInitializer:
    def __init__(self, graph):
        self.seed = graph.config.tensorflow_initializer.seed

    def init(self):
        set_random_seed(self.seed)
