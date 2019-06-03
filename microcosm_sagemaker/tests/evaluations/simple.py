from microcosm.api import binding, defaults
from microcosm_logging.decorators import logger

from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.evaluation import Evaluation
from microcosm_sagemaker.input_data import InputData


@binding("simple_evaluation")
@defaults(
    simple_param=1.0,
)
@logger
class SimpleEvaluation(Evaluation):
    def __init__(self, graph):
        config = graph.config.simple_bundle

        self.simple_param = config.simple_param

    def __call__(self, bundle: Bundle, input_data: InputData):
        pass
