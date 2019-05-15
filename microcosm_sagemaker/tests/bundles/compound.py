from typing import List

from microcosm.api import binding

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.tests.data_models.simple_prediction import SimplePrediction


@binding("compound_bundle")
class CompoundBundle(Bundle):
    def __init__(self, graph):
        self.simple_bundle = graph.simple_bundle

    @property
    def dependencies(self) -> List[Bundle]:
        return [
            self.simple_bundle,
        ]

    def fit(self, input_data: InputData) -> None:
        pass

    def save(self, output_artifact: OutputArtifact) -> None:
        pass

    def load(self, input_artifact: InputArtifact) -> None:
        pass

    def predict(self, simple_arg: float) -> List[SimplePrediction]:
        """
        We just add 1.0 to all predictions of the simple bundle.

        """
        return [
            SimplePrediction(
                uri=prediction.uri,
                score=prediction.score + 1.0,
            )
            for prediction in self.simple_bundle.predict(simple_arg)
        ]
