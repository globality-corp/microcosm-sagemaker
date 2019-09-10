from typing import List

from microcosm.api import binding

from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.pickling_bundle import PicklingBundle
from microcosm_sagemaker.tests.data_models.simple_prediction import SimplePrediction


@binding("simple_pickling_bundle")
class SimplePicklingBundle(PicklingBundle):
    pickle_attrs: List[str] = [
        "simple_trained_param",
    ]

    def __init__(self, graph):
        pass

    @property
    def dependencies(self) -> List[Bundle]:
        return []

    def fit(self, input_data: InputData) -> None:
        with open(input_data.path / "simple.txt") as input_file:
            self.simple_trained_param = float(input_file.read())

    def predict(self, simple_arg: float) -> List[SimplePrediction]:
        """
        Predict using the trained model.

        For this simple bundle, we just add the configured param, the learned
        param, and the argument in order to demonstrate how to use all three.

        """
        return [
            SimplePrediction(
                uri="http://simple.com",
                score=(
                    self.simple_trained_param +
                    simple_arg
                ),
            ),
        ]
