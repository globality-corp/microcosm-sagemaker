from typing import List

from microcosm.api import binding, defaults

from microcosm_sagemaker.artifact import BundleInputArtifact, BundleOutputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.tests.data_models.simple_prediction import SimplePrediction


@binding("simple_bundle")
@defaults(
    simple_param=1.0,
)
class SimpleBundle(Bundle):
    def __init__(self, graph):
        config = graph.config.simple_bundle

        self.simple_param = config.simple_param

    @property
    def dependencies(self) -> List[Bundle]:
        """
        List of bundles upon which this bundle depends.  Whenever the `fit`,
        `save` or `load` methods are called on this bundle, it is guaranteed
        that the corresponding methods will have first been called all all
        `dependency` bundles.

        This simple bundle has no dependencies.

        """
        return []

    def fit(self, input_data: InputData) -> None:
        """
        Perform training

        For this simple bundle, we just expect the input dataset to contain a
        file with a number in it, and we store that as our trained param.

        """
        with open(input_data.path / "simple.txt") as input_file:
            self.simple_trained_param = float(input_file.read())

    def save(self, output_artifact: BundleOutputArtifact) -> None:
        """
        Save the trained model

        For this simple bundle, we just store the param we read during
        training.

        """
        # Save the trained model
        # For this simple bundle, we just store the param we read during
        # training
        with open(output_artifact.path / "simple.txt", "w") as output_file:
            output_file.write(str(self.simple_trained_param))

    def load(self, input_artifact: BundleInputArtifact) -> None:
        """
        Load the trained model

        For this simple bundle, we just load the param we stored during save.

        """
        with open(input_artifact.path / "simple.txt") as input_file:
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
                    self.simple_param +
                    self.simple_trained_param +
                    simple_arg
                ),
            ),
        ]
