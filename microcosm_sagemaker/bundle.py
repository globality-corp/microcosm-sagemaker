from abc import ABC, abstractmethod
from typing import Any, List

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.input_data import InputData


class Bundle(ABC):
    @abstractmethod
    def fit(self, input_data: InputData) -> None:
        """
        Perform training

        """
        ...

    @abstractmethod
    def predict(self) -> Any:
        """
        Predict using the trained model

        Note that derived classes can define their own parameters and are
        expected to return something.

        """
        ...

    @abstractmethod
    def save(self, output_artifact: OutputArtifact) -> None:
        """
        Save the trained model

        """
        ...

    @abstractmethod
    def load(self, input_artifact: InputArtifact) -> None:
        """
        Load the trained model

        """
        ...

    @property
    @abstractmethod
    def dependencies(self) -> List["Bundle"]:
        """
        List of bundles upon which this bundle depends.  Whenever the `fit`,
        `save` or `load` methods are called on this bundle, it is guaranteed
        that the corresponding methods will have first been called all all
        `dependency` bundles.

        """
        ...
