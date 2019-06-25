from abc import ABC, abstractmethod
from typing import Callable, List

from microcosm_sagemaker.artifact import BundleInputArtifact, BundleOutputArtifact
from microcosm_sagemaker.input_data import InputData


class Bundle(ABC):
    """
    Abstract base class for all bundles.

    """
    predict: Callable

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

    def fit(self, input_data: InputData) -> None:
        """
        Perform training

        """
        raise NotImplementedError

    def save(self, output_artifact: BundleOutputArtifact) -> None:
        """
        Save the trained model

        """
        raise NotImplementedError

    def fit_and_save(
        self,
        input_data: InputData,
        output_artifact: BundleOutputArtifact,
    ) -> None:
        """
        Perform training and save the trained artifact. By default just calls
        `fit` and `save`, but the derived class can just override this
        function instead if the two steps are not separable.

        """
        self.fit(input_data)
        self.save(output_artifact)

    @abstractmethod
    def load(self, input_artifact: BundleInputArtifact) -> None:
        """
        Load the trained model

        """
        ...
