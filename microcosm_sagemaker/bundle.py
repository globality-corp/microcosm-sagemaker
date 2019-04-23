from abc import ABC, abstractmethod

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.input_data import InputData


class Bundle(ABC):
    @abstractmethod
    def fit(self, input_data: InputData):
        """
        Perform training

        """
        pass

    @abstractmethod
    def predict(self):
        """
        Predict using the trained model

        Note that derived classes can define their own expected parameters.

        """
        pass

    @abstractmethod
    def save(self, output_artifact: OutputArtifact):
        """
        Save the trained model

        """
        pass

    @abstractmethod
    def load(self, input_artifact: InputArtifact):
        """
        Load the trained model

        """
        pass
