from abc import abstractproperty
from pathlib import Path
from typing import Any, Dict, Optional

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from microcosm_sagemaker.artifact import BundleInputArtifact, BundleOutputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.frameworks.allennlp.constants import ARTIFACT_NAME, CUDA_DEVICE
from microcosm_sagemaker.input_data import InputData


class AllenNLPBundle(Bundle):
    """
    Higher-order AllenNLP component that can wrap other models, serializing
    our configuration format the way that they expect.

    """
    predictor_name: Optional[str] = None

    @abstractproperty
    def allennlp_parameters(self) -> Dict[str, Any]:
        ...

    def fit_and_save(
        self,
        input_data: InputData,
        output_artifact: BundleOutputArtifact,
    ) -> None:
        allennlp_params = Params(self.allennlp_parameters)
        train_model(
            allennlp_params,
            self._allenlp_path(output_artifact.path),
        )

        self._set_predictor(output_artifact.path)

    def load(self, input_artifact: BundleInputArtifact) -> None:
        self._set_predictor(input_artifact.path)

    def _set_predictor(self, path: Path) -> None:
        allennlp_path = self._allenlp_path(path)
        weights_path = allennlp_path / ARTIFACT_NAME

        archive = load_archive(
            allennlp_path / "model.tar.gz",
            weights_file=weights_path,
            cuda_device=CUDA_DEVICE,
        )

        self.predictor = Predictor.from_archive(archive, self.predictor_name)

    def _allenlp_path(self, artifact_path):
        return Path(artifact_path) / "allennlp"
