import json
from pathlib import Path
from typing import Union

from microcosm.config.model import Configuration
from microcosm.metadata import Metadata

from microcosm_sagemaker.constants import ARTIFACT_CONFIGURATION_PATH


class OutputArtifact:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def init(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Configuration) -> None:
        config_path = self.path / ARTIFACT_CONFIGURATION_PATH

        with open(config_path, "w") as config_file:
            json.dump(config, config_file)


class InputArtifact:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def load_config(self, metadata: Metadata) -> Configuration:
        """
        When we train a model, we freeze all of the current graph variables and store it alongside
        the artifact. Whenever we boot up the model again, we want to hydrate this from disk.

        Note that this function is designed to be used as a microcosm loader,
        hence the unused `metadata` parameter.

        """
        config_path = self.path / ARTIFACT_CONFIGURATION_PATH

        with open(config_path) as config_file:
            return Configuration(json.load(config_file))
