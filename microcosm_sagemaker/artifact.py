import json
from pathlib import Path

from microcosm.config.model import Configuration
from microcosm.metadata import Metadata

from microcosm_sagemaker.constants import ARTIFACT_CONFIGURATION_PATH


class OutputArtifact:
    def __init__(self, path):
        self.path = Path(path)

    def init(self):
        self.path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Configuration):
        config_path = self.path / ARTIFACT_CONFIGURATION_PATH

        with open(config_path, "w") as config_file:
            json.dump(config, config_file)


class InputArtifact:
    def __init__(self, path):
        self.path = Path(path)

    def load_config(self, metadata: Metadata):
        """
        When we train a model, we freeze all of the current graph variables and store it alongside
        the artifact. Whenever we boot up the model again, we want to hydrate this from disk.

        """
        config_path = self.path / ARTIFACT_CONFIGURATION_PATH

        with open(config_path) as config_file:
            return json.load(config_file)
