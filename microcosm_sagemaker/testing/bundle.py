import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from microcosm_sagemaker.app_hooks import create_evaluate_app, create_train_app
from microcosm_sagemaker.artifact import RootInputArtifact, RootOutputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.testing.directory_comparison import directory_comparison


class TestBundle(ABC):
    def handle_setup(
        self,
        active_bundle: str,
        input_data: InputData,
        input_artifact: RootInputArtifact,
        gold_output_artifact: RootOutputArtifact,
    ) -> None:
        self.graph = create_train_app(
            extra_config=dict(
                active_bundle=active_bundle,
            )
        )
        self.training_initializers = self.graph.training_initializers
        self.active_bundle = active_bundle

        self.input_data = input_data
        self.input_artifact = input_artifact
        self.gold_output_artifact = gold_output_artifact

    @abstractmethod
    def check_bundle_prediction(self, bundle: Bundle) -> None:
        ...

    def check_train(self) -> None:
        self.training_initializers.init()
        graph = create_train_app(
            extra_config=dict(
                active_bundle=self.active_bundle,
            )
        )
        graph.train_active_bundle_and_dependencies(
            gold_bundle_output_artifact,
        )
        self.check_bundle_prediction(self.active_bundle)

    def check_save(self) -> None:
        self.graph.simple_bundle.load(self.input_artifact)

        with tempfile.TemporaryDirectory() as output_artifact_path:
            output_artifact = RootOutputArtifact(output_artifact_path)

            graph.train_active_bundle_and_dependencies(
                gold_bundle_output_artifact,
            )

            directory_comparison(
                gold_dir=self.gold_output_artifact_path,
                actual_dir=Path(output_artifact_path),
            )

    def check_load(
        self,
        root_input_artifact: RootInputArtifact,
    ) -> None:
        graph = create_evaluate_app(
            extra_config=dict(
                active_bundle=self.active_bundle,
            )
        )
        graph.load_active_bundle_and_dependencies(root_input_artifact)
        self.check_bundle_prediction(self.active_bundle)
