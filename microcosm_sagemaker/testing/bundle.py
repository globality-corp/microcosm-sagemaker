import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from microcosm.loaders import load_from_dict

from microcosm_sagemaker.app_hooks import create_evaluate_app, create_train_app
from microcosm_sagemaker.artifact import BundleOutputArtifact, RootInputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.testing.directory_comparison import directory_comparison


BundleType = TypeVar('BundleType', bound=Bundle)


class BundleTestCase(ABC, Generic[BundleType]):
    # These should be defined in actual test case derived class
    bundle_name: str
    input_data_path: Path
    root_input_artifact_path: Path
    gold_bundle_output_artifact_path: Path

    @abstractmethod
    def check_bundle_prediction(self, bundle: BundleType) -> None:
        ...

    @property
    def _input_data(self) -> InputData:
        return InputData(self.input_data_path)

    @property
    def _root_input_artifact(self) -> RootInputArtifact:
        return RootInputArtifact(self.root_input_artifact_path)

    @property
    def _gold_bundle_output_artifact(self) -> BundleOutputArtifact:
        return BundleOutputArtifact(self.gold_bundle_output_artifact_path)

    def test_fit(self) -> None:
        graph = create_train_app(
            extra_loader=load_from_dict(
                active_bundle=self.bundle_name,
            )
        )

        graph.load_bundle_and_dependencies(
            bundle=graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
            dependencies_only=True,
        )

        graph.training_initializers.init()
        graph.active_bundle.fit(self._input_data)

        self.check_bundle_prediction(graph.active_bundle)

    def test_load(self) -> None:
        graph = create_evaluate_app(
            extra_loader=load_from_dict(
                active_bundle=self.bundle_name,
            )
        )

        graph.load_bundle_and_dependencies(
            bundle=graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
            dependencies_only=True,
        )

        graph.active_bundle.load(
            self._root_input_artifact / self.bundle_name,
        )

        self.check_bundle_prediction(graph.active_bundle)

    def test_save(self) -> None:
        graph = create_train_app(
            extra_loader=load_from_dict(
                active_bundle=self.bundle_name,
            )
        )

        graph.load_bundle_and_dependencies(
            bundle=graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
        )

        with tempfile.TemporaryDirectory() as bundle_output_artifact_path:
            bundle_output_artifact = BundleOutputArtifact(bundle_output_artifact_path)

            graph.active_bundle.save(bundle_output_artifact)

            directory_comparison(
                gold_dir=self._gold_bundle_output_artifact.path,
                actual_dir=bundle_output_artifact.path,
            )
