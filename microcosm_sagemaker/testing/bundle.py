import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Mapping, Optional

from microcosm.loaders import load_each, load_from_dict

from microcosm_sagemaker.app_hooks import create_evaluate_app, create_train_app
from microcosm_sagemaker.artifact import BundleOutputArtifact, RootInputArtifact
from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.testing.bundle_prediction_check import BundlePredictionCheck
from microcosm_sagemaker.testing.bytes_extractor import ExtractorMatcherPair
from microcosm_sagemaker.testing.directory_comparison import directory_comparison


class BundlePredictionChecker:
    bundle_prediction_checks: List[BundlePredictionCheck]

    # If necessary, this function can be overridden to do something more
    # sophisticated to check the bundle prediction
    def check_bundle_prediction(self, bundle: Bundle) -> None:
        for bundle_prediction_check in self.bundle_prediction_checks:
            bundle_prediction_check.check_bundle(bundle)


class BundleTestCase(ABC):
    # These should be defined in actual test case derived class
    bundle_name: str
    root_input_artifact_path: Path

    @property
    def extra_config(self) -> dict:
        """
        Derived classes can override this to provide extra config to the
        various create_app functions.

        """
        return {}

    @property
    def _root_input_artifact(self) -> RootInputArtifact:
        return RootInputArtifact(self.root_input_artifact_path)


class BundleFitTestCase(BundleTestCase, BundlePredictionChecker):
    input_data_path: Path

    @property
    def _input_data(self) -> InputData:
        return InputData(self.input_data_path)

    def setup(self) -> None:
        self.graph = create_train_app(
            extra_loader=load_each(
                load_from_dict(
                    active_bundle=self.bundle_name,
                ),
                load_from_dict(self.extra_config),
            ),
        )

        self.graph.bundle_and_dependencies_loader(
            bundle=self.graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
            dependencies_only=True,
        )

        self.graph.training_initializers.init()

    def test_fit(self) -> None:
        self.graph.active_bundle.fit(self._input_data)
        self.check_bundle_prediction(self.graph.active_bundle)


class BundleSaveTestCase(BundleTestCase):
    gold_bundle_output_artifact_path: Path
    output_artifact_matchers: Optional[Mapping[Path, ExtractorMatcherPair]] = None
    ignore_file_contents: bool = False

    @property
    def _gold_bundle_output_artifact(self) -> BundleOutputArtifact:
        return BundleOutputArtifact(self.gold_bundle_output_artifact_path)

    def setup(self) -> None:
        self.graph = create_evaluate_app(
            extra_loader=load_each(
                load_from_dict(
                    active_bundle=self.bundle_name,
                    root_input_artifact_path=self.root_input_artifact_path,
                ),
                load_from_dict(self.extra_config),
            ),
        )

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.bundle_output_artifact = BundleOutputArtifact(self.temporary_directory.name)

    def teardown(self) -> None:
        self.temporary_directory.cleanup()

    def test_save(self) -> None:
        self.graph.active_bundle.save(self.bundle_output_artifact)

        directory_comparison(
            gold_dir=self._gold_bundle_output_artifact.path,
            actual_dir=self.bundle_output_artifact.path,
            matchers=self.output_artifact_matchers,
            ignore_file_contents=self.ignore_file_contents,
        )


class BundleLoadTestCase(BundleTestCase, BundlePredictionChecker):
    def setup(self) -> None:
        self.graph = create_evaluate_app(
            extra_loader=load_each(
                load_from_dict(
                    active_bundle=self.bundle_name,
                    root_input_artifact_path=self.root_input_artifact_path,
                    load_active_bundle_and_dependencies=dict(
                        perform_load=False,
                    ),
                ),
                load_from_dict(self.extra_config),
            ),
        )

        self.graph.bundle_and_dependencies_loader(
            bundle=self.graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
            dependencies_only=True,
        )

    def test_load(self) -> None:
        self.graph.active_bundle.load(
            self._root_input_artifact / self.bundle_name,
        )

        self.check_bundle_prediction(self.graph.active_bundle)


class BundleFitSaveLoadTestCase(BundleTestCase, BundlePredictionChecker):
    input_data_path: Path

    @property
    def _input_data(self) -> InputData:
        return InputData(self.input_data_path)

    def setup(self) -> None:
        self.setup_train_graph()
        self.setup_evaluate_graph()

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.bundle_output_artifact = BundleOutputArtifact(self.temporary_directory.name)

    def teardown(self) -> None:
        self.temporary_directory.cleanup()

    def setup_train_graph(self) -> None:
        self.train_graph = create_train_app(
            extra_loader=load_each(
                load_from_dict(
                    active_bundle=self.bundle_name,
                ),
                load_from_dict(self.extra_config),
            ),
        )

        self.train_graph.bundle_and_dependencies_loader(
            bundle=self.train_graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
            dependencies_only=True,
        )

        self.train_graph.training_initializers.init()

    def setup_evaluate_graph(self):
        self.evaluate_graph = create_evaluate_app(
            extra_loader=load_each(
                load_from_dict(
                    active_bundle=self.bundle_name,
                    root_input_artifact_path=self.root_input_artifact_path,
                    load_active_bundle_and_dependencies=dict(
                        perform_load=False,
                    ),
                ),
                load_from_dict(self.extra_config),
            ),
        )

        self.evaluate_graph.bundle_and_dependencies_loader(
            bundle=self.evaluate_graph.active_bundle,
            root_input_artifact=self._root_input_artifact,
            dependencies_only=True,
        )

    def test_fit_save_load(self) -> None:
        self.train_graph.active_bundle.fit_and_save(
            self._input_data,
            self.bundle_output_artifact,
        )
        self.check_bundle_prediction(self.train_graph.active_bundle)

        evaluate_graph = self.setup_evaluate_graph()
        evaluate_graph.active_bundle.load(
            self._root_input_artifact / self.bundle_name,
        )
        self.check_bundle_prediction(evaluate_graph.active_bundle)
