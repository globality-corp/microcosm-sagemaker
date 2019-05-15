import tempfile
from pathlib import Path

from hamcrest import assert_that, contains, has_properties

from microcosm_sagemaker.artifact import InputArtifact, OutputArtifact
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.testing.directory_comparison import directory_comparison
from microcosm_sagemaker.tests.app_hooks.train.app import create_app
from microcosm_sagemaker.tests.fixtures import get_fixture_path


class TestSimpleBundle:
    def setup(self) -> None:
        self.graph = create_app(extra_deps=["simple_bundle"])
        self.training_initializers = self.graph.training_initializers

        self.input_data = InputData(get_fixture_path("simple_input_data"))
        self.input_artifact = (
            InputArtifact(
                get_fixture_path("input_artifact") / "simple_bundle"
            )
        )
        self.gold_output_artifact_path = (
            get_fixture_path("gold_output_artifact") / "simple_bundle"
        )

    def check_bundle_prediction(self) -> None:
        assert_that(
            self.graph.simple_bundle.predict(1.0),
            contains(has_properties(
                uri="http://simple.com",
                score=3.0,
            )),
        )

    def test_fit(self) -> None:
        self.training_initializers.init()
        self.graph.simple_bundle.fit(self.input_data)
        self.check_bundle_prediction()

    def test_load(self) -> None:
        self.graph.simple_bundle.load(self.input_artifact)
        self.check_bundle_prediction()

    def test_save(self) -> None:
        self.graph.simple_bundle.load(self.input_artifact)

        with tempfile.TemporaryDirectory() as output_artifact_path:
            output_artifact = OutputArtifact(output_artifact_path)

            self.graph.simple_bundle.save(output_artifact)

            directory_comparison(
                gold_dir=self.gold_output_artifact_path,
                actual_dir=Path(output_artifact_path),
            )
