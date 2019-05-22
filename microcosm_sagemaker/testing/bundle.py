import tempfile
from pathlib import Path

from hamcrest import assert_that, contains, has_properties

from microcosm_sagemaker.app_hooks import create_train_app
from microcosm_sagemaker.artifact import RootInputArtifact, RootOutputArtifact
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.testing.directory_comparison import directory_comparison


class TestBundle:
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

        self.input_data = input_data
        self.input_artifact = input_artifact
        self.gold_output_artifact = gold_output_artifact

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
