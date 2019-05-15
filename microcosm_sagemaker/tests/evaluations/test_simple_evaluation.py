from microcosm_sagemaker.artifact import InputArtifact
from microcosm_sagemaker.input_data import InputData

from microcosm_sagemaker.tests.app_hooks.train.app import create_app
from microcosm_sagemaker.tests.fixtures import get_fixture_path


class TestSimpleEvaluation:
    def setup(self) -> None:
        self.graph = create_app(extra_deps=["simple_evaluation"])

        self.input_data = InputData(get_fixture_path("simple_input_data"))
        self.input_artifact = (
            InputArtifact(
                get_fixture_path("input_artifact") / "simple_bundle"
            )
        )

    def test_evaluation(self) -> None:
        bundle = self.graph.active_bundle
        bundle.load(self.input_artifact)

        self.graph.simple_evaluation(bundle, self.input_data)
