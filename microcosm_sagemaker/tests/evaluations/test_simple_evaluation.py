from microcosm_sagemaker.artifact import InputArtifact
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.tests.app_hooks.train.app import create_app
from microcosm_sagemaker.tests.fixtures import get_fixture_path


class TestSimpleEvaluation:
    def setup(self) -> None:
        self.graph = create_app(extra_deps=["simple_evaluation"])

        self.input_data = InputData(get_fixture_path("simple_input_data"))
        input_artifact = (
            InputArtifact(
                get_fixture_path("input_artifact")
            )
        )
        self.graph.load_active_bundle_and_dependencies(input_artifact)

    def test_evaluation(self) -> None:
        self.graph.simple_evaluation(self.graph.active_bundle, self.input_data)
