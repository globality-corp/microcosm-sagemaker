from microcosm_sagemaker.artifact import RootInputArtifact
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.tests.app_hooks.train.app import create_app
from microcosm_sagemaker.tests.fixtures import get_fixture_path


class TestSimpleEvaluation:
    def setup(self) -> None:
        self.graph = create_app()

        self.input_data = InputData(get_fixture_path("simple_input_data"))

        self.graph.load_bundle_and_dependencies(
            bundle=self.graph.active_bundle,
            root_input_artifact=RootInputArtifact(
                get_fixture_path("input_artifact")
            )
        )

    def test_evaluation(self) -> None:
        self.graph.simple_evaluation(self.graph.active_bundle, self.input_data)
