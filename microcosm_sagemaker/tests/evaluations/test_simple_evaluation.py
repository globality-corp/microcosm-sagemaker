from microcosm_sagemaker.testing.evaluation import EvaluationTestCase
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestSimpleEvaluation(EvaluationTestCase):
    evaluation_name = "simple_evaluation"
    root_input_artifact_path = get_fixture_path("artifact")
    input_data_path = get_fixture_path("input_data")

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
