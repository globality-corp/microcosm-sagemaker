from microcosm_sagemaker.testing.evaluate import EvaluateCliTestCase
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestEvaluateCli(EvaluateCliTestCase):
    input_data_path = get_fixture_path("input_data")
    input_artifact_path = get_fixture_path("artifact")

    @mock_app_hooks()
    def test_evaluate(self) -> None:
        super().test_evaluate()
