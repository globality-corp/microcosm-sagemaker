from microcosm_sagemaker.testing.evaluate import EvaluateCliTestCase
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestEvaluateCli(EvaluateCliTestCase):
    @mock_app_hooks()
    def test_evaluate(self) -> None:
        self.run_and_check_evaluate(
            input_data_path=get_fixture_path("simple_input_data"),
            input_artifact_path=get_fixture_path("input_artifact"),
        )
