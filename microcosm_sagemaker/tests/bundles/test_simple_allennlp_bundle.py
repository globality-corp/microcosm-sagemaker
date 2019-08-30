from hamcrest import has_entries

from microcosm_sagemaker.testing.bundle import BundleFitSaveLoadTestCase, BundlePredictionCheck
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestSimpleAllenNLPBundleFitSaveLoad(BundleFitSaveLoadTestCase):
    bundle_name = "simple_allennlp_bundle"
    root_input_artifact_path = get_fixture_path("artifact")
    input_data_path = get_fixture_path("input_data")
    bundle_prediction_checks = [
        BundlePredictionCheck(
            args=["hello there"],
            return_value_matcher=has_entries(
                label="1",
            ),
        )
    ]

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
