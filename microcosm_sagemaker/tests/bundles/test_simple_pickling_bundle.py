from hamcrest import contains, has_properties

from microcosm_sagemaker.testing.bundle import BundleFitSaveLoadTestCase, BundlePredictionCheck
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestSimplePicklingBundleFitSaveLoad(BundleFitSaveLoadTestCase):
    bundle_name = "simple_pickling_bundle"
    root_input_artifact_path = get_fixture_path("artifact")
    input_data_path = get_fixture_path("input_data")
    bundle_prediction_checks = [
        BundlePredictionCheck(
            args=[1.0],
            return_value_matcher=contains(
                has_properties(
                    uri="http://simple.com",
                    score=2.0,
                ),
            )
        )
    ]

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
