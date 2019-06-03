from hamcrest import contains, has_properties

from microcosm_sagemaker.testing.bundle import (
    BundleFitTestCase,
    BundleLoadTestCase,
    BundlePredictionCheck,
    BundleSaveTestCase,
    BundleTestCase,
)
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


bundle_prediction_checks = [
    BundlePredictionCheck(
        args=[1.0],
        return_value_matcher=contains(
            has_properties(
                uri="http://simple.com",
                score=3.0,
            ),
        )
    )
]


class SimpleBundleTestCase(BundleTestCase):
    bundle_name = "simple_bundle"
    root_input_artifact_path = get_fixture_path("input_artifact")


class TestSimpleBundleFit(BundleFitTestCase, SimpleBundleTestCase):
    input_data_path = get_fixture_path("simple_input_data")
    bundle_prediction_checks = bundle_prediction_checks

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestSimpleBundleSave(BundleSaveTestCase, SimpleBundleTestCase):
    gold_bundle_output_artifact_path = get_fixture_path("gold_output_artifact") / "simple_bundle"

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestSimpleBundleLoad(BundleLoadTestCase, SimpleBundleTestCase):
    bundle_prediction_checks = bundle_prediction_checks

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
