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


class CompoundBundleTestCase(BundleTestCase):
    bundle_name = "compound_bundle"
    root_input_artifact_path = get_fixture_path("input_artifact")
    bundle_prediction_checks = [
        BundlePredictionCheck(
            args=[1.0],
            return_value_matcher=contains(
                has_properties(
                    uri="http://simple.com",
                    score=5.0,
                ),
            )
        )
    ]


class TestCompoundBundleFit(BundleFitTestCase, CompoundBundleTestCase):
    input_data_path = get_fixture_path("simple_input_data")

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestCompoundBundleSave(BundleSaveTestCase, CompoundBundleTestCase):
    gold_bundle_output_artifact_path = get_fixture_path("gold_output_artifact") / "compound_bundle"

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestCompoundBundleLoad(BundleLoadTestCase, CompoundBundleTestCase):
    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
