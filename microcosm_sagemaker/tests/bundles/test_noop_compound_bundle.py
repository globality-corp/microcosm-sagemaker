from hamcrest import contains, has_properties

from microcosm_sagemaker.testing.bundle import (
    BundleEmptySaveTestCase,
    BundleFitTestCase,
    BundleLoadTestCase,
    BundlePredictionCheck,
    BundleTestCase,
)
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class NoopCompoundBundleTestCase(BundleTestCase):
    bundle_name = "noop_compound_bundle"
    root_input_artifact_path = get_fixture_path("artifact")
    bundle_prediction_checks = [
        BundlePredictionCheck(
            args=[1.0],
            return_value_matcher=contains(
                has_properties(
                    uri="http://simple.com",
                    score=6.0,
                ),
            )
        )
    ]


class TestNoopCompoundBundleFit(BundleFitTestCase, NoopCompoundBundleTestCase):
    input_data_path = get_fixture_path("input_data")

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestNoopCompoundBundleSave(BundleEmptySaveTestCase, NoopCompoundBundleTestCase):
    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestNoopCompoundBundleLoad(BundleLoadTestCase, NoopCompoundBundleTestCase):
    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
