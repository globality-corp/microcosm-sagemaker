from hamcrest import assert_that, contains, has_properties

from microcosm_sagemaker.testing.bundle import (
    BundleFitTestCase,
    BundleLoadTestCase,
    BundleSaveTestCase,
    BundleTestCase,
)
from microcosm_sagemaker.tests.bundles.simple import SimpleBundle
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class SimpleBundleTestCase(BundleTestCase):
    bundle_name = "simple_bundle"
    root_input_artifact_path = get_fixture_path("input_artifact")

    def check_bundle_prediction(self, bundle: SimpleBundle) -> None:
        assert_that(
            bundle.predict(1.0),
            contains(has_properties(
                uri="http://simple.com",
                score=3.0,
            )),
        )


class TestSimpleBundleFit(BundleFitTestCase, SimpleBundleTestCase):
    input_data_path = get_fixture_path("simple_input_data")

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestSimpleBundleSave(BundleSaveTestCase, SimpleBundleTestCase):
    gold_bundle_output_artifact_path = get_fixture_path("gold_output_artifact") / "simple_bundle"

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()


class TestSimpleBundleLoad(BundleLoadTestCase, SimpleBundleTestCase):
    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()
