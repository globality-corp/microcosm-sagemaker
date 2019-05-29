
from hamcrest import assert_that, contains, has_properties

from microcosm_sagemaker.testing.bundle import BundleTestCase
from microcosm_sagemaker.tests.bundles.simple import SimpleBundle
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestSimpleBundle(BundleTestCase):
    bundle_name = "simple_bundle"
    input_data_path = get_fixture_path("simple_input_data")
    root_input_artifact_path = get_fixture_path("input_artifact")
    gold_bundle_output_artifact_path = get_fixture_path("gold_output_artifact") / "simple_bundle"

    def check_bundle_prediction(self, bundle: SimpleBundle) -> None:
        assert_that(
            bundle.predict(1.0),
            contains(has_properties(
                uri="http://simple.com",
                score=3.0,
            )),
        )

    # NB: The below definitions won't be required by client services using
    # BundleTestCase.  This is only necessary because we can't define app_hooks
    # within microcosm_sagemaker as would be done by a client service.
    @mock_app_hooks()
    def test_fit(self) -> None:
        super().test_fit()

    @mock_app_hooks()
    def test_save(self) -> None:
        super().test_save()

    @mock_app_hooks()
    def test_load(self) -> None:
        super().test_load()
