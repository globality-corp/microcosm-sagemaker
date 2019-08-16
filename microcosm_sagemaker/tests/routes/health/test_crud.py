"""
Simple CRUD routes tests.

Tests are sunny day cases under the assumption that framework conventions
handle most error conditions.

"""
from hamcrest import (
    assert_that,
    equal_to,
    has_entries,
    is_,
)

from microcosm_sagemaker.testing.route import RouteTestCase
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestHealthRoute(RouteTestCase):
    root_input_artifact_path = get_fixture_path("artifact")

    @mock_app_hooks()
    def setup(self) -> None:
        super().setup()

    def test_health_check(self) -> None:
        uri = "/api/health"

        response = self.client.get(uri)

        assert_that(response.status_code, is_(equal_to(200)))
        assert_that(
            response.json,
            has_entries(
                name="microcosm_sagemaker",
                ok=True,
            ),
        )
