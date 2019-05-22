"""
Simple CRUD routes tests.

Tests are sunny day cases under the assumption that framework conventions
handle most error conditions.

"""
from hamcrest import contains, has_entries

from microcosm_sagemaker.testing.invocations import InvocationsRouteTestCase
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestInvocationsRoute(InvocationsRouteTestCase):
    @mock_app_hooks()
    def setup(self) -> None:
        self.handle_setup(
            input_artifact_path=get_fixture_path("input_artifact")
        )

    def test_search(self) -> None:
        request_json = dict(
            simpleArg=1.0,
        )

        response_items_matcher = contains(
            has_entries(
                uri="http://simple.com",
                score=5.0,
            ),
        )

        self.check_search(
            request_json=request_json,
            response_items_matcher=response_items_matcher,
        )
