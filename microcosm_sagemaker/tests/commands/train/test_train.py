from pathlib import Path

from hamcrest import has_entries
from hamcrest.core.base_matcher import BaseMatcher

from microcosm_sagemaker.testing.bytes_extractor import ExtractorMatcherPair, json_extractor
from microcosm_sagemaker.testing.train import TrainCliTestCase
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


def construct_configuration_matcher(gold_configuration) -> BaseMatcher:
    return has_entries(**gold_configuration)


class TestTrainCli(TrainCliTestCase):
    input_data_path = get_fixture_path("input_data")
    gold_output_artifact_path = get_fixture_path("artifact")
    output_artifact_matchers = {
        Path("configuration.json"): ExtractorMatcherPair(
            extractor=json_extractor,
            matcher_constructor=construct_configuration_matcher,
        ),
    }

    @mock_app_hooks()
    def test_train(self) -> None:
        super().test_train()
