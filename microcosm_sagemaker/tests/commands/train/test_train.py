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
    @mock_app_hooks()
    def test_train(self) -> None:
        configuration_extractor_matcher = ExtractorMatcherPair(
            extractor=json_extractor,
            matcher_constructor=construct_configuration_matcher,
        )

        self.run_and_check_train(
            input_data_path=get_fixture_path("simple_input_data"),
            gold_output_artifact_path=get_fixture_path("gold_output_artifact"),
            output_artifact_matchers={
                Path("configuration.json"): configuration_extractor_matcher,
            }
        )
