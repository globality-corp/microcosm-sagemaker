from pathlib import Path
from tempfile import TemporaryDirectory

from hamcrest import (
    assert_that,
    calling,
    has_entries,
    raises,
)
from hamcrest.core.base_matcher import BaseMatcher
from parameterized import parameterized

from microcosm_sagemaker.testing.bytes_extractor import ExtractorMatcherPair, json_extractor
from microcosm_sagemaker.testing.directory_comparison import directory_comparison
from microcosm_sagemaker.tests.fixtures.directory_comparison import DIRECTORY_COMPARISON_TEST_CASES


def construct_configuration_matcher(gold_configuration) -> BaseMatcher:
    return has_entries(**gold_configuration)


@parameterized(DIRECTORY_COMPARISON_TEST_CASES)
def test_directory_comparison(
    gold,
    actual,
    should_pass,
    use_json_matcher,
    ignore_hidden,
):
    if use_json_matcher:
        matchers = {
            Path("hello.json"): ExtractorMatcherPair(
                extractor=json_extractor,
                matcher_constructor=construct_configuration_matcher,
            ),
            Path("subdir/hi.json"): ExtractorMatcherPair(
                extractor=json_extractor,
                matcher_constructor=construct_configuration_matcher,
            ),
        }
    else:
        matchers = None

    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        gold_dir = temp_dir / "gold"
        gold.instantiate(gold_dir)
        actual_dir = temp_dir / "actual"
        actual.instantiate(actual_dir)

        if should_pass:
            directory_comparison(
                gold_dir=gold_dir,
                actual_dir=actual_dir,
                matchers=matchers,
                ignore_hidden=ignore_hidden,
            )
        else:
            assert_that(
                calling(directory_comparison).with_args(
                    gold_dir=gold_dir,
                    actual_dir=actual_dir,
                    matchers=matchers,
                    ignore_hidden=ignore_hidden,
                ),
                raises(AssertionError),
            )
