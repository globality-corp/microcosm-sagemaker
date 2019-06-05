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
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.fixtures.directory_comparison import DIRECTORY_COMPARISON_TEST_CASES


def construct_configuration_matcher(gold_configuration) -> BaseMatcher:
    return has_entries(**gold_configuration)


@parameterized(DIRECTORY_COMPARISON_TEST_CASES)
def test_simple_directory_comparison(name, should_pass, use_json_matcher):
    test_directory = get_fixture_path(f"directory_comparison/{name}")

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

    if should_pass:
        directory_comparison(
            gold_dir=test_directory / "gold",
            actual_dir=test_directory / "actual",
            matchers=matchers,
            ignore_empty_directories=False,
        )
    else:
        assert_that(
            calling(directory_comparison).with_args(
                gold_dir=test_directory / "gold",
                actual_dir=test_directory / "actual",
                matchers=matchers,
                ignore_empty_directories=False,
            ),
            raises(AssertionError),
        )


def test_empty_directory_comparison():
    with TemporaryDirectory() as gold, TemporaryDirectory() as actual:
        directory_comparison(
            gold_dir=Path(gold),
            actual_dir=Path(actual),
            ignore_empty_directories=False,
        )


def test_empty_subdirectory_comparison():
    """
    We construct directories programmatically because git would just ignore any
    directories that we checked in.

    The directories look as follows:

    gold/
    ├── subdir/
    │  └── hello.txt
    └── hello.txt

    actual/
    ├── empty_subdir/
    ├── subdir/
    │  ├── empty_nested_subdir/
    │  └── hello.txt
    └── hello.txt

    """

    with TemporaryDirectory() as gold_path_str, TemporaryDirectory() as actual_path_str:
        gold = Path(gold_path_str)
        actual = Path(actual_path_str)

        (gold / "hello.txt").write_text("hello")
        (actual / "hello.txt").write_text("hello")

        (gold / "subdir").mkdir()
        (actual / "subdir").mkdir()
        (gold / "subdir" / "hello.txt").write_text("hello")
        (actual / "subdir" / "hello.txt").write_text("hello")

        (actual / "empty_subdir").mkdir()
        (actual / "subdir" / "empty_nested_subdir").mkdir()

        directory_comparison(
            gold_dir=gold,
            actual_dir=actual,
            ignore_empty_directories=True,
        )
