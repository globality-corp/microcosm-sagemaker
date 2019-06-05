from pathlib import Path
from typing import Mapping, Optional

from hamcrest import (
    assert_that,
    contains,
    equal_to,
    is_,
)

from microcosm_sagemaker.testing.bytes_extractor import ExtractorMatcherPair


def _is_empty_dir(path: Path):
    if not path.is_dir():
        return False

    try:
        next(path.iterdir())
    except StopIteration:
        return True

    return False


def _get_relevant_subpaths(directory: Path, ignore_empty_directories: bool):
    """
    Return all subpaths of directory, recursively, optionally ignoring empty
    directories.

    """
    subpaths = sorted([
        subpath.relative_to(directory)
        for subpath in directory.glob('**/*')
    ])

    if ignore_empty_directories:
        subpaths = list(filter(
            lambda path: not _is_empty_dir(directory / path),
            subpaths,
        ))

    return subpaths


def _identity(x):
    return x


def directory_comparison(
    gold_dir: Path,
    actual_dir: Path,
    matchers: Optional[Mapping[Path, ExtractorMatcherPair]] = None,
    ignore_empty_directories: bool = True,
):
    """
    Recursively checks the contents of `actual_dir` against the expected
    contents in `gold_dir`.  It is also possible to leave certain files out of
    the gold dir, and instead specify an (extractor, matcher) pair that should
    be used to extract and match the contents of the given file instead.

    """
    if matchers is None:
        matchers = dict()

    assert_that(gold_dir.exists(), is_(True))
    assert_that(actual_dir.exists(), is_(True))

    actual_paths = _get_relevant_subpaths(
        directory=actual_dir,
        ignore_empty_directories=ignore_empty_directories,
    )
    gold_paths = _get_relevant_subpaths(
        directory=gold_dir,
        ignore_empty_directories=ignore_empty_directories,
    )

    assert_that(actual_paths, contains(*gold_paths))

    for path in gold_paths:
        gold_path = gold_dir / path
        actual_path = actual_dir / path

        assert_that(
            actual_path.is_dir(),
            is_(equal_to(gold_path.is_dir())),
        )
        if not gold_path.is_dir():
            if path in matchers:
                extractor, matcher_constructor = matchers[path]
            else:
                extractor, matcher_constructor = ExtractorMatcherPair(
                    _identity,
                    lambda x: is_(equal_to(x)),
                )

            assert_that(
                extractor(actual_path.read_bytes()),
                matcher_constructor(
                    extractor(gold_path.read_bytes()),
                ),
                path,
            )
