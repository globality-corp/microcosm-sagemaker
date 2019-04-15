from pathlib import Path
from typing import Mapping, Optional

from hamcrest import (
    assert_that,
    contains,
    equal_to,
    is_,
)
from hamcrest.core.base_matcher import BaseMatcher


def directory_comparison(gold_dir: Path,
                         actual_dir: Path,
                         matchers: Optional[Mapping[Path, BaseMatcher]] = None):
    """
    Recursively checks the contents of `actual_dir` against the expected
    contents in `gold_dir`.  It is also possible to leave certain files out of
    the gold dir, and instead specify a matcher that should be used for the
    contents of the given file instead.

    """
    matchers = matchers or dict()

    actual_paths = sorted([
        subpath.relative_to(actual_dir)
        for subpath in actual_dir.glob('**/*')
    ])
    gold_paths = sorted(
        [
            subpath.relative_to(gold_dir)
            for subpath in gold_dir.glob('**/*')
        ] + list(matchers.keys())
    )

    assert_that(actual_paths, contains(*gold_paths))

    for path in gold_paths:
        gold_path = gold_dir / path
        actual_path = actual_dir / path

        if gold_path.is_dir():
            assert_that(actual_path.is_dir(), is_(True))
        else:
            if path in matchers:
                matcher = matchers[path]
            else:
                matcher = is_(equal_to(gold_path.read_bytes()))

            assert_that(actual_path.read_bytes(), matcher, path)
