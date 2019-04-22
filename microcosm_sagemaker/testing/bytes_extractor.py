import json
from typing import Any, Callable, NamedTuple

from hamcrest.core.base_matcher import BaseMatcher


class ExtractorMatcherPair(NamedTuple):
    """
    Contains a pair of extractor and matcher which is used during testing.  The
    extractor will be applied to the raw bytes of a file output during testing,
    and then the matcher will be run on the output of the extractor.

    """
    # NB: We cannot use a dataclass due to python/mypy#5485
    extractor: Callable[[bytes], Any]
    matcher: BaseMatcher


def json_extractor(raw_bytes):
    return json.loads(raw_bytes.decode())
