import json

from hamcrest.core.base_matcher import BaseMatcher


class JSONMatcher(BaseMatcher):
    def __init__(self, matcher):
        self.matcher = matcher

    def _matches(self, raw_bytes):
        return self.matcher.matches(json.loads(raw_bytes.decode()))

    def describe_to(self, description):
        description.append_text('JSON matches ')
        self.matcher.describe_to(description)


def json_matches(matcher):
    """
    When matching, it expects raw bytes, which it will decode, json.loads, and
    then run through `matcher`

    """
    return JSONMatcher(matcher)
