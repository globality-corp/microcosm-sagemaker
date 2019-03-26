from unittest import TestCase

from hamcrest import assert_that, equal_to, is_

from microcosm_sagemaker.s3 import S3Object


class TestLoaders(TestCase):
    def test_parse_parse(self):
        object = S3Object.from_url("s3://foo/bar/config_file.json")

        assert_that(object.bucket, is_(equal_to("foo")))
        assert_that(object.key, is_(equal_to("bar/config_file.json")))

    def test_construct_path(self):
        object = S3Object(bucket="foo", key="bar/config_file.json")

        assert_that(object.path, is_(equal_to("s3://foo/bar/config_file.json")))
