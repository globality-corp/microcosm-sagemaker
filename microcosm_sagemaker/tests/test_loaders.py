from contextlib import contextmanager
from json import dump as json_dump
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

from hamcrest import assert_that, equal_to, is_
from microcosm.metadata import Metadata

from microcosm_sagemaker.loaders import (
    load_from_hyperparameters,
    load_from_s3,
    load_train_conventions,
)


class TestLoaders(TestCase):
    def test_load_from_hyperparameters(self):
        metadata = Metadata("foo")
        hyperparameters = {
            "bar": "baz"
        }

        with self.patch_hyperparameter_value(hyperparameters):
            config = load_from_hyperparameters(metadata)

        assert_that(config, is_(equal_to(hyperparameters)))

    def test_load_from_s3(self):
        metadata = Metadata("foo")
        remote_configuration = {
            "bar": "baz"
        }

        with self.patch_s3_value(remote_configuration):
            loader = load_from_s3("s3://foo/config.json")
            config = loader(metadata)

        assert_that(config, is_(equal_to(remote_configuration)))

    def test_load_train_conventions(self):
        metadata = Metadata("foo")
        hyperparameters = {
            "base_configuration": "s3://foo/config.json",
            "bar2": "baz2",
        }
        remote_configuration = {
            "bar": "baz"
        }

        with self.patch_s3_value(remote_configuration):
            with self.patch_hyperparameter_value(hyperparameters):
                config = load_train_conventions(metadata)

        assert_that(config, is_(equal_to({
            "bar2": "baz2",
            **remote_configuration,
        })))

    @staticmethod
    @contextmanager
    def patch_hyperparameter_value(value):
        with NamedTemporaryFile("w+") as tmp:
            with patch(
                "microcosm_sagemaker.constants.SagemakerPath.HYPERPARAMETERS",
                tmp.name
            ):
                json_dump(value, tmp)
                tmp.seek(0)
                yield

    @staticmethod
    @contextmanager
    def patch_s3_value(value):
        with NamedTemporaryFile("w+") as tmp:
            with patch("microcosm_sagemaker.loaders.client") as boto_client:
                json_dump(value, tmp)
                tmp.seek(0)

                boto_client.return_value.get_object.return_value = {
                    "Body": tmp
                }

                yield
