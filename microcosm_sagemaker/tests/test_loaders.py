from contextlib import contextmanager
from json import dump as json_dump
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

from hamcrest import (
    assert_that,
    equal_to,
    has_entries,
    is_,
)
from microcosm.loaders import load_from_dict
from microcosm.metadata import Metadata

from microcosm_sagemaker.loaders import (
    evaluate_conventions_loader,
    hyperparameter_loader,
    serve_conventions_loader,
    train_conventions_loader,
)
from microcosm_sagemaker.tests.fixtures import get_fixture_path


class TestLoaders(TestCase):
    def test_load_from_hyperparameters(self):
        metadata = Metadata("foo")
        hyperparameters = {
            "bar": "baz"
        }

        with self.patch_hyperparameter_value(hyperparameters):
            config = hyperparameter_loader(metadata)

        assert_that(config, is_(equal_to(hyperparameters)))

    def test_train_conventions_loader(self):
        metadata = Metadata("foo")
        hyperparameters = dict(
            base_configuration="s3://foo/config.json",
            bar2="baz2",
        )
        remote_configuration = dict(
            bar="baz"
        )
        initial_configuration = dict(
            bongo="bazman"
        )

        with self.patch_s3_value(remote_configuration):
            with self.patch_hyperparameter_value(hyperparameters):
                loader = train_conventions_loader(
                    initial_loader=load_from_dict(initial_configuration),
                )
                config = loader(metadata)

        assert_that(config, is_(equal_to({
            **hyperparameters,
            **remote_configuration,
            **initial_configuration,
        })))

    def test_train_conventions_loader_order(self):
        metadata = Metadata("foo")
        hyperparameters = dict(
            base_configuration="s3://foo/config.json",
            bar2="baz2",
        )
        remote_configuration = dict(
            bar="baz"
        )
        initial_configuration = dict(
            bar2="baz1"
        )

        with self.patch_s3_value(remote_configuration):
            with self.patch_hyperparameter_value(hyperparameters):
                loader = train_conventions_loader(
                    initial_loader=load_from_dict(initial_configuration),
                )
                config = loader(metadata)

        expected_config = dict()
        for _config in [initial_configuration, hyperparameters, remote_configuration]:
            expected_config.update(**_config)

        assert_that(config, is_(equal_to(expected_config)))

    def test_serve_conventions_loader(self):
        metadata = Metadata("foo")
        root_input_artifact_path = get_fixture_path("artifact")
        initial_configuration = dict(
            active_bundle="spaghetti",
            bongo="bazman",
            root_input_artifact_path=root_input_artifact_path,
        )

        loader = serve_conventions_loader(
            initial_loader=load_from_dict(initial_configuration),
        )
        config = loader(metadata)

        assert_that(config, has_entries(
            build_route_path=has_entries(
                prefix="",
            ),
            **initial_configuration,
        ))

    def test_evaluate_conventions_loader(self):
        metadata = Metadata("foo")
        root_input_artifact_path = get_fixture_path("artifact")
        initial_configuration = dict(
            active_bundle="spaghetti",
            bongo="bazman",
            root_input_artifact_path=root_input_artifact_path,
        )

        loader = evaluate_conventions_loader(
            initial_loader=load_from_dict(initial_configuration),
        )
        config = loader(metadata)

        assert_that(config, has_entries(
            **initial_configuration,
        ))

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
