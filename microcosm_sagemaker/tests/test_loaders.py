from mock import patch
from microcosm_sagemaker.constants import SagemakerPath
from json import dump
from tempfile import NamedTemporaryFile
from microcosm_sagemaker.loaders import load_from_hyperparameters, load_from_s3
from microcosm.metadata import Metadata
from hamcrest import assert_that, equal_to, is_


def test_load_from_hyperparameters():
    metadata = Metadata("foo")
    hyperparameters = {
        "bar": "baz"
    }

    with NamedTemporaryFile("w+") as tmp:
        with patch(
            "microcosm_sagemaker.constants.SagemakerPath.HYPERPARAMETERS",
            tmp.name
        ) as path:
            dump(hyperparameters, tmp)
            tmp.seek(0)

            config = load_from_hyperparameters(metadata)

    assert_that(config, is_(equal_to(hyperparameters)))


def test_load_from_s3():
    metadata = Metadata("foo")
    hyperparameters = {
        "bar": "baz"
    }

    with NamedTemporaryFile("w+") as tmp:
        with patch("microcosm_sagemaker.loaders.client") as boto_client:
            dump(hyperparameters, tmp)
            tmp.seek(0)

            boto_client.return_value.get_object.return_value = {
                "Body": tmp
            }

            loader = load_from_s3("s3://foo/config.json")
            config = loader(metadata)

    assert_that(config, is_(equal_to(hyperparameters)))
