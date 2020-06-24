import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hamcrest import (
    assert_that,
    equal_to,
    has_entries,
    is_,
)
from microcosm.loaders import load_from_dict

from microcosm_sagemaker.app_hooks import create_train_app
from microcosm_sagemaker.artifact import RootOutputArtifact
from microcosm_sagemaker.commands.train import run_train
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


@dataclass
class MockedWandb:
    path = "WANDB_RUN_PATH"


class TestWandbRunPath(TestCase):

    def setUp(self):
        self.artifact_path = Path(TemporaryDirectory().name)

    @mock_app_hooks()
    def test_train(self):
        with patch("wandb.init", return_value=MockedWandb()):
            graph = create_train_app(
                extra_loader=load_from_dict(dict(
                    active_bundle="bundle_with_metric"
                )),
                testing=False,
            )
            graph.unlock()
            graph.use("wandb")
            graph.lock()

            run_train(
                graph=graph,
                input_data=InputData(get_fixture_path("artifact")),
                root_output_artifact=RootOutputArtifact(self.artifact_path),
            )

        cached_configuration = json.load((self.artifact_path / "configuration.json").open())

        assert_that(
            cached_configuration,
            has_entries(
                wandb_run_path=is_(equal_to("WANDB_RUN_PATH"))
            )
        )
