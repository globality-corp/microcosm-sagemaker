from tempfile import TemporaryDirectory
from unittest import TestCase

from hamcrest import (
    assert_that,
    close_to,
    equal_to,
    has_entries,
    is_,
)
from microcosm.loaders import load_from_dict

from microcosm_sagemaker.app_hooks import create_serve_app, create_train_app
from microcosm_sagemaker.artifact import RootOutputArtifact
from microcosm_sagemaker.commands.train import run_train
from microcosm_sagemaker.input_data import InputData
from microcosm_sagemaker.testing.pipeline_harness import PipelineHarness
from microcosm_sagemaker.tests.fixtures import get_fixture_path
from microcosm_sagemaker.tests.mocks import mock_app_hooks


class TestSimpleBundlePipeline(PipelineHarness, TestCase):
    config = {
        "active_bundle": "simple_bundle",
        "active_evaluation": "simple_evaluation",
    }

    @property
    def _steps(self):
        return [
            self.create_filesystem,
            self.step_train,
            self.step_runserver,
            self.step_check_prediction,
            self.remove_filesystem,
        ]

    def create_filesystem(self):
        directory = TemporaryDirectory()

        return dict(artifact_directory=directory, artifact_path=directory.name)

    @mock_app_hooks()
    def step_train(self, artifact_path):
        graph = create_train_app(extra_loader=load_from_dict(self.config), testing=True)
        run_train(
            graph=graph,
            input_data=InputData(get_fixture_path("input_data")),
            root_output_artifact=RootOutputArtifact(artifact_path),
        )

    @mock_app_hooks()
    def step_runserver(self, artifact_path):
        graph = create_serve_app(
            extra_loader=load_from_dict(root_input_artifact_path=artifact_path),
            debug=True,
        )
        return dict(graph=graph, client=graph.flask.test_client())

    @mock_app_hooks()
    def step_check_prediction(self, client):

        response = client.post("/api/v1/invocations", json=dict(simpleArg=1.0))
        assert_that(response.status_code, is_(equal_to(200)))
        assert_that(
            response.json["items"][0],
            has_entries(uri="http://simple.com", score=close_to(3.0, 0)),
        )

    def remove_filesystem(self, artifact_directory):
        artifact_directory.cleanup()
