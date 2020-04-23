from tempfile import TemporaryDirectory
from unittest import TestCase

from hamcrest import (
    assert_that,
    close_to,
    equal_to,
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

    # TODO: add more test cases
    expected = [{"uri": "http://simple.com", "score": 3.0}]
    test_cases = [{"input": {"simpleArg": 1.0}, "expected": expected}]

    @property
    def _steps(self):
        return [
            self.create_filesystem,
            self.step_training,
            self.step_prepare_predict,
            self.step_predict,
            self.remove_filesystem,
        ]

    def create_filesystem(self):
        directory = TemporaryDirectory()

        return {"artifact_directory": directory, "artifact_path": directory.name}

    @mock_app_hooks()
    def step_training(self, artifact_path):
        graph = create_train_app(extra_loader=load_from_dict(self.config), testing=True)
        run_train(
            graph=graph,
            input_data=InputData(get_fixture_path("input_data")),
            root_output_artifact=RootOutputArtifact(artifact_path),
        )

    @mock_app_hooks()
    def step_prepare_predict(self, artifact_path):
        graph = create_serve_app(
            extra_loader=load_from_dict(root_input_artifact_path=artifact_path,),
            debug=True,
        )
        return {"graph": graph, "client": graph.flask.test_client()}

    @mock_app_hooks()
    def step_predict(self, client):

        for test_case in self.test_cases:
            response = client.post("/api/v1/invocations", json=test_case["input"])
            response_items = response.json["items"]
            assert_that(response.status_code, is_(equal_to(200)))

            expected = test_case["expected"]
            assert_that(
                len(response_items), is_(equal_to(len(expected))),
            )
            for response_item, expected_item in zip(response_items, expected):
                assert_that(response_item["uri"], is_(equal_to(expected_item["uri"])))
                assert_that(
                    response_item["score"], is_(close_to(expected_item["score"], 1e-6))
                )

    def remove_filesystem(self, artifact_directory):
        artifact_directory.cleanup()
