"""
Main training CLI

"""
import json

from click import (
    File,
    Path,
    command,
    option,
)
from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.app_hooks import create_train_app
from microcosm_sagemaker.artifact import OutputArtifact
from microcosm_sagemaker.constants import SagemakerPath
from microcosm_sagemaker.exceptions import raise_sagemaker_exception
from microcosm_sagemaker.input_data import InputData


@command()
@option(
    "--configuration",
    type=File('r'),
    help="Manual import of configuration file, used for local testing",
)
@option(
    "--input-path",
    type=Path(
        resolve_path=True,
        file_okay=False,
        exists=True,
    ),
    default=SagemakerPath.INPUT_DATA,
    help="Path of the folder that houses the train/test datasets",
)
@option(
    "--artifact-path",
    type=Path(
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=SagemakerPath.MODEL,
    help="Path for outputting artifacts, used for local testing",
)
@option(
    "--auto-evaluate/--no-auto-evaluate",
    default=True,
    help="Whether to automatically evaluate after the training has completed",
)
def main(configuration, input_path, artifact_path, auto_evaluate):
    try:
        extra_config = json.load(configuration) if configuration else dict()

        graph = create_train_app(extra_config=extra_config)

        # NB: We don't use the microcosm graph infrastructure for these because
        # we don't want absolute paths to end up in the configuration dump
        input_data = InputData(input_path)
        output_artifact = OutputArtifact(artifact_path)

        run_train(graph, input_data, output_artifact)

        if auto_evaluate:
            run_auto_evaluate(graph, input_data)
    except Exception as e:
        raise_sagemaker_exception(e)


def run_train(graph: ObjectGraph,
              input_data: InputData,
              output_artifact: OutputArtifact):
    output_artifact.init()
    output_artifact.save_config(graph.config)

    graph.random.init()
    graph.frameworks.init()

    bundle = graph.active_bundle
    bundle.fit(input_data.path)
    bundle.save(output_artifact.path)


def run_auto_evaluate(graph: ObjectGraph,
                      input_data: InputData):
    graph.active_evaluation(graph.active_bundle, input_data.path)
