"""
Main training CLI

"""
import json

from click import File, command, option
from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.app_hooks import create_train_app
from microcosm_sagemaker.artifact import RootOutputArtifact
from microcosm_sagemaker.commands.options import input_data_option, output_artifact_option
from microcosm_sagemaker.exceptions import raise_sagemaker_exception
from microcosm_sagemaker.input_data import InputData


@command()
@option(
    "--configuration",
    type=File('r'),
    help="Manual import of configuration file, used for local testing",
)
@input_data_option()
@output_artifact_option()
@option(
    "--auto-evaluate/--no-auto-evaluate",
    default=True,
    help="Whether to automatically evaluate after the training has completed",
)
def main(configuration, input_data, output_artifact, auto_evaluate):
    try:
        extra_config = json.load(configuration) if configuration else dict()

        graph = create_train_app(extra_config=extra_config)

        run_train(graph, input_data, output_artifact)

        if auto_evaluate:
            run_auto_evaluate(graph, input_data)
    except Exception as e:
        raise_sagemaker_exception(e)


def run_train(
    graph: ObjectGraph,
    input_data: InputData,
    output_artifact: RootOutputArtifact,
) -> None:
    output_artifact.init()
    output_artifact.save_config(graph.config)

    active_bundle_output_artifact = output_artifact / graph.config.active_bundle
    active_bundle_output_artifact.init()

    graph.training_initializers.init()

    bundle = graph.active_bundle
    bundle.fit(input_data)
    bundle.save(active_bundle_output_artifact)


def run_auto_evaluate(graph: ObjectGraph, input_data: InputData) -> None:
    graph.active_evaluation(graph.active_bundle, input_data)
