"""
Main training CLI

"""
from json import load
from os import chdir
from os.path import dirname, abspath, join

import click

from microcosm_sagemaker.app_hooks import AppHooks


@click.command()
@click.option(
    "--configuration",
    type=click.Path(),
    required=False,
    help="Manual import of configuration file, used for local testing",
)
@click.option(
    "--input_path",
    type=click.Path(),
    required=False,
    help="Path of the folder that houses the train/test datasets",
)
@click.option(
    "--artifact_path",
    type=click.Path(),
    required=False,
    help="Path for outputting artifacts, used for local testing",
)
@click.option(
    "--auto_evaluate",
    type=bool,
    default=True,
    help="Whether to automatically evaluate after the training has completed",
)
def train_cli(configuration, input_path, artifact_path, auto_evaluate):
    if not artifact_path:
        artifact_path = SagemakerPath.MODEL
    if not input_path:
        input_path = SagemakerPath.INPUT

    if configuration:
        with open(configuration) as configuration_file:
            extra_config = load(configuration_file)
    else:
        extra_config = {}

    graph = AppHooks.create_train_graph(extra_config=extra_config)

    chdir(input_path)

    try:
        model = graph.bundle_manager
        model.fit(artifact_path)
    except Exception as e:
        handle_sagemaker_exception(e)

    if auto_evaluate:
        evaluate(input_path, artifact_path)
