"""
Main evaluation CLI

"""
from os import chdir

from click import Path, command, option

from microcosm_sagemaker.app_hooks import AppHooks


def evaluate(input_path, artifact_path):
    graph = AppHooks.create_serve_graph(artifact_path=artifact_path)

    chdir(input_path)

    # Load the saved artifact
    graph.bundle_manager.load(artifact_path)

    # Evaluate
    graph.evaluation_manager(graph.bundle_manager)


@command()
@option(
    "--input_path",
    type=Path(),
    required=True,
    help="Path of the folder that houses the datasets",
)
@option(
    "--artifact_path",
    type=Path(),
    required=True,
    help="Path for reading artifacts, used for local testing",
)
def evaluate_cli(input_path, artifact_path):
    evaluate(input_path, artifact_path)
