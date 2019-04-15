"""
Main evaluation CLI

"""
from click import Path, command, option
from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.app_hooks import create_evaluate_app
from microcosm_sagemaker.artifact import InputArtifact
from microcosm_sagemaker.click import make_click_callback
from microcosm_sagemaker.constants import SagemakerPath
from microcosm_sagemaker.input_data import InputData


@command()
@option(
    "--input-data",
    type=Path(
        resolve_path=True,
        file_okay=False,
        exists=True,
    ),
    callback=make_click_callback(InputData),
    default=SagemakerPath.INPUT_DATA,
    help="Path of the folder that houses the train/test datasets",
)
@option(
    "--input-artifact",
    type=Path(
        resolve_path=True,
        file_okay=False,
        exists=True,
    ),
    callback=make_click_callback(InputArtifact),
    default=SagemakerPath.MODEL,
    help="Path from which to load artifact",
)
def main(input_data, input_artifact):
    graph = create_evaluate_app(
        loaders=[input_artifact.load_config],
    )

    run_evaluate(graph, input_data, input_artifact)


def run_evaluate(graph: ObjectGraph,
                 input_data: InputData,
                 input_artifact: InputArtifact):
    # Load the saved artifact
    graph.active_bundle.load(input_artifact.path)

    # Evaluate
    graph.active_evaluation(graph.active_bundle, input_data.path)
