"""
Main evaluation CLI

"""
from click import Path, command, option
from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.app_hooks import create_evaluate_app
from microcosm_sagemaker.artifact import InputArtifact
from microcosm_sagemaker.constants import SagemakerPath
from microcosm_sagemaker.input_data import InputData


@command()
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
        exists=True,
    ),
    default=SagemakerPath.MODEL,
    help="Path for outputting artifacts, used for local testing",
)
def main(input_path, artifact_path):
    graph = create_evaluate_app()

    input_data = InputData(input_path)
    input_artifact = InputArtifact(artifact_path)

    run_evaluate(graph, input_data, input_artifact)


def run_evaluate(graph: ObjectGraph,
                 input_data: InputData,
                 input_artifact: InputArtifact):
    # Load the saved artifact
    graph.active_bundle.load(input_artifact.path)

    # Evaluate
    graph.active_evaluation(graph.active_bundle, input_data.path)
