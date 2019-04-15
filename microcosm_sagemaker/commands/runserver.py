"""
Main web service CLI

"""
from click import Path, command, option
from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.app_hooks import create_serve_app
from microcosm_sagemaker.artifact import InputArtifact
from microcosm_sagemaker.constants import SagemakerPath


@command()
@option(
    "--host",
    default="127.0.0.1",
)
@option(
    "--port",
    type=int,
)
@option(
    "--debug/--no-debug",
    default=False,
)
@option(
    "--input-artifact-path",
    type=Path(
        resolve_path=True,
        file_okay=False,
        exists=True,
    ),
    default=SagemakerPath.MODEL,
    help="Path from which to load artifact",
)
def main(host, port, debug, input_artifact_path):
    input_artifact = InputArtifact(input_artifact_path)

    graph = create_serve_app(
        debug=debug,
        loaders=[input_artifact.load_config],
    )

    run_serve(graph, input_artifact)


def run_serve(graph: ObjectGraph,
              input_artifact: InputArtifact,
              host: str,
              port: int):
    graph.active_bundle.load(input_artifact.path)

    graph.flask.run(
        host=host,
        port=port or graph.config.flask.port,
    )
