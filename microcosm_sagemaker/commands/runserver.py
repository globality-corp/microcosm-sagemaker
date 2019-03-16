"""
Main web service CLI

"""
from click import Path, command, option

from microcosm_sagemaker.app_hooks import AppHooks


@command()
@option(
    "--artifact_path",
    type=Path(),
    required=True,
    help="Path for reading artifacts, used for local testing",
)
def runserver_cli(artifact_path):
    graph = AppHooks.create_serve_graph(debug=True, artifact_path=artifact_path)

    graph.flask.run(
        host="127.0.0.1",
        port=graph.config.flask.port,
    )
