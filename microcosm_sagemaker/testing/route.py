"""
Example CRUD routes tests.

Tests are sunny day cases under the assumption that framework conventions
handle most error conditions.

"""
from pathlib import Path

from microcosm_sagemaker.app_hooks import create_serve_app
from microcosm_sagemaker.artifact import RootInputArtifact
from microcosm_sagemaker.commands.config import load_default_runserver_config


class RouteTestCase:
    """
    Helper base class for writing tests of a route.

    """
    def handle_setup(self, input_artifact_path: Path) -> None:
        self.input_artifact = RootInputArtifact(input_artifact_path)

        self.graph = create_serve_app(
            testing=True,
            loaders=[
                load_default_runserver_config,
                self.input_artifact.load_config,
            ],
        )

        self.client = self.graph.flask.test_client()

        self.graph.load_active_bundle_and_dependencies(self.input_artifact)
