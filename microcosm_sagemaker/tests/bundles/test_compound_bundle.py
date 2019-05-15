from hamcrest import assert_that, contains, has_properties

from microcosm_sagemaker.artifact import InputArtifact
from microcosm_sagemaker.tests.app_hooks.train.app import create_app
from microcosm_sagemaker.tests.fixtures import get_fixture_path


class TestCompoundBundle:
    def setup(self) -> None:
        self.graph = create_app(extra_deps=["simple_bundle", "compound_bundle"])

        self.graph.simple_bundle.load(
            InputArtifact(
                get_fixture_path("input_artifact") / "simple_bundle"
            )
        )

    def test_prediction(self) -> None:
        assert_that(
            self.graph.compound_bundle.predict(1.0),
            contains(has_properties(
                uri="http://simple.com",
                score=4.0,
            )),
        )
