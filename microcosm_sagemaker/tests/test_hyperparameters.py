from hamcrest import assert_that, equal_to, is_
from microcosm.api import (
    binding,
    create_object_graph,
    defaults,
    typed,
)

from microcosm_sagemaker.hyperparameters import get_graph_hyperparams, hyperparemeted


@binding("test_factory")
@defaults(
    foo_typed=typed(str, "foo"),
    bar_hyperparameterd=hyperparemeted("bar"),
    baz="baz",
)
def generate_factory(graph):
    def test_func(x):
        return x+1
    return test_func


class TestHyperparameters:
    def setup(self):
        self.graph = create_object_graph("test", testing=True)
        self.graph.use("test_factory")

    def test_detects_hyperparameters(self):
        assert_that(
            get_graph_hyperparams(),
            is_(equal_to(
                [
                    ("test_factory", "bar_hyperparameterd")
                ]
            ))
        )
