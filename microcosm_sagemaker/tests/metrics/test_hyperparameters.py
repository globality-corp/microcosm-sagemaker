from hamcrest import assert_that, contains, contains_inanyorder
from microcosm.api import (
    binding,
    create_object_graph,
    defaults,
    typed,
)

from microcosm_sagemaker.hyperparameters import get_hyperparameters, hyperparameter


@binding("used_component")
@defaults(
    param=1,
    typed_param=typed(int, 2),
    hyperparam=hyperparameter(3),
)
class UsedComponent():
    def __init__(self, graph):
        self.param = graph.config.used_component.param
        self.hyperparam = graph.config.used_component.hyperparam


@binding("unused_component")
@defaults(
    param=1,
    hyperparam=hyperparameter(1),
)
class UnusedComponent():
    def __init__(self, graph):
        self.param = graph.config.unused_component.param
        self.hyperparam = graph.config.unused_component.hyperparam


@binding("component_with_nested_hyperparam")
@defaults(
    dict_param=dict(
        param=1,
        hyperparam=hyperparameter(2),
    )
)
class ComponentWithNestedHyperparam1():
    def __init__(self, graph):
        self.nested_param = graph.config.component_with_nested_hyperparam.dict_param


class TestHyperparameters:
    def setup(self):
        self.graph = create_object_graph("test", testing=True)
        self.graph.use(
            "used_component",
            "component_with_nested_hyperparam",
        )
        self.graph.lock()

    def test_hyperparameters(self):
        assert_that(
            list(get_hyperparameters(self.graph)),
            contains_inanyorder(
                contains("used_component__hyperparam", 3),
                contains("component_with_nested_hyperparam__dict_param__hyperparam", 2),
            ),
        )
