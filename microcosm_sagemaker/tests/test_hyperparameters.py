from hamcrest import assert_that, equal_to, is_
from microcosm.api import (
    binding,
    create_object_graph,
    defaults,
    typed,
)

from microcosm_sagemaker.hyperparameters import GraphHyperparameters, hyperparameter


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


@binding("component_with_dict_hyperparam")
@defaults(
    dict_hyperparam=hyperparameter(dict(
        param1=1,
        param2=2,
    ))
)
class ComponentWithDictHyperparam():
    def __init__(self, graph):
        self.dict_hyperparam = graph.config.component_with_dict_hyperparam.dict_hyperparam


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
            "component_with_dict_hyperparam",
            "component_with_nested_hyperparam"
        )
        self.graph.lock()

    def test_detects_hyperparameters(self):
        hyperparameters = [hp for hp in GraphHyperparameters(self.graph).find_all()]

        expected_hyperparameters = [
            "config__used_component__hyperparam",
            "config__component_with_dict_hyperparam__dict_hyperparam__param1",
            "config__component_with_dict_hyperparam__dict_hyperparam__param2",
            "config__component_with_nested_hyperparam__dict_param__hyperparam"
        ]

        assert_that(
            sorted(hyperparameters),
            is_(equal_to(
                sorted(expected_hyperparameters)
            ))
        )
