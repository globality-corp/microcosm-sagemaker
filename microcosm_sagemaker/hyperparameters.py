from typing import List, Optional, Tuple

from microcosm.config.model import Requirement
from microcosm.errors import LockedGraphError


class HyperParameter(Requirement):
    """
    This class subclasses from Requirement and adds the `is_hyperparameter=True` flag.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hyperparameter = True


def hyperparameter(default_value, parameter_type=None):
    """
    Fluent hyperparameter declaration.
    In most cases, the type is not needed and can be inferred from the default value.
    For example, to declare `epochs` as a hyperparameter, use:
    ```
    from microcosm.api import defaults
    from microcosm_sagemaker.hyperparameters import hyperparameter

    @defaults(
        epochs=hyperparameter(100)
    )
    class ClassifierBundle():
        ...
    ```

    """
    if not parameter_type:
        parameter_type = type(default_value)
    return HyperParameter(type=parameter_type, default_value=default_value)


def get_nested_value(d: dict, keys: Optional[List[str]]):
    """
    Given a dictionary `d` and a list of keys `[k1, k2, ..., kn]`, retuns `d[k1][k2]...[kn]`.

    """

    for k in keys:
        d = d[k]
    return d


class GraphHyperparameters:

    def __init__(self, graph) -> None:
        self.graph = graph

    def find_all(self) -> List[str]:
        """
        Given a graph, yields all of the hyperparameters in the config as `__` seperated attributes.

        Consider the following example:
        ```
        @binding("ann_classifier_bundle")
        @defaults(
            no_of_epochs=hyperparameter(100),
            no_of_layers=hyperparameter(
                dict(
                    layer1=10,
                    layer2=20,
                )
            ),
            other_params=dict(
                dropout=0.2,
                learning_rate=hyperparameter(0.05),
            )
        )
        class ANNClassifierBundle():
            ...
        ```

        This functions yields the following:
        ```
        [
            "config__ann_classifier_bundle__no_of_epochs",
            "config__ann_classifier_bundle__no_of_layers__layer1",
            "config__ann_classifier_bundle__no_of_layers__layer2",
            "config__ann_classifier_bundle__other_params__learning_rate",
        ]
        ```

        The ability to define hyperparameters at the dict-level (like the `no_of_layers` in the example)
        is just for convenience to define multiple hyperparameters at one. Otherwise, the same outcome
        could be achieved by defining `layer1` and `layer2` as invididual hyperparameters.

        """

        for binding_name, class_definition in self.graph._registry.all.items():
            # skip if not bound to the graph
            try:
                _ = getattr(self.graph, binding_name)
            except LockedGraphError:
                continue

            if "_defaults" in class_definition.__dict__:  # this class definition has some defaults
                default_params = self._get_default_parameters(binding_name)

                # First, we recursively look for any instances from the Hyperparameter class
                # in the default paremeters and return each one as the list of its attributes.
                # For the above example, `hyperparameters` will be:
                # ```
                # [
                #     ("no_of_epochs", ),
                #     ("no_of_layers", ),
                #     ("other_params", "learning_rate"),
                # ]
                # ```
                hyperparams = self._find_hyperparameter_instances(default_params, keys=())

                # Next, if the hyperparameter is a dictionary, we replace it in the list of hyperparameters
                # with all of its children. That means for the above example, `("no_of_layers", )` will be
                # replaced with its two children:
                # ```
                # ("no_of_layers", "layer1"),
                # ("no_of_layers", "layer2"),
                # ```
                expanded_hyperparams = []
                for hyperparam in hyperparams:
                    expanded_hyperparams.extend(
                        self._expand_hyperparameter(binding_name, hyperparam)
                    )

                # create and yield the final `__` separated strings
                yield from ["__".join(("config", binding_name) + p) for p in expanded_hyperparams]

    def _get_default_parameters(self, binding_name: str) -> dict:
        """
        Given a binding name, returns its dictionary of default parameters.

        """
        return self.graph._registry.all[binding_name].__dict__['_defaults']

    def _find_hyperparameter_instances(self, default_params: dict, keys: Tuple[str]):
        """
        Recursively looks for parameters in `default_params` that are instances of the Hyperparameter class.
        Returns each hyperparameter as a `(attr1, attr2, ...)` tuple.

        """

        param = get_nested_value(default_params, keys)
        hyperparams = []

        if isinstance(param, HyperParameter):
            return [keys]

        if isinstance(param, dict):
            for k in param.keys():
                hyperparams.extend(self._find_hyperparameter_instances(default_params, keys+(k,)))

        return hyperparams

    def _expand_hyperparameter(self, binding_name, hyperparam):
        """
        Recursively, expands a dictionary hyperparameter into all of its children.
        For non-dictionary hyperparameters, returns the input as is.

        """
        binding_config = getattr(self.graph.config, binding_name)
        param = get_nested_value(binding_config, hyperparam)

        if not isinstance(param, dict):
            return [hyperparam]

        children = []
        for k, v in param.items():
            child = hyperparam + (k,) if hyperparam else (k,)
            if isinstance(v, dict):
                children.extend(self._expand_hyperparameter(binding_name, child))
            else:
                children.append(child)
        return children

    def get_hyperparameter_value(self, hyperparameter_string):
        hyperparameter = hyperparameter_string.split("__")
        return get_nested_value(self.graph.config, hyperparameter[1:])
