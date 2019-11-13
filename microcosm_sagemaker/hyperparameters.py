from typing import Generator, List, Tuple

from microcosm.config.model import Requirement
from microcosm.config.validation import zip_dicts
from microcosm.errors import LockedGraphError
from microcosm.registry import get_defaults


class Hyperparameter(Requirement):
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
    return Hyperparameter(type=parameter_type, default_value=default_value)


def get_hyperparameters(graph):
    """
    Given a graph, yields all of the hyperparameters in the config as
    `__`-separated keys.

    Consider the following example:

    ```
    @binding("ann_classifier_bundle")
    @defaults(
        no_of_epochs=hyperparameter(100),
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
        "ann_classifier_bundle__no_of_epochs",
        "ann_classifier_bundle__other_params__learning_rate",
    ]
    ```

    """
    defaults = {
        key: get_defaults(graph.factory_for(key))
        for key, _ in graph.items()
    }

    for path, _, default, _, value in zip_dicts(defaults, graph.config):
        if isinstance(default, Hyperparameter):
            yield "__".join(path), value
