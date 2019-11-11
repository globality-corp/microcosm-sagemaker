from microcosm.config.model import Requirement
from microcosm.registry import _registry


class HyperParameter(Requirement):
    """
    This class subclasses from Requirement and adds the `is_hyperparameter=True` flag.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hyperparameter = True


def hyperparemeted(default_value, parameter_type=None):
    """
    Fluent hyperparameter declaration.
    In most cases, the type is not needed and can be infered from the default value.
    For example, to declare `epochs` as a hyperparameter, use:
    ```
    from microcosm.api import defaults
    from microcosm_sagemaker.hyperparameters import hyperparameter

    @defaults(
        epochs=hyperparemeted(100)
    )
    class ClassifierBundle():
        ...
    ```

    """
    if not parameter_type:
        parameter_type = type(default_value)
    return HyperParameter(type=parameter_type, default_value=default_value)


def get_graph_hyperparams():
    """
    Returns all of the graph hyperparameters as a list of (binding name, parameter name) pairs.

    To do so, it inspects the entire microcosm registry for class definitions with `_defaults`
    that have an `is_hyperparameter=True` flag.

    """

    hyperparams = []

    for binding_name, class_definition in _registry.all.items():
        if "_defaults" in class_definition.__dict__:
            hyperparams.extend([
                (binding_name, hyperparamer_name)
                for hyperparamer_name, hyperparamer in class_definition.__dict__["_defaults"].items()
                if getattr(hyperparamer, "is_hyperparameter", False)
            ])

    return hyperparams
