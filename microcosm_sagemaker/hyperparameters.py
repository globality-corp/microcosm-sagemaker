from microcosm.config.model import Requirement
from microcosm.registry import _registry


class HyperParameter(Requirement):
    """
    This class subclasses from Requirement and adds the `is_hyperparameter=True` flag.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hyperparameter = True


def hyperparemeted(*args, **kwargs):
    """
    Fluent hyperParameter declaration.

    """
    return HyperParameter(*args, **kwargs)


def get_graph_hyperparams():
    """
    returns all of the graph hyperparameters as a list of (binding name, parameter name) pairs.

    """

    hyperparams = []

    for binding_name, binding_factory in _registry.all.items():
        if "_defaults" in binding_factory.__dict__:
            hyperparams.extend([
                (binding_name, hyperparamer_name)
                for hyperparamer_name, hyperparamer in binding_factory.__dict__["_defaults"].items()
                if getattr(hyperparamer, "is_hyperparameter", False)
            ])

    return hyperparams
