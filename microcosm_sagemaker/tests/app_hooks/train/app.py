"""
Create the application.

"""
from microcosm.api import create_object_graph
from microcosm.loaders import empty_loader, load_each, load_from_environ
from microcosm.object_graph import ObjectGraph
from microcosm.typing import Loader

import microcosm_sagemaker.tests.bundles  # noqa: 401
from microcosm_sagemaker.loaders import train_conventions_loader
from microcosm_sagemaker.tests.app_hooks.train.config import load_default_config


def create_app(
    debug: bool = False,
    testing: bool = False,
    extra_loader: Loader = empty_loader,
    use_wandb: bool = False,
) -> ObjectGraph:
    """
    Create the object graph for the application.

    """
    loader = train_conventions_loader(
        initial_loader=load_each(
            load_default_config,
            load_from_environ,
            extra_loader,
        )
    )

    graph = create_object_graph(
        name=__name__.split(".")[0],
        debug=debug,
        testing=testing,
        loader=loader,
    )

    graph.use(
        "logging",

        # Sagemaker basics
        "sagemaker",
    )

    if use_wandb:
        graph.use("wandb")

    return graph.lock()
