"""
Create the application.

"""
from typing import Callable

from microcosm.api import create_object_graph
from microcosm.config.model import Configuration
from microcosm.loaders import load_each
from microcosm.metadata import Metadata
from microcosm.object_graph import ObjectGraph

import microcosm_sagemaker.tests.bundles  # noqa: 401
import microcosm_sagemaker.tests.evaluations  # noqa: 401
import microcosm_sagemaker.tests.routes  # noqa: 401
from microcosm_sagemaker.tests.app_hooks.serve.config import load_default_config


Loader = Callable[[Metadata], Configuration]
empty_loader = load_each()


def create_app(
    debug: bool = False,
    testing: bool = False,
    model_only: bool = False,
    extra_loader: Loader = empty_loader,
) -> ObjectGraph:
    """
    Create the object graph for serving.

    """
    loader = load_each(
        load_default_config,
        extra_loader,
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

    if not model_only:
        graph.use(
            # SageMaker conventions
            "ping_convention",

            # Routes
            "invocations_route",
        )

    return graph.lock()