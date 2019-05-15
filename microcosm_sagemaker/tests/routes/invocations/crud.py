from microcosm.api import binding
from microcosm_flask.conventions.base import EndpointDefinition

from microcosm_sagemaker.conventions.invocations import configure_invocations
from microcosm_sagemaker.tests.resources.invocations_resources import (
    ClassificationResultSchema,
    NewPredictionSchema,
)


@binding("invocations_route")
def configure_invocations_route(graph):
    controller = graph.invocations_controller

    return configure_invocations(
        graph,
        EndpointDefinition(
            func=controller.search,
            request_schema=NewPredictionSchema(),
            response_schema=ClassificationResultSchema(),
        ),
    )
