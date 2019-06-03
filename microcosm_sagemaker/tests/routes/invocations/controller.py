"""
Invocations controller.

"""
from microcosm.api import binding
from microcosm_logging.decorators import logger


@binding("invocations_controller")
@logger
class InvocationsController():
    def __init__(self, graph):
        self.bundle = graph.active_bundle

    def search(self, simple_arg, **kwargs):
        results = self.bundle.predict(simple_arg)

        return results, len(results)
