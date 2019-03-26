"""
Consumer factories.

"""


def configure_active_bundle(graph):
    if not getattr(graph.config, "active_bundle", default=""):
        return None
    return getattr(graph, graph.config.active_bundle)


def configure_active_evaluation(graph):
    if not getattr(graph.config, "active_evaluation", default=""):
        return None
    return getattr(graph, graph.config.active_evaluation)
