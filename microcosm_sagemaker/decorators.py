import logging
import time
from typing import Any, Callable

from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.artifact import BundleInputArtifact, BundleOutputArtifact
from microcosm_sagemaker.input_data import InputData


def training_initializer():
    """
    Register a microcosm component as a training initializer, so that its init
    method will automatically be called.  This function is designed to be used
    as a decorator on a factory.

    """
    def decorator(func: Callable[[ObjectGraph], Any]):
        def factory(graph):
            component = func(graph)
            graph.training_initializers.register(component)
            return component
        return factory
    return decorator


# TODO: remove this when get_component_name is released with microcosm
try: 
    from microcosm.api import get_component_name

except ImportError:
    def get_component_name(graph: ObjectGraph, component) -> str:
        """
        Given an object that is attached to the graph, it returns the object name.
        """
        return next(
            key
            for key, possible_component in graph.items()
            if possible_component == component
        )


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_seconds = time.perf_counter() - self._start


def _method_with_logging(original_method):
    def new_method(*args, **kwargs):
        self = args[0]
        logging.info(
            f"Started method `{original_method.__name__}` of the `{self.bundle_name}`."
        )
        with Timer() as t:
            original_method(*args, **kwargs)
        logging.info(
            f"Completed method `{original_method.__name__}` of the `{self.bundle_name}` after {t.elapsed_seconds:.1f} seconds."
        )
    return new_method


def log_bundle_methods(cls):

    _init = cls.__init__

    def __init__(self, graph: ObjectGraph, **kwargs) -> None:
        _init(self, graph, **kwargs)
        self._graph = graph

    @property
    def bundle_name(self):
        return get_component_name(self._graph, self)

    cls.__init__ = __init__
    cls.bundle_name = bundle_name
    cls.fit = _method_with_logging(cls.fit)
    cls.load = _method_with_logging(cls.load)
    cls.save = _method_with_logging(cls.save)

    return cls
