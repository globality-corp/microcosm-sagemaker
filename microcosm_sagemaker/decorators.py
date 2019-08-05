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


def log_bundle_methods(cls):

    _init = cls.__init__
    _fit = cls.fit
    _load = cls.load
    _save = cls.save

    def __init__(self, graph: ObjectGraph, **kwargs) -> None:
        _init(self, graph, **kwargs)
        self._graph = graph

    def fit(self, input_data: InputData) -> None:
        logging.info(
            f"Started `fitting` the `{self.bundle_name}` with `{str(input_data.path)}` input data."
        )
        with Timer() as t:
            _fit(self, input_data)
        logging.info(
            f"Completed `fitting` the `{self.bundle_name}` after {t.elapsed_seconds:.1f} seconds."
        )

    def load(self, input_artifact: BundleInputArtifact) -> None:
        logging.info(
            f"Started `loading` the `{self.bundle_name}` from `{str(input_artifact.path)}` input artifact."
        )
        with Timer() as t:
            _load(self, input_artifact)
        logging.info(
            f"Completed `loading` the `{self.bundle_name}` after {t.elapsed_seconds:.1f} seconds."
        )

    def save(self, output_artifact: BundleOutputArtifact) -> None:
        logging.info(
            f"Started `saving` the `{self.bundle_name}` artifacts into `{str(output_artifact.path)}`."
        )
        with Timer() as t:
            _save(self, output_artifact)
        logging.info(
            f"Completed `saving` the `{self.bundle_name}` artifacts after {t.elapsed_seconds:.1f} seconds."
        )

    @property
    def bundle_name(self):
        return get_component_name(self._graph, self)

    cls.__init__ = __init__
    cls.bundle_name = bundle_name
    cls.fit = fit
    cls.load = load
    cls.save = save

    return cls
