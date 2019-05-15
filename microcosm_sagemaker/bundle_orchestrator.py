from abc import ABC, abstractmethod
from typing import Callable

from microcosm.object_graph import ObjectGraph

from microcosm_sagemaker.bundle import Bundle
from microcosm_sagemaker.dependency_traverser import traverse_component_and_dependencies


class BundleOrchestrator(ABC):
    def __init__(self, graph: ObjectGraph):
        pass

    @abstractmethod
    def __call__(
        self,
        active_bundle: Bundle,
        bundle_handler: Callable[[Bundle], None],
    ):
        """
        Given an `active_bundle`, call `bundle_handler` on `active_bundle` and
        all its transitive dependencies, ensuring that `bundle_handler` has
        been called on all dependencies of any bundle before it is called on
        the bundle itself.

        """
        ...


class SingleThreadedBundleOrchestrator(BundleOrchestrator):
    """
    Performs topological sort on dependencies of `active_bundle` and calls
    `bundle_handler` on each bundle one at a time.

    """
    def __call__(
        self,
        active_bundle: Bundle,
        bundle_handler: Callable[[Bundle], None],
    ):
        for bundle in traverse_component_and_dependencies(active_bundle):
            bundle_handler(bundle)
