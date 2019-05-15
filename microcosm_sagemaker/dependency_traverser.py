from contextlib import contextmanager
from operator import attrgetter
from typing import (
    Callable,
    Iterable,
    Set,
    TypeVar,
)

from microcosm_sagemaker.exceptions import DependencyCycleError


T = TypeVar('T')


def traverse_component_and_dependencies(
    component: T,
    dependency_getter: Callable[[T], Iterable[T]] = attrgetter("dependencies"),
):
    """
    Given a component in a dependency graph, traverses the graph in topological
    order, ie such that all dependencies of a component will be yielded before
    the component itself.

    """
    seen: Set[T] = set()
    reserved: Set[T] = set()

    yield from _traverse_helper(
        dependency_getter=dependency_getter,
        seen=seen,
        reserved=reserved,
        component=component,
    )


def _traverse_helper(
    dependency_getter: Callable[[T], Iterable[T]],
    seen: Set[T],
    reserved: Set[T],
    component: T,
):
    if component in seen:
        return

    with _reserve(reserved, component):
        for dependency in dependency_getter(component):
            yield from _traverse_helper(
                dependency_getter=dependency_getter,
                seen=seen,
                reserved=reserved,
                component=dependency,
            )

    yield component
    seen.add(component)


@contextmanager
def _reserve(reserved: Set[T], component: T):
    """
    Adds component to a list of reserved components, erroring if the component
    has already been reserved, then remove compenet from list of reserved
    components.

    """
    if component in reserved:
        raise DependencyCycleError(component)

    reserved.add(component)
    try:
        yield
    finally:
        reserved.remove(component)
