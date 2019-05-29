from dataclasses import dataclass, field
from typing import List

from hamcrest import (
    any_of,
    assert_that,
    calling,
    contains,
    raises,
)

from microcosm_sagemaker.dependency_traverser import traverse_component_and_dependencies
from microcosm_sagemaker.exceptions import DependencyCycleError


@dataclass(eq=True, frozen=True)
class Component:
    name: str
    dependencies: List["Component"] = field(compare=False)


def test_no_dependencies():
    a = Component("a", [])

    assert_that(
        list(traverse_component_and_dependencies(a)),
        contains(a),
    )


def test_simple_dependencies():
    a = Component("a", [])
    b = Component("b", [a])

    assert_that(
        list(traverse_component_and_dependencies(b)),
        contains(a, b),
    )


def test_diamond_dependencies():
    a = Component("a", [])
    b = Component("b", [a])
    c = Component("c", [a])
    d = Component("d", [b, c])

    assert_that(
        list(traverse_component_and_dependencies(d)),
        any_of(
            contains(a, b, c, d),
            contains(a, c, b, d),
        ),
    )


def test_complex_graph():
    """
    Test the following graph:

      f
     ╱ ╲
    e   d
       ╱ ╲
      b   c
       ╲ ╱
        a

    """
    a = Component("a", [])
    b = Component("b", [a])
    c = Component("c", [a])
    d = Component("d", [b, c])
    e = Component("e", [])
    f = Component("f", [d, e])

    assert_that(
        list(traverse_component_and_dependencies(f)),
        any_of(
            contains(e, a, b, c, d, f),
            contains(a, e, b, c, d, f),
            contains(a, b, e, c, d, f),
            contains(a, b, c, e, d, f),
            contains(a, b, c, d, e, f),
            contains(e, a, c, b, d, f),
            contains(a, e, c, b, d, f),
            contains(a, c, e, b, d, f),
            contains(a, c, b, e, d, f),
            contains(a, c, b, d, e, f),
        ),
    )


def test_one_node_cycle():
    a = Component("a", [])
    a.dependencies.append(a)

    assert_that(
        calling(list).with_args(
            traverse_component_and_dependencies(a),
        ),
        raises(DependencyCycleError)
    )


def test_two_node_cycle():
    a = Component("a", [])
    b = Component("b", [a])
    a.dependencies.append(b)

    assert_that(
        calling(list).with_args(
            traverse_component_and_dependencies(a),
        ),
        raises(DependencyCycleError)
    )

    assert_that(
        calling(list).with_args(
            traverse_component_and_dependencies(b),
        ),
        raises(DependencyCycleError)
    )


def test_complex_cycle():
    a = Component("a", [])
    b = Component("b", [a])
    c = Component("c", [a])
    d = Component("d", [b, c])
    a.dependencies.append(d)

    for component in [a, b, c, d]:
        assert_that(
            calling(list).with_args(
                traverse_component_and_dependencies(component),
            ),
            raises(DependencyCycleError)
        )
