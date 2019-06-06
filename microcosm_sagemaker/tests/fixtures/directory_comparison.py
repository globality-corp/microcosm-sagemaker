from typing import NamedTuple

from microcosm_sagemaker.tests.directory_test_description import Dir, File


class DirectoryComparisonTest(NamedTuple):
    gold: Dir
    actual: Dir
    should_pass: bool
    use_json_matcher: bool = False
    ignore_hidden: bool = True


DIRECTORY_COMPARISON_TEST_CASES = [
    # Simple pass
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        actual=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        should_pass=True,
    ),

    # Extra path
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        actual=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
                "extra.txt": File(),
            }),
            "hello.txt": File("hello"),
        }),
        should_pass=False,
    ),

    # Missing an expected path
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "keep": File(),
            }),
            "hello.txt": File("hello"),
        }),
        actual=Dir({
            "subdir": Dir({
                "keep": File(),
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        should_pass=False,
    ),

    # Contents of a file differ
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        actual=Dir({
            "subdir": Dir({
                "hi.txt": File("hi uh oh"),
            }),
            "hello.txt": File("hello"),
        }),
        should_pass=False,
    ),

    # Expected file but found directory
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        actual=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": Dir(),
        }),
        should_pass=False,
    ),

    # Expected directory but found file
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
        }),
        actual=Dir({
            "subdir": File(),
            "hello.txt": File("hello"),
        }),
        should_pass=False,
    ),

    # Pass using a matcher to ignore some json entries
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.json": File('{"foo":"bar"}'),
            }),
            "hello.json": File('{"foo":"bar"}'),
        }),
        actual=Dir({
            "subdir": Dir({
                "hi.json": File('{"foo":"bar","extra":"fine"}'),
            }),
            "hello.json": File('{"foo":"bar","extra":"fine"}'),
        }),
        should_pass=True,
        use_json_matcher=True,
    ),

    # Fail using a matcher to ignore some json entries
    DirectoryComparisonTest(
        gold=Dir({
            "subdir": Dir({
                "hi.json": File('{"foo":"baz"}'),
            }),
            "hello.json": File('{"foo":"baz"}'),
        }),
        actual=Dir({
            "subdir": Dir({
                "hi.json": File('{"foo":"bar","extra":"fine"}'),
            }),
            "hello.json": File('{"foo":"bar","extra":"fine"}'),
        }),
        should_pass=False,
        use_json_matcher=True,
    ),

    # Both directories empty
    DirectoryComparisonTest(
        gold=Dir(),
        actual=Dir(),
        should_pass=True,
    ),

    # Using ignore_hidden=True
    DirectoryComparisonTest(
        gold=Dir({
            "subdir1": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
            "subdir2": Dir({
                ".keep": File(),
            }),
        }),
        actual=Dir({
            "subdir1": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
            "subdir2": Dir(),
        }),
        should_pass=True,
        ignore_hidden=True,
    ),

    # Using ignore_hidden=False
    DirectoryComparisonTest(
        gold=Dir({
            "subdir1": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
            "subdir2": Dir({
                ".keep": File(),
            }),
        }),
        actual=Dir({
            "subdir1": Dir({
                "hi.txt": File("hi"),
            }),
            "hello.txt": File("hello"),
            "subdir2": Dir(),
        }),
        should_pass=False,
        ignore_hidden=False,
    ),

    # Doubly nested directory using ignore_hidden=True
    DirectoryComparisonTest(
        gold=Dir({
            "subdir1": Dir({
                "hi.txt": File("hi"),
                "subdir1-1": Dir()
            }),
            "hello.txt": File("hello"),
            "subdir2": Dir({
                ".keep": File(),
            }),
        }),
        actual=Dir({
            "subdir1": Dir({
                "hi.txt": File("hi"),
                "subdir1-1": Dir({
                    ".keep": File(),
                })
            }),
            "hello.txt": File("hello"),
            "subdir2": Dir(),
        }),
        should_pass=True,
        ignore_hidden=True,
    ),
]
