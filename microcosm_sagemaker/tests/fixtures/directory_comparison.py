from typing import NamedTuple


class DirectoryComparisonTest(NamedTuple):
    name: str
    should_pass: bool
    use_json_matcher: bool = False


DIRECTORY_COMPARISON_TEST_CASES = [
    # simple-pass/
    # ├── actual/
    # │  ├── subdir/
    # │  │  └── hi.txt = hi
    # │  └── hello.txt = hello
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.txt = hi
    #    └── hello.txt = hello
    DirectoryComparisonTest("simple-pass", True),

    # added-path-fail/
    # ├── actual/
    # │  ├── subdir/
    # │  │  ├── extra.txt
    # │  │  └── hi.txt = hi
    # │  └── hello.txt = hello
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.txt = hi
    #    └── hello.txt = hello
    DirectoryComparisonTest("added-path-fail", False),

    # missing-path-fail/
    # ├── actual/
    # │  ├── subdir/
    # │  │  └── keep
    # │  └── hello.txt = hello
    # └── gold/
    #    ├── subdir/
    #    │  ├── hi.txt = hi
    #    │  └── keep
    #    └── hello.txt = hello
    DirectoryComparisonTest("missing-path-fail", False),

    # content-fail/
    # ├── actual/
    # │  ├── subdir/
    # │  │  └── hi.txt = hi
    # │  └── hello.txt = hello
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.txt = hi uh oh
    #    └── hello.txt = hello
    DirectoryComparisonTest("content-fail", False),

    # file-to-dir-fail/
    # ├── actual/
    # │  ├── hello.txt/
    # │  └── subdir/
    # │     └── hi.txt = hi
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.txt = hi
    #    └── hello.txt = hello
    DirectoryComparisonTest("file-to-dir-fail", False),

    # dir-to-file-fail/
    # ├── actual/
    # │  ├── hello.txt = hello
    # │  └── subdir
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.txt = hi
    #    └── hello.txt = hello
    DirectoryComparisonTest("dir-to-file-fail", False),

    # matcher-pass/
    # ├── actual/
    # │  ├── subdir/
    # │  │  └── hi.json = {"foo":"bar","extra":"fine"}
    # │  └── hello.json = {"foo":"baz","extra":"fine"}
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.json = {"foo":"bar"}
    #    └── hello.json = {"foo":"bar"}
    DirectoryComparisonTest("matcher-pass", True, use_json_matcher=True),

    # matcher-fail/
    # ├── actual/
    # │  ├── subdir/
    # │  │  └── hi.json = {"foo":"baz","extra":"fine"}
    # │  └── hello.json = {"foo":"baz","extra":"fine"}
    # └── gold/
    #    ├── subdir/
    #    │  └── hi.json = {"foo":"bar"}
    #    └── hello.json = {"foo":"bar"}
    DirectoryComparisonTest("matcher-fail", False, use_json_matcher=True),
]
