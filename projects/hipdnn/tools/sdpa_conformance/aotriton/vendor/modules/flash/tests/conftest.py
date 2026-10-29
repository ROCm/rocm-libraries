# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest configuration for the flash AOT/pyaotriton integration+perf suite.

This suite has no __init__.py, so pytest's default (rootless) import mode already
puts this directory on sys.path before loading any module in it — the suite's
sibling modules (_common_test, attn_torch_function, aotriton_flash, fwd_kernel, …)
import cleanly without naming individual test files.

test_forward.py is intentionally EXCLUDED from directory collection: the forward
kernel is already exercised (with broader coverage) as part of test_backward.py,
which runs the forward pass to produce the inputs it differentiates. test_forward.py
is kept only as a standalone entry point for users who want to check the forward
kernel in isolation — run it explicitly (`pytest .../test_forward.py`); its coverage
is a subset of test_backward.py's.
"""

from pathlib import Path

# Exclude the forward-only suite from `pytest modules/flash/tests` (see docstring).
collect_ignore = ['test_forward.py']

# Explicit run order for the suite, most-important-first. test_backward.py carries the
# bulk of the coverage, so it must be dispatched before test_varlen.py rather than
# relying on the alphabetical accident that currently orders them (b < v) -- a future
# test_aaa_*.py would silently jump the queue.
#
# Files not listed here keep their relative collection order and run after the listed
# ones.
_FILE_ORDER = ['test_backward.py', 'test_varlen.py']


def pytest_collection_modifyitems(config, items):
    """Order collected tests by _FILE_ORDER.

    Under `-n`, this orders the *dispatch queue*, not completion: with xdist's default
    `--dist load` the controller hands tests out in collection order, so varlen work is
    only handed out once the backward queue is drained -- but a straggler backward test
    can still be in flight when the first varlen test starts. A hard barrier would need
    two separate pytest invocations.
    """
    def rank(item):
        # item.location[0] is the repo-relative path; stable public API, unlike fspath.
        name = Path(item.location[0]).name
        return _FILE_ORDER.index(name) if name in _FILE_ORDER else len(_FILE_ORDER)

    items.sort(key=rank)  # stable: preserves within-file order
