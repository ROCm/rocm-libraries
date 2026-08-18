################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Regression guard: no characterization test may construct a path into the
live hipBLASLt product tree (``library/src/amd_detail/rocblaslt/...``).

Incident this guards against: ``test_bigfile_capped_emit`` (see
``_codegen/test_emit_bigfiles_char.py``) used to read production tuned-logic
YAMLs directly out of ``library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full``.
Those files are live, frequently-retuned tuning data, so unrelated tuning PRs
(e.g. #10877) could shift the pinned golden and fail unrelated PRs (e.g.
#10750). The fix vendors trimmed, self-contained copies of the needed data
under ``_codegen/data/bigfiles/`` instead (see DECISIONS.md).

This test operationalizes "the TensileLite characterization tests should never
look under library/src" as a standing CI check: it scans every ``.py`` file in
the characterization suite for string-literal path segments that are specific
enough to the real product tree that they have no legitimate use here
(``amd_detail``, ``rocblaslt``, ``asm_full``) -- as opposed to a bare
``"library"``, which is a common, unrelated dict key/dirname elsewhere in this
suite and would produce false positives.
"""

import ast
import os

import pytest

pytestmark = pytest.mark.unit

_CHAR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../characterization
_THIS_FILE = os.path.abspath(__file__)

# String-literal path segments that only make sense when reaching into the
# real hipBLASLt source tree (library/src/amd_detail/rocblaslt/.../asm_full);
# none of these have a legitimate, unrelated use inside the characterization
# suite's own sources (unlike a bare "library", which is a common dict key).
_FORBIDDEN_SEGMENTS = {"amd_detail", "rocblaslt", "asm_full"}


def _iter_char_py_files():
    for dirpath, _dirnames, filenames in os.walk(_CHAR_ROOT):
        for fname in filenames:
            if fname.endswith(".py"):
                yield os.path.join(dirpath, fname)


def _forbidden_literals_in_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _FORBIDDEN_SEGMENTS:
                found.add(node.value)
    return found


def test_no_library_src_path_construction():
    """No characterization ``.py`` source may build a path through
    ``library/src/amd_detail/rocblaslt/.../asm_full``.

    The characterization suite owns no code under ``library/``; any of these
    literal path segments appearing as a string constant here means a test is
    (re)coupling itself to the live, frequently-retuned hipBLASLt production
    tuning tree instead of using an in-tree vendored fixture under
    ``_codegen/data/``.
    """
    offenders = {}
    for path in _iter_char_py_files():
        if os.path.abspath(path) == _THIS_FILE:
            continue  # this guard legitimately names the forbidden segments
        found = _forbidden_literals_in_file(path)
        if found:
            offenders[os.path.relpath(path, _CHAR_ROOT)] = sorted(found)

    assert not offenders, (
        "Characterization test(s) construct a path into the live hipBLASLt "
        "product tree, which couples the codegen golden to unrelated tuning "
        "churn (see DECISIONS.md, test_emit_bigfiles_char.py history):\n"
        + "\n".join(f"  {f}: {segs}" for f, segs in sorted(offenders.items()))
    )
