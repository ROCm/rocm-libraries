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
look under library/src" as a standing CI check. It statically folds the
string-*expressions* each ``.py`` file in the characterization suite builds
(plain literals, ``+`` / f-string concatenation, ``os.path.join``, and
``pathlib.Path`` ``/`` joins) and checks the resulting **path components**
against a set of segments specific enough to the real product tree that they
have no legitimate use here (``amd_detail``, ``rocblaslt``, ``asm_full``) --
as opposed to a bare ``"library"``, which is a common, unrelated dict
key/dirname elsewhere in this suite and would produce false positives.

Checking whole path *components* (not "is this substring present anywhere")
matters in both directions: it doesn't flag an unrelated literal that merely
contains one of these words as a substring (e.g. ``"rocblaslt-bench"``), and
it doesn't miss the forbidden path when it's assembled from split literals
(e.g. ``"amd_" + "detail"``, or ``os.path.join(...)`` over such pieces).
"""

import ast
import os

import pytest

pytestmark = pytest.mark.unit

_CHAR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../characterization
_THIS_FILE = os.path.abspath(__file__)

# Path components that only make sense when reaching into the real hipBLASLt
# source tree (library/src/amd_detail/rocblaslt/.../asm_full); none of these
# have a legitimate, unrelated use inside the characterization suite's own
# sources (unlike a bare "library", which is a common dict key).
_FORBIDDEN_SEGMENTS = {"amd_detail", "rocblaslt", "asm_full"}


def _iter_char_py_files():
    for dirpath, _dirnames, filenames in os.walk(_CHAR_ROOT):
        for fname in filenames:
            if fname.endswith(".py"):
                yield os.path.join(dirpath, fname)


def _resolve_str_expr(node):
    """Best-effort constant-fold a string-*building* AST expression.

    Returns the resolved string for anything a source file could plausibly use
    to assemble a path segment-by-segment (literal, ``+``/f-string
    concatenation, ``os.path.join``/``posixpath.join``/``ntpath.join``, or a
    ``pathlib.Path`` ``/`` join) — or ``None`` if the expression depends on a
    runtime value and can't be folded statically.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):  # f-string
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None  # a formatted (non-constant) part; can't fold
        return "".join(parts)

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left = _resolve_str_expr(node.left)
            right = _resolve_str_expr(node.right)
            if left is not None and right is not None:
                return left + right
        elif isinstance(node.op, ast.Div):
            # pathlib-style `Path(...) / "..."` / `"..." / "..."` joining.
            left = _resolve_str_expr(node.left)
            right = _resolve_str_expr(node.right)
            if left is not None and right is not None:
                return left.rstrip("/") + "/" + right
        return None

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "join":
            # os.path.join(...) / posixpath.join(...) / ntpath.join(...)
            args = [_resolve_str_expr(a) for a in node.args]
            if args and all(a is not None for a in args):
                return "/".join(args)
        if (isinstance(func, ast.Name) and func.id == "Path") or (
            isinstance(func, ast.Attribute) and func.attr == "Path"
        ):
            # pathlib.Path("...") / Path("...") — pass the single arg through.
            if len(node.args) == 1:
                return _resolve_str_expr(node.args[0])
        return None

    return None


def _forbidden_components(resolved):
    """Forbidden segments present as whole path components of ``resolved``."""
    components = set()
    for part in resolved.replace("\\", "/").split("/"):
        components.add(part)
    return _FORBIDDEN_SEGMENTS & components


def _forbidden_literals_in_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Call)):
            resolved = _resolve_str_expr(node)
            if resolved is not None:
                found |= _forbidden_components(resolved)
    return found


def test_no_library_src_path_construction():
    """No characterization ``.py`` source may build a path through
    ``library/src/amd_detail/rocblaslt/.../asm_full``.

    The characterization suite owns no code under ``library/``; any of these
    path components appearing (as a literal, or assembled via concatenation /
    ``os.path.join`` / ``pathlib`` joins) means a test is (re)coupling itself
    to the live, frequently-retuned hipBLASLt production tuning tree instead
    of using an in-tree vendored fixture under ``_codegen/data/``.
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


# --- Direct tests of the guard's own detection logic -------------------------
#
# The guard above walks real files on disk; these exercise
# `_forbidden_literals_in_file` against synthetic sources (via `tmp_path`) so
# the detection contract itself — including the two failure modes an earlier
# review found (a harmless substring false-flagged, and a split/joined
# forbidden path missed entirely) — is pinned directly, not just implied by
# "the real suite happens to be clean today".


def _check(tmp_path, source):
    p = tmp_path / "probe.py"
    p.write_text(source, encoding="utf-8")
    return _forbidden_literals_in_file(p)


def test_guard_catches_the_original_forbidden_path(tmp_path):
    found = _check(
        tmp_path,
        'PATH = "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"\n',
    )
    assert found == {"amd_detail", "rocblaslt", "asm_full"}


def test_guard_catches_os_path_join_variant(tmp_path):
    found = _check(
        tmp_path,
        'import os\n'
        'PATH = os.path.join("library", "src", "amd_detail", "rocblaslt", "asm_full")\n',
    )
    assert found == {"amd_detail", "rocblaslt", "asm_full"}


def test_guard_catches_split_concatenated_literals(tmp_path):
    # Same forbidden path, but each segment is assembled from smaller
    # constant pieces -- this used to slip past a per-literal substring check.
    found = _check(
        tmp_path,
        'import os\n'
        'PATH = os.path.join('
        '"library", "src", "amd_" + "detail", "roc" + "blaslt", "asm_" + "full"'
        ')\n',
    )
    assert found == {"amd_detail", "rocblaslt", "asm_full"}


def test_guard_catches_pathlib_join(tmp_path):
    found = _check(
        tmp_path,
        'from pathlib import Path\n'
        'PATH = Path("library") / "src" / "amd_detail" / "rocblaslt" / "asm_full"\n',
    )
    assert found == {"amd_detail", "rocblaslt", "asm_full"}


def test_guard_catches_fstring_concatenation(tmp_path):
    found = _check(
        tmp_path,
        'ARCH = "gfx942"\n'
        'PATH = f"library/src/amd_detail/rocblaslt/{ARCH}/asm_full"\n',
    )
    # The f-string has a non-constant formatted part (`{ARCH}`), so it can't
    # be folded whole -- but its constant literal segments are still visible
    # to the AST walk as their own nested Constant nodes.
    assert found == {"amd_detail", "rocblaslt", "asm_full"}


def test_guard_does_not_false_positive_on_unrelated_substring(tmp_path):
    # "rocblaslt" appears as a substring, but not as a whole path component --
    # this is a harmless tool/product name, not a path into the product tree.
    found = _check(tmp_path, 'TOOL_NAME = "rocblaslt-bench"\n')
    assert found == set()


def test_guard_does_not_false_positive_on_bare_library(tmp_path):
    found = _check(
        tmp_path,
        'import os\n'
        'PATH = os.path.join("library", "src", "codegen_fixture.yaml")\n',
    )
    assert found == set()


def test_guard_does_not_false_positive_on_docstring_prose(tmp_path):
    found = _check(
        tmp_path,
        '"""Mentions rocblaslt-bench and asm_fullstack tooling in prose, not a path."""\n',
    )
    assert found == set()
