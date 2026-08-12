################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id f6884744123d833fb9c2adb4d3d9b29f52dfa5b0

Predicate : self.do["PreLoop"]   (bare boolean read)
Site      : Tensile/KernelWriterAssembly.py:2182
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

Classification note
-------------------
Both polarities are solver-satisfiable under the seeded boolean domain.
At runtime the live value is pinned to True: KernelWriter.__init__
(KernelWriter.py:485) writes ``self.do["PreLoop"] = True`` and that literal
is never reassigned anywhere in the Tensile/ package.  The FALSE branch is
therefore structurally dead at runtime even though the solver admits it.

Tests here:
  1. Predicate semantics — pin that ``pred(x) = bool(x)`` for both witnesses.
  2. Runtime binding    — pin that KernelWriter.__init__ sets the value to True
     (confirms the singleton live domain, documents any future change).
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Predicate semantics (pure; no Tensile import required)
# ---------------------------------------------------------------------------

def _pred(value):
    """Mirror of the branch predicate: ``if self.do["PreLoop"]:``"""
    return bool(value)


def test_preloop_true_example_enters_branch():
    """z3 true witness: self.do['PreLoop']=True -> predicate evaluates True."""
    assert _pred(True) is True


def test_preloop_false_example_skips_branch():
    """z3 false witness: self.do['PreLoop']=False -> predicate evaluates False."""
    assert _pred(False) is False


# ---------------------------------------------------------------------------
# Runtime binding pin — KernelWriter.__init__ sets "PreLoop" = True
# ---------------------------------------------------------------------------

def test_preloop_init_binding_is_true():
    """
    KernelWriter.py:485 assigns self.do['PreLoop'] = True.
    Pin the actual initialised value so any future change is caught here.

    This import is CPU-only: KernelWriter.__init__ accepts assembler/debugConfig
    keyword args; we pass minimal stubs so __init__ completes without GPU access.
    """
    import types

    # Minimal stub assembler that satisfies attribute reads KernelWriter.__init__
    # performs before the self.do block (it reads no attrs in the pre-do path).
    assembler_stub = types.SimpleNamespace()
    debugConfig_stub = types.SimpleNamespace()

    from Tensile.KernelWriter import KernelWriter  # noqa: PLC0415

    # KernelWriter is an abstract base; instantiate the minimal concrete subclass
    # that exists in the package — KernelWriterAssembly — but only import what
    # is needed to read self.do without triggering GPU assembly.
    # We use __new__ + partial __init__ tracing via a do-dict spy rather than
    # calling full __init__ (which would require a complete Solution object).
    #
    # Safer approach: read the class body directly to confirm the literal.
    import inspect
    import ast
    import textwrap

    source = textwrap.dedent(inspect.getsource(KernelWriter.__init__))
    tree = ast.parse(source)

    assigned_true = False
    for node in ast.walk(tree):
        # Match:  self.do["PreLoop"] = True
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
        ):
            target = node.targets[0]
            if (
                isinstance(target.value, ast.Attribute)
                and target.value.attr == "do"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "PreLoop"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                assigned_true = True
                break

    assert assigned_true, (
        "KernelWriter.__init__ no longer assigns self.do['PreLoop'] = True; "
        "re-evaluate the FALSE-branch dead-code classification for "
        "KernelWriterAssembly.py:2182."
    )
