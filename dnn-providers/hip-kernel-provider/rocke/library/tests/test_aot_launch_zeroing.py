# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A per-launch clear in the AOT sweep must be queued, never drained for.

Three wgrad legs have to zero something before every launch: the packed-atomic
split-K epilogue accumulates into dW, the two-stage path accumulates into its
f32 scratch, and a stream-K reduction clears dW, the scratch or its flag table
depending on which reduction it uses. That clear is part of the algorithm's
cost and is inside the timed region on purpose.

``hipMemset`` is synchronous with respect to the whole device, so one inside a
timed ``run()`` drains everything queued before it. On the serial path that
costs little -- the next thing was a drain anyway. Under ``--fast-rank`` it
undoes the flag: :func:`benchmarks.common.batched_timing.rank` queues a chunk
of candidates between two drains precisely so they do not serialise, and one
blocking clear in the middle serialises them again. Queued on the launch
stream (``hipMemsetAsync``), the clear is also what makes a shared output safe
to reuse across a batch, since same-stream FIFO orders it behind the previous
candidate's launches.

This is an easy thing to undo by writing the obvious ``rt.memset`` in a new
leg, and the symptom is a silent loss of speed rather than a failure, so the
rule is checked by reading the source rather than left to review.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from benchmarks.common import benchmark_implicit_gemm_conv as sweep

#: The AOT functions that build a launch closure for a cached binary.
_LAUNCH_BUILDERS = ("_cached_bind", "_streamk_launch")


def _function(name: str) -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(getattr(sweep, name)))
    return ast.parse(src).body[0]


def _blocking_memsets(node: ast.AST) -> list:
    """``rt.memset(...)`` calls anywhere under ``node``."""
    out = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "memset"
        ):
            out.append(sub)
    return out


def _run_closures(fn: ast.FunctionDef) -> list:
    """The nested ``run`` definitions -- the callable the timing loop calls."""
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.FunctionDef) and node.name == "run" and node is not fn
    ]


@pytest.mark.parametrize("name", _LAUNCH_BUILDERS)
def test_a_timed_launch_closure_never_drains_the_device(name):
    fn = _function(name)
    closures = _run_closures(fn)
    assert closures, f"{name} builds no run() closure -- has it been renamed?"
    for closure in closures:
        blocking = _blocking_memsets(closure)
        assert not blocking, (
            f"{name}: run() calls a blocking memset at line "
            f"{blocking[0].lineno} of the function. A clear inside a timed "
            f"launch has to be rt.memset_async(..., stream) on the launch "
            f"stream, or it drains every candidate the probe queued with it."
        )


@pytest.mark.parametrize("name", _LAUNCH_BUILDERS)
def test_the_clear_goes_on_the_launch_stream(name):
    """An async clear on the wrong stream is worse than a blocking one: it
    would be unordered against the launches it has to precede."""
    fn = _function(name)
    for closure in _run_closures(fn):
        for call in ast.walk(closure):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "memset_async"
            ):
                assert len(call.args) == 4, (
                    f"{name}: memset_async without an explicit stream falls "
                    f"back to the null stream, which synchronises with every "
                    f"other stream on ROCm"
                )
                assert isinstance(call.args[3], ast.Name)


def test_the_runtime_offers_a_queued_fill():
    from rocke.runtime.hip_module import Runtime

    assert hasattr(Runtime, "memset_async")
    params = list(inspect.signature(Runtime.memset_async).parameters)
    assert params == ["self", "ptr", "value", "nbytes", "stream"]


def test_setup_outside_a_launch_still_drains():
    """The rule is about the timed region, not about memset. Allocating a
    two-stage workspace clears it once, before anything is queued, and that
    one must stay blocking -- the launch that reads it is bound afterwards."""
    fn = _function("_extras_and_grid")
    assert _blocking_memsets(fn), (
        "_extras_and_grid no longer clears the workspace it allocates; if the "
        "clear moved into the launch, it has to be queued instead"
    )
