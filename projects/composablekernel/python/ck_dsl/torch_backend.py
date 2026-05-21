# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Optional TorchInductor-style backend.

This module provides ``torch.compile(fn, backend=ck_dsl.compile)``
support. The backend is intentionally conservative: it walks the
captured FX graph and replaces every supported subgraph (single matmul
plus an epilogue chain, pointwise-only chain, or rowwise reduction)
with a CK DSL fused kernel call, while leaving everything else alone.

Approach
--------

1. Run :func:`helpers.fuse.compile_fn` against the whole graph first.
   When it succeeds the backend returns the fused callable directly --
   this is the fast path for the small atomically-fused programs
   ``compile_fn`` was designed for.

2. When the whole-graph match fails the backend falls back to walking
   the graph and inlining per-output fusion attempts. Any output the
   pattern matcher can express is replaced with a fused call; the
   rest stays as the original ``gm.forward``. Unsupported graphs
   gracefully fall back to ``gm.forward`` instead of raising, so
   ``torch.compile`` users can opt into the backend incrementally.

The backend never imports torch eagerly (the import lives inside the
``compile`` function), so the surrounding module is safe to import
in tests that run without torch installed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence


__all__ = ["compile", "BackendError"]


class BackendError(RuntimeError):
    """Raised by the backend for non-fallback errors (programmer error)."""


def _build_subgraph(gm, output_node):
    """Build a small ``torch.fx.GraphModule`` whose single output is ``output_node``.

    Used by the partial-fusion path: we extract the subgraph that
    produces one supported output, run :func:`compile_fn` against it,
    and splice the resulting callable back into the parent graph.
    """
    import torch.fx as fx

    sub = fx.Graph()
    env: Dict[Any, Any] = {}

    def visit(node):
        if node in env:
            return env[node]
        if node.op == "placeholder":
            new = sub.placeholder(node.target)
        elif node.op == "get_attr":
            new = sub.get_attr(node.target)
        elif node.op == "call_function":
            args = tuple(visit(a) if isinstance(a, fx.Node) else a for a in node.args)
            kwargs = {
                k: (visit(v) if isinstance(v, fx.Node) else v)
                for k, v in node.kwargs.items()
            }
            new = sub.call_function(node.target, args, kwargs)
        elif node.op == "call_method":
            args = tuple(visit(a) if isinstance(a, fx.Node) else a for a in node.args)
            new = sub.call_method(node.target, args, {})
        elif node.op == "call_module":
            args = tuple(visit(a) if isinstance(a, fx.Node) else a for a in node.args)
            new = sub.call_module(node.target, args, {})
        else:
            raise BackendError(f"unsupported FX op {node.op!r} for subgraph extraction")
        env[node] = new
        return new

    out = visit(output_node)
    sub.output(out)
    return fx.GraphModule(gm, sub)


def _try_full_graph(gm, example_inputs) -> Callable[..., Any]:
    """First attempt: ``compile_fn`` against the whole graph."""
    from .helpers.fuse import FusionMatchError, compile_fn

    try:
        compiled = compile_fn(gm.forward)
        return compiled
    except (FusionMatchError, NotImplementedError):
        return None  # type: ignore[return-value]


def _partial_fuse(gm) -> Callable[..., Any]:
    """Fallback: try per-output FX subgraph extraction + fusion.

    The current torch.fx graph types produced by ``torch.compile``
    typically expose one ``output`` node whose ``args[0]`` may be a
    tuple of node outputs. For each such output we try to extract a
    sub-graph module and run :func:`compile_fn`. When successful we
    splice the call back into the parent graph as a ``call_function``
    on the compiled callable.
    """
    import torch.fx as fx
    from .helpers.fuse import FusionMatchError, compile_fn

    graph = gm.graph
    output_node = next(n for n in graph.nodes if n.op == "output")
    raw = output_node.args[0]
    outputs = raw if isinstance(raw, (tuple, list)) else (raw,)

    new_graph = fx.Graph()
    env: Dict[Any, Any] = {}

    def map_arg(a):
        if isinstance(a, fx.Node):
            return env[a]
        if isinstance(a, (list, tuple)):
            cls = type(a)
            return cls(map_arg(x) for x in a)
        return a

    placeholders = {}
    for node in graph.nodes:
        if node.op == "placeholder":
            new = new_graph.placeholder(node.target)
            env[node] = new
            placeholders[node.target] = new
            continue
        if node.op == "output":
            continue
        if node in outputs:
            # Try fusing the sub-graph that produces this output.
            try:
                sub = _build_subgraph(gm, node)
                compiled = compile_fn(sub.forward)
            except (FusionMatchError, NotImplementedError, BackendError):
                compiled = None
            if compiled is not None:
                # Identify which placeholders this output uses, in
                # call order. The sub-builder preserves placeholder
                # insertion order.
                used_names = []
                for sub_node in sub.graph.nodes:
                    if sub_node.op == "placeholder":
                        used_names.append(sub_node.target)
                args = tuple(placeholders[name] for name in used_names)
                env[node] = new_graph.call_function(compiled, args, {})
                continue
        # Default: copy the op unchanged.
        if node.op == "call_function":
            new = new_graph.call_function(
                node.target,
                map_arg(node.args),
                {k: map_arg(v) for k, v in node.kwargs.items()},
            )
        elif node.op == "call_method":
            new = new_graph.call_method(node.target, map_arg(node.args), {})
        elif node.op == "call_module":
            new = new_graph.call_module(node.target, map_arg(node.args), {})
        elif node.op == "get_attr":
            new = new_graph.get_attr(node.target)
        else:
            raise BackendError(f"unsupported FX op {node.op!r}")
        env[node] = new
    new_graph.output(map_arg(raw))
    new_gm = fx.GraphModule(gm, new_graph)
    return new_gm.forward


def compile(gm, example_inputs: Sequence[Any]):
    """Torch backend entry point: ``torch.compile(fn, backend=compile)``.

    Returns a callable. The implementation tries a whole-graph CK DSL
    fusion first; failing that, walks the FX graph and replaces every
    supported subgraph with a fused call while leaving the rest as
    the original ``gm.forward``. Unsupported graphs gracefully fall
    back to ``gm.forward`` instead of raising, so users can enable
    this backend incrementally.
    """
    full = _try_full_graph(gm, example_inputs)
    if full is not None:
        return full
    try:
        return _partial_fuse(gm)
    except BackendError:
        return gm.forward


__all__ = ["compile", "BackendError"]
