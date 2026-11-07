# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SLO107: count repeated low-level code shapes, one counter per kind of
copy-pasted index or mask math. Pure `ast`. Every matcher says exactly what it counts, so a number
can be checked by hand.

`builder_call(node)` is supplied by the caller: it returns the method name when
`node` is a call on the IR builder (`b.add(...)`), else None.
"""
from __future__ import annotations

import ast
from collections import Counter

# id -> one-line definition (shown in reports and help).
PATTERNS = {
    "const": "builder const_i32/const_i64/const_f32 calls, plus bare const_*() calls",
    "add_mul": "builder add() with a builder mul() as a direct argument (row*stride+col)",
    "lane_decode": "builder div()/mod() of a value that comes from thread_id_x()/lane_id(), directly or through earlier div()/mod()",
    "compare": "builder cmp_*, fcmp, vector_cmp calls",
    "select": "builder select, masked_select, vector_select calls",
    "mask": "builder land, lor, lnot calls",
    "ceil_div": "builder div(add(x, ...k-1...), k), and Python (x + k - 1) // k or -(-x // k)",
    "global_load": "builder global_load*, masked_global_load, buffer_load* calls",
    "lds": "builder smem_load*, smem_store*, ds_read* calls",
    "sync": "builder sync*, s_waitcnt, s_wait_*, s_barrier*, sched_barrier, sched_group_barrier, iglp_opt calls",
    "vec": "builder vec_extract, vec_pack, vec_concat, vec_insert, bitcast, vec_bitcast calls",
    "exp2": "builder exp2 / exp2_fast calls (the online-softmax step)",
    "epilogue_cast": "builder trunc_f32_to_*, vec_trunc_f32_to_*, cast_f32_to, vec_cast_f32_to, cast_to_f32 calls",
    "params_dict": "params['...'] subscripts (kernel arguments unpacked from a dict)",
}

_EXACT = {
    "compare": {"fcmp", "vector_cmp"},
    "select": {"select", "masked_select", "vector_select"},
    "mask": {"land", "lor", "lnot"},
    "sync": {"s_waitcnt", "sched_barrier", "sched_group_barrier", "iglp_opt"},
    "vec": {
        "vec_extract",
        "vec_pack",
        "vec_concat",
        "vec_insert",
        "bitcast",
        "vec_bitcast",
    },
    "exp2": {"exp2", "exp2_fast"},
    "epilogue_cast": {"cast_f32_to", "vec_cast_f32_to", "cast_to_f32"},
    "global_load": {"masked_global_load"},
}
_PREFIX = {
    "const": ("const_",),
    "compare": ("cmp_",),
    "global_load": ("global_load", "buffer_load"),
    "lds": ("smem_load", "smem_store", "ds_read"),
    "sync": ("sync", "s_wait_", "s_barrier"),
    "epilogue_cast": ("trunc_f32_to_", "vec_trunc_f32_to_"),
}
_LANE_SOURCES = {"thread_id_x", "lane_id"}


def _simple_kind(op: str) -> str | None:
    for kind, names in _EXACT.items():
        if op in names:
            return kind
    for kind, prefixes in _PREFIX.items():
        if op.startswith(prefixes):
            return kind
    return None


def _has_minus_one(node: ast.AST) -> bool:
    """True if `node` contains a `<expr> - 1` subtraction."""
    for n in ast.walk(node):
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Sub)
            and isinstance(n.right, ast.Constant)
            and n.right.value == 1
        ):
            return True
    return False


def count_patterns(tree: ast.AST, builder_call) -> Counter:
    c: Counter = Counter({k: 0 for k in PATTERNS})

    # lane_decode needs a per-function view of which names hold lane/thread ids.
    def lane_names(fn: ast.AST) -> set[str]:
        tainted: set[str] = set()
        changed = True
        while changed:
            changed = False
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                    op = builder_call(n.value)
                    src = op in _LANE_SOURCES or (
                        op in ("div", "mod") and _lane_arg(n.value, tainted)
                    )
                    if src:
                        for t in n.targets:
                            if isinstance(t, ast.Name) and t.id not in tainted:
                                tainted.add(t.id)
                                changed = True
        return tainted

    def _lane_arg(call: ast.Call, tainted: set[str]) -> bool:
        if not call.args:
            return False
        a = call.args[0]
        if isinstance(a, ast.Name):
            return a.id in tainted
        return isinstance(a, ast.Call) and builder_call(a) in _LANE_SOURCES

    counted_lane: set[int] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tainted = lane_names(fn)
            for n in ast.walk(fn):
                if (
                    isinstance(n, ast.Call)
                    and id(n) not in counted_lane
                    and builder_call(n) in ("div", "mod")
                    and _lane_arg(n, tainted)
                ):
                    counted_lane.add(id(n))
    c["lane_decode"] = len(counted_lane)

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            op = builder_call(n)
            if op is None:
                if isinstance(n.func, ast.Name) and n.func.id.startswith("const_"):
                    c["const"] += 1
                continue
            kind = _simple_kind(op)
            if kind:
                c[kind] += 1
            if op == "add" and any(
                isinstance(a, ast.Call) and builder_call(a) == "mul" for a in n.args
            ):
                c["add_mul"] += 1
            if (
                op == "div"
                and n.args
                and isinstance(n.args[0], ast.Call)
                and builder_call(n.args[0]) == "add"
            ):
                inner = n.args[0]
                if any(
                    isinstance(a, ast.Call) and builder_call(a) == "sub"
                    for a in inner.args
                ) or _has_minus_one(inner):
                    c["ceil_div"] += 1
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv):
            # needs an Add too, so (n - 1) // 2 is not counted
            if (
                isinstance(n.left, ast.BinOp)
                and _has_minus_one(n.left)
                and any(
                    isinstance(m, ast.BinOp) and isinstance(m.op, ast.Add)
                    for m in ast.walk(n.left)
                )
            ):
                c["ceil_div"] += 1
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            o = n.operand
            if (
                isinstance(o, ast.BinOp)
                and isinstance(o.op, ast.FloorDiv)
                and isinstance(o.left, ast.UnaryOp)
                and isinstance(o.left.op, ast.USub)
            ):
                c["ceil_div"] += 1
        elif isinstance(n, ast.Subscript):
            if (
                isinstance(n.value, ast.Name)
                and n.value.id == "params"
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)
            ):
                c["params_dict"] += 1
    return c
