# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hand-written adapters for Python calls whose C twin has a different shape.

Each entry is written once and reused by every translated file (part of the
plan's routine library, section 7.1). A template gets {b} (the builder), {obj}
(the receiver) and {args} (the lowered arguments, comma-separated).
"""

FUNCTIONS = {
    # rocke.helpers.spec.ceil_div(a, b) = -(-a // b)
    "spec:ceil_div": ("py::ceil_div({args})", "int64_t"),
    # math.gcd(a, b) of two ints
    "math:gcd": ("py::gcd({args})", "int64_t"),
    # itertools.pairwise(xs): consecutive pairs
    "itertools:pairwise": ("py::pairwise({args})", None),
    # eval_layout(layout) -> per lane, per slot (row, col); the C twin fills a flat int array
    "_attention_bwd_frag:eval_layout": (
        "py_adapt::eval_layout({b}, {args})",
        "py::tuple<py::tuple<std::array<int64_t, 2>>>",
        "py_adapt::eval_layout_scratch({args})",  # no builder in scope
    ),
}

METHODS = {
    # MmaOp.c_layout(): the verified lane map, NotImplementedError when the op has none
    ("const rocke_mma_op_t*", "c_layout"): (
        "py_adapt::c_layout({obj})",
        "const rocke_layout_map_t*",
    ),
    ("const rocke_mma_op_t*", "a_layout"): (
        "py_adapt::layout_of({obj}, {obj}->a_layout)",
        "const rocke_layout_map_t*",
    ),
    ("const rocke_mma_op_t*", "b_layout"): (
        "py_adapt::layout_of({obj}, {obj}->b_layout)",
        "const rocke_layout_map_t*",
    ),
}


# Attributes a C struct does not carry, derived as Python derives them.
ATTRS = {
    # ArchTarget.has_mfma / has_wmma: any atom of the catalog in that family
    ("const rocke_arch_target_t*", "has_mfma"): (
        'py_adapt::arch_has_family({obj}, "mma")',
        "bool",
    ),
    ("const rocke_arch_target_t*", "has_wmma"): (
        'py_adapt::arch_has_family({obj}, "wmma")',
        "bool",
    ),
}


def attr_adapter(recv_t: str, name: str):
    return ATTRS.get((recv_t, name))


# Methods with keyword arguments: (receiver type, name) -> (template, return
# type, parameter order, defaults). {args} gets the arguments in that order.
KW_METHODS = {
    # MmaCatalog.enumerate(*, family="mma", a_dtype, b_dtype, c_dtype, m=None, n=None)
    ("rocke_mma_catalog_t", "enumerate"): (
        "py_adapt::mma_enumerate(&{obj}, {args})",
        "std::vector<const rocke_mma_op_t*>",
        ["family", "a_dtype", "b_dtype", "c_dtype", "m", "n"],
        {"family": '"mma"', "m": "-1", "n": "-1"},
    ),
}


def kw_method_adapter(recv_t: str, name: str):
    return KW_METHODS.get((recv_t.replace("const ", "").rstrip("*").strip(), name))


def function_adapter(qual: str):
    return FUNCTIONS.get(qual)


def method_adapter(recv_t: str, name: str):
    return METHODS.get((recv_t, name))


# C4: an emitting generator becomes a round count plus one function per round.
# "yields" lists the yielded tuple: the round function's return ("ret") or an
# out parameter named in the round template.
GENERATORS = {
    "_attention_bwd_addr:deal_chunks": {
        "params": ["b", "tid", "total", "threads"],
        "count": "rocke_attn_bwd_deal_chunks_count({b}, {total}, {threads})",
        "round": "rocke_attn_bwd_deal_chunks_round({b}, {tid}, {total}, {threads}, {i}, {tail})",
        "yields": [("rocke_value_t*", "ret"), ("bool", "tail")],
    },
}


def generator_split(qual: str):
    return GENERATORS.get(qual)
