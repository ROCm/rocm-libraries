# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Known inputs, known counts, for every per-file metric."""
from conftest import src

import slo_metrics as M

HM = {"attention", "transforms"}
K = "library/kernels/common/k.py"


def run(text, rel=K):
    return M.analyze_source(src(text), rel, HM)


def test_sloc_skips_blank_comment_and_docstring_lines():
    r = run(
        '''
        """module doc
        two lines"""
        # comment

        def f():
            """doc"""
            x = 1  # trailing comment still counts
            return x
    '''
    )
    assert r["sloc"] == 3
    assert r["loc"] == 8


def test_longest_function_counts_def_to_last_line_and_reports_name():
    r = run(
        """
        def small():
            return 1

        def big():
            a = 1
            # comment counts here
            return a
    """
    )
    assert (r["longest_fn"], r["longest_fn_name"], r["longest_fn_line"]) == (
        4,
        "big",
        4,
    )


def test_raw_ops_found_by_type_not_only_by_name():
    r = run(
        """
        from rocke.core.ir import IRBuilder

        def build(spec):
            ir = IRBuilder("k")
            ir.add(1, 2)
            alias = ir
            alias.mul(1, 2)
            return ir

        def helper(bld: IRBuilder, x):
            return bld.sub(x, 1)

        def opt(bld2: "IRBuilder | None"):
            return bld2.mod(1, 2)

        class Body:
            def __init__(self, b):
                self.b = b
            def go(self):
                return self.b.div(1, 2)

        def plain(b):
            return b.add(const_i32(1), 2)
    """
    )
    assert (
        r["raw_ops"] == 7
    )  # ir.add, alias.mul, bld.sub, bld2.mod, self.b.div, b.add, const_i32
    assert set(r["builder_names"]) == {"alias", "bld", "bld2", "ir", "self.b"}


def test_callable_annotation_is_not_a_builder():
    r = run(
        """
        from typing import Callable, Optional
        from rocke.core.ir import IRBuilder

        def f(cb: Optional[Callable[[IRBuilder], int]]):
            return cb.whatever()
    """
    )
    assert r["raw_ops"] == 0
    assert r["builder_names"] == []


def test_helper_calls_include_methods_on_helper_objects():
    r = run(
        """
        from rocke.helpers.transforms import TensorDescriptor
        from rocke.helpers import attention

        def f(b):
            d = TensorDescriptor.naive(1, 2)    # helper: method on a helper class
            d.offset(b, 1)                      # helper: method on a helper object
            TensorDescriptor(1).offset(b, 2)    # helper x2: the call and its method
            attention.mask(b, 1)                # helper: helper module function
            b.add(1, 2)                         # raw
    """
    )
    assert r["helper_calls"] == 5
    assert r["raw_ops"] == 1
    assert r["hcr"] == round(5 / 6, 4)


def test_relative_helper_import_from_an_instance():
    r = run(
        """
        from ...helpers.attention import mask   # instances/common -> rocke.helpers

        def f(b):
            return mask(b, 1)
    """,
        rel="platform/python/rocke/instances/common/x.py",
    )
    assert r["helper_calls"] == 1


def test_bool_fields_counted_on_frozen_dataclasses_only():
    r = run(
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ASpec:
            a: bool = False
            b: bool = True
            c: int = 1
            d: "bool" = False

        @dataclass
        class NotFrozen:
            x: bool
            y: bool
            z: bool
            w: bool
    """
    )
    assert (r["spec_bools_max"], r["spec_bools_class"]) == (3, "ASpec")


def test_patterns():
    r = run(
        """
        def f(b, params, n, k):
            tid = b.thread_id_x()
            wave = b.div(tid, 64)              # lane_decode
            lane = b.mod(tid, 64)              # lane_decode
            q = b.div(lane, b.const_i32(16))   # lane_decode, const
            other = b.div(n, 2)                # not lane decode
            off = b.add(b.mul(q, 8), lane)     # add_mul
            m = b.land(b.cmp_lt(off, n), b.cmp_ge(off, 0))   # mask, compare x2
            v = b.select(m, off, 0)            # select
            g = b.div(b.add(n, b.sub(k, 1)), k)  # ceil_div
            h = (n + k - 1) // k               # ceil_div (python)
            i = -(-n // k)                     # ceil_div (python)
            j = (n - 1) // 2                   # not ceil_div
            x = b.global_load_f16(v)           # global_load
            b.smem_store_v(x)                  # lds
            b.sync()                           # sync
            b.exp2(x)                          # exp2
            b.trunc_f32_to_f16(x)              # epilogue_cast
            b.vec_extract(x, 0)                # vec
            p = params["p_X"]                  # params_dict
            return const_i32(3)                # const (bare)
    """
    )
    p = r["patterns"]
    assert p["lane_decode"] == 3
    assert p["const"] == 2
    assert p["add_mul"] == 1
    assert p["mask"] == 1
    assert p["compare"] == 2
    assert p["select"] == 1
    assert p["ceil_div"] == 3
    assert (
        p["global_load"],
        p["lds"],
        p["sync"],
        p["exp2"],
        p["epilogue_cast"],
        p["vec"],
        p["params_dict"],
    ) == (1, 1, 1, 1, 1, 1, 1)


def test_scope_rules():
    assert M.in_scope("library/kernels/common/x.py")
    assert M.in_scope("platform/python/rocke/instances/gfx950/y.py")
    assert not M.in_scope("library/kernels/tests/x.py")
    assert not M.in_scope("library/kernels/common/test_x.py")
    assert not M.in_scope("library/builders/x.py")
    assert not M.in_scope("platform/python/rocke/helpers/x.py")
