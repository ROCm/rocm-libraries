#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/elementary_emit.py -- Python reference emitter for the
# elementary-function programs (rocke.helpers.elementary). Config idx selects
# one kernel of _configs(): every function through elementary.emit (f64 / f32,
# anchored / not), then every family program with run-time predicates built as
# hipTensor's math kernels build them (all members, and each member alone),
# anchored with the family's pooled constants, plus the unanchored form. The C
# emitter elementary_emit.c builds the same kernels with rocke/elementary.h.
from rocke.core.ir import F32, F64, I32, IRBuilder, PtrType
from rocke.helpers import elementary as el
from _emit_common import run_emit


def _configs():
    out = []
    for name in el.FUNCTIONS:
        for dt in ("f64", "f32"):
            for anchored in (True, False):
                out.append(("emit", name, dt, anchored, ()))
    for fam, names in el.FAMILIES.items():
        variants = [tuple(names)] + ([(n,) for n in names] if len(names) > 1 else [])
        for dt in ("f64", "f32"):
            for v in variants:
                out.append(("family", fam, dt, True, v))
            out.append(("family", fam, dt, False, tuple(names)))
    return out


def _spec(idx: int):
    cfgs = _configs()
    if not 0 <= idx < len(cfgs):
        raise SystemExit(f"unknown config index {idx}")
    # a (spec, arch) pair: the config itself is a tuple
    return cfgs[idx], "gfx950"


def _build(cfg, arch=None):
    kind, name, dt, anchored, variant = cfg
    ty, nb = (F64, 8) if dt == "f64" else (F32, 4)
    tag = "_".join(variant)
    b = IRBuilder(
        f"elementary_{kind}_{name}_{dt}_{'a' if anchored else 'u'}"
        + (f"_{tag}" if tag else "")
    )
    y = b.param("y", PtrType(ty, "global"), align=nb)
    x = b.param("x", ty)
    if kind == "emit":
        r = el.emit(b, name, x, anchored)
    else:
        code = b.param("code", I32)
        preds = {}
        for k, n in enumerate(variant):
            preds[n] = b.cmp_eq(code, b.const_i32(k + 1))
        m = el.IRMath(b, el.opaque_zero(b) if anchored else None)
        if anchored:
            m.materialize(el.family_constants(name, preds))
        xx = x if dt == "f64" else b.cvt_f32_to_f64(x)
        r = el.family(m, name, xx, preds)
        if dt == "f32":
            r = b.cvt_f64_to_f32(r)
    b.global_store(y, b.const_i32(0), r, align=nb)
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: elementary_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
