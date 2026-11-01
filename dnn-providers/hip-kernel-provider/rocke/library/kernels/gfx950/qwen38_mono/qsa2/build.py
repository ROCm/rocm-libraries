# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build one QSA stage function: the ``stage_fn`` ABI (``tag, task, step,
layer, <41 LAYER_ARGS>, l0..l3, lds_base``) around a Q3-Q7 body, lowered by
:func:`lower.lower_qsa_function`.

The symbol is ATOM's (``rocke_q38_<stage>_<variant>``), so ATOM's
``ATOM_Q38_EXTERN_STAGES`` links the function in place of the FlyDSL stage.
"""

from __future__ import annotations

from types import SimpleNamespace

from rocke.core.ir import I32, I64, IRBuilder

from ..device_function import DeviceFunctionKnobs
from ..stage_fn import ABI_PARAMS, SYMBOL_PREFIX, variant_name
from . import layout as QL
from .ctx import QCtx
from .knobs import QsaKnobs
from .lower import lower_qsa_function

STAGES = ("q3", "q4", "q5", "q6", "q7")
NARROW = ("r1", "r2", "r3", "r4", "r2s", "r3s", "r4s")


def _body(stage: str):
    import importlib

    mod = importlib.import_module(f"{__package__}.{'q34' if stage in ('q3', 'q4') else 'q567'}")
    return getattr(mod, stage)


def supported(stage: str, rows: int, seq: bool, group: int, knobs: QsaKnobs) -> bool:
    if stage not in STAGES or group or not 1 <= rows <= 4 or (seq and rows < 2):
        return False
    if knobs.fuse != "none":
        return stage in ("q5", "q6", "q7")
    return True


def build_kernel(stage: str, lay: QL.QsaLayout, knobs: QsaKnobs):
    if not supported(stage, lay.rows, lay.seq, 0, knobs):
        raise ValueError(f"no QSA function {stage} for rows={lay.rows} seq={lay.seq} knobs={knobs.diff()}")
    sym = f"{SYMBOL_PREFIX}{stage}_{variant_name(lay.rows, lay.seq)}"
    b = IRBuilder(sym)
    b.kernel.attrs["max_workgroup_size"] = lay.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer", "l0", "l1", "l2", "l3"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(
        **vals,
        tid=tid,
        lane=b.land(tid, b.const_i32(63)),
        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))),
        ls=[vals["l0"], vals["l1"], vals["l2"], vals["l3"]],
    )
    x = QCtx(b, a, lay, knobs, stage)
    _body(stage)(x)
    b.ret()
    return b.kernel, sym


def lower_stage(stage: str, rows: int, seq: bool, knobs: QsaKnobs = QsaKnobs(), *,  # noqa: B008
                lay: QL.QsaLayout | None = None, atom_root=None, arch="gfx950", flavor=None):  # fmt: skip
    lay = lay or QL.from_atom(rows, seq, atom_root)
    kern, sym = build_kernel(stage, lay, knobs)
    fnk = DeviceFunctionKnobs(lds="arg", inline=knobs.inline)
    return lower_qsa_function(kern, arch=arch, symbol=sym, llvm_flavor=flavor, knobs=fnk,
                              fma=knobs.fma, fmax=knobs.fmax)  # fmt: skip
