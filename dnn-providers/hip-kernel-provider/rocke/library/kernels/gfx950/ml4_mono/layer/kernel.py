# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ML4 decoder layer as one persistent rocKE launch (gfx950).

``build_ml4_layer`` emits ``spec.stages`` in program order into one kernel
of ``spec.grid`` resident CTAs. Stages hand off through tagged scratch
mailboxes; a build with a subset of the stages reads the others' mailboxes as
a previous launch left them, which is how each stage is checked on its own
against the reference kernel before the chain is closed.
"""

from __future__ import annotations

from typing import Callable

from rocke.core.ir import IRBuilder, KernelDef

from .ctx import Ml4Ctx
from .spec import Ml4LayerSpec
from .stage_cache import emit_cache
from .stage_index import emit_index_score, emit_index_select
from .stage_mlp import emit_mlp_down, emit_mlp_ug
from .stage_moe import emit_down, emit_router, emit_ug
from .stage_moe_em import emit_down_em, emit_ug_em
from .stage_norm import emit_q_norm
from .stage_q_b import emit_index_q, emit_q_b
from .stage_qkv_a import emit_qkv_a
from .stage_split import emit_split
from .stage_uk import emit_uk
from .stage_uv import emit_o, emit_uv

EMITTERS: dict[str, Callable[[Ml4Ctx], None]] = {
    "qkv_a": emit_qkv_a,
    "q_norm": emit_q_norm,
    "cache": emit_cache,
    "q_b": emit_q_b,
    "index_q": emit_index_q,
    "uk": emit_uk,
    "index_score": emit_index_score,
    "split": emit_split,
    "uv": emit_uv,
    "o": emit_o,
    "router": emit_router,
    "ug": lambda cx: emit_ug_em(cx) if cx.spec.ug_em or cx.spec.moe_em else emit_ug(cx),
    "down": lambda cx: emit_down_em(cx) if cx.spec.moe_em else emit_down(cx),
    "mlp_ug": emit_mlp_ug,
    "mlp_down": emit_mlp_down,
}


def build_ml4_layer(spec: Ml4LayerSpec, *, arch: str = "gfx950") -> KernelDef:
    if arch != "gfx950":
        raise ValueError(f"the ML4 layer kernel targets gfx950, got {arch!r}")
    b = IRBuilder(spec.kernel_name)
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    cx = Ml4Ctx(b, spec)
    order = list(spec.stages)
    if spec.score_before_uk and "uk" in order and "index_score" in order:
        # index scoring does not read uk's output: its tasks on uk's CTAs need
        # not queue behind uk (whose q_b wait is the later one); every input
        # of both still comes from a stage earlier in the order
        order.remove("index_score")
        order.insert(order.index("uk"), "index_score")
    # cache_kv_late: the cache CTA's latent rows after its score tiles (split
    # reads them after the top-k); one place for both parts without scoring
    late_kv = spec.cache_kv_late and spec.with_indexer and "cache" in order and "index_score" in order
    for name in order:
        if name == "cache" and late_kv:
            emit_cache(cx, "index")
        elif name.startswith("index_sel"):
            emit_index_select(cx, name)
        else:
            EMITTERS[name](cx)
        if name == "index_score" and late_kv:
            emit_cache(cx, "kv")
    b.ret()
    return b.kernel
