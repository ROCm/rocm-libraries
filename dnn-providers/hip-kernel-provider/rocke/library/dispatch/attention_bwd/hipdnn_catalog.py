# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernel catalog of the native hipDNN engine ``hipkernel:AttentionGeneralBwd``.

Every kernel the engine's plans (:func:`.hipdnn.engine_verdict`) can launch on
an arch, as rows of one main kernel per (class, tuned-table key, head
packing) with the prep, convert and split-dQ kernels its plan launches:

* the class defaults: every head size, dtype, sequence mode, head mode
  (MHA, GQA, MQA, ``h_k != h_v``), mask (none, causal), access width
  (``stage_vec`` 8 and 1) and short and long query spans (head packing on
  and off), with no device fill hint;
* every key of the arch's general tuned table, over the representative
  requests of the key (:func:`.tuned_tables.general_key_requests`) that a
  hipDNN graph can express (no padding, no XCD hint).

Selection is deterministic: a request's class and tuned-table key give one
row. The descriptor emitter and the engine's generated tables read these rows.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kernels.common.attention_bwd import (
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
)
from kernels.common.attention_bwd_aux import (
    attn_bwd_aux_block_threads,
    attn_bwd_aux_rows_per_cta,
)
from kernels.common.attention_bwd_plan import (
    RUNTIME_KNOBS,
    AttnBwdPlan,
    AttnBwdRequest,
)

from .hipdnn import (
    ENGINE_ARCHS,
    engine_policy,
    engine_verdict,
    main_fields,
    plan_table_key,
    prep_fields,
    tuning_key_id,
    tuning_key_string,
)
from .routing import ARCH_TUNING
from .tuned_tables import general_key_requests

__all__ = [
    "CLASS_FIELDS",
    "CatalogRow",
    "PrepRow",
    "catalog_rows",
    "default_block_n",
    "default_requests",
    "pack_of",
    "prep_rows",
    "row_key",
    "tuned_requests",
]

CLASS_FIELDS = (
    "dtype",
    "head_size",
    "seq_mode",
    "stage_vec",
    "dkv_mode",
    "mask_class",
    "dq_mode",
    "head_pack",
    "lb_mirror",
    "lb_interleave",
)
_DEFAULT_HEADS = ((8, 8, 8), (8, 2, 2), (8, 4, 2), (8, 1, 1))
_DEFAULT_S_Q = (128, 1, 2)
_DEFAULT_S_KV = 300


@dataclass(frozen=True)
class CatalogRow:
    """One main kernel of the engine and the kernels its plan launches."""

    arch: str
    pack: str
    dtype: str
    head_size: int
    seq_mode: str
    stage_vec: int
    dkv_mode: str
    mask_class: str
    dq_mode: str
    head_pack: bool
    lb_mirror: bool
    lb_interleave: bool
    tuning_key: tuple | None
    tuning_key_str: str
    tuning_key_id: int
    main_spec: Any
    prep_spec: Any
    convert_spec: Any
    dq_spec: Any  # None unless dq_mode == "split"
    # launch -> kernel symbol: prep, main, dq (the dQ convert or the split
    # dQ kernel), convert_dk, convert_dv (the convert kernel)
    kernels: Mapping[str, str]
    main_block_m: int
    main_block_n: int
    main_waves: int
    main_threads: int
    main_worklist: bool
    main_static_scale: str
    prep_rows_per_cta: int
    prep_threads: int
    prep_worklist: bool
    prep_narrow_vec: int
    convert_rows_per_cta: int
    convert_threads: int
    convert_vec: int
    dq_block_m: int
    dq_block_n: int
    dq_threads: int
    # the tuned-table entry's runtime values (RUNTIME_KNOBS) and its whole entry
    runtime_knobs: Mapping[str, Any]
    tuned_knobs: Mapping[str, Any]
    representative: AttnBwdRequest

    @property
    def class_key(self) -> tuple:
        return tuple(getattr(self, f) for f in CLASS_FIELDS)

    @property
    def main_kmd(self) -> dict:
        """The main slot's KMD fields of this row (raw values)."""
        out = {f: getattr(self, f) for f in CLASS_FIELDS}
        out.update(
            tuning_key=self.tuning_key_str,
            main_block_n=self.main_block_n,
            main_worklist=self.main_worklist,
            main_static_scale=self.main_static_scale,
        )
        return out


@dataclass(frozen=True)
class PrepRow:
    """One prep kernel of the engine."""

    arch: str
    pack: str
    fields: Mapping[str, Any]  # the prep slot's KMD fields
    spec: Any
    kernel: str
    threads: int
    representative: AttnBwdRequest


def pack_of(arch: str, slot: str, dq_mode: str = "atomic") -> str:
    if slot == "prep":
        return f"prep_{arch}"
    return f"main_splitdq_{arch}" if dq_mode == "split" else f"main_{arch}"


def row_key(row: CatalogRow) -> tuple:
    """``(arch, class, tuning key id)``: the identity of a row (head packing is
    a class field)."""
    return (row.arch, row.class_key, row.tuning_key_id)


def default_block_n(arch: str, d: int, dtype: str, stage_vec: int, seq_mode: str):
    return attn_bwd_default_geometry(arch, d, dtype, stage_vec, seq_mode)["block_n"]


def default_requests(arch: str) -> list[AttnBwdRequest]:
    """The class-default requests (no device fill hint, so no tuned key)."""
    out = []
    for d in BWD_HEAD_SIZES:
        for dtype in BWD_DTYPES:
            for seq in ("batched", "thd"):
                for h_q, h_k, h_v in _DEFAULT_HEADS:
                    for causal in (False, True):
                        for align in (16, 8):
                            for s_q in _DEFAULT_S_Q:
                                kw = {
                                    "b": 2,
                                    "h_q": h_q,
                                    "h_k": h_k,
                                    "h_v": h_v,
                                    "s_q": s_q,
                                    "s_kv": _DEFAULT_S_KV,
                                    "d_qk": d,
                                    "d_v": d,
                                    "dtype": dtype,
                                    "tensor_alignment": align,
                                    "right_bound": 0 if causal else -1,
                                }
                                if seq == "thd":
                                    kw["layout"] = "thd"
                                out.append(AttnBwdRequest(**kw))
    return out


def tuned_requests(arch: str) -> list[tuple[tuple, AttnBwdRequest]]:
    """``(key, request)`` for every hipDNN-expressible representative request
    of every general tuned-table key of ``arch``."""
    mod = ARCH_TUNING.get(arch)
    out, seen = [], set()
    for key in [] if mod is None else sorted(mod.GENERAL_TUNED_TABLE, key=repr):
        for req in general_key_requests(arch, key):
            if req.padding:
                continue
            req = dataclasses.replace(req, num_xcds=None)
            if (key, req) not in seen:
                seen.add((key, req))
                out.append((key, req))
    return out


def _row(plan: AttnBwdPlan, req: AttnBwdRequest) -> CatalogRow:
    arch = plan.arch
    main, prep, conv = plan.specs["main"], plan.specs["prep"], plan.specs["convert"]
    dq = plan.specs.get("dq")
    steps = {s.stage: s for s in plan.launches}
    conv_kernel = conv.kernel_name("convert")
    kernels = {
        "prep": steps["prep"].kernel,
        "main": steps["main"].kernel,
        "dq": steps["dq"].kernel if dq is not None else conv_kernel,
        "convert_dk": conv_kernel,
        "convert_dv": conv_kernel,
    }
    table_key = plan_table_key(plan)
    entry = dict(engine_policy().tuned_table.get(table_key, {})) if table_key else {}
    mf, pf = main_fields(plan), prep_fields(plan)
    wave = attn_bwd_arch_facts(arch).wave_size
    return CatalogRow(
        arch=arch,
        pack=pack_of(arch, "main", main.dq_mode),
        **{f: mf[f] for f in CLASS_FIELDS},
        tuning_key=table_key,
        tuning_key_str=tuning_key_string(table_key),
        tuning_key_id=tuning_key_id(arch, table_key),
        main_spec=main,
        prep_spec=prep,
        convert_spec=conv,
        dq_spec=dq,
        kernels=kernels,
        main_block_m=main.block_m,
        main_block_n=main.block_n,
        main_waves=main.waves,
        main_threads=main.waves * wave,
        main_worklist=bool(main.worklist),
        main_static_scale=main.static_scale,
        prep_rows_per_cta=pf["rows_per_cta"],
        prep_threads=attn_bwd_aux_block_threads(prep, arch=arch),
        prep_worklist=bool(prep.worklist),
        prep_narrow_vec=prep.narrow_vec,
        convert_rows_per_cta=attn_bwd_aux_rows_per_cta(conv, arch=arch),
        convert_threads=attn_bwd_aux_block_threads(conv, arch=arch),
        convert_vec=conv.convert_vec,
        dq_block_m=dq.block_m if dq is not None else 0,
        dq_block_n=dq.block_n if dq is not None else 0,
        dq_threads=dq.waves * wave if dq is not None else 0,
        runtime_knobs={k: entry[k] for k in RUNTIME_KNOBS if k in entry},
        tuned_knobs=entry,
        representative=req,
    )


def _sort_key(row: CatalogRow) -> tuple:
    return (row.pack, *map(str, row.class_key), row.tuning_key_id)


@functools.cache
def _catalog(arch: str) -> tuple[tuple[CatalogRow, ...], tuple[PrepRow, ...]]:
    if arch not in ENGINE_ARCHS:
        raise ValueError(f"{arch} is not an engine arch {ENGINE_ARCHS}")
    rows: dict[tuple, CatalogRow] = {}
    preps: dict[str, PrepRow] = {}
    requests = [(None, r) for r in default_requests(arch)] + tuned_requests(arch)
    for _key, req in requests:
        code, detail, plan = engine_verdict(req, arch)
        if plan is None:
            raise AssertionError(f"catalog request declined {code}: {detail}: {req}")
        row = _row(plan, req)
        known = rows.get(row_key(row))
        if known is None:
            rows[row_key(row)] = row
        elif known.kernels != row.kernels or known.main_kmd != row.main_kmd:
            raise AssertionError(
                f"{arch} row {row_key(row)} has two kernel sets: "
                f"{known.kernels} vs {row.kernels}"
            )
        prep = PrepRow(
            arch=arch,
            pack=pack_of(arch, "prep"),
            fields=prep_fields(plan),
            spec=plan.specs["prep"],
            kernel=row.kernels["prep"],
            threads=row.prep_threads,
            representative=req,
        )
        seen = preps.get(prep.kernel)
        if seen is None:
            preps[prep.kernel] = prep
        elif dict(seen.fields) != dict(prep.fields):
            raise AssertionError(f"{arch} prep {prep.kernel} has two KMD records")
    ordered = tuple(sorted(rows.values(), key=_sort_key))
    prep_ordered = tuple(
        sorted(preps.values(), key=lambda p: tuple(map(str, p.fields.values())))
    )
    return ordered, prep_ordered


def catalog_rows(arch: str) -> list[CatalogRow]:
    """Every main-kernel row of the engine on ``arch``, in a fixed order."""
    return list(_catalog(arch)[0])


def prep_rows(arch: str) -> list[PrepRow]:
    """Every distinct prep kernel of the engine on ``arch``, in a fixed order."""
    return list(_catalog(arch)[1])
