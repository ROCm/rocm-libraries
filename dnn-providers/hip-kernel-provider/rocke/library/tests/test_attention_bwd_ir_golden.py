# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR hashes of the attention backward kernels (CPU: pure text
lowering, no device, no comgr).

The fixture ``tests/golden/attention_bwd_ir_sha256.json`` pins the emitted
LLVM IR of four kernel kinds on gfx942, gfx950, gfx1151 and gfx1201, at
every LLVM flavor (llvm20, llvm22, llvm23):

* ``main``: the layout-general main kernel (``build_attn_bwd_main``);
* ``dq``: the split dQ kernel (``build_attn_bwd_dq``);
* ``aux``: the prep, convert, slice-reduce and paged-workspace kernels
  (``attention_bwd_aux``);
* ``unified``: the unified (paged KV) main kernel
  (``build_attn_bwd_unified_main``).

The dense family has its own fixture (``test_attention_dense_bwd_ir.py``).

Case set. Each kind has a few problem bases (head size, dtype, mask class,
sequence mode, stage width, dK / dV mode, page mode, ...). Every legal base is
a case. Then every registered non-default value of every knob of the kind
(``dispatch/attention_bwd/tuning_common.py``: the compile-time axes of the
main kernel and the split dQ kernel; the unified kernel's knobs are the
general knobs it carries, ``UNIFIED_KNOBS``; the aux fields are listed here)
is switched on, on the first base where it is legal and built, or, when no
base takes it alone, together with one enabling value of another knob (the
first legal pair in registry order). A value no base takes even with one
enabler is not a case; the fixture lists it under ``uncovered`` with the arch
and kind, and :func:`test_uncovered_values_are_exactly_the_recorded_ones`
keeps that list exact. The cases (kind, arch, spec fields) are stored in the
fixture next to their hashes, so a case is rebuilt from the fixture alone.

Every test fails, and none skips, when the fixture, a flavor, an arch, a kind
or a case is missing. Re-bless after an intended IR change with
``python tests/test_attention_bwd_ir_golden.py --write`` (the case search
runs again, so the case set follows the knob registry and the validators).
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

_TESTS = Path(__file__).resolve().parent
_LIB_ROOT = str(_TESTS.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)

GOLDEN = _TESTS / "golden" / "attention_bwd_ir_sha256.json"
SCHEMA = "attention_bwd.ir_golden_sha256/v1"
FLAVORS = ("llvm20", "llvm22", "llvm23")
ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
KINDS = ("main", "dq", "aux", "unified")
# The split dQ kernel is built on the MFMA targets only.
KIND_ARCHS = {
    "main": ARCHS,
    "dq": ("gfx942", "gfx950"),
    "aux": ARCHS,
    "unified": ARCHS,
}
KIND_ARCH = tuple((k, a) for k in KINDS for a in KIND_ARCHS[k])
# Row-major LDS buffers named by an `lds_swizzle` layout.
SWIZZLE_BUFFERS = (
    "dkv_out", "do", "ds", "k_res", "k_stage", "pt", "q", "v_res", "v_stage",
)  # fmt: skip

# --------------------------------------------------------------------------- bases
# Problem bases per kind, in search order (the first legal base takes a knob).
MAIN_BASES: Mapping[str, Mapping[str, Any]] = {
    "d64_fp16_band": dict(head_size=64, dtype="fp16", mask_class="band"),
    "d128_bf16_band": dict(head_size=128, dtype="bf16", mask_class="band"),
    "d128_bf16_none": dict(head_size=128, dtype="bf16", mask_class="none"),
    "d32_fp16_none": dict(head_size=32, dtype="fp16", mask_class="none"),
    "d64_bf16_thd_band_atomic": dict(
        head_size=64, dtype="bf16", mask_class="band", seq_mode="thd",
        dkv_mode="atomic",
    ),  # fmt: skip
    "d128_fp16_thd_none": dict(
        head_size=128, dtype="fp16", mask_class="none", seq_mode="thd"
    ),
    "d64_fp16_band_atomic": dict(
        head_size=64, dtype="fp16", mask_class="band", dkv_mode="atomic"
    ),
    "d64_fp16_band_sv1": dict(
        head_size=64, dtype="fp16", mask_class="band", stage_vec=1
    ),
    "d128_bf16_band_split": dict(
        head_size=128, dtype="bf16", mask_class="band", dq_mode="split"
    ),
}
# Tile geometries per arch (spec fields only), on top of a problem base: the
# one-wave tile, and on CDNA the tiles of the shipped tuned tables (the
# geometry values most other knobs are measured on), on RDNA the two-wave tile.
_W1 = dict(
    waves=1, block_m=16, block_n=16, block_k4=16, warp_grid_g4=(1, 1),
    atom_g02="16x16x16", atom_g13="16x16x16", atom_g4="16x16x16",
    kv_residency="lds", kt_source="lds", transpose_source="lds_plain",
    pt_route="relabel", sched="none",
)  # fmt: skip
_RDNA_W2 = dict(waves=2, block_m=16, block_n=32, block_k4=32, warp_grid_g4=(1, 2))
_GFX942_W2 = dict(
    atom_g02="16x16x16", atom_g13="16x16x16", atom_g4="16x16x16", block_k4=32,
    block_m=16, block_n=32, global_path="vgpr", kt_source="reg",
    kv_residency="reg", pt_route="relabel", ring_depth=2, stats_fma=True,
    stats_vec=4, transpose_source="lds_plain", warp_grid_g02=(1, 2),
    warp_grid_g13=(2, 1), warp_grid_g4=(1, 2), waves=2, waves_per_eu=1,
)  # fmt: skip
_GFX942_W4_ASM = dict(
    agpr_alloc=(192, 192), agpr_route="asm_g13", atom_g02="16x16x16",
    atom_g13="16x16x16", atom_g4="16x16x16", block_k4=32, block_m=16,
    block_n=128, global_path="vgpr", kt_source="reg", kv_residency="reg",
    pt_route="relabel", ring_depth=2, stats_fma=True, stats_vec=4,
    transpose_source="lds_plain", warp_grid_g02=(1, 4), warp_grid_g13=(4, 1),
    warp_grid_g4=(1, 4), waves=4, waves_per_eu=1,
)  # fmt: skip
_GFX950_W8_DMA = dict(
    atom_g02="16x16x32", atom_g13="16x16x16", atom_g4="16x16x32", block_k4=32,
    block_m=32, block_n=128, global_path="dma", kt_source="reg",
    kv_residency="reg", pt_route="relabel", ring_depth=2, stats_fma=True,
    stats_vec=4, transpose_source="tr_read", warp_grid_g02=(1, 8),
    warp_grid_g13=(8, 1), warp_grid_g4=(2, 4), waves=8, waves_per_eu=1,
)  # fmt: skip
_GFX950_W4_ASM = dict(
    agpr_alloc=(128, 128), agpr_route="asm_g13", atom_g02="16x16x32",
    atom_g13="16x16x16", atom_g4="16x16x32", block_k4=32, block_m=32,
    block_n=128, global_path="dma", kt_source="reg", kv_residency="reg",
    pt_route="relabel", ring_depth=2, stats_fma=True, stats_vec=4,
    transpose_source="tr_read", warp_grid_g02=(1, 4), warp_grid_g13=(4, 1),
    warp_grid_g4=(1, 4), waves=4, waves_per_eu=1,
)  # fmt: skip
_RDNA_W4 = dict(waves=4, block_m=32, block_n=64, block_k4=32, warp_grid_g4=(2, 2))
_D64 = dict(head_size=64, dtype="fp16", mask_class="band")
_D128 = dict(head_size=128, dtype="bf16", mask_class="band")
MAIN_GEOMETRY_BASES: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "gfx942": {
        "d64_fp16_band_w1": dict(_D64, **_W1),
        "d128_bf16_band_t2w": dict(_D128, **_GFX942_W2),
        "d128_bf16_band_t4w_asm": dict(_D128, **_GFX942_W4_ASM),
    },
    "gfx950": {
        "d64_fp16_band_w1": dict(_D64, **_W1),
        "d128_bf16_band_t8w_dma": dict(_D128, **_GFX950_W8_DMA),
        "d128_bf16_band_t4w_asm": dict(_D128, **_GFX950_W4_ASM),
    },
    "gfx1151": {
        "d64_fp16_band_w2": dict(_D64, **_RDNA_W2),
        "d64_fp16_band_w4": dict(_D64, **_RDNA_W4),
    },
    "gfx1201": {
        "d64_fp16_band_w2": dict(_D64, **_RDNA_W2),
        "d64_fp16_band_w4": dict(_D64, **_RDNA_W4),
    },
}
# The unified shell builds neither the prefetch ring nor the DMA path.
_NOT_UNIFIED = ("global_path", "ring_depth")
DQ_BASES: Mapping[str, Mapping[str, Any]] = {
    "d64_fp16_band": dict(head_size=64, dtype="fp16", mask_class="band"),
    "d128_bf16_none": dict(head_size=128, dtype="bf16", mask_class="none"),
    "d32_fp16_thd_band": dict(
        head_size=32, dtype="fp16", mask_class="band", seq_mode="thd"
    ),
    "d128_bf16_band_tm": dict(
        head_size=128, dtype="bf16", mask_class="band", ws_layout="token_major"
    ),
    "d64_fp16_band_sv1": dict(
        head_size=64, dtype="fp16", mask_class="band", stage_vec=1
    ),
}
UNIFIED_BASES: Mapping[str, Mapping[str, Any]] = {
    "d64_bf16_atomic_band": dict(head_size=64),
    "d128_fp16_atomic_none": dict(head_size=128, dtype="fp16", mask_class="none"),
    "d32_bf16_direct_band": dict(head_size=32, page_mode="direct"),
    "d64_fp16_atomic_band_sp16": dict(head_size=64, dtype="fp16", sub_page=16),
    "d64_bf16_atomic_band_sp32": dict(head_size=64, sub_page=32),
    "d128_bf16_direct_none_sp64": dict(
        head_size=128, page_mode="direct", mask_class="none", sub_page=64
    ),
    "d64_bf16_atomic_band_sv1": dict(head_size=64, stage_vec=1),
}
# Aux kernels: one base per (stage, problem); each other field is varied on it.
AUX_STAGES = ("prep", "convert", "reduce", "page_zero", "page_convert")
AUX_BASES: Mapping[str, Mapping[str, Any]] = {
    "d32_fp16_batched": dict(head_size=32, dtype="fp16"),
    "d64_bf16_thd": dict(head_size=64, dtype="bf16", seq_mode="thd"),
    "d128_fp16_batched": dict(head_size=128, dtype="fp16"),
    "d128_bf16_thd_paged": dict(
        head_size=128, dtype="bf16", seq_mode="thd", kv_target="paged"
    ),
}
AUX_AXES = (
    ("stage_vec", (8, 1)),
    ("ws_layout", ("head_major", "token_major")),
    ("kv_target", ("contiguous", "paged")),
    ("rows_per_cta", (None, 1, 2, 4)),
    ("convert_vec", (8, 4, 16)),
    ("worklist", (False, True)),
    ("narrow_vec", (1, 2, 4)),
)


# --------------------------------------------------------------------------- kinds


def _main_axes(arch: str):
    from dispatch.attention_bwd import tuning_common as tc

    out = []
    for a in tc.backward_tuning_axes(arch, "full"):
        if a.kind != tc.KIND_COMPILE:
            continue
        vals = [
            v for v in a.values(arch, "full") if v != a.default and v not in a.not_built
        ]
        if a.name == "lds_swizzle":
            # a registry value is a per-buffer layout: every row-major
            # buffer of the kernel takes it (absent buffers are ignored)
            vals = [",".join(f"{b}={v}" for b in SWIZZLE_BUFFERS) for v in vals]
        out.append((a.name, tuple(vals)))
    return out


def _dq_axes(arch: str):
    from dispatch.attention_bwd import tuning_common as tc

    out = []
    for a in tc.backward_tuning_axes(arch, "full"):
        if a.kind != tc.KIND_DQ or a.name == "dq_xcd_chunk":
            continue
        vals = [
            v for v in a.values(arch, "full") if v != a.default and v not in a.not_built
        ]
        out.append((a.name[len("dq_") :], tuple(vals)))
    # xcd_remap is plan-set from dq_xcd_chunk: the compile-time switch
    out.append(("xcd_remap", (True,)))
    return out


def _unified_axes(arch: str):
    from kernels.common.attention_bwd_unified import _SHELL_KNOBS, UNIFIED_KNOBS

    # the general-shell knobs and the DMA path are not built for the unified
    # shell (validate_attn_bwd_unified_spec): only their default is legal
    names = set(UNIFIED_KNOBS) - set(_SHELL_KNOBS) - {"global_path"}
    return [(n, v) for n, v in _main_axes(arch) if n in names]


def _aux_axes(arch: str):
    return [(n, tuple(v for v in vals[1:])) for n, vals in AUX_AXES]


def _build_main(arch, f):
    from kernels.common.attention_bwd import AttnBwdSpec, build_attn_bwd_main

    return build_attn_bwd_main(AttnBwdSpec(**f), arch=arch)


def _legal_main(arch, f) -> bool:
    from dispatch.attention_bwd.tuning_common import backward_spec_legality
    from kernels.common.attention_bwd import AttnBwdSpec

    try:
        spec = AttnBwdSpec(**f)
    except (TypeError, ValueError):
        return False
    return backward_spec_legality(spec, arch).ok


def _build_dq(arch, f):
    from kernels.common.attention_bwd_dq import AttnBwdDqSpec, build_attn_bwd_dq

    return build_attn_bwd_dq(AttnBwdDqSpec(**f), arch=arch)


def _legal_dq(arch, f) -> bool:
    from kernels.common.attention_bwd_dq import (
        AttnBwdDqSpec,
        validate_attn_bwd_dq_spec,
    )

    try:
        validate_attn_bwd_dq_spec(AttnBwdDqSpec(**f), arch)
    except (TypeError, ValueError):
        return False
    return True


def _build_unified(arch, f):
    from kernels.common.attention_bwd_unified import (
        AttnBwdUnifiedSpec,
        build_attn_bwd_unified_main,
    )

    return build_attn_bwd_unified_main(AttnBwdUnifiedSpec(**f), arch=arch)


def _legal_unified(arch, f) -> bool:
    from kernels.common.attention_bwd import _not_built
    from kernels.common.attention_bwd_unified import (
        AttnBwdUnifiedSpec,
        validate_attn_bwd_unified_spec,
    )

    try:
        s = validate_attn_bwd_unified_spec(AttnBwdUnifiedSpec(**f), arch)
    except (TypeError, ValueError):
        return False
    return not _not_built(s.general_spec())


_AUX_BUILDERS = {
    "prep": "build_attn_bwd_prep",
    "convert": "build_attn_bwd_convert",
    "reduce": "build_attn_bwd_reduce",
    "page_zero": "build_attn_bwd_page_zero",
    "page_convert": "build_attn_bwd_page_convert",
}


def _build_aux(arch, f):
    from kernels.common import attention_bwd_aux as aux

    f = dict(f)
    stage = f.pop("stage")
    return getattr(aux, _AUX_BUILDERS[stage])(aux.AttnBwdAuxSpec(**f), arch=arch)


def _legal_aux(arch, f) -> bool:
    from kernels.common import attention_bwd_aux as aux

    g = dict(f)
    g.pop("stage")
    try:
        aux.AttnBwdAuxSpec(**g)
    except (TypeError, ValueError):
        return False
    return True


_KIND = {
    "main": (_build_main, _legal_main, _main_axes),
    "dq": (_build_dq, _legal_dq, _dq_axes),
    "unified": (_build_unified, _legal_unified, _unified_axes),
    "aux": (_build_aux, _legal_aux, _aux_axes),
}


def _bases(kind: str, arch: str) -> dict:
    if kind == "main":
        return {**MAIN_BASES, **MAIN_GEOMETRY_BASES[arch]}
    if kind == "dq":
        return dict(DQ_BASES)
    if kind == "unified":
        from kernels.common.attention_bwd_unified import UNIFIED_KNOBS

        geo = {
            bid.replace("_band", "_atomic_band"): {
                k: v
                for k, v in g.items()
                if k in ("head_size", "dtype", "mask_class")
                or (k in UNIFIED_KNOBS and k not in _NOT_UNIFIED)
            }
            for bid, g in MAIN_GEOMETRY_BASES[arch].items()
        }
        return {**UNIFIED_BASES, **geo}
    return {
        f"{stage}_{bid}": dict(base, stage=stage)
        for stage in AUX_STAGES
        for bid, base in AUX_BASES.items()
    }


# --------------------------------------------------------------------------- search
def _tok(v) -> str:
    return repr(v).replace(" ", "").replace("'", "")


def _search(kind: str, arch: str) -> tuple[dict, list]:
    """``({case id: fields}, [uncovered "name=value"])`` of ``kind`` on
    ``arch`` (see the module docstring)."""
    build, legal, axes_of = _KIND[kind]

    def ok(f) -> bool:
        if not legal(arch, f):
            return False
        try:
            build(arch, f)
        except (TypeError, ValueError):
            return False
        return True

    bases = {bid: b for bid, b in _bases(kind, arch).items() if ok(b)}
    cases = dict(bases)
    covered = {f"{k}={_tok(v)}" for f in cases.values() for k, v in f.items()}
    axes = axes_of(arch)
    uncovered = []
    for name, values in axes:
        for v in values:
            tok = f"{name}={_tok(v)}"
            if tok in covered:
                continue  # a base or an earlier pair already switches it on
            hit = None
            for bid, base in bases.items():
                f = dict(base, **{name: v})
                if ok(f):
                    hit = (f"{bid}+{tok}", f)
                    break
            if hit is None:
                hit = _pair(bases, axes, name, v, ok)
            if hit is None:
                uncovered.append(tok)
                continue
            cases[hit[0]] = hit[1]
            covered |= {f"{k}={_tok(x)}" for k, x in hit[1].items()}
    return cases, uncovered


def _pair(bases, axes, name, v, ok: Callable[[dict], bool]):
    for bid, base in bases.items():
        for n2, vals2 in axes:
            if n2 == name:
                continue
            for w in vals2:
                f = dict(base, **{name: v, n2: w})
                if ok(f):
                    return (f"{bid}+{name}={_tok(v)}+{n2}={_tok(w)}", f)
    return None


# --------------------------------------------------------------------------- hashing
def _lower(kernel, arch: str, flavor: str) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


def _sha(kernel, arch: str, flavor: str) -> str:
    return hashlib.sha256(_lower(kernel, arch, flavor).encode("utf-8")).hexdigest()


def case_fields(kind: str, arch: str, cid: str) -> dict:
    """Spec fields of case ``cid`` (``<base>[+<knob>=<value>]...``): the base
    of the kind and arch with each named registry value switched on."""
    base, *knobs = cid.split("+")
    bases = _bases(kind, arch)
    if base not in bases:
        raise KeyError(f"{kind}/{arch}: no base {base!r}")
    values = {n: {_tok(v): v for v in vals} for n, vals in _KIND[kind][2](arch)}
    out = dict(bases[base])
    for item in knobs:
        name, tok = item.split("=", 1)
        if tok not in values.get(name, {}):
            raise KeyError(f"{kind}/{arch}: {name}={tok} is not a registered value")
        out[name] = values[name][tok]
    return out


def _build_doc() -> dict:
    """The fixture: the uncovered values and, per case, the sha256 of its
    LLVM IR at every flavor."""
    cases, uncovered = {}, {}
    for kind, arch in KIND_ARCH:
        found, unc = _search(kind, arch)
        uncovered[f"{kind}/{arch}"] = unc
        for cid, f in found.items():
            if case_fields(kind, arch, cid) != f:
                raise AssertionError(
                    f"{kind}/{arch}/{cid}: id does not name its fields"
                )
            kernel = _KIND[kind][0](arch, f)
            cases[f"{kind}/{arch}/{cid}"] = {
                fl: _sha(kernel, arch, fl) for fl in FLAVORS
            }
    return {"schema": SCHEMA, "uncovered": uncovered, "cases": cases}


# --------------------------------------------------------------------------- tests
@cache
def _golden() -> dict:
    if not GOLDEN.exists():
        pytest.fail(f"backward IR golden fixture missing ({GOLDEN.name}); --write")
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        pytest.fail(f"backward IR golden schema {doc.get('schema')!r} != {SCHEMA!r}")
    return doc


@cache
def _kernel(key: str):
    kind, arch, cid = key.split("/", 2)
    return _KIND[kind][0](arch, case_fields(kind, arch, cid))


def _keys(kind: str, arch: str) -> list:
    pre = f"{kind}/{arch}/"
    return sorted(k for k in _golden()["cases"] if k.startswith(pre))


def test_fixture_has_every_flavor_kind_and_arch():
    doc = _golden()
    for kind, arch in KIND_ARCH:
        assert _keys(kind, arch), f"no {kind} case on {arch}"
        assert f"{kind}/{arch}" in doc["uncovered"], (kind, arch)
    for key, shas in doc["cases"].items():
        assert set(shas) == set(FLAVORS), key
    assert {k.rsplit("/", 1)[0] for k in doc["uncovered"]} <= set(KINDS)


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("kind,arch", KIND_ARCH)
def test_ir_matches_golden(kind, arch, flavor):
    keys = _keys(kind, arch)
    if not keys:
        pytest.fail(f"no {kind} golden case on {arch}")
    drift = []
    for key in keys:
        want = _golden()["cases"][key].get(flavor)
        if want is None:
            drift.append(f"{key}: no {flavor} hash in the golden")
            continue
        got = _sha(_kernel(key), arch, flavor)
        if got != want:
            drift.append(f"{key}: {want[:16]} -> {got[:16]}")
    assert not drift, f"backward IR drift ({flavor}):\n  " + "\n  ".join(drift)


@pytest.mark.parametrize("kind,arch", KIND_ARCH)
def test_case_set_follows_the_registry_and_validators(kind, arch):
    """The search over the current knob registry and validators finds exactly
    the fixture's cases and uncovered values (a new knob value, a value that
    became legal or illegal, or a changed base fails here)."""
    found, unc = _search(kind, arch)
    pre = f"{kind}/{arch}/"
    assert sorted(found) == [k[len(pre) :] for k in _keys(kind, arch)]
    for cid, f in found.items():
        assert case_fields(kind, arch, cid) == f, cid
    assert unc == _golden()["uncovered"][f"{kind}/{arch}"]


@pytest.mark.parametrize("kind,arch", KIND_ARCH)
def test_uncovered_values_are_exactly_the_recorded_ones(kind, arch):
    """Every registered non-default value of the kind is switched on in some
    case on ``arch``, except the recorded ``uncovered`` ones."""
    want = {f"{n}={_tok(v)}" for n, vals in _KIND[kind][2](arch) for v in vals}
    covered = set()
    for key in _keys(kind, arch):
        fields = case_fields(kind, arch, key.split("/", 2)[2])
        covered |= {f"{k}={_tok(v)}" for k, v in fields.items()}
    missing = want - covered
    assert sorted(missing) == sorted(_golden()["uncovered"][f"{kind}/{arch}"])


def _write() -> None:
    doc = _build_doc()
    lines = ["{", f' "schema": {json.dumps(SCHEMA)},', ' "uncovered": {']
    unc = sorted(doc["uncovered"].items())
    for i, (k, v) in enumerate(unc):
        comma = "," if i + 1 < len(unc) else ""
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)}{comma}")
    lines += [" },", ' "cases": {']
    cases = sorted(doc["cases"].items())
    for i, (k, v) in enumerate(cases):
        comma = "," if i + 1 < len(cases) else ""
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}{comma}")
    lines += [" }", "}"]
    text = "\n".join(lines) + "\n"
    if json.loads(text) != doc:
        raise AssertionError("fixture text does not round-trip")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {GOLDEN}")
    for kind, arch in KIND_ARCH:
        n = sum(1 for k in doc["cases"] if k.startswith(f"{kind}/{arch}/"))
        print(
            f"{kind:8s} {arch:8s} cases={n:4d} uncovered={len(doc['uncovered'][f'{kind}/{arch}'])}"
        )


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write()
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
