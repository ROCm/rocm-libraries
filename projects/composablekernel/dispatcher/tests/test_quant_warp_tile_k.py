#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Table-unanimity gate for the arch-derived quant ``warp_tile_k`` rule.

``warp_tile_k`` is the silent-all-zeros predicate of the block-scale quant
bridges: 128 on gfx942 compiles cleanly and then produces all-zeros output,
because there is no valid 16x16x128 fp8/bf8 warp-gemm on gfx942.  A drifting
second copy therefore cannot be caught by a build, only by a numeric check on
the right arch -- which is why the rule must exist exactly once.

This module enumerates the full ``arch x variant x pipeline`` table and asserts
every per-op spelling agrees with the canonical
``codegen_common.quant_warp_tile_k``, and that no per-op file re-implements the
rule as literal constants.

No GPU / hipcc required.
"""

import re
import sys
from pathlib import Path

import pytest

_DISP = Path(__file__).resolve().parent.parent
for _p in (_DISP / "python", _DISP / "codegen"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from codegen_common import (  # noqa: E402
    fp8_warp_tile_k_for_arch,
    quant_warp_tile_k,
    variant_is_8bit_float,
)

import gemm_abquant_utils  # noqa: E402
import gemm_aquant_utils  # noqa: E402
import gemm_bquant_utils  # noqa: E402
import gemm_rowcolquant_utils  # noqa: E402
import gemm_tensor_quant_utils  # noqa: E402
import grouped_gemm_abquant_utils  # noqa: E402
import grouped_gemm_aquant_utils  # noqa: E402
import grouped_gemm_bquant_utils  # noqa: E402
import grouped_gemm_rowcolquant_utils  # noqa: E402
import grouped_gemm_tensorquant_utils  # noqa: E402

# Every module that ships default_*_config factories.  The gate below covers all
# ten; an op added without being listed here fails ``test_all_quant_ops_gated``.
_ALL_QUANT_UTILS = (
    gemm_tensor_quant_utils,
    gemm_rowcolquant_utils,
    gemm_aquant_utils,
    gemm_abquant_utils,
    gemm_bquant_utils,
    grouped_gemm_tensorquant_utils,
    grouped_gemm_rowcolquant_utils,
    grouped_gemm_aquant_utils,
    grouped_gemm_abquant_utils,
    grouped_gemm_bquant_utils,
)


# The full arch axis, including the target-feature suffixes hipcc reports.
_ARCHES = (
    "gfx942",
    "gfx942:sramecc+:xnack-",
    "gfx950",
    "gfx950:sramecc+:xnack-",
    "gfx90a",
    "gfx1200",
)

# 8-bit-float variants (fp8_t / bf8_t PrecType) and the one that is not.
_EIGHT_BIT_FLOAT = ("fp8", "bf8", "fp8i4", "bf8i4")
_NOT_EIGHT_BIT_FLOAT = ("fp4",)
_VARIANTS = _EIGHT_BIT_FLOAT + _NOT_EIGHT_BIT_FLOAT

_FLAT_MM = (False, True)


# Every per-op spelling of the rule, adapted to (arch, variant, is_flat_mm).
# ``None`` means the spelling does not carry that axis and so is only compared
# on the sub-table it claims to cover.
_IMPLEMENTATIONS = (
    (
        "codegen_common.fp8_warp_tile_k_for_arch",
        lambda arch, variant, flat: fp8_warp_tile_k_for_arch(
            arch, preshuffle_quant=flat),
        _EIGHT_BIT_FLOAT,
        _FLAT_MM,
    ),
    (
        "gemm_aquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: gemm_aquant_utils._warp_tile_k_for(
            arch, preshuffle_aquant=flat),
        _EIGHT_BIT_FLOAT,
        _FLAT_MM,
    ),
    (
        "gemm_bquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: gemm_bquant_utils._warp_tile_k_for(
            arch, is_flatmm=flat),
        _EIGHT_BIT_FLOAT,
        _FLAT_MM,
    ),
    (
        "gemm_abquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: gemm_abquant_utils._warp_tile_k_for(
            variant, arch, is_flat_mm=flat),
        _VARIANTS,
        _FLAT_MM,
    ),
    (
        "gemm_rowcolquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: gemm_rowcolquant_utils._warp_tile_k_for(
            variant, arch),
        _VARIANTS,
        (False,),
    ),
    (
        "gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch",
        lambda arch, variant, flat: gemm_tensor_quant_utils.fp8_warp_tile_k_for_arch(
            arch),
        _EIGHT_BIT_FLOAT,
        (False,),
    ),
    (
        "grouped_gemm_abquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: grouped_gemm_abquant_utils._warp_tile_k_for(
            arch, variant, is_flatmm=flat),
        _VARIANTS,
        _FLAT_MM,
    ),
    (
        "grouped_gemm_aquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: grouped_gemm_aquant_utils._warp_tile_k_for(
            arch, variant, is_flatmm=flat),
        _VARIANTS,
        _FLAT_MM,
    ),
    (
        "grouped_gemm_bquant_utils._warp_tile_k_for",
        lambda arch, variant, flat: grouped_gemm_bquant_utils._warp_tile_k_for(
            arch, variant, is_flatmm=flat),
        _VARIANTS,
        _FLAT_MM,
    ),
)


@pytest.mark.parametrize(
    "name,fn,variants,flats", _IMPLEMENTATIONS, ids=[i[0] for i in _IMPLEMENTATIONS]
)
def test_warp_tile_k_table_unanimity(name, fn, variants, flats):
    """Every per-op spelling must agree with the canonical rule, everywhere."""
    disagreements = []
    for arch in _ARCHES:
        for variant in variants:
            for flat in flats:
                expected = quant_warp_tile_k(
                    arch,
                    is_8bit_float=variant_is_8bit_float(variant),
                    is_flat_mm=flat,
                )
                got = fn(arch, variant, flat)
                if got != expected:
                    disagreements.append(
                        f"{name}({arch!r}, {variant!r}, is_flat_mm={flat}) "
                        f"-> {got}, canonical -> {expected}"
                    )
    assert not disagreements, "\n".join(disagreements)


def test_canonical_rule_values():
    """Pin the canonical table itself, so a delegation cascade cannot drift as one."""
    # gfx950: 8-bit float is 128 regardless of IsFlatMM; anything else is 32.
    assert quant_warp_tile_k("gfx950") == 128
    assert quant_warp_tile_k("gfx950", is_flat_mm=True) == 128
    assert quant_warp_tile_k("gfx950", is_8bit_float=False) == 32
    assert quant_warp_tile_k("gfx950", is_8bit_float=False, is_flat_mm=True) == 32
    # gfx942/other: IsFlatMM decides, 32 vs 64. 128 here is the all-zeros trap.
    for arch in ("gfx942", "gfx90a", "gfx1200"):
        assert quant_warp_tile_k(arch) == 32
        assert quant_warp_tile_k(arch, is_flat_mm=True) == 64
        assert quant_warp_tile_k(arch, is_8bit_float=False) == 32
        assert quant_warp_tile_k(arch, is_8bit_float=False, is_flat_mm=True) == 64


# ---------------------------------------------------------------------------
# Per-factory gate over every shipped default config of all ten ops
# ---------------------------------------------------------------------------

def _default_factories(mod):
    """(name, callable) for every default_*_config in ``mod``, sorted."""
    return sorted(
        (a, getattr(mod, a))
        for a in dir(mod)
        if a.startswith("default_") and a.endswith("_config")
        and callable(getattr(mod, a))
    )


def _all_default_factories():
    for mod in _ALL_QUANT_UTILS:
        for name, fn in _default_factories(mod):
            yield mod.__name__, name, fn


# Factories whose Old-TE GemmConfig derives K_Warp_Tile with IsFlatMM=true.
#
# This cannot be inferred from the pipeline string: aquant's preshuffle-quant
# decode (GemmConfigPreshuffleQuantDecode, gemm_utils.hpp:146-162) is flat-mm
# and runs on the compv3 pipeline, while bquant's preshuffle-quant prefill
# (GemmConfigPreshuffleBQuantPrefill -> GemmConfigQuantPrefill, :256-269,
# :322-333) is not.  The two ops genuinely differ, so the axis is data.
_FLAT_MM_FACTORIES = frozenset({
    ("gemm_aquant_utils", "default_fp8_preshufflequant_config"),
    ("gemm_aquant_utils", "default_bf8_preshufflequant_config"),
    ("gemm_aquant_utils", "default_fp8i4_preshufflequant_config"),
    ("gemm_aquant_utils", "default_bf8i4_preshufflequant_config"),
    ("gemm_abquant_utils", "default_fp8_preshuffleb_config"),
    ("gemm_abquant_utils", "default_bf8_preshuffleb_config"),
    ("gemm_abquant_utils", "default_fp4_preshuffleb_config"),
    ("gemm_abquant_utils", "default_fp8_preshuffleb_preshufflequant_config"),
    ("gemm_bquant_utils", "default_fp8_preshuffleb_config"),
    ("gemm_bquant_utils", "default_bf8_preshuffleb_config"),
    ("gemm_bquant_utils", "default_fp8i4_preshuffleb_config"),
    ("gemm_bquant_utils", "default_bf8i4_preshuffleb_config"),
    ("gemm_bquant_utils", "default_fp8_preshuffleb_bquant_config"),
    ("gemm_bquant_utils", "default_bf8_preshuffleb_bquant_config"),
    ("gemm_bquant_utils", "default_fp8i4_preshuffleb_bquant_config"),
    ("gemm_bquant_utils", "default_bf8i4_preshuffleb_bquant_config"),
    ("grouped_gemm_aquant_utils", "default_fp8_preshuffleaq_config"),
    ("grouped_gemm_aquant_utils", "default_bf8_preshuffleaq_config"),
    ("grouped_gemm_abquant_utils", "default_fp8_preshuffleb_config"),
    ("grouped_gemm_abquant_utils", "default_bf8_preshuffleb_config"),
    ("grouped_gemm_bquant_utils", "default_fp8_preshuffleb_config"),
    ("grouped_gemm_bquant_utils", "default_bf8_preshuffleb_config"),
    ("grouped_gemm_bquant_utils", "default_fp8i4_preshuffleb_config"),
    ("grouped_gemm_bquant_utils", "default_bf8i4_preshuffleb_config"),
    ("grouped_gemm_bquant_utils", "default_fp8_preshuffleb_bquant_config"),
    ("grouped_gemm_bquant_utils", "default_bf8_preshuffleb_bquant_config"),
    ("grouped_gemm_bquant_utils", "default_fp8i4_preshuffleb_bquant_config"),
    ("grouped_gemm_bquant_utils", "default_bf8i4_preshuffleb_bquant_config"),
})

# Factories that legitimately do NOT follow get_k_warp_tile, with the reason and
# the arch -> value they must ship instead.  Every entry is a claim about
# Old-TE source, and the gate fails if the shipped value stops matching it.
_LITERAL_WARP_TILE_K = {
    # GemmConfigMixedPrecision (gemm_utils.hpp:284-297) hardcodes
    # `K_Warp_Tile = 64` -- it is the A=16bit/B=8bit config with KPack=16 and
    # does not call get_k_warp_tile at all.
    ("gemm_bquant_utils", "default_mx_bf16bf8_config"): (
        {"gfx942": 64, "gfx950": 64},
        "GemmConfigMixedPrecision hardcodes K_Warp_Tile=64",
    ),
    ("grouped_gemm_bquant_utils", "default_mx_bf16bf8_config"): (
        {"gfx942": 64, "gfx950": 64},
        "GemmConfigMixedPrecision hardcodes K_Warp_Tile=64",
    ),
    # KNOWN DEVIATION, deliberately not changed in this round.
    #
    # These two ops use M_Warp_Tile=32 (ROWCOL_TENSOR_QUANT_DEFAULT_TILE), where
    # get_k_warp_tile<fp8_t, 32>() is 64 on gfx950 and 16 on gfx942.  They ship
    # 16 unconditionally, i.e. the gfx942 value on both arches.
    #
    # It is NOT a correctness bug in either direction: 16 with M_Warp_Tile=32 is
    # mfma_f32_32x32x16_fp8, a valid instruction on gfx942 and gfx950, and all
    # four configs verify on device (global max_rel 3.2e-4 .. 4.2e-4).  It is
    # purely a smaller-than-Old-TE K tile on gfx950, i.e. a performance
    # deviation -- and this bridge publishes no defensible performance number
    # yet (the timing instrument has no cache-flush / rotating-buffer path), so
    # there is no gate that could distinguish a good change here from a bad one.
    # Recorded as an asserted fact rather than left silent.
    ("grouped_gemm_rowcolquant_utils", "default_fp8_config"): (
        {"gfx942": 16, "gfx950": 16},
        "M_Warp_Tile=32 op shipping the gfx942 value on both arches; "
        "correct-but-smaller K tile on gfx950, perf-only, needs a perf gate",
    ),
    ("grouped_gemm_rowcolquant_utils", "default_bf8_config"): (
        {"gfx942": 16, "gfx950": 16},
        "see default_fp8_config",
    ),
    ("grouped_gemm_tensorquant_utils", "default_fp8_config"): (
        {"gfx942": 16, "gfx950": 16},
        "see grouped_gemm_rowcolquant_utils.default_fp8_config",
    ),
    ("grouped_gemm_tensorquant_utils", "default_bf8_config"): (
        {"gfx942": 16, "gfx950": 16},
        "see grouped_gemm_rowcolquant_utils.default_fp8_config",
    ),
}


def _expected_warp_tile_k(mod_name, factory_name, cfg, arch):
    key = (mod_name, factory_name)
    if key in _LITERAL_WARP_TILE_K:
        table, _reason = _LITERAL_WARP_TILE_K[key]
        return table[arch]
    variant = getattr(cfg, "variant_key", None) or getattr(cfg, "dtype", "")
    return quant_warp_tile_k(
        arch,
        is_8bit_float=variant_is_8bit_float(variant),
        is_flat_mm=key in _FLAT_MM_FACTORIES,
        m_warp_tile=cfg.warp_tile_m,
    )


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_every_default_config_warp_tile_k_matches_the_rule(arch):
    """Every shipped default config of all ten ops must carry the derived value.

    This is the gate that D3 needed and did not have.  Before it, three
    different values (128 / 32 / 16) shipped for the same ``fp8i4`` variant on
    the same arch, one of which returned a C that was 98% exactly zero, and
    nothing in the suite objected.
    """
    wrong = []
    for mod_name, name, fn in _all_default_factories():
        cfg = fn(gfx_arch=arch)
        expected = _expected_warp_tile_k(mod_name, name, cfg, arch)
        if cfg.warp_tile_k != expected:
            wrong.append(
                f"{mod_name}.{name}(gfx_arch={arch!r}).warp_tile_k == "
                f"{cfg.warp_tile_k}, rule says {expected} "
                f"(warp_tile_m={cfg.warp_tile_m})"
            )
    assert not wrong, "\n".join(wrong)


def test_no_gfx942_all_zeros_config():
    """No shipped default config may hand gfx942 the 128 that silently all-zeros.

    Covers all ten ops.  The earlier revision of this test iterated six modules
    and happened to exclude every violator, so it passed while ten shipped
    factories handed gfx942 a 128.
    """
    factories = list(_all_default_factories())
    # Coverage is "every module contributes", not a magic total: the total moves
    # legitimately (grouped_gemm_bquant dropped 12 preshuffle defaults that could
    # not compile).  A module contributing nothing means the gate lost an op.
    per_module = {}
    for mod_name, attr, _fn in factories:
        per_module.setdefault(mod_name, []).append(attr)
    empty = [m.__name__ for m in _ALL_QUANT_UTILS if m.__name__ not in per_module]
    assert not empty, f"these ops contributed no default_*_config factory: {empty}"
    for mod_name, attr, factory in factories:
        cfg = factory(gfx_arch="gfx942")
        assert cfg.warp_tile_k != 128, (
            f"{mod_name}.{attr}(gfx_arch='gfx942').warp_tile_k == 128; "
            "128 compiles on gfx942 and outputs all zeros"
        )


def test_all_quant_ops_gated():
    """_ALL_QUANT_UTILS must list every quant utils module that ships defaults."""
    listed = {m.__name__ for m in _ALL_QUANT_UTILS}
    on_disk = set()
    for path in (_DISP / "python").glob("*quant*_utils.py"):
        text = path.read_text()
        if "def default_" in text and "_config(" in text:
            on_disk.add(path.stem)
    missing = on_disk - listed
    assert not missing, (
        "quant utils modules shipping default_*_config factories but not gated "
        f"by this file: {sorted(missing)}"
    )


def test_literal_exemptions_are_minimal():
    """Every documented deviation must still be a deviation."""
    stale = []
    for (mod_name, name), (table, reason) in _LITERAL_WARP_TILE_K.items():
        mod = next((m for m in _ALL_QUANT_UTILS if m.__name__ == mod_name), None)
        assert mod is not None, f"{mod_name} is not in _ALL_QUANT_UTILS"
        fn = getattr(mod, name, None)
        assert fn is not None, f"{mod_name}.{name} no longer exists"
        deviates = False
        for arch, value in table.items():
            cfg = fn(gfx_arch=arch)
            derived = quant_warp_tile_k(
                arch,
                is_8bit_float=variant_is_8bit_float(
                    getattr(cfg, "variant_key", None) or getattr(cfg, "dtype", "")),
                is_flat_mm=(mod_name, name) in _FLAT_MM_FACTORIES,
                m_warp_tile=cfg.warp_tile_m,
            )
            deviates = deviates or derived != value
        if not deviates:
            stale.append(
                f"{mod_name}.{name}: the exemption values now equal the derived "
                f"values on every gated arch; drop the exemption ({reason})")
    assert not stale, "\n".join(stale)


# The literal rule -- the pair of magic numbers -- may appear in exactly one file.
_RULE_LITERAL = re.compile(r"return\s+128\s+if\b|\breturn\s+128\s*$", re.MULTILINE)

_RULE_OWNER = _DISP / "codegen" / "codegen_common.py"


def test_warp_tile_k_rule_has_one_implementation():
    """No per-op file may re-derive the arch -> warp_tile_k constants itself."""
    offenders = []
    for path in sorted(
        list((_DISP / "python").glob("*quant*_utils.py"))
        + list((_DISP / "codegen").glob("unified_*quant*_codegen.py"))
    ):
        if path == _RULE_OWNER:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if _RULE_LITERAL.search(code):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "the arch -> warp_tile_k rule must exist exactly once, in "
        f"{_RULE_OWNER.name}; found re-implementations:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# The codegen enumeration sweeps
# ---------------------------------------------------------------------------
#
# Each generator's built-in ``_default_config()`` is the sweep that populates the
# enumerated header set.  It carries its own literal warp_tile_k, which is a
# second place the rule can drift -- and does.  These are not among the 74
# shipped ``default_*_config`` factories, and changing them moves the emitted
# kernel *names* for the whole enumerated registry, so they are fixed under
# their own gate rather than here.  They are asserted so the drift is a named
# fact and cannot spread further.

_CODEGEN_SWEEP_DEVIATIONS = {
    # generator stem -> (shipped gfx950 warp_tile_k, canonical, note)
    "unified_grouped_gemm_aquant_codegen": (
        16, 128,
        "16 is the gfx1250 WMMA value; on gfx950 the rule is 128",
    ),
    "unified_grouped_gemm_abquant_codegen": (
        16, 128,
        "16 is the gfx1250 WMMA value; on gfx950 the rule is 128",
    ),
    "unified_gemm_abquant_codegen": (
        32, 128,
        "32 is the gfx942 value; the op's shipped default_*_config factories "
        "are already arch-derived and hand gfx950 128, so the sweep disagrees "
        "with the bridge it belongs to",
    ),
    # The two grouped scalar-quant ops use M_Warp_Tile=32, where the rule gives
    # 64 on gfx950 and 16 on gfx942.  Same declared deviation as their shipped
    # default configs; see _LITERAL_WARP_TILE_K.
    "unified_grouped_gemm_rowcolquant_codegen": (
        16, 64,
        "M_Warp_Tile=32 op shipping the gfx942 value; perf-only, needs a perf gate",
    ),
    "unified_grouped_gemm_tensorquant_codegen": (
        16, 64,
        "M_Warp_Tile=32 op shipping the gfx942 value; perf-only, needs a perf gate",
    ),
}


def _codegen_sweep_tile(stem):
    import importlib
    mod = importlib.import_module(stem)
    fn = getattr(mod, "_default_config", None)
    if fn is None:
        return None
    try:
        cfg = fn("gfx950")
    except TypeError:
        cfg = fn()
    tiles = cfg.get("tile_configs") or []
    if not tiles:
        return None
    return tiles[0]


def test_codegen_sweeps_follow_the_rule_or_are_declared():
    """A codegen sweep may deviate from the rule only if it is declared here."""
    undeclared, stale = [], []
    for path in sorted((_DISP / "codegen").glob("unified_*quant*_codegen.py")):
        stem = path.stem
        tile = _codegen_sweep_tile(stem)
        if tile is None:
            continue
        got = tile.get("warp_tile_k")
        # Every quant sweep is 8-bit float and non-flat-mm; M_Warp_Tile comes
        # from the sweep's own tile (the two grouped scalar-quant ops use 32).
        expected = quant_warp_tile_k(
            "gfx950", is_8bit_float=True, is_flat_mm=False,
            m_warp_tile=tile.get("warp_tile_m", 16))
        if stem in _CODEGEN_SWEEP_DEVIATIONS:
            declared, canonical, _note = _CODEGEN_SWEEP_DEVIATIONS[stem]
            if got == expected:
                stale.append(
                    f"{stem}: sweep warp_tile_k is now {got}, matching the rule; "
                    "drop the declared deviation")
            elif got != declared:
                undeclared.append(
                    f"{stem}: sweep warp_tile_k is {got}, declared deviation is "
                    f"{declared} (canonical {canonical})")
        elif got != expected:
            undeclared.append(
                f"{stem}: sweep warp_tile_k is {got}, rule says {expected}; "
                "fix it or declare it in _CODEGEN_SWEEP_DEVIATIONS with a reason")
    assert not undeclared + stale, "\n".join(undeclared + stale)
