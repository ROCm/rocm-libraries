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
        "grouped_gemm_abquant_utils._eightwaves_warp_tile_k",
        lambda arch, variant, flat: grouped_gemm_abquant_utils._eightwaves_warp_tile_k(
            arch, variant),
        _VARIANTS,
        (False,),
    ),
    (
        "grouped_gemm_abquant_utils._preshuffleb_warp_tile_k",
        lambda arch, variant, flat: grouped_gemm_abquant_utils._preshuffleb_warp_tile_k(
            arch, variant),
        _VARIANTS,
        (True,),
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


def test_no_gfx942_all_zeros_config():
    """No shipped default config may hand gfx942 the 128 that silently all-zeros."""
    factories = []
    for mod in (
        gemm_aquant_utils,
        gemm_bquant_utils,
        gemm_abquant_utils,
        gemm_rowcolquant_utils,
        gemm_tensor_quant_utils,
        grouped_gemm_abquant_utils,
    ):
        for attr in dir(mod):
            if attr.startswith("default_") and attr.endswith("_config"):
                factories.append((mod.__name__, attr, getattr(mod, attr)))
    assert factories, "no default_*_config factories discovered"

    for mod_name, attr, factory in factories:
        try:
            cfg = factory(gfx_arch="gfx942")
        except TypeError:
            continue
        assert cfg.warp_tile_k in (32, 64), (
            f"{mod_name}.{attr}(gfx_arch='gfx942').warp_tile_k == "
            f"{cfg.warp_tile_k}; 128 compiles on gfx942 and outputs all zeros"
        )


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
