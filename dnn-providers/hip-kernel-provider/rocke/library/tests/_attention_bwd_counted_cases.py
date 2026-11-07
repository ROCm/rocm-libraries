# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Case generation shared by the counted-sync unit and explicit gates.

The exhaustive matrices are functions rather than module constants so ordinary
pytest discovery can import the unit contracts without validating and
materializing hundreds of gate rows.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping

from kernels.common._attention_bwd_counted import COUNTED_ISSUES
from kernels.common.attention_bwd import AttnBwdSpec, validate_attn_bwd_spec

from ._attention_bwd_test_utils import distinct_configs
from .sdpa.bwd_kernel_cases import config_knobs

__all__ = [
    "COUNTED_COMPILER_FLAVORS",
    "COUNTED_VARIANTS",
    "DTYPES",
    "HEADS",
    "ISA_VARIANTS",
    "MFMA_ARCHS",
    "RDNA_ARCHS",
    "counted_rows",
    "is_legal_counted_row",
    "isa_rows",
    "row_digest",
    "require_explicit_gate_entrypoint",
    "row_id",
    "selected_isa_arch",
    "spec_of",
]

COUNTED_COMPILER_FLAVORS = ("llvm22", "llvm23", "llvm24")
MFMA_ARCHS = ("gfx942", "gfx950")
RDNA_ARCHS = ("gfx1151", "gfx1201")
HEADS = (32, 64, 128)
DTYPES = ("fp16", "bf16")

COUNTED_VARIANTS = {
    "band": dict(mask_class="band"),
    "none": dict(mask_class="none"),
    "band_edge": dict(mask_class="band", edge_tiles=True),
    "band_pack": dict(mask_class="band", head_pack=True),
    "band_atomic_pack": dict(mask_class="band", dkv_mode="atomic", head_pack=True),
    "thd_band_atomic": dict(mask_class="band", seq_mode="thd", dkv_mode="atomic"),
    "sv1_band": dict(mask_class="band", stage_vec=1),
    "band_hout": dict(mask_class="band", step_order="head_outer"),
    "band_pack_hout": dict(mask_class="band", head_pack=True, step_order="head_outer"),
    "band_qrot_edge": dict(mask_class="band", q_rotate=True, edge_tiles=True),
    "band_qrot_hout": dict(mask_class="band", q_rotate=True, step_order="head_outer"),
    "band_ptlds": dict(mask_class="band", pt_route="lds"),
    "band_g4g1g3": dict(mask_class="band", stage_order="g4g1g3"),
    "band_dqfrag": dict(mask_class="band", dq_issue="fragment_regs"),
    "band_accl": dict(mask_class="band", acc_in_lds=True, agpr_alloc=(0, 0)),
    "thd_band_slab": dict(mask_class="band", seq_mode="thd", dkv_slab_hoist=True),
    "band_tokmajor": dict(mask_class="band", ws_layout="token_major"),
    "band_mirror": dict(mask_class="band", lb_mirror=True),
    "band_mirror_edge": dict(mask_class="band", lb_mirror=True, edge_tiles=True),
}

ISA_VARIANTS = (
    "band",
    "band_edge",
    "band_pack",
    "band_atomic_pack",
    "thd_band_atomic",
    "band_hout",
    "band_ptlds",
    "band_dqfrag",
    "band_mirror",
)

CountedRow = tuple[str, str, int, str, str, str]


def require_explicit_gate_entrypoint(environ: Mapping[str, str] | None = None) -> None:
    """Reject direct gate imports that bypass runner and CTest policy."""
    environment = os.environ if environ is None else environ
    enabled = environment.get("ROCKE_EXPLICIT_GATE_RUN", "").strip() == "1"
    preflight = environment.get("ROCKE_EXPLICIT_GATE_PREFLIGHT", "").strip() == "1"
    from tests.gates import preflight_loaded

    if not enabled or not preflight or not preflight_loaded():
        raise RuntimeError(
            "explicit gates must run through library/tests/run_all.py or their CTest "
            "with gate conftest enabled"
        )


def selected_isa_arch(environ: Mapping[str, str] | None = None) -> str | None:
    """Normalized optional architecture selector for a partial ISA gate."""
    environment = os.environ if environ is None else environ
    return environment.get("ROCKE_BWD_ISA_ARCH", "").strip() or None


def spec_of(
    arch: str, config: str, head: int, dtype: str, variant: str, issue: str
) -> AttnBwdSpec:
    """Construct the counted spec named by one matrix row."""
    knobs = dict(config_knobs(config, arch), **COUNTED_VARIANTS[variant])
    return AttnBwdSpec(
        head_size=head,
        dtype=dtype,
        loop_sync="counted",
        counted_issue=issue,
        **knobs,
    )


def is_legal_counted_row(row: CountedRow) -> bool:
    """Whether the validator admits one counted-sync matrix row."""
    try:
        validate_attn_bwd_spec(spec_of(*row), row[0])
    except ValueError:
        return False
    return True


def counted_rows() -> tuple[CountedRow, ...]:
    """Every legal counted-sync structural-gate row, in stable order."""
    rows = []
    for arch in MFMA_ARCHS:
        for config in distinct_configs(arch):
            for head in HEADS:
                for dtype in DTYPES:
                    for variant in COUNTED_VARIANTS:
                        for issue in COUNTED_ISSUES:
                            row = (arch, config, head, dtype, variant, issue)
                            if is_legal_counted_row(row):
                                rows.append(row)
    return tuple(rows)


def isa_rows() -> tuple[CountedRow, ...]:
    """The counted rows whose final ISA and resources are gated."""
    return tuple(
        row
        for row in counted_rows()
        if row[4] in ISA_VARIANTS and (row[3] == "fp16" or row[4] == "band")
    )


def row_id(row: CountedRow) -> str:
    """Stable pytest id for a counted-sync matrix row."""
    return "-".join(map(str, row))


def row_digest(rows: Iterable[CountedRow]) -> str:
    """SHA-256 of an ordered row manifest, including every axis value."""
    manifest = "\n".join(row_id(row) for row in rows).encode()
    return hashlib.sha256(manifest).hexdigest()
