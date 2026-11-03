# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Named cumulative MX FP4 TDM GEMM variants for Step 6 feature increments.

Each variant is a preset over :func:`build_mxfp4_tdm_gemm`. Entries are
cumulative: ``step6_N`` includes every earlier Step 6 flag, so benchmarking
the chain shows what each feature of the hipBLASLt YAML winner is worth.
Every variant is meant for the YAML tile (256x256x512, 2x2 waves); step 6.1
is the base variant at that tile and needs no flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .mxfp4_tdm_gemm import MxFp4TdmGemmSpec

BASE_VARIANT = "base_kernel_mxfp4_gemm"


def _chain(*steps: tuple[str, str, dict[str, Any]]):
    rows: list[tuple[str, dict[str, Any]]] = [(BASE_VARIANT, {})]
    labels = {BASE_VARIANT: "6.1"}
    flags: dict[str, Any] = {}
    for label, name, added in steps:
        flags = {**flags, **added}
        rows.append((name, dict(flags)))
        labels[name] = label
    return rows, labels


# Ordered cumulative table: (variant_name, flag overrides vs defaults).
# Only landed steps appear here. Blocked steps are listed separately.
MXFP4_GEMM_VARIANTS, VARIANT_STEPS = _chain(
    ("6.2", "step6_2_mxfp4_gemm", {"vector_epilogue": True}),
    ("6.3", "step6_3_mxfp4_gemm", {"local_read_prefetch": True}),
    ("6.4", "step6_4_mxfp4_gemm", {"wave_parity_tdm": True}),
    ("6.5", "step6_5_mxfp4_gemm", {"scale_tdm": True}),
    ("6.6", "step6_6_mxfp4_gemm", {"lds_pad": True}),
    ("6.7", "step6_7_mxfp4_gemm", {"lds_segment_interleave": True}),
    (
        "6.8",
        "step6_8_mxfp4_gemm",
        {"cluster_dim": (4, 4, 1), "tdm_multicast": True},
    ),
    ("6.9", "step6_9_mxfp4_gemm", {"expert_scheduling": True}),
    ("6.10", "step6_10_mxfp4_gemm", {"tdm_desc_hoist": True}),
    ("6.11", "step6_11_mxfp4_gemm", {"wmma_n_outer": True}),
    ("6.12", "step6_12_mxfp4_gemm", {"cluster_aid_remap": True}),
    (
        "6.13",
        "step6_13_mxfp4_gemm",
        {"epilogue_nan_clamp": True, "epilogue_pk8_cvt": True},
    ),
    # The winner's main loop. Each feature alone was slower (T5 to T12,
    # T29); together they are faster, and dropping any of them from the
    # set is slower (T31 to T40). gl2_prefetch_early is left out (T37).
    (
        "6.14",
        "step6_14_mxfp4_gemm",
        {
            "init_c_wmma": True,
            "tile_refill_mid": True,
            "tile_read_ahead": True,
            "source_order": True,
            "cluster_signal_wmmas": 6,
            "cluster_signal_wave0": True,
            "prefetch_gl2": True,
            "coarse_ds_wait": True,
            "loop_unroll2": True,
        },
    ),
)

# Trials build on 6.8 and stay off the chain. The YAML's
# VectorWidthB=4 (each lane owns 4 adjacent rocKE A rows) is not a trial: it
# is the M mapping ``scale_tdm`` already uses from 6.5 on, just as
# VectorWidthA=8 is the N mapping of ``vector_epilogue``.
# prefetch_gl2, init_c_wmma and tile_refill_mid were chain steps; each was
# slower than the step before it on the YAML shape, so they are trials now.
# trial_ck_lds_pad replaces 6.6 instead: it is 6.5 with CK Tile's comp_tdm
# padding (16 B after every 256 B LDS row on A and B, no B iterate mode,
# unpadded scales), so it compares against 6.6 directly. The on6_9 trials
# build on 6.9 instead.
_CHAIN_END = dict(MXFP4_GEMM_VARIANTS)["step6_8_mxfp4_gemm"]
_STEP6_5 = dict(MXFP4_GEMM_VARIANTS)["step6_5_mxfp4_gemm"]
_STEP6_9 = dict(MXFP4_GEMM_VARIANTS)["step6_9_mxfp4_gemm"]
_READ_AHEAD_SRCO = {
    **_STEP6_9,
    "init_c_wmma": True,
    "tile_refill_mid": True,
    "tile_read_ahead": True,
    "source_order": True,
}
MXFP4_GEMM_TRIAL_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("trial_prefetch_gl2_mxfp4_gemm", {**_CHAIN_END, "prefetch_gl2": True}),
    ("trial_init_c_wmma_mxfp4_gemm", {**_CHAIN_END, "init_c_wmma": True}),
    (
        "trial_tile_refill_mid_mxfp4_gemm",
        {**_CHAIN_END, "init_c_wmma": True, "tile_refill_mid": True},
    ),
    ("trial_ck_lds_pad_mxfp4_gemm", {**_STEP6_5, "lds_pad_ck": True}),
    (
        "trial_tile_refill_mid_on6_9_mxfp4_gemm",
        {**_STEP6_9, "init_c_wmma": True, "tile_refill_mid": True},
    ),
    (
        "trial_read_ahead_on6_9_mxfp4_gemm",
        {
            **_STEP6_9,
            "init_c_wmma": True,
            "tile_refill_mid": True,
            "tile_read_ahead": True,
        },
    ),
    ("trial_read_ahead_srco_on6_9_mxfp4_gemm", _READ_AHEAD_SRCO),
    ("trial_srco_on6_9_mxfp4_gemm", {**_STEP6_9, "source_order": True}),
    *(
        (
            f"trial_csig{wmmas}_on6_9_mxfp4_gemm",
            {**_READ_AHEAD_SRCO, "cluster_signal_wmmas": wmmas},
        )
        for wmmas in (6, 0)
    ),
    (
        "trial_pgl2_csig6_on6_9_mxfp4_gemm",
        {**_READ_AHEAD_SRCO, "cluster_signal_wmmas": 6, "prefetch_gl2": True},
    ),
    (
        "trial_pgl2e_csig6_on6_9_mxfp4_gemm",
        {
            **_READ_AHEAD_SRCO,
            "cluster_signal_wmmas": 6,
            "prefetch_gl2": True,
            "gl2_prefetch_early": True,
        },
    ),
    # T5 and T7 without the peeled first tile: the loop starts at tile 0.
    (
        "trial_refill_mid_nopeel_on6_9_mxfp4_gemm",
        {**_STEP6_9, "tile_refill_mid": True},
    ),
    (
        "trial_read_ahead_srco_nopeel_on6_9_mxfp4_gemm",
        {key: value for key, value in _READ_AHEAD_SRCO.items() if key != "init_c_wmma"},
    ),
]

# The winner's main-loop features, one at a time on T12 and together.
# tdm_desc_hoist and wmma_n_outer, the ones that also speed up 6.9, are
# chain steps 6.10 and 6.11.
_T12 = dict(MXFP4_GEMM_TRIAL_VARIANTS)["trial_pgl2e_csig6_on6_9_mxfp4_gemm"]
_LOOP_SAFE = {"tdm_desc_hoist": True, "wmma_n_outer": True, "coarse_ds_wait": True}
MXFP4_GEMM_TRIAL_VARIANTS += [
    ("trial_dhst_on_t12_mxfp4_gemm", {**_T12, "tdm_desc_hoist": True}),
    ("trial_nout_on_t12_mxfp4_gemm", {**_T12, "wmma_n_outer": True}),
    ("trial_cdsw_on_t12_mxfp4_gemm", {**_T12, "coarse_ds_wait": True}),
    ("trial_csw0_on_t12_mxfp4_gemm", {**_T12, "cluster_signal_wave0": True}),
    ("trial_loop_safe_on_t12_mxfp4_gemm", {**_T12, **_LOOP_SAFE}),
    (
        "trial_loop_all_on_t12_mxfp4_gemm",
        {**_T12, **_LOOP_SAFE, "cluster_signal_wave0": True},
    ),
]

# The winner's kernel setup on 6.11, one at a time and together.
# cluster_aid_remap, the one that sped 6.11 up in every run, is chain step
# 6.12; kernarg_preload is tried on 6.12 instead.
_STEP6_11 = dict(MXFP4_GEMM_VARIANTS)["step6_11_mxfp4_gemm"]
_STEP6_12 = dict(MXFP4_GEMM_VARIANTS)["step6_12_mxfp4_gemm"]
_SETUP = {"wmma_arb_stall_off": True, "kernarg_preload": True, "cluster_aid_remap": True}
MXFP4_GEMM_TRIAL_VARIANTS += [
    ("trial_noarb_on6_11_mxfp4_gemm", {**_STEP6_11, "wmma_arb_stall_off": True}),
    ("trial_pka_on6_12_mxfp4_gemm", {**_STEP6_12, "kernarg_preload": True}),
    ("trial_setup_on6_11_mxfp4_gemm", {**_STEP6_11, **_SETUP}),
    (
        "trial_caid_on6_10_mxfp4_gemm",
        {**dict(MXFP4_GEMM_VARIANTS)["step6_10_mxfp4_gemm"], "cluster_aid_remap": True},
    ),
]

# The winner's epilogue on 6.12: its FP8 store sequence, piece by piece, and
# its SW instruction prefetch. The two pieces together are chain step 6.13.
# Six vector stores stay under the 4 KiB a prefetch covers on both beta
# paths of 6.12.
_STEP6_13 = dict(MXFP4_GEMM_VARIANTS)["step6_13_mxfp4_gemm"]
_EPI_PREFETCH = {"epilogue_inst_prefetch": 6}
MXFP4_GEMM_TRIAL_VARIANTS += [
    ("trial_nclp_on6_12_mxfp4_gemm", {**_STEP6_12, "epilogue_nan_clamp": True}),
    ("trial_pk8_on6_12_mxfp4_gemm", {**_STEP6_12, "epilogue_pk8_cvt": True}),
    ("trial_ipf_on6_13_mxfp4_gemm", {**_STEP6_13, **_EPI_PREFETCH}),
    ("trial_ipf_on6_12_mxfp4_gemm", {**_STEP6_12, **_EPI_PREFETCH}),
    ("trial_unr2_on6_13_mxfp4_gemm", {**_STEP6_13, "loop_unroll2": True}),
]

# T20's main loop with constant LDS buffers, alone and on 6.13.
_T20 = dict(MXFP4_GEMM_TRIAL_VARIANTS)["trial_loop_all_on_t12_mxfp4_gemm"]
MXFP4_GEMM_TRIAL_VARIANTS += [
    ("trial_unr2_on_t20_mxfp4_gemm", {**_T20, "loop_unroll2": True}),
    ("trial_t20_unr2_on6_13_mxfp4_gemm", {**_STEP6_13, **_T20, "loop_unroll2": True}),
]

# T31 with one feature off at a time; flags that require it go with it.
_T31 = dict(MXFP4_GEMM_TRIAL_VARIANTS)["trial_t20_unr2_on6_13_mxfp4_gemm"]
_DROP_CSIG = {"cluster_signal_wmmas": None, "cluster_signal_wave0": False}
_ABLATIONS = (
    ("unr2", {"loop_unroll2": False}),
    ("initc", {"init_c_wmma": False}),
    ("srco", {"source_order": False, **_DROP_CSIG}),
    ("csw0", {"cluster_signal_wave0": False}),
    ("csig", _DROP_CSIG),
    ("pgl2e", {"gl2_prefetch_early": False}),
    ("pgl2", {"prefetch_gl2": False, "gl2_prefetch_early": False}),
    ("cdsw", {"coarse_ds_wait": False}),
    (
        "rahd",
        {
            "tile_read_ahead": False,
            "gl2_prefetch_early": False,
            "coarse_ds_wait": False,
            **_DROP_CSIG,
        },
    ),
)
MXFP4_GEMM_TRIAL_VARIANTS += [
    (f"trial_t31_no_{tag}_mxfp4_gemm", {**_T31, **off}) for tag, off in _ABLATIONS
]

# 6.14 with LLVM's machine schedulers on: the only change source_order makes.
_STEP6_14 = dict(MXFP4_GEMM_VARIANTS)["step6_14_mxfp4_gemm"]
MXFP4_GEMM_TRIAL_VARIANTS += [
    ("trial_misched_on6_14_mxfp4_gemm", {**_STEP6_14, "source_order": False}),
]

MXFP4_GEMM_BLOCKED_VARIANTS: list[tuple[str, str]] = []


def variant_names(*, trials: bool = False) -> list[str]:
    names = [name for name, _ in MXFP4_GEMM_VARIANTS]
    if trials:
        names += [name for name, _ in MXFP4_GEMM_TRIAL_VARIANTS]
    return names


def variant_step(name: str) -> str:
    """The plan step label (``"6.1"`` .. ``"6.14"``, or ``"T*"``) of ``name``."""
    if name in VARIANT_STEPS:
        return VARIANT_STEPS[name]
    for index, (trial, _) in enumerate(MXFP4_GEMM_TRIAL_VARIANTS, start=1):
        if trial == name:
            return f"T{index}"
    raise KeyError(f"unknown variant {name!r}")


def variant_flags(name: str) -> dict[str, Any]:
    for variant, flags in (*MXFP4_GEMM_VARIANTS, *MXFP4_GEMM_TRIAL_VARIANTS):
        if variant == name:
            return dict(flags)
    blocked = {n: reason for n, reason in MXFP4_GEMM_BLOCKED_VARIANTS}
    if name in blocked:
        raise ValueError(f"variant {name!r} is blocked: {blocked[name]}")
    known = ", ".join(variant_names(trials=True)) or "(none)"
    raise KeyError(f"unknown variant {name!r}; known: {known}")


def variant_spec(
    variant: str,
    M: int,
    N: int,
    K: int,
    **geometry: Any,
) -> MxFp4TdmGemmSpec:
    """Build a spec for ``variant`` with the given problem and tile geometry."""
    flags = variant_flags(variant)
    return MxFp4TdmGemmSpec(
        name=variant,
        M=M,
        N=N,
        K=K,
        **geometry,
        **flags,
    )


def assert_variants_cumulative(
    table: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
) -> None:
    """Require each entry's flags to be a superset of the previous entry."""
    rows = list(table if table is not None else MXFP4_GEMM_VARIANTS)
    if not rows:
        raise AssertionError("MXFP4_GEMM_VARIANTS is empty")
    if rows[0][0] != BASE_VARIANT:
        raise AssertionError(f"first variant must be {BASE_VARIANT!r}")
    if rows[0][1]:
        raise AssertionError("base variant must have no Step 6 flags")
    prev: dict[str, Any] = {}
    for name, flags in rows:
        for key, value in prev.items():
            if flags.get(key, False if isinstance(value, bool) else value) != value:
                raise AssertionError(
                    f"{name} is not cumulative: missing prior flag {key}={value!r}"
                )
        prev = dict(flags)
