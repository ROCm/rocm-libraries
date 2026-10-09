# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The AOT instance table: one row per kernel object this provider ships.

This module is the single source of truth for *what gets built*. The generator
compiles from it, the packer keys its table of contents off it, and the
descriptor emitter turns each row into a kernelDescriptor with its ``.kmd.json``
metadata and its ``priority``. Adding an instance is a row here, not an edit in
three places that can disagree.

**Two tiers per op.** A specialized instance bakes a knob in and matches that
value exactly; a generic instance leaves the knob to runtime, covers everything,
and is slower. The selector picks the specialist when one applies and falls back
to the generic otherwise, so the declared surface has no holes. Which tier wins
a shape is expressed purely as data -- ``priority`` is the ingestor's tie-break
when no heuristic model is loaded -- so no ranking code is written for it.

For RMSNorm the specialized knob is the row width ``N``: FlyDSL's vectorized
128-bit path caches tiles in a Python list of registers, which only exists under
compile-time loop unrolling, so it genuinely cannot take a runtime ``N``. The
generic tier is the vendored runtime-``N`` variant (``kernels_src`` modification
1), which reads ``N`` from the input tensor descriptor and predicates the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ranking is `score desc -> priority desc -> kernelId asc`, and with no
# heuristic model every score ties, so priority is what orders the catalog.
# Two bands, far enough apart to insert tiers between them later.
PRIORITY_SPECIALIZED = 100
PRIORITY_GENERIC = 10

# FlyDSL dtype spellings, as `build_*_module` takes them.
DTYPES = ("bf16", "f16")


@dataclass(frozen=True)
class Instance:
    """One compiled kernel object and everything the pipeline needs about it."""

    op: str
    dtype: str
    priority: int
    # Baked knob values, verbatim into `.kmd.json`. A knob whose value is None is
    # the runtime/generic tier for that axis -- the matcher reads null as "any".
    knobs: dict = field(default_factory=dict)

    @property
    def is_generic(self) -> bool:
        return any(value is None for value in self.knobs.values())

    @property
    def name(self) -> str:
        """Stable instance name: the object's filename stem and TOC key leaf."""
        parts = [self.op, self.dtype]
        for key, value in self.knobs.items():
            if key == "dtype":
                continue
            # A knob added after objects shipped is left out of the name at its
            # default, so existing names -- filenames, TOC keys -- do not move.
            if key in _NAME_OMITTED_WHEN and _NAME_OMITTED_WHEN[key](self.knobs):
                continue
            parts.append(f"{_abbrev(key)}{'generic' if value is None else value}")
        return "_".join(parts)

    def to_record(self) -> dict:
        return {
            "name": self.name,
            "op": self.op,
            "dtype": self.dtype,
            "priority": self.priority,
            "tier": "generic" if self.is_generic else "specialized",
            "knobs": dict(self.knobs),
        }


_ABBREV = {
    "N": "n",
    "block_threads": "bt",
    "head_dim": "d",
    "block_m": "bm",
    "block_n": "bn",
    "dv_split": "dv",
    "has_bias": "bias",
    "head_dim_max": "dmax",
    "decode": "decode",
    "hd_pair": "pair",
    "kernel": "",
    "num_heads": "hq",
    "num_kv_heads": "hkv",
    "kv_align": "kva",
    "any_diagonal": "anydiag",
    "waves_per_eu": "wpe",
    "setprio": "prio",
    "bias_q": "biasq",
    "kv_gpfetch": "gpf",
}

# Knobs omitted from an instance's name when they add nothing to it; see
# Instance.name. Each was added after objects shipped.
_NAME_OMITTED_WHEN = {
    "dv_split": lambda knobs: knobs["dv_split"] == 1,
    "has_bias": lambda knobs: knobs["has_bias"] == 0,
    "head_dim_max": lambda knobs: knobs["head_dim_max"] == knobs["head_dim"],
    "decode": lambda knobs: knobs["decode"] == 0,
    "hd_pair": lambda knobs: knobs["hd_pair"] == 0,
    "kv_align": lambda knobs: knobs["kv_align"] == 0,
    "any_diagonal": lambda knobs: knobs["any_diagonal"] == 1,
    "waves_per_eu": lambda knobs: knobs["waves_per_eu"] == 0,
    "setprio": lambda knobs: knobs["setprio"] == 1,
    "bias_q": lambda knobs: knobs["bias_q"] == 0,
    "kv_gpfetch": lambda knobs: knobs["kv_gpfetch"] == 1,
    # A pick rule, not a build choice: two objects never differ in it alone.
    "prefer_feature": lambda knobs: True,
    "prefer_min": lambda knobs: True,
    "prefer_max": lambda knobs: True,
    # A symbol, not a choice: the object's name already says it is a decode one.
    "merge_symbol": lambda knobs: True,
}


def _abbrev(key: str) -> str:
    return _ABBREV.get(key, key.lower())


# --- RMSNorm ----------------------------------------------------------------

# Below this width `build_rmsnorm_module` does not specialize the kernel above --
# it returns `_build_rmsnorm_large_m_small_n_module`, a *different* kernel with a
# different ABI (an extra `m_in` kernarg) and a different launch geometry (grid
# divided by BLOCK_M rather than one block per row). Mirrors SMALL_N_THRESHOLD at
# kernels_src/kernels/norm/rmsnorm_kernel.py:73; the generator asserts the two
# still agree, because a drift here is the kind that produces a correct-looking
# object dispatched with the wrong grid.
#
# That path is worth having -- it is upstream's large-M/small-N specialization --
# but it is a second launch record, so it lands as its own instance family rather
# than hiding inside this one. Until then small N is served by the generic tier.
RMSNORM_SMALL_N_THRESHOLD = 2048

# The specialized widths: the hidden sizes that dominate the models this release
# targets, not a sweep. 3072/4096/5120/8192 are Llama-family, 3584 is Qwen2-7B.
# Every other N -- and every N a future model introduces -- is served by the
# generic tier, which is the point of having one. Widening this list is cheap; it
# is not a coverage fix.
#
# 2048 was in the original plan and is deliberately absent: `N <= threshold` is
# the small-N branch above, so asking for it here would not produce a
# specialization of this kernel at all.
RMSNORM_SPECIALIZED_N = (3072, 3584, 4096, 5120, 8192)

# The kernel's `BLOCK_THREADS`. Single-valued today: it is a launch-geometry
# knob, not a correctness one, so the second value only earns its instances once
# benchmarking says it wins somewhere. Kept in the knob dict so descriptors and
# the native matcher already carry the axis when it grows.
RMSNORM_BLOCK_THREADS = (256,)


def rmsnorm_instances() -> list[Instance]:
    """The RMSNorm instance table: generic tier first, then the specialists."""
    instances: list[Instance] = []
    for block_threads in RMSNORM_BLOCK_THREADS:
        for dtype in DTYPES:
            instances.append(
                Instance(
                    op="rmsnorm",
                    dtype=dtype,
                    priority=PRIORITY_GENERIC,
                    knobs={"N": None, "block_threads": block_threads},
                )
            )
        for dtype in DTYPES:
            for n in RMSNORM_SPECIALIZED_N:
                instances.append(
                    Instance(
                        op="rmsnorm",
                        dtype=dtype,
                        priority=PRIORITY_SPECIALIZED,
                        knobs={"N": n, "block_threads": block_threads},
                    )
                )
    _assert_unique(instances)
    return instances


# --- SDPA forward ------------------------------------------------------------

# The vendored kernel (kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py)
# bakes only head_dim, the causal flag, dtype and its tile. Batch, both sequence
# lengths, both head counts (GQA/MQA), every stride, the softmax scale and the
# causal offset are runtime arguments, so each row below serves every shape and
# layout of its (dtype, head_dim, causal) class. That is what keeps this table
# small: the axes it enumerates are model-determined, never request-determined.
#
# head_dim: 64 and 128 cover the large majority of current LLM and diffusion
# attention; 96 is the Phi-family width; 256 is Gemma's. Up to 128 the kernel
# holds Q and every O accumulator in registers. At 256 those alone would fill
# the 256-VGPR file, so its objects re-read Q per K-step and split the output
# columns across SDPA_DV_SPLIT workgroups (kernel modification 13). That is a
# schedule of its own, baked per head_dim like the rest, so the 64-128 objects
# are built exactly as before and pay nothing for it.
SDPA_HEAD_DIMS = (64, 96, 128, 256)


def sdpa_dv_split(head_dim: int) -> int:
    """Workgroups sharing one Q tile's output columns: 2 above 128, else 1."""
    return 2 if head_dim > 128 else 1


# A prefill object's schedule: (block_m, dv_split, hd_pair). Launch geometry, not
# correctness, chosen per class by measurement on gfx1151:
# - 64-row tiles launch twice the workgroups of 128-row ones and win wherever the
#   kernel still holds Q in registers at that size: d64, and d128 without a causal
#   bound or a bias. A 64-row d128 causal or bias object has to re-read Q per
#   K-step to fit, and that costs more than the tile saves.
# - Above 128, waves work in head-dim pairs (kernel modification 18), which
#   computes QK^T once per row rather than once per column workgroup, at 64 rows so
#   the thread count stays 256.
def sdpa_prefill_schedule(head_dim: int, causal: int, has_bias: int) -> tuple:
    if head_dim > 128:
        return 64, 1, 1
    if head_dim == 64 and not has_bias:
        return 64, 1, 0
    if head_dim == 128 and not causal and not has_bias:
        return 64, 1, 0
    return SDPA_BLOCK_M, sdpa_dv_split(head_dim), 0


# The generic tier: any head_dim that is a multiple of 8, read at runtime, by an
# object built for the largest it serves (kernel modification 15). An object
# runs at its largest head's cost, so the tiers are cut where the odd head dims
# models use fall: up to 64 (40, 48, 56), 96 (72, 80, 88), 128, 160 (136-160),
# 224 (168-224) and 256. Each maps to its schedule (block_m, dv_split, hd_pair):
# up to 64 a 64-row tile, as for the specialized head_dim 64 (at 96 and 128 it
# measured slower or mixed); above 128 the waves work in head-dim pairs, and the 256 one also splits its
# columns across two workgroups, because its masked loads on top of the d256
# schedule do not fit the register file in one, which is why 224 is a tier of its
# own. Below the specialized tier in priority, and each narrower one above the
# wider, so the narrowest object that serves a head is the one chosen.
SDPA_GENERIC_HEAD_DIM_MAX = {
    64: (64, 1, 0),
    96: (128, 1, 0),
    128: (128, 1, 0),
    160: (64, 1, 1),
    224: (64, 1, 1),
    256: (64, 2, 1),
}
PRIORITY_SDPA_GENERIC = {
    64: 40,
    96: 30,
    128: 20,
    160: 15,
    224: 12,
    256: PRIORITY_GENERIC,
}

# The decode family (flash_attn_decode_rdna.py): split-KV with the GQA group packed
# into 16-row tiles, for graphs with g * Sq <= 128. Its own kernel and its own launch
# geometry, so a separate set of objects; above the prefill ones in priority, so a
# graph both serve runs here. Each object also carries the merge kernel that combines
# its splits, named by `merge_symbol`. The specialized head dims, with and without a
# bias (a baked axis here too: a runtime bias failed the gate, §6), one V head per K
# head; anything else takes the prefill objects. d256 splits its output columns across
# two workgroups, as the prefill object does, to fit the register file.
SDPA_DECODE_HEAD_DIMS = (64, 96, 128, 256)
# Generic decode objects for every other head_dim that is a multiple of 8: up to 96,
# and above it up to 256 with the d256 column split. Columns past the actual head_dim
# are never fetched, so a decode step reads only the bytes its head has; an up-to-128
# tier does not fit the register file. No bias, as for the generic prefill tier. Above
# every prefill object in priority and below the specialized decode objects, the
# narrower tier above the wider.
SDPA_DECODE_GENERIC_HEAD_DIM_MAX = {96: 1, 256: 2}
PRIORITY_SDPA_DECODE_GENERIC = {96: 145, 256: 140}
SDPA_DECODE_MERGE_SYMBOL = "flash_attn_decode_merge_rdna_kernel_1"
PRIORITY_SDPA_DECODE = 150
SDPA_DECODE_ROWS = 16


# Two variants of one kernel rather than a runtime flag: causal changes the KV
# loop bound and skips fully-masked tiles, and the non-causal variant carries a
# V prefetch across iterations that the causal one drops for register pressure.
SDPA_CAUSAL = (0, 1)

# The kernel's tile: upstream's RDNA4 defaults, unchanged. Launch geometry, not
# correctness -- kept in the knob dict so a second tile is a row, and so the
# dispatcher reads the grid's block_m from the selected object rather than
# restating it.
SDPA_BLOCK_M = 128
SDPA_BLOCK_N = 32

# An additive f32 bias (`attn_mask`, what `F.sdpa(attn_mask=)` lowers to) is
# per-score work in the inner loop, so it is a baked axis: every (dtype,
# head_dim, causal) class has a bias object beside its plain one, and the plain
# objects compile to the same code as before bias existed (COVERAGE.md §6).
SDPA_HAS_BIAS = (0, 1)
# Which dtype the bias is in: f32 (`bias_q` 0), or the query's (1) -- the dtype
# PyTorch requires of a float `attn_mask`, and Hugging Face builds its padding
# masks in. A bias class ships both; the dtype is a load width, so baked.
SDPA_BIAS_Q = (0, 1)


def sdpa_instances() -> list[Instance]:
    """The SDPA-forward instance table: (dtype, head_dim, causal, has_bias)."""
    instances: list[Instance] = []
    for dtype in DTYPES:
        for head_dim in SDPA_HEAD_DIMS:
            for causal in SDPA_CAUSAL:
                for has_bias in SDPA_HAS_BIAS:
                    block_m, dv_split, hd_pair = sdpa_prefill_schedule(
                        head_dim, causal, has_bias
                    )
                    for bias_q in SDPA_BIAS_Q if has_bias else (0,):
                        instances.append(
                            Instance(
                                op="sdpa_rdna",
                                dtype=dtype,
                                priority=PRIORITY_SPECIALIZED,
                                knobs={
                                    "head_dim": head_dim,
                                    "causal": causal,
                                    "block_m": block_m,
                                    "block_n": SDPA_BLOCK_N,
                                    "dv_split": dv_split,
                                    "has_bias": has_bias,
                                    "head_dim_max": head_dim,
                                    "decode": 0,
                                    "merge_symbol": "",
                                    "hd_pair": hd_pair,
                                    "bias_q": bias_q,
                                },
                            )
                        )
        # Generic head_dim, without a bias: a bias on a head no specialized
        # object serves declines.
        for head_dim_max, schedule in SDPA_GENERIC_HEAD_DIM_MAX.items():
            block_m, dv_split, hd_pair = schedule
            for causal in SDPA_CAUSAL:
                instances.append(
                    Instance(
                        op="sdpa_rdna",
                        dtype=dtype,
                        priority=PRIORITY_SDPA_GENERIC[head_dim_max],
                        knobs={
                            "head_dim": None,
                            "causal": causal,
                            "block_m": block_m,
                            "block_n": SDPA_BLOCK_N,
                            "dv_split": dv_split,
                            "has_bias": 0,
                            "head_dim_max": head_dim_max,
                            "decode": 0,
                            "merge_symbol": "",
                            "hd_pair": hd_pair,
                            "bias_q": 0,
                        },
                    )
                )
        for head_dim in SDPA_DECODE_HEAD_DIMS:
            for causal in SDPA_CAUSAL:
                for has_bias in SDPA_HAS_BIAS:
                    for bias_q in SDPA_BIAS_Q if has_bias else (0,):
                        instances.append(
                            Instance(
                                op="sdpa_rdna",
                                dtype=dtype,
                                priority=PRIORITY_SDPA_DECODE,
                                knobs={
                                    "head_dim": head_dim,
                                    "causal": causal,
                                    "block_m": SDPA_DECODE_ROWS,
                                    "block_n": SDPA_BLOCK_N,
                                    "dv_split": sdpa_dv_split(head_dim),
                                    "has_bias": has_bias,
                                    "head_dim_max": head_dim,
                                    "decode": 1,
                                    "merge_symbol": SDPA_DECODE_MERGE_SYMBOL,
                                    "hd_pair": 0,
                                    "bias_q": bias_q,
                                },
                            )
                        )
        for head_dim_max, dv_split in SDPA_DECODE_GENERIC_HEAD_DIM_MAX.items():
            for causal in SDPA_CAUSAL:
                instances.append(
                    Instance(
                        op="sdpa_rdna",
                        dtype=dtype,
                        priority=PRIORITY_SDPA_DECODE_GENERIC[head_dim_max],
                        knobs={
                            "head_dim": None,
                            "causal": causal,
                            "block_m": SDPA_DECODE_ROWS,
                            "block_n": SDPA_BLOCK_N,
                            "dv_split": dv_split,
                            "has_bias": 0,
                            "head_dim_max": head_dim_max,
                            "decode": 1,
                            "merge_symbol": SDPA_DECODE_MERGE_SYMBOL,
                            "hd_pair": 0,
                            "bias_q": 0,
                        },
                    )
                )
    _assert_unique(instances)
    return instances


def _assert_unique(instances: list[Instance]) -> None:
    """Instance names are filenames and TOC keys; a collision silently drops one."""
    seen: dict[str, Instance] = {}
    for instance in instances:
        if instance.name in seen:
            raise ValueError(
                f"duplicate instance name {instance.name!r}: {seen[instance.name]} "
                f"and {instance}. Names are derived from the knob dict, so two rows "
                "differing only in an axis absent from the name collide."
            )
        seen[instance.name] = instance


# --- SDPA forward, CDNA ---------------------------------------------------------

# FlyDSL's own CDNA kernels (kernels_src/kernels/attention_cdna/): the dual-wave
# software-pipelined kernel (gfx950 only; head_dim 64 and 128) and the generic one
# (powers of two from 64 to 256). Unlike every table above, which tables an op for all of its targets,
# this one differs by target, so it takes the arch.
#
# What ships was chosen by measurement, not by hand (COVERAGE.md section 7.6): every
# candidate schedule was built and timed on each target against the forward
# attention graphs of the dnn-benchmarking workloads, and the set below is the
# smallest found that comes within a few percent of the best candidate on that corpus.
#
# A class -- (dtype, head_dim, causal, bias) -- ships one or more tiers. The first is
# its default; the others are companions for the shapes where another schedule
# wins, most often short queries (decode, small prefill), where the dual-wave
# kernel's 256-row tile is mostly idle. A companion that the class's rule prefers
# for a shape ranks above the default there (see `prefer` below); otherwise it ranks
# below it and is reached by autotune only.


@dataclass(frozen=True)
class CdnaTier:
    """One object family of a class: a schedule, and when it is preferred."""

    kernel: str
    block_m: int
    # 0 is the kernel's own default.
    waves_per_eu: int = 0
    # The dual-wave kernel's s_setprio scheduling hint.
    setprio: int = 1
    # The generic kernel's KV global prefetch (gfx942).
    kv_gpfetch: int = 1
    # (feature, min, max): preferred when the graph's feature lies in [min, max),
    # a max of 0 being unbounded. "rows" is batch x query heads x query length;
    # "seq_q" the query length. ("none", 0, 0) for a tier with no preference.
    prefer: tuple[str, int, int] = ("none", 0, 0)
    # Also ship the object that skips the KV tail mask, for key lengths a
    # multiple of SDPA_CDNA_KV_ALIGN (generic, non-causal only), ranked above this one.
    kv_aligned: bool = False
    # Also ship the plain causal build, exact at diagonal offset 0 only, ranked above
    # this one (dual-wave, causal only).
    offset_zero: bool = False


SDPA_CDNA_KV_ALIGN = 64
# Within a tier the specialized variant (kv_aligned / offset_zero) ranks above the
# plain one; a companion tier ranks below the default tier unless its rule prefers it.
PRIORITY_SDPA_CDNA_DEFAULT = PRIORITY_SPECIALIZED
PRIORITY_SDPA_CDNA_DEFAULT_SPECIALIZED = 110
PRIORITY_SDPA_CDNA_COMPANION = 80
PRIORITY_SDPA_CDNA_COMPANION_SPECIALIZED = 85

_DW, _GEN = "dualwave", "generic"
# The short-query companion: the generic kernel's 64-row tile.
_SHORT_Q = 256
_SDPA_CDNA_DUALWAVE = {
    # Non-causal: setprio off measured ~1% faster than upstream's default.
    (64, 0): (
        CdnaTier(_DW, 256, setprio=0),
        CdnaTier(_GEN, 64),
    ),
    # d64 causal: the any-diagonal build measured as fast as the offset-0 one, so it
    # ships alone.
    (64, 1): (
        CdnaTier(_DW, 256),
        CdnaTier(_GEN, 64, prefer=("rows", 0, 24576)),
    ),
    (128, 0): (
        CdnaTier(_DW, 256, setprio=0),
        CdnaTier(_GEN, 64),
    ),
    (128, 1): (
        CdnaTier(_DW, 256, offset_zero=True),
        CdnaTier(_GEN, 64, prefer=("seq_q", 0, _SHORT_Q)),
    ),
}
SDPA_CDNA_TIERS: dict[str, dict[tuple[int, int], tuple[CdnaTier, ...]]] = {
    "gfx950": {
        **_SDPA_CDNA_DUALWAVE,
        # The generic kernel needs one wave per SIMD at 256 to hold its tile.
        (256, 0): (
            CdnaTier(_GEN, 128, waves_per_eu=1, kv_aligned=True),
            CdnaTier(_GEN, 64, waves_per_eu=1, kv_aligned=True),
        ),
        (256, 1): (
            CdnaTier(_GEN, 128, waves_per_eu=1),
            CdnaTier(_GEN, 64, waves_per_eu=1, prefer=("seq_q", 0, _SHORT_Q)),
        ),
    },
    "gfx942": {
        (64, 0): (
            CdnaTier(_GEN, 128, waves_per_eu=3, kv_aligned=True),
            CdnaTier(_GEN, 64, kv_aligned=True, prefer=("rows", 0, 24576)),
        ),
        (64, 1): (
            CdnaTier(_GEN, 128, waves_per_eu=3),
            CdnaTier(_GEN, 64, prefer=("rows", 0, 24576)),
        ),
        # Upstream's own launcher switches 128 -> 256 rows by problem size.
        (128, 0): (
            CdnaTier(_GEN, 128, kv_aligned=True),
            CdnaTier(_GEN, 64, kv_aligned=True, prefer=("rows", 0, 18432)),
            CdnaTier(_GEN, 256, kv_aligned=True, prefer=("rows", 49152, 0)),
        ),
        (128, 1): (
            CdnaTier(_GEN, 128, kv_gpfetch=0),
            CdnaTier(_GEN, 64, prefer=("rows", 0, 18432)),
            CdnaTier(_GEN, 256, prefer=("rows", 49152, 0)),
        ),
        # head_dim 256 fits gfx942's 64 KiB of LDS only with the K and V tiles unpadded
        # (gen_sdpa_cdna._GENERIC_LDS_PADS); the 256-row tile spills.
        (256, 0): (
            CdnaTier(_GEN, 128, kv_aligned=True),
            CdnaTier(_GEN, 64, kv_aligned=True, prefer=("rows", 0, 24576)),
        ),
        (256, 1): (
            CdnaTier(_GEN, 128),
            CdnaTier(_GEN, 64, prefer=("rows", 0, 24576)),
        ),
    },
}
# An additive bias in the query's dtype is a compile-time axis of both kernels (the
# dual-wave kernel upstream's, the generic one modification 12). Bias graphs are rare in
# the corpus, so a bias class ships its default tier only, with upstream's scheduling on
# the dual-wave kernel.
SDPA_CDNA_BIAS_TIERS: dict[str, dict[tuple[int, int], tuple[CdnaTier, ...]]] = {
    "gfx950": {
        (64, 0): (CdnaTier(_DW, 256),),
        (64, 1): (CdnaTier(_DW, 256),),
        (128, 0): (CdnaTier(_DW, 256),),
        (128, 1): (CdnaTier(_DW, 256, offset_zero=True),),
        (256, 0): (CdnaTier(_GEN, 128, waves_per_eu=1),),
        (256, 1): (CdnaTier(_GEN, 128, waves_per_eu=1),),
    },
    "gfx942": {
        # Three waves per EU spill with a bias; the kernel's default does not.
        (64, 0): (CdnaTier(_GEN, 128),),
        (64, 1): (CdnaTier(_GEN, 128),),
        (128, 0): (CdnaTier(_GEN, 128),),
        (128, 1): (CdnaTier(_GEN, 128, kv_gpfetch=0),),
        (256, 0): (CdnaTier(_GEN, 128),),
        (256, 1): (CdnaTier(_GEN, 128),),
    },
}

# Everything else is a kernel argument (vendored-kernel modifications 2-8).
SDPA_CDNA_DTYPES = ("bf16", "f16")


def _cdna_instance(
    dtype, head_dim, causal, has_bias, tier, priority, kv_align, any_diagonal
):
    feature, low, high = tier.prefer
    return Instance(
        op="sdpa_cdna",
        dtype=dtype,
        priority=priority,
        knobs={
            "kernel": tier.kernel,
            "head_dim": head_dim,
            "causal": causal,
            "block_m": tier.block_m,
            "kv_align": kv_align,
            "any_diagonal": any_diagonal,
            "waves_per_eu": tier.waves_per_eu,
            "has_bias": has_bias,
            "setprio": tier.setprio,
            "kv_gpfetch": tier.kv_gpfetch,
            "prefer_feature": feature,
            "prefer_min": low,
            "prefer_max": high,
        },
    )


def sdpa_cdna_instances(arch: str) -> list[Instance]:
    """The CDNA SDPA-forward instance table for one target."""
    if arch not in SDPA_CDNA_TIERS:
        raise KeyError(
            f"no CDNA SDPA kernel for {arch!r}; known: {', '.join(sorted(SDPA_CDNA_TIERS))}"
        )
    classes = [(key, 0, tiers) for key, tiers in SDPA_CDNA_TIERS[arch].items()]
    classes += [(key, 1, tiers) for key, tiers in SDPA_CDNA_BIAS_TIERS[arch].items()]
    instances: list[Instance] = []
    for dtype in SDPA_CDNA_DTYPES:
        for (head_dim, causal), has_bias, tiers in classes:
            for index, tier in enumerate(tiers):
                plain, special = (
                    (PRIORITY_SDPA_CDNA_DEFAULT, PRIORITY_SDPA_CDNA_DEFAULT_SPECIALIZED)
                    if index == 0
                    else (
                        PRIORITY_SDPA_CDNA_COMPANION,
                        PRIORITY_SDPA_CDNA_COMPANION_SPECIALIZED,
                    )
                )
                variants = [(0, 1, plain)]
                if tier.kv_aligned and not causal:
                    variants.append((SDPA_CDNA_KV_ALIGN, 1, special))
                if tier.offset_zero and causal:
                    variants.append((0, 0, special))
                for kv_align, any_diagonal, priority in variants:
                    instances.append(
                        _cdna_instance(
                            dtype,
                            head_dim,
                            causal,
                            has_bias,
                            tier,
                            priority,
                            kv_align,
                            any_diagonal,
                        )
                    )
    _assert_unique(instances)
    return instances


OPS = {
    "rmsnorm": rmsnorm_instances,
    "sdpa_rdna": sdpa_instances,
    "sdpa_cdna": sdpa_cdna_instances,
}
# Ops whose table differs by target and so takes the arch.
_PER_TARGET_OPS = {"sdpa_cdna"}


def instances_for(op: str, arch: str | None = None) -> list[Instance]:
    if op not in OPS:
        raise KeyError(f"unknown op {op!r}; known: {', '.join(sorted(OPS))}")
    if op in _PER_TARGET_OPS:
        if arch is None:
            raise ValueError(
                f"op {op!r} tables its instances per target; pass the arch"
            )
        return OPS[op](arch)
    return OPS[op]()
