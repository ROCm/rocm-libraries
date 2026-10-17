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
    "head_dim_v": "vd",
    "qk_mask": "qkm",
    "split_kv": "split",
    "pack_gqa": "gqa",
    "k_chunk": "kc",
    "has_sink": "sink",
    "bias_f32": "f32bias",
    "bias_kstrided": "kstrbias",
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
    "head_dim_v": lambda knobs: knobs["head_dim_v"] == knobs["head_dim"],
    "qk_mask": lambda knobs: knobs["qk_mask"] == 0,
    "split_kv": lambda knobs: knobs["split_kv"] == 0,
    "pack_gqa": lambda knobs: knobs["pack_gqa"] == 0,
    "k_chunk": lambda knobs: knobs["k_chunk"] == 0,
    "has_sink": lambda knobs: knobs["has_sink"] == 0,
    "bias_f32": lambda knobs: knobs["bias_f32"] == 0,
    "bias_kstrided": lambda knobs: knobs["bias_kstrided"] == 0,
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
# Absorbed multi-head latent attention (DeepSeek-V2/V3, Kimi, GLM: Q/K 576 = a 512-wide
# latent and a 64-wide rotary part, V 512 = the latent) and DeepSeek-V4's 512/512, both
# with a single KV head: decode objects only, the output columns split across four
# workgroups (dv_split 4) so each holds the d256 objects' 128 columns. No bias.
SDPA_DECODE_WIDE_HEAD_DIMS = ((576, 512), (512, 512))
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

# A V head_dim of its own (kernel modification 20): multi-head latent attention
# (DeepSeek-V2/V3, Kimi K2, GLM-5) runs its prefill un-absorbed, with Q and K 192
# wide (128 + a 64-wide rotary part) and V 128. One baked pair, both causal
# variants, no bias; its schedule is the QK width's.
SDPA_MLA_HEAD_DIMS = ((192, 128),)

# Attention sinks (prefill modification 21, and the decode kernel's): one logit per
# query head joining each row's softmax, as gpt-oss uses them. A baked axis, so the
# plain objects build as before; the head dims gpt-oss and its relatives use, both
# causal variants, prefill and decode, no bias. A sink object serves only a graph
# with a sink, so it ranks as its plain counterpart does.
SDPA_SINK_HEAD_DIMS = (64, 128)


def _sdpa_rdna_rows() -> list[Instance]:
    """Every RDNA SDPA-forward row, prefill and decode alike, before the two are split
    into their engines' tables (`sdpa_instances`, `sdpa_rdna_decode_instances`)."""
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
        for head_dim, head_dim_v in SDPA_MLA_HEAD_DIMS:
            for causal in SDPA_CAUSAL:
                block_m, dv_split, hd_pair = sdpa_prefill_schedule(head_dim, causal, 0)
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
                            "has_bias": 0,
                            "head_dim_max": head_dim,
                            "decode": 0,
                            "merge_symbol": "",
                            "hd_pair": hd_pair,
                            "bias_q": 0,
                            "head_dim_v": head_dim_v,
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
        for head_dim, head_dim_v in SDPA_DECODE_WIDE_HEAD_DIMS:
            for causal in SDPA_CAUSAL:
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
                            "dv_split": head_dim_v // 128,
                            "has_bias": 0,
                            "head_dim_max": head_dim,
                            "decode": 1,
                            "merge_symbol": SDPA_DECODE_MERGE_SYMBOL,
                            "hd_pair": 0,
                            "bias_q": 0,
                            "head_dim_v": head_dim_v,
                        },
                    )
                )
        for head_dim in SDPA_SINK_HEAD_DIMS:
            for causal in SDPA_CAUSAL:
                block_m, dv_split, hd_pair = sdpa_prefill_schedule(head_dim, causal, 0)
                for decode in (0, 1):
                    instances.append(
                        Instance(
                            op="sdpa_rdna",
                            dtype=dtype,
                            priority=(
                                PRIORITY_SDPA_DECODE if decode else PRIORITY_SPECIALIZED
                            ),
                            knobs={
                                "head_dim": head_dim,
                                "causal": causal,
                                "block_m": SDPA_DECODE_ROWS if decode else block_m,
                                "block_n": SDPA_BLOCK_N,
                                "dv_split": (
                                    sdpa_dv_split(head_dim) if decode else dv_split
                                ),
                                "has_bias": 0,
                                "head_dim_max": head_dim,
                                "decode": decode,
                                "merge_symbol": (
                                    SDPA_DECODE_MERGE_SYMBOL if decode else ""
                                ),
                                "hd_pair": 0 if decode else hd_pair,
                                "bias_q": 0,
                                "has_sink": 1,
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
    # Every other object's V is as wide as its Q and K (a generic one's too: both
    # are the runtime head_dim).
    # Only the sink objects add one.
    for instance in instances:
        instance.knobs.setdefault("head_dim_v", instance.knobs["head_dim"])
        instance.knobs.setdefault("has_sink", 0)
    return instances


# One kernel family per engine. The prefill kernel (flash_attn_func_rdna.py) and the
# decode kernel (flash_attn_decode_rdna.py) are different algorithms -- one workgroup per
# query tile against split-KV with the GQA group packed into 16-row tiles and a merge
# pass -- with their own argument lists and launches, so each is an engine of its own
# (`hipkernel:flydsl_sdpa_rdna`, `hipkernel:flydsl_sdpa_rdna_decode`) rather than a knob
# that picks between them. Which serves a graph both can is the heuristic's to decide.
# Each family drops the knobs that only told the two apart: `decode` from both, the
# merge kernel's symbol from the prefill objects, `hd_pair` (always 0) from the decode
# ones.
_SDPA_RDNA_PREFILL_ONLY = ("decode", "merge_symbol")
_SDPA_RDNA_DECODE_ONLY = ("decode", "hd_pair")


def _without(instance: Instance, op: str, dropped: tuple[str, ...]) -> Instance:
    knobs = {k: v for k, v in instance.knobs.items() if k not in dropped}
    return Instance(
        op=op, dtype=instance.dtype, priority=instance.priority, knobs=knobs
    )


def sdpa_instances() -> list[Instance]:
    """The RDNA SDPA prefill table: (dtype, head_dim, causal, has_bias, has_sink)."""
    instances = [
        _without(row, "sdpa_rdna", _SDPA_RDNA_PREFILL_ONLY)
        for row in _sdpa_rdna_rows()
        if not row.knobs["decode"]
    ]
    _assert_unique(instances)
    return instances


def sdpa_rdna_decode_instances() -> list[Instance]:
    """The RDNA SDPA decode table: split-KV objects, each carrying its merge kernel."""
    instances = [
        _without(row, "sdpa_rdna_decode", _SDPA_RDNA_DECODE_ONLY)
        for row in _sdpa_rdna_rows()
        if row.knobs["decode"]
    ]
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


# --- SDPA forward, RDNA4: AOTriton's FlyDSL kernel -------------------------------

# AOTriton's FlyDSL gfx1201 forward (kernels_src/kernels/attention_rdna4/), vendored with
# the modifications its README lists. A kernel family of its own, so an engine of its own
# (`hipkernel:flydsl_sdpa_aiw`), gfx12-generic only: it is faster than the RDNA kernel on
# medium and long sequences and slower on short ones, and which serves a graph both can is
# the heuristic's to decide. Its schedule (tile, waves, prefetch distances, row subtiles)
# is the kernel's own tuning table's answer for each row (`fmha_tuning_gfx1201.plan`); the
# generator records it with the object, as knobs a ranker can read.
#
# Rows, bf16 and f16 x causal 0/1: the exact head dims 64, 128 and 256, each plain, with an
# additive mask in the query's dtype and with an f32 one; attention sinks at 64 and 128;
# multi-head latent attention's un-absorbed 192/128; and padded tiers -- an object built
# for a tile on the kernel's ladder that serves any head_dim up to it that is a multiple of
# 8, the real one a runtime argument -- at the RDNA generic tiers' widths. A padded tier
# ranks below an exact object, the narrower tier above the wider.
SDPA_AIW_TARGETS = ("gfx12-generic",)
SDPA_AIW_HEAD_DIMS = (64, 128, 256)
SDPA_AIW_SINK_HEAD_DIMS = (64, 128)
SDPA_AIW_MLA_HEAD_DIMS = ((192, 128),)
SDPA_AIW_PADDED_TIERS = (64, 96, 128, 160, 224, 256)
PRIORITY_SDPA_AIW_PADDED = {
    64: 40,
    96: 30,
    128: 20,
    160: 15,
    224: 12,
    256: PRIORITY_GENERIC,
}
# Mask variants: (has_bias, bias_f32). The mask in Q's dtype is (1, 0).
SDPA_AIW_BIAS = ((0, 0), (1, 0), (1, 1))


def _aiw_instance(
    dtype,
    head_dim,
    causal,
    priority,
    head_dim_max,
    head_dim_v=None,
    has_bias=0,
    bias_f32=0,
    has_sink=0,
) -> Instance:
    return Instance(
        op="sdpa_aiw",
        dtype=dtype,
        priority=priority,
        knobs={
            "head_dim": head_dim,
            "causal": causal,
            "head_dim_max": head_dim_max,
            "head_dim_v": head_dim if head_dim_v is None else head_dim_v,
            "has_bias": has_bias,
            "bias_f32": bias_f32,
            "bias_kstrided": 0,
            "has_sink": has_sink,
        },
    )


def sdpa_aiw_instances(arch: str) -> list[Instance]:
    """AOTriton's gfx1201 kernel's table for one target (gfx12-generic only; empty elsewhere)."""
    if arch not in SDPA_AIW_TARGETS:
        return []
    instances: list[Instance] = []
    for dtype in DTYPES:
        for causal in SDPA_CAUSAL:
            for head_dim in SDPA_AIW_HEAD_DIMS:
                for has_bias, bias_f32 in SDPA_AIW_BIAS:
                    instances.append(
                        _aiw_instance(
                            dtype,
                            head_dim,
                            causal,
                            PRIORITY_SPECIALIZED,
                            head_dim,
                            has_bias=has_bias,
                            bias_f32=bias_f32,
                        )
                    )
            for head_dim in SDPA_AIW_SINK_HEAD_DIMS:
                instances.append(
                    _aiw_instance(
                        dtype,
                        head_dim,
                        causal,
                        PRIORITY_SPECIALIZED,
                        head_dim,
                        has_sink=1,
                    )
                )
            for head_dim, head_dim_v in SDPA_AIW_MLA_HEAD_DIMS:
                instances.append(
                    _aiw_instance(
                        dtype,
                        head_dim,
                        causal,
                        PRIORITY_SPECIALIZED,
                        head_dim,
                        head_dim_v=head_dim_v,
                    )
                )
            for tier in SDPA_AIW_PADDED_TIERS:
                instances.append(
                    _aiw_instance(
                        dtype, None, causal, PRIORITY_SDPA_AIW_PADDED[tier], tier
                    )
                )
    _assert_unique(instances)
    return instances


OPS = {
    "rmsnorm": rmsnorm_instances,
    "sdpa_rdna": sdpa_instances,
    "sdpa_rdna_decode": sdpa_rdna_decode_instances,
    "sdpa_aiw": sdpa_aiw_instances,
}
# Ops whose table differs by target and so takes the arch.
_PER_TARGET_OPS = {"sdpa_aiw"}


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
