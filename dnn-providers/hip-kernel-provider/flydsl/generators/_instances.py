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
    "scale_f32": "f32scale",
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
    "scale_f32": lambda knobs: knobs["scale_f32"] == 0,
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

# Dtypes built in the generic tier only. The kernel's specialized-N fast path is
# 128-bit tiles of 16-bit elements (`elem_bits <= 16` in rmsnorm_kernel.py), so an
# f32 object baked for one width would run the same scalar loop the generic one
# does; the specialized tier would add objects and no speed.
RMSNORM_GENERIC_ONLY_DTYPES = ("f32",)


# `scale_f32`: gamma read as f32 beside bf16/f16 x and y (the kernel's
# `weight_dtype_str="f32"`), the mixed form cuDNN and PyTorch commonly send. 0 is
# gamma in x's own dtype. f32 x has only 0: its gamma is already f32. Both tiers
# are built for it; the 16-bit fast path reads gamma with its own 128-bit copies.
RMSNORM_SCALE_F32 = (0, 1)


def rmsnorm_instances() -> list[Instance]:
    """The RMSNorm instance table: generic tier first, then the specialists."""
    instances: list[Instance] = []
    for scale_f32 in RMSNORM_SCALE_F32:
        dtypes = DTYPES if scale_f32 else DTYPES + RMSNORM_GENERIC_ONLY_DTYPES
        for block_threads in RMSNORM_BLOCK_THREADS:
            for dtype in dtypes:
                instances.append(
                    Instance(
                        op="rmsnorm",
                        dtype=dtype,
                        priority=PRIORITY_GENERIC,
                        knobs={
                            "N": None,
                            "block_threads": block_threads,
                            "scale_f32": scale_f32,
                        },
                    )
                )
            for dtype in DTYPES:
                for n in RMSNORM_SPECIALIZED_N:
                    instances.append(
                        Instance(
                            op="rmsnorm",
                            dtype=dtype,
                            priority=PRIORITY_SPECIALIZED,
                            knobs={
                                "N": n,
                                "block_threads": block_threads,
                                "scale_f32": scale_f32,
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


OPS = {
    "rmsnorm": rmsnorm_instances,
}
# Ops whose table differs by target and so takes the arch.
_PER_TARGET_OPS: set[str] = set()


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
