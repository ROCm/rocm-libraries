# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knobs of the QSA stage functions. ``QsaKnobs()`` is the FlyDSL mono's
stages, bitwise; every other value is a lever (default off), named in the
function's sidecar (``stage_knobs``) and in :meth:`QsaKnobs.suffix`.

Exact (bitwise) levers:
  pause      poll retry: ``s_nop`` (mono) | ``s_sleep1`` | ``none``
  pbar       pre-poll barrier: ``workgroup`` (mono's fenced ``gpu.barrier``) | ``compiler``
  mailbox_impl  ``fly`` (FlyDSL's shape: buffer stores / loads with sc1, the first load peeled before the
             pause + reload loop, 32-bit tag compares; default) | ``asm`` (rocKE inline-asm global puts / polls,
             64-bit compares) | ``irf`` (rocKE ``tagged_poll`` with the reload fix)
  put_nt     ``nt`` on mailbox stores
  kv_nt      ``nt`` on the K / V cache loads of Q5 / Q7 (FlyDSL: plain)
  tbl        scalar page-table loads: ``smem`` (s_load) | ``vmem`` (vector load + readfirstlane)
  q7_pp      Q7's probability poll: ``batch`` (mono: 5 specs in one poll) | ``split`` (one poll per spec)
  q4_scan    Q4's radix-digit scan: ``dpp`` (the mono's DPP inclusive scan) | ``lds``
  q4_umax    Q4's wave max: ``asm`` (the mono's ``s_nop 1``-padded asm DPP chain) |
             ``ir`` (``update.dpp`` max chain the scheduler interleaves) | ``readlane``
  q3_switch  Q3: the one-block-per-wave cached-score path up to nw / V blocks (1: the mono's nw = 2048 blocks at
             one row; 0: always the batched path). P8: the small path is slower at its top end (exact either way)
  q4_chunks  Q4: chunk tasks per request row (8: the mono's 8 x 8192 blocks; 16 / 32 spread the chunk selections
             over the free CTAs after Q4's, the final task merges 16 / 32 candidates per thread; exact, P8 G5)
  q4_thr     Q4: the two-level threshold in blocks (0: the chunk size, the mono's rule; 8192 with q4_chunks=16:
             the mono's single-level path up to 8192 blocks, 16 chunks of 4096 above; exact)
  prepoll    stages without the hint pre-poll throttle ("q5+q7", ...): the payload polls spin themselves
  q5_kpre    Q5: every thread polls its two rows' selected blocks and issues the key-row loads BEFORE the q / k
             head poll (the loads overlap it; FlyDSL issues them after the poll and a barrier)
  q3_batch   Q3 cached scores: ``8`` blocks per wave per batch (mono UC) | 16 | 4 (exact: the
             wave sums of a batch are the same bits)
  numa       0 | 1: the mailbox layout of a mono build with ATOM's ``numa=1`` lever (dual copies in the NUMA
             area, IOD-local polls); needed to link into ``best`` / stack builds (sidecar ``mailboxes: numa1``)
  numa_hdr   byte offset of ATOM's NUMA header (0: ``moe_fused.NUMA_HDR``)
Tolerance-class levers (P11 / DESIGN 12.4 gates; off by default):
  qdense     N4a (P13): 0 off | 1 Q4 publishes the position order | 2 1 + Q3 keeps only the
             completing block's key | 3 2 + Q5 / Q7 take block j for list slot j (no ``sel`` poll)
  fuse       N4b: ``none`` | ``single`` (Q7 takes Q5's scores: Q6 idles, one hop fewer; f32 probabilities,
             one CTA-wide max per head, one 1 / den multiply at the end) | ``splitkv`` (the same, each wave a
             split of 17 x 16 positions: wave max, exp inside PV, the 8 splits merged in wave order)
             | ``tile`` (Q5's 32-position tiles are the splits: per head max / exp / sum in Q5, Q6 idles, Q7
             merges the 65 tiles' (m, l) per head in wave order and runs PV with p = e exp(m_t - M) / L)
  n4b_e      ``tile``'s e values: ``f32`` (one per pair) | ``bf16`` (two per pair, the mono's pp volume)
  split      (reserved for a Q5-side split-KV form; 64)
Codegen (exact): fma ``fma`` | ``fmuladd`` (f32), fmax ``maximum`` | ``maxnum`` (f32: differs only on
  NaN / signed zero), inline ``always`` | ``default`` | ``never``.
Test-only mutants (``mutate``): see ``MUTANTS``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

MUTANTS = (
    "q5_scale",  # Q5: the 1/16 score scale's lowest bit flipped
    "q5_cur",  # Q5: the current token's key from memory, not the mailbox
    "q6_den",  # Q6: the denominator without wave 3's partial sum
    "q7_gate",  # Q7: the gate of the other half of the pair
    "q7_tail",  # Q7: the incomplete block's tokens from the block before
    "q3_relu",  # Q3: no relu on head 3
    "q3_key_row",  # Q3: the completing block's key from row 4 b + 1
    "q4_tie",  # Q4: ties to the HIGHER block (the planted-tie / chunk-boundary gate)
    "q4_rank",  # Q4: rank 0 and 1 swapped in sel
    "qdense_drop",  # qdense: list slot 0 takes block 1 (P13's mutant 8 / 9)
    "tile_norescale",  # N4b tile: the merge without the exp(m_t - M) rescale
    "fuse_den",  # N4b single: the denominator without wave 7's partial
    "splitkv_norescale",  # N4b: the merge without the exp(m_s - M) rescale
    "splitkv_drop",  # N4b: split 0 dropped from the merge
)


# prepoll: the stages (Q5 / Q6 / Q7) whose hint pre-poll is skipped. Q3 / Q4 keep theirs: up to 256 x 512
# threads would spin on the score pairs (the mono's livelock note in mailbox.pre_poll).
PREPOLL_VALUES = ("", "q7", "q6", "q5", "q6+q7", "q5+q7", "q5+q6+q7")


@dataclass(frozen=True)
class QsaKnobs:
    pause: str = "s_nop"
    pbar: str = "workgroup"
    bar: str = "workgroup"
    mailbox_impl: str = "fly"
    put_nt: bool = False
    kv_nt: bool = False
    tbl: str = "smem"
    q7_pp: str = "batch"
    q4_scan: str = "dpp"
    q4_umax: str = "asm"
    q3_batch: int = 8
    q3_switch: int = 1
    q4_chunks: int = 8
    q4_thr: int = 0
    prepoll: str = ""
    q5_kpre: bool = False
    numa: int = 0
    numa_hdr: int = 0
    qdense: int = 0
    fuse: str = "none"
    split: int = 64
    n4b_e: str = "f32"
    fma: str = "fma"
    fmax: str = "maximum"
    inline: str = "always"
    mutate: str = ""

    def __post_init__(self):
        ok = {
            "pause": ("s_nop", "s_sleep1", "none"),
            "pbar": ("workgroup", "compiler"),
            "bar": ("workgroup", "compiler"),
            "mailbox_impl": ("fly", "asm", "irf", "ir"),
            "tbl": ("smem", "vmem"),
            "q7_pp": ("batch", "split"),
            "q4_scan": ("dpp", "lds"),
            "q4_umax": ("asm", "ir", "readlane"),
            "q3_batch": (4, 8, 16),
            "prepoll": PREPOLL_VALUES,
            "q3_switch": (1, 2, 4, 8, 0),
            "q4_chunks": (8, 16, 32),
            "q4_thr": (0, 8192, 16384),
            "numa": (0, 1),
            "qdense": (0, 1, 2, 3),
            "fuse": ("none", "single", "splitkv", "tile"),
            "n4b_e": ("f32", "bf16"),
            "split": (32, 64, 128),
            "fma": ("fma", "fmuladd"),
            "fmax": ("maximum", "maxnum"),
            "inline": ("always", "default", "never"),
        }
        for name, vals in ok.items():
            if getattr(self, name) not in vals:
                raise ValueError(f"QsaKnobs.{name} must be one of {vals}, got {getattr(self, name)!r}")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"unknown mutant {self.mutate!r}")

    @property
    def prepoll_skip(self) -> tuple:
        return tuple(x for x in self.prepoll.split("+") if x)

    @property
    def tolerance(self) -> bool:
        """A tolerance-class build (not bitwise the mono by design)."""
        return self.qdense != 0 or self.fuse != "none"

    def diff(self) -> dict:
        d = QsaKnobs()
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(d, f.name)}

    def suffix(self) -> str:
        return "_".join(f"{k}{v}" for k, v in sorted(self.diff().items()))

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def parse(cls, spec: str | dict | None) -> "QsaKnobs":
        """``"a=1,b=x"`` or a dict."""
        if not spec:
            return cls()
        if isinstance(spec, dict):
            items = spec.items()
        else:
            items = [kv.split("=", 1) for kv in spec.split(",") if kv.strip()]
        kw = {}
        types = {f.name: f.type for f in fields(cls)}
        for k, v in items:
            k = k.strip()
            if k not in types:
                raise ValueError(f"unknown QSA knob {k!r}")
            d = getattr(cls(), k)
            if isinstance(d, bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif isinstance(d, int):
                v = int(v)
            kw[k] = v
        return cls(**kw)
