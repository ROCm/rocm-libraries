# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The correctness gate: which kernels of a pool compute wrong results.

Every kernel of the pool (the library's table and the bank, `kernel_dat_info`
entries) is run by the exact harness (`tools/errmap.cpp`, built against the
run's hipBLASLt build when needed) on the shapes of `SHAPES`, picked by
solution index, in the config's layout (`ProblemType`): a padded leading
dimension exceeds the rows its matrix is stored with in that layout. The
harness fills A, B and C with small integers, so a correct kernel returns the
reference bit for bit and any other element is wrong. 8-bit and 4-bit A and
B with block-32 E8M0 scales (MX) hold the same integers, every scale of an
operand one power of 2, and run the shapes of `mx_shapes` instead: K a
multiple of 32, M not a multiple of 8, operands of 4 GiB or more at the K
their element size needs, and the bias epilogue on and off. 8-bit A and B
with one f32 scale per tensor (2 for A, 4 for B) run `SCALAR_SHAPES`: those
of SHAPES, whose K and alignment rules such kernels share, with the operands
over 4 GiB at the K 8-bit elements need, K of one iteration of DepthU 64 to
1024 and the bias epilogue on and off.
Shapes that trigger failures only some of the time run `trigger_repetitions`
times, the others `repetitions` times, each repetition a separate launch. On
top of that:

  first launches  a kernel the policy `first_launch` names (`custom`: the
                  hand-written ones, whose logic file entry says its kernel
                  was not generated; `all`; `none`) also runs as the only
                  GEMM of `repetitions` fresh processes, at K = its DepthU:
                  some kernels fail only on the first launch of a process;
  K round-up      a kernel a shape is not supported by (a hand-written
                  kernel that needs K to be a multiple of its DepthU) runs it
                  again with K rounded up to that multiple;
  follow-ups      when a kernel fails on a shape, the kernels of its family
                  (equal macro tile and matrix instruction) that passed it in
                  this gate run it again `trigger_repetitions` times.
  tiny shapes     M of 1 to 3 and N of 1 to 3 (`tiny_shapes_of`): a kernel
                  whose vector reads of A or B its predicates do not keep
                  within the operand at such a free size reads before or past
                  it there. Each runs with every operand at the start of an
                  allocation of its own behind an unmapped range (the
                  harness's --fresh, when it takes it, `harness_caps`), where
                  a read before an operand faults, which is a crash, instead
                  of landing in another buffer unseen. They are a stage of
                  their own (`TINY`, with its own protocol and cached
                  verdicts), so that a kernel with cached verdicts of the
                  other shapes runs only these.

Before the harness runs anything, the static check (`static_check`, on
unless `correctness.static_check` is false) compares every hand-written
kernel's predicates with what its source declares: TensileLite builds a
kernel written by hand with global-read vector widths of 1, and derives its
LeadingFree size predicates from those, while the kernel reads A and B as
wide as its custom.config says (GlobalReadVectorWidthA and B); along a
leading free dimension (M of a non-transposed A, N of a transposed B) a
smaller free size reads outside the operand. A kernel whose predicates admit
a free size below the width it declares on such a side is `unguarded`,
blocked and never run; the source is TensileLite's CustomKernels of the
hipBLASLt source the build was configured from, or
`correctness.custom_kernels`.

A harness process that prints nothing for `timeout_s` (before it is ready,
between two records, after its last one) hangs and is killed. When a process
crashes or hangs after it is ready, its culprit is the first kernel the
harness found (`ready` names them) that lacks a repetition; the culprit runs
again alone in a fresh process, and the kernels after it in another. Only a
crash or hang of that lone run blames the culprit; a process killed by a
signal from outside (an operator, the OOM killer) blames no kernel and stops
the gate.

A kernel fails (`fail`) when a repetition returned a wrong element, crashes
(`crash`) when it crashed or hung alone, passes when it ran at least once and
did neither, and is untested when no shape supported it or the harness did
not find it; its verdict is the worse of its two stages'. Failing, crashing
and unguarded kernels are blocked.

The harness runs without TensileLite's launch overrides (lib/bench_env.py),
so a kernel is checked as the runtime launches it in production, whatever
overrides a run's benches time under, and on the whole device in the
default schedule only: the reduced-CU and co-tenant contexts launch it with
other grids and are not checked.

Verdicts are cached (`VerdictCache`, `verdicts.jsonl` in the cache
directory, shared by every run that names it) by kernel id
(lib/kernel_identity.py), device (processor, PCI chip id and CU count) and
the digest of the library file that holds the kernel and of its code objects
(`code_digests`), so a rebuilt library is checked again. A cached failure
holds under any protocol, a later harness's too, since a kernel that returned
a wrong element may do so rarely and a check that missed it would serve it
(removing its records from the cache file checks it again); a cached crash,
pass or untested verdict only under the protocol it was made with
(`protocol`: the shapes, repetitions, first-launch policy and the harness),
and a crash of hangs alone only while the timeout is no longer than the one
it hung under (its `timeout_s`); a pass or untested verdict of the tiny
shapes not when one of its runs lacked fresh operands, where a read before an
operand may go unseen (`unfresh`). A verdict is cached as soon as no later
pass of the gate can change it, and a gate that stops early still caches the
failures it saw. Each line of the harness is checked as it arrives: a kernel
it names otherwise than the library's logic file does stops the gate at once
(HarnessMismatch), and so does a process that finds none of its kernels, but
a first launch: its one kernel is then untested, unless another process
found it. Kernels that a configured list or the hazard canary's quarantine
(lib/hazards.py) blocks already are not run. When every other kernel of the
pool has a verdict, the gate runs nothing and takes no GPU.
"""
from __future__ import annotations

import collections
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional
from typing import Sequence, Set, Tuple

from . import correctness as cx
from . import kernel_identity as ki
from . import procs
from .dat import library_logic_path, read_tensile_logic

PIPELINE_DIR = Path(__file__).resolve().parent.parent
HARNESS_SOURCE = PIPELINE_DIR / "tools" / "errmap.cpp"
VERDICTS_FILE = "verdicts.json"
SUMMARY_FILE = "gate_summary.json"
CACHE_FILE = "verdicts.jsonl"
FIRST_LAUNCH_POLICIES = ("custom", "all", "none")
DEFAULTS: Dict[str, Any] = {
    "repetitions": 3,
    "trigger_repetitions": 10,
    "first_launch": "custom",
    "timeout_s": 600.0,
}
MIN_REPETITIONS = 3
# The static check's verdict of a hand-written kernel whose predicates admit
# a free size below the vector width it reads that operand with.
UNGUARDED = "unguarded"
BLOCKING_VERDICTS = ("fail", "crash", UNGUARDED)
# The stages of the harness runs: the shapes of `shapes_of` (with the first
# launches), and the tiny shapes of `tiny_shapes_of`, each with a protocol
# and cached verdicts of its own.
MAIN = "main"
TINY = "tiny"
# Problem types the harness runs: hipblaslt-bench names of A/B/C/D types and
# of the compute type, as the harness's options take them.
HARNESS_TYPES = {"bf16_r": "bf16", "f32_r": "f32"}
HARNESS_COMPUTE = {"c_f32_r": "f32", "f32_r": "f32", "c_xf32_r": "xf32"}
# A and B types it runs with block-32 E8M0 scales (hipblaslt-bench scaling
# format MX_SCALING for scaleA and scaleB), as (option, element bits): E4M3,
# E5M2 and E2M1. The harness stores such an operand with K rounded up to a
# multiple of MX_STORED_K, zeros past K, which the scale tensor's padding
# multiplies.
HARNESS_MX_TYPES = {"f8_r": ("f8", 8), "bf8_r": ("bf8", 8), "f4_r": ("f4", 4)}
MX_SCALING = 3
MX_STORED_K = 256
# A and B types it runs with one f32 scale per tensor (hipblaslt-bench scaling
# format SCALAR_SCALING for scaleA and scaleB, the libraries' UseScaleAB
# Scalar): E4M3 and E5M2, stored with K as it is.
HARNESS_SCALAR_TYPES = {"f8_r": "f8", "bf8_r": "bf8"}
SCALAR_SCALING = 1
# A first launch runs on M x N with K = the kernel's DepthU, as the only GEMM
# of its process.
FIRST_LAUNCH_MN = (625, 77)
FIRST_LAUNCH_REPS = 1
# Seconds to wait for another process to leave a GPU before a task runs.
FOREIGN_WAIT_S = 600.0
POLL_S = 0.5


class GateError(RuntimeError):
    """The gate cannot check the pool."""


class HarnessMismatch(GateError):
    """The harness ran other kernels than the library's at those indices."""


@dataclass(frozen=True)
class Shape:
    """One problem of the gate. `pad_a` .. `pad_d`: how far the leading
    dimension of A, B, C and D exceeds the rows the matrix is stored with
    (0: by none); `bias`: the epilogue adds a bias vector, of the harness's
    type `bias_type` ("": the type hipBLASLt defaults to); `fresh`: every
    operand starts an allocation of its own behind an unmapped range (the
    harness's --fresh), a shape of the tiny stage (`TINY`)."""

    name: str
    m: int
    n: int
    k: int
    batch: int = 1
    alpha: float = 1.0
    beta: float = 0.0
    pad_a: int = 0
    pad_b: int = 0
    pad_c: int = 0
    pad_d: int = 0
    separate_d: bool = False
    trigger: bool = False
    bias: bool = False
    bias_type: str = ""
    fresh: bool = False

    def __post_init__(self) -> None:
        if self.bias_type and not self.bias:
            raise ValueError(f"shape {self.name}: a bias type without a bias")

    @property
    def stage(self) -> str:
        return TINY if self.fresh else MAIN

    def record(self) -> Dict[str, Any]:
        """The shape's fields, `bias`, `bias_type` and `fresh` only when set:
        the records of a shape without them stay those of the shapes before
        they existed."""
        out = asdict(self)
        for key in ("bias", "bias_type", "fresh"):
            if not out[key]:
                del out[key]
        return out

    def rows(
        self, trans_a: bool, trans_b: bool, stored_k: int = 1
    ) -> Tuple[int, int, int, int]:
        """The rows A, B, C and D are stored with (column major): A is K x M
        when transposed, else M x K; B is N x K when transposed, else K x N;
        C and D are M x N; K rounded up to a multiple of `stored_k`."""
        k = -(-self.k // stored_k) * stored_k
        return (
            k if trans_a else self.m,
            self.n if trans_b else k,
            self.m,
            self.m,
        )

    def args(
        self, trans_a: bool, trans_b: bool, stored_k: int = 1, fresh: bool = False
    ) -> List[str]:
        """The harness's arguments of the shape; with `fresh` (the harness
        takes --fresh) a fresh shape asks for its operands at fresh
        allocations."""
        out = [str(self.m), str(self.n), str(self.k), str(self.batch)]
        if self.alpha != 1.0:
            out += ["--alpha", repr(self.alpha)]
        if self.beta != 0.0:
            out += ["--beta", repr(self.beta)]
        pads = (self.pad_a, self.pad_b, self.pad_c, self.pad_d)
        for flag, rows, pad in zip(
            ("--lda", "--ldb", "--ldc", "--ldd"),
            self.rows(trans_a, trans_b, stored_k),
            pads,
        ):
            if pad:
                out += [flag, str(rows + pad)]
        if self.separate_d:
            out += ["--sep", "1"]
        if self.bias:
            out += ["--bias", "1"]
        if self.bias_type:
            out += ["--bias-type", self.bias_type]
        if self.fresh and fresh:
            out += ["--fresh", "1"]
        return out

    def text(self) -> str:
        t = f"{self.m}x{self.n}x{self.k}x{self.batch}"
        if self.alpha != 1.0 or self.beta != 0.0:
            t += f" alpha {self.alpha:g} beta {self.beta:g}"
        if self.separate_d:
            t += " C != D"
        if any((self.pad_a, self.pad_b, self.pad_c, self.pad_d)):
            t += " padded"
        if self.bias:
            t += " bias"
        if self.bias_type:
            t += f" {self.bias_type}"
        if self.fresh:
            t += " fresh operands"
        return t


# Edge tiles in M and N throughout (dimensions are rarely tile multiples),
# K not a multiple of DepthU unless the shape is about K, and launches the
# kernels' launch rules make data-parallel, persistent with several tiles
# per workgroup and Stream-K with split tiles.
SHAPES: Tuple[Shape, ...] = (
    # One partial M tile, persistent launches looping over many tiles.
    Shape("small_m_batched", 13, 65, 383, 2048, trigger=True),
    Shape("one_m_tile_batched", 64, 256, 887, 1000, trigger=True),
    Shape("small_m_long_n", 65, 131071, 4355, trigger=True),
    Shape("small_m_long_n_k_aligned", 65, 131071, 4096, trigger=True),
    Shape("tiny_m_batched", 2, 256, 887, 1000),
    # Long K over many tiles, data-parallel and split along K.
    Shape("long_k", 1024, 960, 8192),
    Shape("long_k_thin_batched", 33301, 3, 29715, 7),
    Shape("mid_k_tiles", 4096, 2560, 1000),
    Shape("wide_n", 257, 23199, 513),
    # Operands over 4 GiB.
    Shape("a_over_4gib", 524287, 3, 4355),
    Shape("b_over_4gib", 5, 524287, 4355),
    # alpha != 1 and beta != 0 with a partial last M tile; padded leading
    # dimensions; a separate D.
    Shape(
        "alpha_beta_padded",
        1537,
        515,
        517,
        alpha=1.5,
        beta=0.5,
        pad_a=9,
        pad_b=31,
        pad_c=7,
        pad_d=7,
    ),
    Shape(
        "alpha_beta_separate_d", 1537, 515, 517, alpha=1.5, beta=0.5, separate_d=True
    ),
    # One DepthU iteration of the most common DepthU.
    Shape("k_64", 625, 77, 64),
)

# The operand size the over-4-GiB shapes reach.
FOUR_GIB = 1 << 32


def k_over_4gib(rows: int, batch: int, bits: int, multiple: int = 32) -> int:
    """The least K, an odd multiple of `multiple`, at which `batch` operands
    of `rows` (M or N) x K elements of `bits` bits take FOUR_GIB bytes or
    more."""
    k = -(-FOUR_GIB * 8 // (rows * batch * bits))
    k = -(-k // multiple) * multiple
    return k + multiple if k // multiple % 2 == 0 else k


def mx_shapes(bits_a: int, bits_b: int) -> Tuple[Shape, ...]:
    """The shapes of 8- and 4-bit A and B (elements of `bits_a` and `bits_b`
    bits) with block-32 scales: those of SHAPES with K a multiple of 32, an
    odd one (no DepthU divides it) unless the shape is about K, M not a
    multiple of 8 and mostly prime, since a store vector width that does not
    divide M is where out-of-bounds stores would hide, the operands over 4 GiB
    unbatched and batched at the K their element size needs, A and B leading
    dimensions padded by whole scale blocks past the K the harness stores
    (even in every layout, so packed 4-bit columns start on a byte), and the
    bias epilogue on and off."""
    return (
        # One partial M tile, persistent launches looping over many tiles.
        Shape("small_m_batched", 13, 65, 416, 2048, trigger=True),
        Shape("small_m_batched_bias", 13, 65, 416, 2048, trigger=True, bias=True),
        Shape("one_m_tile_batched", 61, 256, 864, 1000, trigger=True),
        Shape("small_m_long_n", 67, 131071, 4384, trigger=True),
        Shape("small_m_long_n_k_aligned", 67, 131071, 4096, trigger=True),
        Shape("tiny_m_batched", 3, 256, 864, 1000),
        # Long K over many tiles, data-parallel and split along K.
        Shape("long_k", 1021, 960, 8224),
        Shape("long_k_thin_batched", 33301, 3, 29728, 7),
        Shape("mid_k_tiles", 4093, 2560, 992),
        Shape("wide_n", 257, 23199, 544),
        # Operands of 4 GiB or more: one matrix, and many small batches
        # whose offsets pass 4 GiB.
        Shape("a_over_4gib", 524287, 3, k_over_4gib(524287, 1, bits_a)),
        Shape("b_over_4gib", 5, 524287, k_over_4gib(524287, 1, bits_b)),
        Shape(
            "a_over_4gib_batched",
            257,
            3,
            k_over_4gib(257, 4100, bits_a),
            4100,
            bias=True,
        ),
        Shape("b_over_4gib_batched", 3, 257, k_over_4gib(257, 4100, bits_b), 4100),
        # alpha != 1 and beta != 0 with a partial last M tile; padded leading
        # dimensions; a separate D, with and without a bias.
        Shape(
            "alpha_beta_padded",
            1530,
            514,
            544,
            alpha=1.5,
            beta=0.5,
            pad_a=32,
            pad_b=64,
            pad_c=7,
            pad_d=7,
        ),
        Shape(
            "alpha_beta_separate_d",
            1531,
            515,
            544,
            alpha=1.5,
            beta=0.5,
            separate_d=True,
        ),
        Shape(
            "alpha_beta_separate_d_bias",
            1531,
            515,
            544,
            alpha=1.5,
            beta=0.5,
            separate_d=True,
            bias=True,
        ),
        # One DepthU iteration of the two most common DepthUs.
        Shape("k_128", 625, 77, 128),
        Shape("k_256", 625, 77, 256),
    )


# The shapes of 8-bit A and B with one f32 scale per tensor. Their kernels
# take any K, M and N and leading dimensions of any alignment, as bf16 and f32
# kernels do (their libraries assert no multiple of K, M or N and have no
# predicate on a leading dimension), so these are the problems of SHAPES (K
# not a multiple of a DepthU unless the shape is about K, odd pads) with what
# one-byte elements and the libraries' bias epilogue add: the operands over
# 4 GiB at the least odd K that takes them there, also as many small batches
# whose offsets pass 4 GiB; K of one iteration of each DepthU from 64 to
# 1024, and of 512 and 1024 also with an odd tail; and a bias vector over the
# rows of D, of bf16 (hipBLASLt's default for 8-bit A and B) and once of f32,
# both of which the kernels take, on a batched persistent shape, on
# alpha != 1 with beta != 0 and a separate D, and on the batched operand over
# 4 GiB, each also run without it.
SCALAR_SHAPES: Tuple[Shape, ...] = (
    # One partial M tile, persistent launches looping over many tiles.
    Shape("small_m_batched", 13, 65, 383, 2048, trigger=True),
    Shape("small_m_batched_bias", 13, 65, 383, 2048, trigger=True, bias=True),
    Shape("one_m_tile_batched", 64, 256, 887, 1000, trigger=True),
    Shape("small_m_long_n", 65, 131071, 4355, trigger=True),
    Shape("small_m_long_n_k_aligned", 65, 131071, 4096, trigger=True),
    Shape("tiny_m_batched", 2, 256, 887, 1000),
    # Long K over many tiles, data-parallel and split along K.
    Shape("long_k", 1024, 960, 8192),
    Shape("long_k_thin_batched", 33301, 3, 29715, 7),
    Shape("mid_k_tiles", 4096, 2560, 1000),
    Shape("wide_n", 257, 23199, 513),
    # Operands of 4 GiB or more: one matrix, and many small batches whose
    # offsets pass 4 GiB.
    Shape("a_over_4gib", 524287, 3, k_over_4gib(524287, 1, 8, 1)),
    Shape("b_over_4gib", 5, 524287, k_over_4gib(524287, 1, 8, 1)),
    Shape(
        "a_over_4gib_batched",
        257,
        3,
        k_over_4gib(257, 4100, 8, 1),
        4100,
        bias=True,
    ),
    Shape("b_over_4gib_batched", 3, 257, k_over_4gib(257, 4100, 8, 1), 4100),
    # alpha != 1 and beta != 0 with a partial last M tile; padded leading
    # dimensions; a separate D, without a bias and with either type of it.
    Shape(
        "alpha_beta_padded",
        1537,
        515,
        517,
        alpha=1.5,
        beta=0.5,
        pad_a=9,
        pad_b=31,
        pad_c=7,
        pad_d=7,
    ),
    Shape(
        "alpha_beta_separate_d", 1537, 515, 517, alpha=1.5, beta=0.5, separate_d=True
    ),
    Shape(
        "alpha_beta_separate_d_bias",
        1537,
        515,
        517,
        alpha=1.5,
        beta=0.5,
        separate_d=True,
        bias=True,
    ),
    Shape(
        "alpha_beta_separate_d_bias_f32",
        1537,
        515,
        517,
        alpha=1.5,
        beta=0.5,
        separate_d=True,
        bias=True,
        bias_type="f32",
    ),
    # One DepthU iteration of each DepthU from 64 to 1024 (those of gfx950's
    # F8BS TN kernels): no other K here is one iteration of DepthU 512 or runs
    # the loop over K of a DepthU-1024 kernel once. On the same M and N, K of
    # DepthU 512 and 1024 plus 7 runs that one iteration and then a tail of 7,
    # odd as every K here that is not about K, so that a kernel that fails
    # there and not on the plain one fails in its tail.
    Shape("k_64", 625, 77, 64),
    Shape("k_128", 625, 77, 128),
    Shape("k_256", 625, 77, 256),
    Shape("k_512", 625, 77, 512),
    Shape("k_512_tail", 625, 77, 519),
    Shape("k_1024", 625, 77, 1024),
    Shape("k_1024_tail", 625, 77, 1031),
)

# The free sizes of the tiny stage: M and N of 1 to 3, below the vector
# width of a kernel that reads A along M (A not transposed) or B along N (B
# transposed) four or more elements at a time, the other free size
# TINY_OTHER, a partial second tile of every macro tile up to 256.
TINY_SIZES = (1, 2, 3)
TINY_OTHER = 257


def tiny_shapes(k: int) -> Tuple[Shape, ...]:
    """The tiny stage's shapes at K `k`: M of 1 to 3 (`tiny_m_<M>`), where a
    kernel reading A along M is outside A by up to its width less M, then N
    of 1 to 3 (`tiny_n_<N>`) for B; every operand at a fresh allocation."""
    return tuple(
        Shape(f"tiny_m_{s}", s, TINY_OTHER, k, fresh=True) for s in TINY_SIZES
    ) + tuple(Shape(f"tiny_n_{s}", TINY_OTHER, s, k, fresh=True) for s in TINY_SIZES)


# K odd and prime, as in SHAPES and SCALAR_SHAPES; for block-scaled A and B an
# odd multiple of 32, as in `mx_shapes`.
TINY_SHAPES = tiny_shapes(383)
MX_TINY_SHAPES = tiny_shapes(416)


@dataclass(frozen=True)
class Settings:
    repetitions: int = DEFAULTS["repetitions"]
    trigger_repetitions: int = DEFAULTS["trigger_repetitions"]
    first_launch: str = DEFAULTS["first_launch"]
    timeout_s: float = DEFAULTS["timeout_s"]

    @classmethod
    def of(cls, cfg: Optional[Mapping[str, Any]]) -> "Settings":
        sec = dict((cfg or {}).get("correctness") or {})
        st = cls(
            **{k: type(DEFAULTS[k])(sec[k]) for k in DEFAULTS if sec.get(k) is not None}
        )
        if min(st.repetitions, st.trigger_repetitions) < MIN_REPETITIONS:
            raise GateError(
                f"correctness.repetitions and trigger_repetitions must be at least "
                f"{MIN_REPETITIONS}"
            )
        if st.first_launch not in FIRST_LAUNCH_POLICIES:
            raise GateError(
                f"correctness.first_launch must be one of {FIRST_LAUNCH_POLICIES}"
            )
        return st

    def reps(self, shape: Shape) -> int:
        return self.trigger_repetitions if shape.trigger else self.repetitions


@dataclass(frozen=True)
class ProblemType:
    """The config's problem type as the harness takes it: its type, compute
    and scale options, whether A and B are transposed, the element bits of A
    and B when they hold block-scaled 8- or 4-bit data (else 0), and whether
    they hold 8-bit data with one f32 scale per tensor."""

    options: Tuple[str, ...]
    trans_a: bool
    trans_b: bool
    mx_bits: Tuple[int, int] = (0, 0)
    scalar_scales: bool = False

    @property
    def stored_k(self) -> int:
        """The multiple of K the harness stores A and B with."""
        return MX_STORED_K if any(self.mx_bits) else 1

    def args(self) -> List[str]:
        return [
            *self.options,
            "--trans-a",
            "T" if self.trans_a else "N",
            "--trans-b",
            "T" if self.trans_b else "N",
        ]


def harness_problem(cfg: Mapping[str, Any]) -> ProblemType:
    """The config's problem type for the harness. GateError for a type the
    harness does not run: A and B of bf16 or f32 without scales, f8 or bf8
    with one f32 scale per tensor (scaleA and scaleB SCALAR_SCALING), or f8,
    bf8 or f4 with block-32 E8M0 scales (scaleA and scaleB MX_SCALING); C and
    D of bf16 or f32."""
    h = dict(cfg.get("hipblaslt") or {})
    scales = (int(h.get("scaleA") or 0), int(h.get("scaleB") or 0))
    mx = scales == (MX_SCALING, MX_SCALING)
    scalar = scales == (SCALAR_SCALING, SCALAR_SCALING)
    if any(scales) and not (mx or scalar):
        raise GateError(
            f"the exact harness does not run scaled inputs (scaleA/scaleB "
            f"{scales[0]}/{scales[1]}) other than one f32 scale per tensor "
            f"({SCALAR_SCALING}) or block-32 E8M0 scales ({MX_SCALING}) of both "
            f"A and B"
        )
    out: List[str] = []
    bits: List[int] = []
    for key, flag in (("a_type", "--a-type"), ("b_type", "--b-type")):
        name = str(h.get(key))
        if mx:
            found = HARNESS_MX_TYPES.get(name)
            if found is None:
                raise GateError(
                    f"the exact harness does not run hipblaslt.{key} {h.get(key)!r} "
                    f"with block-32 scales (it runs {sorted(HARNESS_MX_TYPES)})"
                )
            out += [flag, found[0]]
            bits.append(found[1])
            continue
        if scalar:
            t = HARNESS_SCALAR_TYPES.get(name)
            if t is None:
                raise GateError(
                    f"the exact harness does not run hipblaslt.{key} {h.get(key)!r} "
                    f"with one f32 scale per tensor (it runs "
                    f"{sorted(HARNESS_SCALAR_TYPES)})"
                )
            out += [flag, t]
            continue
        t = HARNESS_TYPES.get(name)
        if t is None:
            raise GateError(
                f"the exact harness does not run hipblaslt.{key} {h.get(key)!r} "
                f"(it runs {sorted(HARNESS_TYPES)}; {sorted(HARNESS_SCALAR_TYPES)} "
                f"with one f32 scale per tensor, scaleA and scaleB "
                f"{SCALAR_SCALING}; and {sorted(HARNESS_MX_TYPES)} with block-32 "
                f"E8M0 scales, scaleA and scaleB {MX_SCALING})"
            )
        out += [flag, t]
    for key, flag in (("c_type", "--c-type"), ("d_type", "--d-type")):
        t = HARNESS_TYPES.get(str(h.get(key)))
        if t is None:
            raise GateError(
                f"the exact harness does not run hipblaslt.{key} {h.get(key)!r} "
                f"(it runs {sorted(HARNESS_TYPES)})"
            )
        out += [flag, t]
    compute = HARNESS_COMPUTE.get(str(h.get("compute_type")))
    if compute is None:
        raise GateError(
            f"the exact harness does not run compute type {h.get('compute_type')!r}"
        )
    if h.get("c_type") != h.get("d_type"):
        raise GateError("the exact harness needs equal C and D types")
    out += ["--compute", compute]
    if any(scales):
        out += ["--scale-a", str(scales[0]), "--scale-b", str(scales[1])]
    return ProblemType(
        tuple(out),
        str(h.get("transA", "T")) == "T",
        str(h.get("transB", "N")) == "T",
        (bits[0], bits[1]) if mx else (0, 0),
        scalar,
    )


def shapes_of(problem: Optional[ProblemType]) -> Tuple[Shape, ...]:
    """The shapes the gate runs for a problem type: `mx_shapes` for
    block-scaled A and B, SCALAR_SHAPES for A and B with one f32 scale per
    tensor, else (also with no problem type the harness runs) SHAPES."""
    if problem is not None and any(problem.mx_bits):
        return mx_shapes(*problem.mx_bits)
    if problem is not None and problem.scalar_scales:
        return SCALAR_SHAPES
    return SHAPES


def tiny_shapes_of(problem: Optional[ProblemType]) -> Tuple[Shape, ...]:
    """The tiny stage's shapes for a problem type: MX_TINY_SHAPES for
    block-scaled A and B, else TINY_SHAPES."""
    if problem is not None and any(problem.mx_bits):
        return MX_TINY_SHAPES
    return TINY_SHAPES


def protocol(
    settings: Settings,
    harness_id: str,
    shapes: Sequence[Shape] = SHAPES,
    stage: str = MAIN,
) -> str:
    """Digest of what a verdict of a stage means: shapes, repetitions,
    first-launch policy and harness; a stage other than MAIN is named in
    it, so that the two stages' verdicts never stand for each other."""
    body: Dict[str, Any] = {
        "shapes": [s.record() for s in shapes],
        "repetitions": settings.repetitions,
        "trigger_repetitions": settings.trigger_repetitions,
        "first_launch": settings.first_launch,
        "first_launch_mn": FIRST_LAUNCH_MN,
        "first_launch_reps": FIRST_LAUNCH_REPS,
        "harness": harness_id,
    }
    if stage != MAIN:
        body["stage"] = stage
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def device_key(device: Any) -> str:
    """`processor:chip:cus` of a `lib.bank_pool.BankDevice`."""
    chip = "none" if device.chip_id is None else f"{int(device.chip_id):04x}"
    return f"{device.processor}:{chip}:{int(device.cu_count)}"


def code_digests(dat_dir: Any, stems: Iterable[str]) -> Dict[str, str]:
    """{stem: digest of its logic file, its code object `<stem>.co` (or
    `.hsaco`) and the directory's `Kernels.so-*.hsaco` (hand-written and
    helper kernels)}, whichever exist."""
    d = Path(dat_dir)
    helpers = sorted(p for p in d.glob("Kernels.so-*.hsaco") if p.is_file())
    out: Dict[str, str] = {}
    for stem in stems:
        files = [library_logic_path(d, stem)]
        files += [p for p in (d / f"{stem}.co", d / f"{stem}.hsaco") if p.is_file()]
        h = hashlib.sha256()
        for p in files + helpers:
            h.update(f"{p.name}\t{sha256_file(p)}\n".encode())
        out[stem] = h.hexdigest()[:16]
    return out


def hand_written(dat_dir: Any, stems: Iterable[str]) -> Set[int]:
    """Solution indices of the stems' logic files whose kernel was written
    by hand (`customKernel` with `generated` false)."""
    out: Set[int] = set()
    for stem in stems:
        data = read_tensile_logic(library_logic_path(dat_dir, stem)) or {}
        for sol in data.get("solutions") or []:
            ck = sol.get("customKernel") or {}
            if ck.get("name") and ck.get("generated") is False:
                out.add(int(sol["index"]))
    return out


# ── the static check of hand-written kernels ─────────────────────────────────

# Where TensileLite keeps the sources of hand-written kernels, under the
# hipBLASLt source tree.
CUSTOM_KERNELS_DIR = Path("tensilelite") / "Tensile" / "CustomKernels"
# The operands the static check compares: (operand, free size, the index of
# that free size, the custom.config key of its global-read vector width).
STATIC_SIDES = (
    ("A", "M", 0, "GlobalReadVectorWidthA"),
    ("B", "N", 1, "GlobalReadVectorWidthB"),
)


def custom_kernel_name(sol: Mapping[str, Any]) -> Optional[str]:
    """The name of a logic file solution's hand-written kernel: its
    `customKernel`'s, unless that says it was generated (an absent
    `generated` reads as false, as TensileLite reads it); None for a
    generated kernel."""
    ck = sol.get("customKernel")
    if not isinstance(ck, Mapping) or not ck.get("name"):
        return None
    if ck.get("generated") is True:
        return None
    return str(ck["name"])


def custom_kernels_dir(
    cfg: Optional[Mapping[str, Any]], build_dir: Optional[Path]
) -> Tuple[Optional[Path], str]:
    """(the directory of the hand-written kernels' sources, where it comes
    from): `correctness.custom_kernels`, else TensileLite's CustomKernels in
    the hipBLASLt source the build was configured from; (None, why) when
    there is none."""
    given = ((cfg or {}).get("correctness") or {}).get("custom_kernels")
    if given:
        d = Path(str(given))
        if d.is_dir():
            return d, "correctness.custom_kernels"
        return None, f"correctness.custom_kernels {d} is no directory"
    src = hipblaslt_source(Path(build_dir)) if build_dir is not None else None
    if src is None:
        return None, "the hipBLASLt build names no source tree (its CMakeCache.txt)"
    d = src / CUSTOM_KERNELS_DIR
    if not d.is_dir():
        return None, f"the hipBLASLt source of the build has no {CUSTOM_KERNELS_DIR}"
    return d, "TensileLite's CustomKernels of the build's hipBLASLt source"


def custom_kernel_sources(directory: Path) -> Dict[str, Path]:
    """{kernel name: its `<name>.s`} below `directory`, as TensileLite finds
    a kernel's file: at the top of the directory, else the first of a sorted
    walk."""
    out: Dict[str, Path] = {}
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".s"):
                top = Path(root) == Path(directory)
                if top or f[:-2] not in out:
                    out[f[:-2]] = Path(root) / f
    return out


def custom_config(path: Path) -> Dict[str, Any]:
    """The `custom.config` of a hand-written kernel's source: the YAML
    between the `---` and `...` lines of its .amdgpu_metadata, as TensileLite
    reads it. ValueError when it has none."""
    import yaml

    lines: List[str] = []
    inside = False
    for line in Path(path).read_text(errors="replace").splitlines():
        if line == "---":
            inside = True
            continue
        if line == "..." and inside:
            break
        if inside:
            lines.append(line)
    try:
        doc = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: its metadata is not YAML ({e})") from None
    config = doc.get("custom.config") if isinstance(doc, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{path} has no custom.config in its .amdgpu_metadata")
    return config


def predicate_leaves(node: Any) -> List[Mapping[str, Any]]:
    """The predicates of a solution's `problemPredicate` that hold whatever
    else does: those of its And nodes, depth first."""
    out: List[Mapping[str, Any]] = []
    stack: List[Any] = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, Mapping):
            continue
        if n.get("type") == "And":
            stack.extend(reversed(list(n.get("value") or [])))
        else:
            out.append(n)
    return out


def free_size_floor(
    predicates: Iterable[Mapping[str, Any]], index: int
) -> Tuple[int, List[str]]:
    """(the least free size of `index` (0: M, 1: N) the predicates admit, at
    least 1; the predicates that bound it, as `<type> <value>`):
    LeadingFree<i>SizesGreaterOrEqual and Free<i>SizeMultiple by their name,
    SizeEqual, SizeMultiple and SizeGreaterThan by their index."""
    floor, by = 1, []
    for p in predicates:
        kind, value = str(p.get("type")), p.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if kind in (
            f"LeadingFree{index}SizesGreaterOrEqual",
            f"Free{index}SizeMultiple",
        ):
            low = value
        elif kind in ("SizeEqual", "SizeMultiple") and p.get("index") == index:
            low = value
        elif kind == "SizeGreaterThan" and p.get("index") == index:
            low = value + 1
        else:
            continue
        by.append(f"{kind} {value}")
        floor = max(floor, low)
    return floor, by


@dataclass
class StaticFinding:
    """The static check of one hand-written kernel: `status` unguarded (its
    predicates admit a leading free size below the width it reads that
    operand with: blocked), guarded, undeclared (its custom.config gives no
    width for an operand read along its leading free dimension) or unchecked
    (no source or custom.config); per operand what it compared (`sides`);
    the source file and its sha256."""

    sid: int
    name: str
    status: str
    reason: str = ""
    source: Optional[str] = None
    sha256: Optional[str] = None
    sides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.status == UNGUARDED

    def record(self) -> Dict[str, Any]:
        return asdict(self)


def check_custom_kernel(
    sid: int,
    sol: Mapping[str, Any],
    name: str,
    source: Optional[Path],
    trans: Tuple[bool, bool] = (True, False),
    missing: str = "",
) -> StaticFinding:
    """The static check of the hand-written kernel `name` of solution `sol`
    against its source `source` (None: `missing` says why there is none):
    along a leading free dimension (M of a non-transposed A, N of a
    transposed B; the solution's problem type, else `trans`) the least free
    size its predicates admit must reach the global-read vector width its
    custom.config declares for that operand."""
    if source is None:
        return StaticFinding(sid, name, "unchecked", missing or f"no {name}.s found")
    try:
        config = custom_config(source)
        digest = sha256_file(source)
    except (OSError, ValueError) as e:
        return StaticFinding(sid, name, "unchecked", str(e), str(source))
    pt = sol.get("problemType") or {}
    trans_a = bool(pt["transA"]) if "transA" in pt else trans[0]
    trans_b = bool(pt["transB"]) if "transB" in pt else trans[1]
    built = sol.get("sizeMapping") or {}
    preds = predicate_leaves(sol.get("problemPredicate"))
    sides: Dict[str, Dict[str, Any]] = {}
    unguarded: List[str] = []
    undeclared: List[str] = []
    for op, dim, index, key in STATIC_SIDES:
        raw = config.get(key)
        width = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        width = width if width is not None and width > 0 else None
        floor, by = free_size_floor(preds, index)
        leading = not trans_a if op == "A" else trans_b
        sides[op] = {
            "free_size": dim,
            "leading_free": leading,
            "declared_width": width,
            "built_width": built.get(f"grvw{op}"),
            "floor": floor,
            "predicates": by,
        }
        if not leading:
            continue
        if width is None:
            undeclared.append(op)
        elif width > floor:
            parts = ", ".join(by) or "no size predicate"
            low = f"{dim} of {floor}" + (
                f" to {width - 1}" if width - 1 > floor else ""
            )
            unguarded.append(
                f"its predicates ({parts}) admit {low}, below the global-read "
                f"vector width {width} its custom.config declares for {op}, which it "
                f"reads along {dim}: such a {dim} reads outside {op}"
            )
    if unguarded:
        status = UNGUARDED
        reason = (
            "unguarded in the static check: "
            + "; ".join(unguarded)
            + " (TensileLite builds a hand-written kernel with global-read vector "
            "widths of 1 and derives its LeadingFree predicates from them)"
        )
    elif undeclared:
        status = "undeclared"
        reason = (
            f"its custom.config declares no global-read vector width for "
            f"{' and '.join(undeclared)}, read along a leading free dimension"
        )
    else:
        status, reason = "guarded", ""
    return StaticFinding(sid, name, status, reason, str(source), digest, sides)


def static_check(
    dat_dir: Any,
    stems: Iterable[str],
    sids: Iterable[int],
    sources: Optional[Mapping[str, Path]],
    trans: Tuple[bool, bool] = (True, False),
    missing: str = "",
) -> Dict[int, StaticFinding]:
    """{solution: StaticFinding} of the hand-written kernels among `sids` in
    the stems' logic files (`check_custom_kernel`), each found in `sources`
    ({kernel name: source}, `custom_kernel_sources`; None: there are none,
    `missing` says why)."""
    wanted = {int(s) for s in sids}
    out: Dict[int, StaticFinding] = {}
    for stem in stems:
        data = read_tensile_logic(library_logic_path(dat_dir, stem)) or {}
        for sol in data.get("solutions") or []:
            sid = int(sol.get("index", -1))
            name = custom_kernel_name(sol)
            if sid not in wanted or sid in out or name is None:
                continue
            src = sources.get(name) if sources is not None else None
            why = missing if sources is None else f"no {name}.s among the sources"
            out[sid] = check_custom_kernel(sid, sol, name, src, trans, why)
    return out


def harness_caps(
    path: Path, env: Optional[Mapping[str, str]] = None, timeout_s: float = 60.0
) -> Set[str]:
    """The options the harness at `path` takes beyond the original ones, as
    its `--caps` lists them (it runs no HIP call); none for a harness built
    before --caps existed, which refuses it."""
    try:
        proc = subprocess.run(
            [str(path), "--caps"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if proc.returncode != 0:
        return set()
    for line in reversed(proc.stdout.splitlines()):
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("caps"), list):
            return {str(c) for c in doc["caps"]}
    return set()


# ── the verdict cache ────────────────────────────────────────────────────────


def unfresh(rec: Mapping[str, Any]) -> bool:
    """Whether a cached verdict is a pass or untested verdict of the tiny
    stage whose runs did not all have their operands at fresh allocations
    (`evidence.fresh` false: the harness took no --fresh, or the device or
    a process could not place them so), where a read before an operand may
    land in another buffer unseen, or whose evidence does not say (no
    `fresh`). A later gate does not take it; a crash or a failure stands
    wherever it happened."""
    if rec.get("stage") != TINY or rec.get("verdict") not in ("pass", "untested"):
        return False
    evidence = rec.get("evidence")
    fresh = evidence.get("fresh", False) if isinstance(evidence, Mapping) else False
    return fresh is False


class VerdictCache:
    """Append-only verdict records in `<dir>/verdicts.jsonl`, one JSON
    object per line, written under an exclusive lock so that runs sharing
    the directory can add to it at once."""

    def __init__(self, directory: Any) -> None:
        self.dir = Path(directory)
        self.path = self.dir / CACHE_FILE
        self._records: Optional[List[Dict[str, Any]]] = None

    def records(self) -> List[Dict[str, Any]]:
        if self._records is None:
            out = []
            if self.path.is_file():
                for line in self.path.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("kernel_id"):
                        out.append(rec)
            self._records = out
        return self._records

    def lookup(
        self,
        kid: str,
        device: str,
        code: str,
        proto: str,
        timeout_s: Optional[float] = None,
        fresh: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """The verdict that holds for a kernel: a failure under any
        protocol, else a crash, pass or untested verdict under `proto`; a
        crash of hangs alone (its `timeout_s`) not when `timeout_s` is
        longer, since the kernel may finish within it; with `fresh`, a
        tiny-stage pass or untested verdict made without fresh operands
        not (`unfresh`)."""
        found = None
        for rec in self.records():
            if (rec["kernel_id"], rec.get("device"), rec.get("code")) != (
                kid,
                device,
                code,
            ):
                continue
            if rec.get("verdict") == "fail":
                return rec
            if rec.get("protocol") != proto or found is not None:
                continue
            hung = rec.get("timeout_s")
            if hung is not None and timeout_s is not None and timeout_s > hung:
                continue
            if fresh and unfresh(rec):
                continue
            found = rec
        return found

    def add(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(dict(r), sort_keys=True) + "\n" for r in records)
        with open(self.path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._records = None


# ── the harness ──────────────────────────────────────────────────────────────


def rocm_path() -> Path:
    return Path(os.environ.get("ROCM_PATH") or "/opt/rocm")


def default_compiler() -> str:
    """ROCm's clang driver, else hipcc on PATH. The harness has no device
    code: it only calls the HIP and hipBLASLt host APIs."""
    clang = rocm_path() / "bin" / "amdclang++"
    if clang.is_file():
        return str(clang)
    return shutil.which("hipcc") or "hipcc"


def hipblaslt_source(build_dir: Path) -> Optional[Path]:
    """The hipBLASLt source tree a build was configured from (its
    CMakeCache.txt)."""
    cache = Path(build_dir) / "CMakeCache.txt"
    if not cache.is_file():
        return None
    found: Dict[str, str] = {}
    for line in cache.read_text(errors="replace").splitlines():
        for key in ("hipblaslt_SOURCE_DIR", "CMAKE_HOME_DIRECTORY"):
            if line.startswith(key + ":"):
                found[key] = line.split("=", 1)[1].strip()
    src = found.get("hipblaslt_SOURCE_DIR") or found.get("CMAKE_HOME_DIRECTORY")
    return Path(src) if src else None


def build_command(compiler: str, build_dir: Path, source: Path, out: Path) -> List[str]:
    """The command that builds the harness against the hipBLASLt build in
    `build_dir`, finding its library there at run time."""
    rocm = rocm_path()
    lib = Path(build_dir) / "library"
    src = hipblaslt_source(build_dir)
    includes = [f"-I{rocm / 'include'}"]
    if src is not None:
        includes.append(f"-I{src / 'library' / 'include'}")
    includes.append(f"-I{lib / 'include'}")
    rpath = ":".join(str(p) for p in (lib, rocm / "lib", rocm / "lib" / "llvm" / "lib"))
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-fopenmp",
        "-D__HIP_PLATFORM_AMD__",
        *includes,
        str(source),
        f"-L{lib}",
        "-lhipblaslt",
        f"-L{rocm / 'lib'}",
        "-lamdhip64",
        f"-Wl,-rpath,{rpath}",
        "-o",
        str(out),
    ]


@dataclass
class Harness:
    path: Path
    digest: str
    built: bool = False
    command: Optional[List[str]] = None


def host_digest(build_dir: Optional[Path]) -> str:
    """Digest of the hipBLASLt host libraries of a build (the launch logic
    every kernel runs under)."""
    if build_dir is None:
        return ""
    files = sorted(
        p
        for pattern in (
            "library/libhipblaslt.so*",
            "tensilelite/libtensilelite-host.so*",
        )
        for p in Path(build_dir).glob(pattern)
        if p.is_file() and not p.is_symlink()
    )
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.name}\t{sha256_file(p)}\n".encode())
    return h.hexdigest()[:16]


def harness_digest(prebuilt: Optional[Path], build_dir: Optional[Path]) -> str:
    """What a harness's verdicts depend on besides the kernels' own code: a
    prebuilt binary's bytes or the harness source, and the build's host
    libraries."""
    own = sha256_file(Path(prebuilt) if prebuilt is not None else HARNESS_SOURCE)
    body = f"{own}\n{host_digest(build_dir)}\n"
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def harness(
    cache_dir: Path,
    build_dir: Path,
    prebuilt: Optional[Path] = None,
    compiler: Optional[str] = None,
    log: Optional[Path] = None,
) -> Harness:
    """The harness binary: `prebuilt`, or the one built against `build_dir`
    in `<cache_dir>/harness/<digest>/`, built now when it is not there."""
    if prebuilt is not None:
        if not Path(prebuilt).is_file():
            raise GateError(f"correctness.harness {prebuilt} does not exist")
        return Harness(Path(prebuilt), harness_digest(Path(prebuilt), build_dir))
    compiler = compiler or default_compiler()
    digest = harness_digest(None, build_dir)
    out = Path(cache_dir) / "harness" / digest / "errmap"
    if out.is_file():
        return Harness(out, digest)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".errmap.{os.getpid()}")
    cmd = build_command(compiler, build_dir, HARNESS_SOURCE, tmp)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise GateError(f"cannot run the compiler {compiler}: {e}") from None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(" ".join(cmd) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0 or not tmp.is_file():
        raise GateError(
            f"building the harness failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[-600:]}"
        )
    os.replace(tmp, out)
    return Harness(out, digest, built=True, command=cmd)


# ── tasks ────────────────────────────────────────────────────────────────────


@dataclass
class Task:
    """One harness process: `shape` on the solutions `sids` (in order),
    `reps` repetitions each; `alone` runs the culprit of a crash again on
    its own, `first_launch` a kernel as the only GEMM of its process."""

    tag: str
    shape: Shape
    sids: List[int]
    reps: int
    alone: bool = False
    first_launch: bool = False

    @property
    def stage(self) -> str:
        return self.shape.stage

    def command(
        self, harness: Path, problem: ProblemType, fresh: bool = False
    ) -> List[str]:
        """The harness's command; `fresh` when the harness takes --fresh."""
        return [
            str(harness),
            *self.shape.args(
                problem.trans_a, problem.trans_b, problem.stored_k, fresh=fresh
            ),
            "--ids",
            ",".join(str(s) for s in self.sids),
            "--reps",
            str(self.reps),
            "--maxlist",
            "8",
            *problem.args(),
        ]


def chunked(sids: Sequence[int], parts: int) -> List[List[int]]:
    """`sids` in at most `parts` runs of equal length, in order."""
    if not sids:
        return []
    size = max(1, math.ceil(len(sids) / max(1, parts)))
    return [list(sids[i : i + size]) for i in range(0, len(sids), size)]


def main_tasks(
    sids: Sequence[int],
    settings: Settings,
    workers: int,
    shapes: Sequence[Shape] = SHAPES,
) -> List[Task]:
    """Every shape of `shapes` on `sids`, in two tasks per worker each."""
    out = []
    for shape in shapes:
        for part in chunked(sids, 2 * workers):
            out.append(Task(shape.name, shape, part, settings.reps(shape)))
    return out


def first_launch_tasks(
    kernels: Mapping[int, Mapping[str, Any]], sids: Iterable[int], settings: Settings
) -> List[Task]:
    """`settings.repetitions` processes per kernel of `sids`, each launching
    it FIRST_LAUNCH_REPS times at K = its DepthU: only the first GEMM of a
    process is a first launch."""
    out = []
    m, n = FIRST_LAUNCH_MN
    for sid in sids:
        du = max(1, int(kernels[sid]["mt_k"]))
        shape = Shape(f"first_launch_k{du}", m, n, du)
        out += [
            Task(shape.name, shape, [int(sid)], FIRST_LAUNCH_REPS, first_launch=True)
            for _ in range(settings.repetitions)
        ]
    return out


def roundup_tasks(
    unsupported: Mapping[str, Set[int]],
    kernels: Mapping[int, Mapping[str, Any]],
    settings: Settings,
    shapes: Sequence[Shape] = SHAPES,
) -> List[Task]:
    """Each shape of `shapes` again, with K rounded up to a multiple of each
    kernel's DepthU, for the kernels it did not support; the leading
    dimensions follow K wherever the layout stores K as rows."""
    by_name = {s.name: s for s in shapes}
    out = []
    for name, sids in sorted(unsupported.items()):
        shape = by_name.get(name)
        if shape is None:
            continue
        groups: Dict[int, List[int]] = collections.defaultdict(list)
        for sid in sorted(sids):
            du = max(1, int(kernels[sid]["mt_k"]))
            if shape.k % du:
                groups[math.ceil(shape.k / du) * du].append(sid)
        for k, group in sorted(groups.items()):
            grown = replace(shape, name=f"{shape.name}_k{k}", k=k)
            out.append(Task(grown.name, grown, group, settings.reps(shape)))
    return out


def family(kernel: Mapping[str, Any]) -> Tuple[int, ...]:
    """Macro tile and matrix instruction: the kernels a failure is followed
    up on."""
    return tuple(
        int(kernel[f]) for f in ("mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k")
    )


def followup_tasks(
    failing: Mapping[str, Set[int]],
    passed: Mapping[str, Set[int]],
    kernels: Mapping[int, Mapping[str, Any]],
    shapes: Mapping[str, Shape],
    settings: Settings,
) -> List[Task]:
    """For each shape some kernels failed, the kernels of their families
    that passed it, `trigger_repetitions` times."""
    out = []
    for name, bad in sorted(failing.items()):
        fams = {family(kernels[s]) for s in bad}
        again = sorted(
            s
            for s in passed.get(name, ())
            if s not in bad and family(kernels[s]) in fams
        )
        if again and name in shapes:
            out.append(
                Task(
                    f"{name}_followup",
                    shapes[name],
                    again,
                    settings.trigger_repetitions,
                )
            )
    return out


# ── running ──────────────────────────────────────────────────────────────────


@dataclass
class Outcome:
    """What the harness reported: per solution its repetition records,
    crashes and unsupported shapes, every record with its task's `tag`."""

    records: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "Outcome") -> None:
        self.records.extend(other.records)
        self.tasks.extend(other.tasks)


# A value of a harness line as printf writes a NaN or an infinity (an element
# a solution did not write keeps a NaN pattern), which JSON has no word for.
_NONFINITE = re.compile(r"(?<=[\[,:])(\s*)(-?)(nan|inf)(?=\s*[,\]}])")


def _finite_json(match: "re.Match[str]") -> str:
    if match[3] == "nan":
        return f"{match[1]}NaN"
    return f"{match[1]}{match[2]}Infinity"


def _parse_line(line: str) -> Optional[Dict[str, Any]]:
    """The harness's `ready` record or a solution's record on one line of
    its output, NaN and infinite values read as such; None for any other
    line."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        rec = json.loads(line)
    except ValueError:
        try:
            rec = json.loads(_NONFINITE.sub(_finite_json, line))
        except ValueError:
            return None
    if isinstance(rec, dict) and (rec.get("ready") is True or "sid" in rec):
        return rec
    return None


def _pump(stream: Any, put: Callable[[Optional[str]], None], end: bool) -> None:
    try:
        for line in stream:
            put(line)
    except (OSError, ValueError):
        pass
    finally:
        if end:
            put(None)


def run_task(
    task: Task,
    cmd_of: Callable[[Task], List[str]],
    env: Mapping[str, str],
    timeout_s: float,
    cpus: Optional[Set[int]] = None,
    check: Optional[Callable[[Mapping[str, Any], Task], None]] = None,
    stop: Optional[threading.Event] = None,
    children: Optional[procs.ChildGroups] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Task]]:
    """(records, the task's record, the tasks to run next) of one harness
    process.

    The process hangs when it prints no line for `timeout_s`: before it is
    ready, between two records, or after its last one; it is then killed.
    `check` sees the ready record and every solution's record as it arrives
    and stops the process by raising; `stop` stops it from outside
    (GateError). The records of a tiny-stage task, and its own, name the
    stage and whether its operands were at fresh allocations (the ready
    line's `fresh`). A process that exits non-zero, dies or hangs after it
    reported ready makes its culprit, the first solution the harness found
    without all its repetitions, a crash: confirmed when the task runs the
    culprit `alone`, else to be confirmed by the task that runs it alone,
    which comes next, before a task of the solutions after it. Before it is
    ready no kernel ran, and a process killed by a signal from outside
    (`procs.KILL_SIGNALS`) blames none: both are a GateError."""

    def pin() -> None:
        if cpus:
            os.sched_setaffinity(0, cpus)

    t0 = time.time()
    kw: Dict[str, Any] = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
        preexec_fn=pin,
    )
    if children is not None:
        proc = children.popen(cmd_of(task), **kw)
    else:
        proc = subprocess.Popen(cmd_of(task), start_new_session=True, **kw)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    err: Deque[str] = collections.deque(maxlen=200)
    readers = [
        threading.Thread(target=_pump, args=(proc.stdout, lines.put, True)),
        threading.Thread(target=_pump, args=(proc.stderr, err.append, False)),
    ]
    for r in readers:
        r.daemon = True
        r.start()
    records: List[Dict[str, Any]] = []
    ready: Optional[Dict[str, Any]] = None
    timed_out = stopped = False

    def stamp(rec: Dict[str, Any]) -> None:
        if task.stage != MAIN:
            rec["stage"] = task.stage
            rec["fresh"] = bool((ready or {}).get("fresh"))

    problem: Optional[BaseException] = None
    last = time.monotonic()
    while True:
        if stop is not None and stop.is_set():
            stopped = True
            break
        left = last + timeout_s - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        try:
            line = lines.get(timeout=min(POLL_S, left))
        except queue.Empty:
            continue
        if line is None:
            break
        last = time.monotonic()
        rec = _parse_line(line)
        if rec is None:
            continue
        if rec.get("ready") is True:
            ready = rec
        else:
            rec["tag"] = task.tag
            stamp(rec)
            records.append(rec)
        if check is not None:
            try:
                check(rec, task)
            except BaseException as e:  # noqa: BLE001
                problem = e
                break
    if timed_out or stopped or problem is not None:
        procs.kill_group(proc)
    else:
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            procs.kill_group(proc)
    for r in readers:
        r.join(timeout=30)
    if problem is not None:
        raise problem
    if stopped:
        raise GateError(f"{task.tag}: stopped")
    done: Dict[int, int] = collections.Counter()
    for r in records:
        done[int(r["sid"])] += task.reps if r.get("supported") is False else 1
    found = (
        [int(s) for s in ready["sids"]]
        if ready is not None and isinstance(ready.get("sids"), list)
        else list(task.sids)
    )
    rec = {
        "tag": task.tag,
        "shape": task.shape.text(),
        "n_sids": len(task.sids),
        "reps": task.reps,
        "alone": task.alone,
        "first_launch": task.first_launch,
        "rc": proc.returncode,
        "timeout": timed_out,
        "wall_s": round(time.time() - t0, 3),
        "n_records": len(records),
    }
    not_found = sorted(set(task.sids) - set(found))
    if not_found:
        rec["not_found"] = not_found
    if ready is not None:
        stamp(rec)
    following: List[Task] = []
    if proc.returncode != 0 or timed_out:
        rec["stderr"] = "".join(err)[-1200:]
        if ready is None:
            raise GateError(
                f"the harness failed before it ran a kernel ({task.tag}, rc="
                f"{proc.returncode}{', timed out' if timed_out else ''}): "
                f"{''.join(err).strip()[-600:]}"
            )
        sig = None if timed_out else procs.killed_by(proc.returncode)
        if sig is not None:
            raise GateError(
                f"the harness running {task.tag} was killed by {sig.name} from "
                f"outside; no kernel is blamed for it"
            )
        culprit = next((s for s in found if done.get(s, 0) < task.reps), None)
        if culprit is not None:
            crash = {
                "tag": task.tag,
                "sid": culprit,
                "crash": True,
                "confirmed": task.alone,
                "rc": proc.returncode,
                "timeout": timed_out,
            }
            stamp(crash)
            records.append(crash)
            if not task.alone:
                following.append(replace(task, sids=[culprit], alone=True))
            after = found[found.index(culprit) + 1 :]
            if after:
                following.append(replace(task, sids=after, alone=False))
    return records, rec, following


def run_tasks(
    tasks: Sequence[Task],
    cmd_of: Callable[[Task], List[str]],
    gpus: Sequence[int],
    env: Mapping[str, str],
    timeout_s: float,
    log_dir: Path,
    cpus: Optional[Sequence[int]] = None,
    wait_foreign: Optional[Callable[[int, threading.Event], float]] = None,
    report: Optional[Callable[[str], None]] = None,
    outcome: Optional[Outcome] = None,
    check: Optional[Callable[[Mapping[str, Any], Task], None]] = None,
    children: Optional[procs.ChildGroups] = None,
) -> Outcome:
    """Run `tasks` on `gpus`, one harness process at a time per GPU (its
    ROCR_VISIBLE_DEVICES the GPU), the CPUs of `cpus` split between them;
    each GPU's records go to `<log_dir>/gpu<N>.jsonl` as they arrive. The
    tasks a task leaves (`run_task`) run before the others. `outcome` (a
    new one by default) collects every finished task, also when this
    raises: the first error stops every GPU's process at once and is
    raised as a GateError (a HarnessMismatch as itself). `wait_foreign(gpu,
    stop)` waits, before each task, for other processes to leave the GPU
    and returns how long it waited."""
    outcome = outcome if outcome is not None else Outcome()
    pending: Deque[Task] = collections.deque(tasks)
    lock = threading.Lock()
    stop = threading.Event()
    errors: List[BaseException] = []
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = list(gpus)
    cpu_list = sorted(set(cpus or os.sched_getaffinity(0)))
    share = max(1, len(cpu_list) // max(1, len(gpus)))
    total = [len(pending)]
    finished = [0]
    unfound: Set[int] = set()

    def worker(i: int, gpu: int) -> None:
        mine = set(cpu_list[i * share : (i + 1) * share]) or set(cpu_list)
        wenv = dict(env)
        wenv["ROCR_VISIBLE_DEVICES"] = str(gpu)
        for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
            wenv.pop(var, None)
        wenv["OMP_NUM_THREADS"] = str(len(mine))
        with open(log_dir / f"gpu{gpu}.jsonl", "a") as log:
            while not stop.is_set():
                with lock:
                    if not pending:
                        return
                    task = pending.popleft()
                try:
                    waited = (
                        wait_foreign(gpu, stop) if wait_foreign is not None else 0.0
                    )
                    records, rec, following = run_task(
                        task, cmd_of, wenv, timeout_s, mine, check, stop, children
                    )
                except BaseException as e:  # noqa: BLE001
                    with lock:
                        errors.append(e)
                        stop.set()
                    return
                rec.update(gpu=gpu, foreign_wait_s=round(waited, 1))
                for r in records:
                    r["gpu"] = gpu
                    log.write(json.dumps(r) + "\n")
                log.write(json.dumps(dict(rec, task_done=True)) + "\n")
                log.flush()
                with lock:
                    outcome.records.extend(records)
                    outcome.tasks.append(rec)
                    pending.extendleft(reversed(following))
                    total[0] += len(following)
                    finished[0] += 1
                    n, of = finished[0], total[0]
                    new = sorted(set(rec.get("not_found") or ()) - unfound)
                    unfound.update(new)
                if report is not None and (rec["rc"] or rec["timeout"]):
                    report(
                        f"GPU {gpu}: {task.tag} ended with rc={rec['rc']}"
                        + (" (timed out)" if rec["timeout"] else "")
                        + (
                            "; its kernel without every repetition crashed alone"
                            if task.alone
                            else "; its kernel without every repetition runs "
                            "again alone"
                        )
                    )
                if report is not None and new:
                    report(
                        f"GPU {gpu}: the harness finds no solution {new[:5]} of "
                        f"{task.tag}; unless another task finds them, they stay "
                        f"untested"
                    )
                if report is not None and (n == of or n % 10 == 0):
                    report(f"{n}/{of} tasks done")

    threads = [
        threading.Thread(target=worker, args=(i, g), daemon=True)
        for i, g in enumerate(gpus)
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except BaseException:
        stop.set()
        for t in threads:
            t.join(timeout=60)
        raise
    if errors:
        first = errors[0]
        if isinstance(first, GateError):
            raise first
        raise GateError(f"the harness could not run: {first!r}")
    return outcome


# ── verdicts ─────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """What the gate found of one kernel: its repetitions, the failing ones,
    its crashes (confirmed alone, of them the hangs, and those its lone run
    did not repeat), per tag the (failing, all) repetitions of the tags it
    failed, and the tags that did not support it."""

    sid: int
    runs: int = 0
    failing_runs: int = 0
    crashes: int = 0
    hangs: int = 0
    unconfirmed_crashes: int = 0
    failing: Dict[str, List[int]] = field(default_factory=dict)
    unsupported: Set[str] = field(default_factory=set)

    @property
    def verdict(self) -> str:
        if self.failing_runs:
            return "fail"
        if self.crashes:
            return "crash"
        return "pass" if self.runs else "untested"

    def evidence(self) -> Dict[str, Any]:
        return {
            "runs": self.runs,
            "failing_runs": self.failing_runs,
            "crashes": self.crashes,
            "hangs": self.hangs,
            "unconfirmed_crashes": self.unconfirmed_crashes,
            "failing_shapes": {k: v for k, v in sorted(self.failing.items())},
            "unsupported_shapes": sorted(self.unsupported),
        }


def verdicts_of(records: Iterable[Mapping[str, Any]]) -> Dict[int, Verdict]:
    out: Dict[int, Verdict] = {}
    counts: Dict[Tuple[int, str], List[int]] = collections.defaultdict(lambda: [0, 0])
    for r in records:
        sid, tag = int(r["sid"]), str(r.get("tag", ""))
        v = out.setdefault(sid, Verdict(sid))
        if r.get("crash"):
            if r.get("confirmed"):
                v.crashes += 1
                v.hangs += 1 if r.get("timeout") else 0
                counts[(sid, tag)][0] += 1
                counts[(sid, tag)][1] += 1
            else:
                v.unconfirmed_crashes += 1
        elif r.get("supported") is False:
            v.unsupported.add(tag)
        else:
            v.runs += 1
            counts[(sid, tag)][1] += 1
            if int(r.get("nbad", 0) or 0) or int(r.get("nnan", 0) or 0):
                v.failing_runs += 1
                counts[(sid, tag)][0] += 1
    for (sid, tag), (bad, n) in counts.items():
        if bad:
            out[sid].failing[tag] = [bad, n]
    return out


def tag_results(
    records: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]], Dict[str, Set[int]]]:
    """({tag: solutions that failed or crashed alone}, {tag: solutions that
    passed every repetition}, {tag: solutions it did not support}) of
    harness records; a crash not yet run alone counts in none of them."""
    failed: Dict[str, Set[int]] = collections.defaultdict(set)
    ran: Dict[str, Set[int]] = collections.defaultdict(set)
    unsupported: Dict[str, Set[int]] = collections.defaultdict(set)
    for r in records:
        tag, sid = str(r.get("tag", "")), int(r["sid"])
        if r.get("crash"):
            if r.get("confirmed"):
                failed[tag].add(sid)
        elif r.get("supported") is False:
            unsupported[tag].add(sid)
        else:
            ran[tag].add(sid)
            if int(r.get("nbad", 0) or 0) or int(r.get("nnan", 0) or 0):
                failed[tag].add(sid)
    passed = {t: s - failed.get(t, set()) for t, s in ran.items()}
    return dict(failed), passed, dict(unsupported)


def check_name(
    record: Mapping[str, Any], kernels: Mapping[int, Mapping[str, Any]]
) -> None:
    """HarnessMismatch when the harness reported a solution under another
    name than the library's logic file gives it: the harness loaded another
    hipBLASLt build."""
    name = record.get("name")
    k = kernels.get(int(record["sid"]))
    if name and k is not None and str(name) != str(k.get("kernel_name")):
        raise HarnessMismatch(
            f"solution {record['sid']}: the harness ran "
            f"{ki.label_of_name(str(name))}, the library holds {ki.kernel_id(k)} "
            f"there: the harness does not load the library the pipeline reads"
        )


def failure_reason(v: Verdict) -> str:
    """What a failing or crashing kernel did, for the blocklist."""
    parts = [f"{t} {f}/{n}" for t, (f, n) in sorted(v.failing.items())]
    what = "wrong results" if v.failing_runs else "crashed or hung, also alone,"
    return f"{what} in the exact check: " + ", ".join(parts)


# ── one gate run ─────────────────────────────────────────────────────────────


@dataclass
class Spec:
    """Where one gate run reads the pool and writes its results."""

    cfg: Mapping[str, Any]
    dat_dir: Path
    library_stem: str
    out_dir: Path
    cache_dir: Path
    build_dir: Optional[Path] = None
    hardware_device: int = 0
    devices: Sequence[int] = (0,)
    lease_wait_s: float = 0.0
    dry_run: bool = False
    quarantine: Optional[Path] = None


def _wait_foreign(gpu: int, stop: Optional[threading.Event] = None) -> float:
    """Wait up to FOREIGN_WAIT_S for other processes to leave physical GPU
    `gpu`, or until `stop` is set; the seconds waited."""
    from . import kfd

    found = next((g for g in kfd.gpus() if g.index == int(gpu)), None)
    if found is None:
        return 0.0
    stop = stop or threading.Event()
    t0 = time.time()
    while (
        kfd.foreign_now(found)
        and time.time() - t0 < FOREIGN_WAIT_S
        and not stop.wait(30)
    ):
        pass
    return time.time() - t0


def _owners(identity: ki.LibraryIdentity) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for stem in identity.stems:
        for sid in identity.file_ids.get(stem, {}):
            out.setdefault(int(sid), stem)
    return out


def _seeds(cfg: Mapping[str, Any], quarantine: Optional[Path] = None) -> cx.Blocklist:
    out = cx.lists_blocklist(cfg)
    if quarantine is not None:
        out = out.merged(cx.read_list(quarantine))
    return out


Settle = Callable[[Outcome, Iterable[int], bool], None]


def first_launches_lost(outcome: Outcome) -> List[int]:
    """Kernels a first-launch process did not find although other processes
    ran them. A first launch that finds its kernel nowhere leaves it
    untested, as the other passes do."""
    ran = {int(r["sid"]) for r in outcome.records}
    lost = {
        int(s)
        for t in outcome.tasks
        if t.get("first_launch")
        for s in t.get("not_found") or ()
    }
    return sorted(lost & ran)


def run_passes(
    kernels: Mapping[int, Mapping[str, Any]],
    sids: Sequence[int],
    custom: Set[int],
    settings: Settings,
    run: Callable[[List[Task], Outcome], None],
    workers: int,
    say: Callable[[str], None],
    settle: Optional[Settle] = None,
    shapes: Sequence[Shape] = SHAPES,
    tiny: Sequence[Shape] = (),
    tiny_sids: Sequence[int] = (),
) -> Outcome:
    """The gate's passes over `sids`: `shapes`, the first launches, and over
    `tiny_sids` the tiny stage's shapes `tiny`; then the shapes of both again
    with K rounded up where they were not supported, and the follow-ups of
    failures (see the module docstring). `settle(outcome, sids,
    failures_only)` hears of the kernels whose verdicts no later pass can
    change: before the follow-ups those no follow-up runs, after them the
    rest; when a pass stops early, with `failures_only`, every kernel (a
    wrong result stands whatever the passes left to run), unless the harness
    turned out to run other kernels."""
    outcome = Outcome()
    tell: Settle = settle or (lambda _o, _s, _f: None)
    everyone = sorted(set(sids) | set(tiny_sids))
    every_shape = [*shapes, *tiny]
    try:
        tasks = main_tasks(sids, settings, workers, shapes)
        tasks += main_tasks(tiny_sids, settings, workers, tiny)
        first = (
            list(sids)
            if settings.first_launch == "all"
            else (
                [s for s in sids if s in custom]
                if settings.first_launch == "custom"
                else []
            )
        )
        tasks += first_launch_tasks(kernels, first, settings)
        say(
            f"{len(sids):,} kernels x {len(shapes)} shapes"
            + (
                f" and {len(tiny_sids):,} x {len(tiny)} tiny shapes"
                if tiny_sids and tiny
                else ""
            )
            + f" in {len(tasks)} harness processes ({len(first)} kernels "
            f"launched first in {settings.repetitions} processes each)"
        )
        run(tasks, outcome)
        lost = first_launches_lost(outcome)
        if lost:
            raise HarnessMismatch(
                f"the harness finds solution {lost[0]} among other kernels but not "
                f"as the only GEMM of a process: it does not load the library the "
                f"pipeline reads alike in every process"
            )
        _, _, unsupported = tag_results(outcome.records)
        again = roundup_tasks(unsupported, kernels, settings, every_shape)
        if again:
            n = sum(len(t.sids) for t in again)
            say(f"{n:,} unsupported (kernel, shape) pairs again with K rounded up")
            run(again, outcome)
        failed, passed, _ = tag_results(outcome.records)
        by_tag = {s.name: s for s in every_shape}
        by_tag.update({t.tag: t.shape for t in again})
        follow = followup_tasks(failed, passed, kernels, by_tag, settings)
        waiting = {s for t in follow for s in t.sids}
        tell(outcome, [s for s in everyone if s not in waiting], False)
        if follow:
            say(f"following up {len(waiting):,} kernels of the families of failures")
            run(follow, outcome)
        tell(outcome, sorted(waiting), False)
    except HarnessMismatch:
        raise
    except BaseException:
        tell(outcome, everyone, True)
        raise
    return outcome


def _severity(rec: Optional[Mapping[str, Any]]) -> int:
    """How much a stage's verdict record weighs in its kernel's verdict."""
    if rec is None:
        return -1
    return {"fail": 3, "crash": 2, "pass": 1}.get(str(rec.get("verdict")), 0)


def kernel_verdict(
    main: Optional[Mapping[str, Any]], tiny: Optional[Mapping[str, Any]]
) -> Tuple[str, Optional[Mapping[str, Any]]]:
    """(a kernel's verdict, the stage record that decides it) from its
    verdicts of the two stages (MAIN, TINY): a failure or a crash of either,
    the worse of them; else `unchecked` while a stage has no verdict, else
    pass when either passed, else untested."""
    worst = max((main, tiny), key=_severity)
    if worst is not None and worst.get("verdict") in ("fail", "crash"):
        return str(worst["verdict"]), worst
    if main is None or tiny is None:
        return "unchecked", None
    return ("pass" if _severity(worst) == 1 else "untested"), main


def _fresh_runs(outcome: Outcome) -> Dict[int, List[bool]]:
    """{solution: whether each of its tiny-stage records ran on operands at
    fresh allocations}."""
    out: Dict[int, List[bool]] = collections.defaultdict(list)
    for r in outcome.records:
        if r.get("stage") == TINY:
            out[int(r["sid"])].append(bool(r.get("fresh")))
    return out


def run_gate(
    spec: Spec,
    say: Callable[[str], None] = lambda _m: None,
    warn: Callable[[str], None] = lambda _m: None,
    children: Optional[procs.ChildGroups] = None,
) -> Dict[str, Any]:
    """The static check of the pool's hand-written kernels, then the harness
    on every other kernel without a verdict of a stage (MAIN, TINY); cache
    the new verdicts, and write `verdicts.json`, `blocklist.json` and
    `gate_summary.json` into `spec.out_dir`; returns the summary. Harness
    processes start through `children` when given (lib/procs.py)."""
    from . import bench_env as benv
    from . import gpu_lease, kfd
    from .bank_pool import bank_device, config_identity, config_pool

    t0 = time.time()
    cfg = spec.cfg
    sec = dict(cfg.get("correctness") or {})
    settings = Settings.of(cfg)
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = config_pool(
        cfg,
        spec.dat_dir,
        spec.library_stem,
        spec.hardware_device,
        apply_blocklist=False,
    )
    kernels = {int(k["sol_idx_global"]): k for k in pool.kernels}
    identity = config_identity(
        cfg, spec.dat_dir, spec.library_stem, spec.hardware_device
    )
    owner = _owners(identity)
    seeded = cx.resolved(_seeds(cfg, spec.quarantine), pool.kernels)
    dev = bank_device(cfg, spec.hardware_device)
    dkey = device_key(dev)
    for d in spec.devices:
        other = device_key(bank_device(cfg, int(d)))
        if other != dkey:
            raise GateError(
                f"device {d} is a {other}, device {spec.hardware_device} a {dkey}: "
                f"the gate's verdicts hold for one kind of device"
            )
    codes = code_digests(spec.dat_dir, sorted(set(owner[s] for s in kernels)))
    prebuilt = Path(sec["harness"]) if sec.get("harness") else None
    try:
        problem_type: Optional[ProblemType] = harness_problem(cfg)
    except GateError:
        # Raised again below when a kernel needs a check.
        problem_type = None
    shapes = shapes_of(problem_type)
    tiny = tiny_shapes_of(problem_type)
    hid = harness_digest(prebuilt, spec.build_dir)
    proto = protocol(settings, hid, shapes)
    tiny_proto = protocol(settings, hid, tiny, TINY)
    cache = VerdictCache(spec.cache_dir)

    def code_of(sid: int) -> str:
        return codes[owner[sid]]

    unlisted = [s for s, k in kernels.items() if not seeded.blocks(k)]
    static: Dict[int, StaticFinding] = {}
    static_on = sec.get("static_check", True) is not False
    sources_dir: Optional[Path] = None
    sources_what = "off (correctness.static_check)"
    if static_on:
        sources_dir, sources_what = custom_kernels_dir(cfg, spec.build_dir)
        sources = (
            custom_kernel_sources(sources_dir) if sources_dir is not None else None
        )
        h = dict(cfg.get("hipblaslt") or {})
        trans = (str(h.get("transA", "T")) == "T", str(h.get("transB", "N")) == "T")
        static = static_check(
            spec.dat_dir,
            sorted(set(owner.values())),
            unlisted,
            sources,
            trans,
            sources_what,
        )
        for status, what in (
            ("unchecked", "cannot be checked statically"),
            ("undeclared", "declare no global-read vector width the check needs"),
        ):
            found = [f for f in static.values() if f.status == status]
            if found:
                warn(
                    f"{len(found)} hand-written kernel(s) {what} (e.g. {found[0].name}: "
                    f"{found[0].reason}); the tiny shapes still run them"
                )
    unguarded = {s for s, f in static.items() if f.blocks}
    known: Dict[int, Dict[str, Any]] = {}
    known_tiny: Dict[int, Dict[str, Any]] = {}
    pending: List[int] = []
    pending_tiny: List[int] = []
    not_fresh = 0
    for sid in unlisted:
        if sid in unguarded:
            continue
        kid = ki.kernel_id(kernels[sid])
        for proto_, have, want in (
            (proto, known, pending),
            (tiny_proto, known_tiny, pending_tiny),
        ):
            rec = cache.lookup(kid, dkey, code_of(sid), proto_, settings.timeout_s)
            if rec is not None:
                have[sid] = rec
                continue
            want.append(sid)
            if proto_ == tiny_proto and cache.lookup(
                kid, dkey, code_of(sid), proto_, settings.timeout_s, fresh=False
            ):
                not_fresh += 1
    say(
        f"pool {len(kernels):,} kernels: {len(seeded)} blocked by lists, "
        f"{len(unguarded)} unguarded in the static check, {len(known):,} with a "
        f"cached verdict, {len(pending):,} to check (device {dkey}, protocol "
        f"{proto}); tiny shapes: {len(known_tiny):,} cached, "
        f"{len(pending_tiny):,} to check (protocol {tiny_proto})"
    )
    if not_fresh:
        say(
            f"{not_fresh:,} of the kernels to check on the tiny shapes have a "
            f"cached verdict of them made without fresh operands, which no gate "
            f"takes: they run the tiny shapes again"
        )
    summary: Dict[str, Any] = {
        "library_stem": spec.library_stem,
        "library_digest": identity.digest,
        "device": dkey,
        "protocol": proto,
        "settings": asdict(settings),
        "shapes": [dict(s.record(), text=s.text()) for s in shapes],
        "pool": len(kernels),
        "listed": len(seeded),
        "cached": len(known),
        "pending": len(pending),
        "checked": 0,
        "harness": None,
        "gpus": [],
        "tasks": 0,
        "harness_crashes": 0,
        "unconfirmed_crashes": 0,
        "not_found": 0,
        "gpu_seconds": 0.0,
        "dry_run": bool(spec.dry_run),
        "tiny": {
            "protocol": tiny_proto,
            "shapes": [dict(s.record(), text=s.text()) for s in tiny],
            "cached": len(known_tiny),
            "pending": len(pending_tiny),
            "cached_not_fresh": not_fresh,
            "checked": 0,
            "harness_fresh": None,
            "tasks": 0,
            "fresh_tasks": 0,
        },
        "static": {
            "enabled": static_on,
            "sources": str(sources_dir) if sources_dir is not None else None,
            "where": sources_what,
            "hand_written": len(static),
            "statuses": dict(
                sorted(collections.Counter(f.status for f in static.values()).items())
            ),
            "unguarded": [
                {"solution_index": s, "name": static[s].name} for s in sorted(unguarded)
            ],
        },
    }
    settled: Dict[int, Dict[str, Any]] = {}
    settled_tiny: Dict[int, Dict[str, Any]] = {}
    if (pending or pending_tiny) and not spec.dry_run:
        problem = harness_problem(cfg)
        h = harness(
            Path(spec.cache_dir),
            Path(spec.build_dir) if spec.build_dir else Path("."),
            prebuilt,
            sec.get("compiler"),
            log=out_dir / "harness_build.log",
        )
        summary["harness"] = {"path": str(h.path), "built": h.built, "digest": h.digest}
        if h.built:
            say(f"built the harness {h.path}")
        custom = hand_written(spec.dat_dir, sorted(set(owner.values())))
        gpus = kfd.bench_gpus(spec.devices)
        if not gpus:
            raise GateError(f"devices {list(spec.devices)} reach no GPU here")
        summary["gpus"] = gpus
        env = benv.bench_env_with()
        fresh_ok = [False]

        def cmd_of(task: Task) -> List[str]:
            return task.command(h.path, problem, fresh_ok[0])

        def check(line: Mapping[str, Any], task: Task) -> None:
            if line.get("ready") is True:
                found = line.get("sids")
                if (
                    isinstance(found, list)
                    and task.sids
                    and not found
                    and not task.first_launch
                ):
                    raise HarnessMismatch(
                        f"the harness finds none of the solutions of {task.tag} "
                        f"({', '.join(str(s) for s in task.sids[:4])}, ...): it "
                        f"does not load the library the pipeline reads"
                    )
                return
            check_name(line, kernels)

        def run(tasks: List[Task], into: Outcome) -> None:
            run_tasks(
                tasks,
                cmd_of,
                gpus,
                env,
                settings.timeout_s,
                out_dir / "runs",
                wait_foreign=_wait_foreign,
                report=say,
                outcome=into,
                check=check,
                children=children,
            )

        stages = (
            (MAIN, set(pending), settled, proto),
            (TINY, set(pending_tiny), settled_tiny, tiny_proto),
        )

        def settle(outcome: Outcome, sids: Iterable[int], failures_only: bool) -> None:
            by_stage = {
                MAIN: verdicts_of(
                    r for r in outcome.records if r.get("stage", MAIN) == MAIN
                ),
                TINY: verdicts_of(r for r in outcome.records if r.get("stage") == TINY),
            }
            fresh = _fresh_runs(outcome)
            new = []
            for sid in sids:
                for stage, wanted, done, stage_proto in stages:
                    if sid not in wanted or sid in done:
                        continue
                    v = by_stage[stage].get(sid) or Verdict(sid)
                    if failures_only and v.verdict != "fail":
                        continue
                    k = kernels[sid]
                    rec: Dict[str, Any] = {
                        "kernel_id": ki.kernel_id(k),
                        "name": k["kernel_name"],
                        "signature": ki.signature_label(k),
                        "device": dkey,
                        "code": code_of(sid),
                        "protocol": stage_proto,
                        "verdict": v.verdict,
                        "evidence": v.evidence(),
                        "reason": (
                            failure_reason(v) if v.verdict in BLOCKING_VERDICTS else ""
                        ),
                        "time": kfd.utc_iso(),
                    }
                    if stage != MAIN:
                        runs = fresh.get(sid) or []
                        rec["stage"] = stage
                        rec["evidence"]["fresh"] = all(runs) if runs else None
                    if v.verdict == "crash" and v.hangs == v.crashes:
                        rec["timeout_s"] = settings.timeout_s
                    done[sid] = rec
                    new.append(rec)
            cache.add(new)

        with gpu_lease.hold_usable(
            spec.devices,
            "correctness gate",
            out_dir,
            wait_s=spec.lease_wait_s,
            report=warn,
        ) as leases:
            gpus = kfd.bench_gpus(leases.devices) or gpus
            summary["gpus"] = gpus
            if pending_tiny:
                probe = dict(env, ROCR_VISIBLE_DEVICES=str(gpus[0]))
                for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
                    probe.pop(var, None)
                fresh_ok[0] = "fresh" in harness_caps(h.path, probe)
                summary["tiny"]["harness_fresh"] = fresh_ok[0]
                if not fresh_ok[0]:
                    warn(
                        f"the harness {h.path} takes no --fresh by its --caps (it "
                        f"was built before the option existed, or --caps failed): "
                        f"the tiny shapes run on its usual allocations, where a read "
                        f"before an operand may land in another buffer and go unseen; "
                        f"the next gate runs them again"
                    )
            outcome = run_passes(
                kernels,
                sorted(pending),
                custom,
                settings,
                run,
                len(gpus),
                say,
                settle,
                shapes,
                tiny,
                sorted(pending_tiny),
            )
        summary["tasks"] = len(outcome.tasks)
        summary["harness_crashes"] = sum(
            1 for t in outcome.tasks if t["rc"] or t["timeout"]
        )
        summary["unconfirmed_crashes"] = sum(
            1 for r in outcome.records if r.get("crash") and not r.get("confirmed")
        )
        found_nowhere = set().union(
            *(set(t.get("not_found") or ()) for t in outcome.tasks)
        ) - {int(r["sid"]) for r in outcome.records}
        summary["not_found"] = len(found_nowhere)
        if found_nowhere:
            warn(
                f"the harness finds {len(found_nowhere)} kernel(s) of the pool "
                f"nowhere (e.g. solution {min(found_nowhere)}); they stay untested"
            )
        tiny_tasks = [t for t in outcome.tasks if t.get("stage") == TINY]
        n_fresh = sum(1 for t in tiny_tasks if t.get("fresh"))
        summary["tiny"].update(tasks=len(tiny_tasks), fresh_tasks=n_fresh)
        if fresh_ok[0] and n_fresh < len(tiny_tasks):
            warn(
                f"{len(tiny_tasks) - n_fresh} of the {len(tiny_tasks)} tiny-shape "
                f"harness processes could not place their operands at fresh "
                f"allocations (no virtual memory management on the device?): those "
                f"ran on its usual allocations, and the next gate runs their "
                f"kernels on the tiny shapes again"
            )
        summary["gpu_seconds"] = round(sum(t["wall_s"] for t in outcome.tasks), 1)
        known.update(settled)
        known_tiny.update(settled_tiny)
        summary["checked"] = len(pending)
        summary["tiny"]["checked"] = len(pending_tiny)
    elif pending or pending_tiny:
        warn(
            f"dry run: {len(set(pending) | set(pending_tiny)):,} kernels left "
            f"without a verdict"
        )

    rows = []
    blocked: List[cx.Entry] = list(seeded.entries)
    listed = {e.name: e for e in seeded.entries}
    counts: Dict[str, int] = collections.Counter()
    for sid, k in sorted(kernels.items()):
        kid = ki.kernel_id(k)
        base = {
            "solution_index": sid,
            "kernel_id": kid,
            "signature": ki.signature_label(k),
            "name": k["kernel_name"],
            "library_stem": owner.get(sid),
        }
        entry = listed.get(str(k["kernel_name"]))
        if entry is not None:
            rows.append(
                dict(base, verdict="listed", source=entry.source, reason=entry.reason)
            )
            counts["listed"] += 1
            continue
        finding = static.get(sid)
        if finding is not None:
            base["static"] = finding.record()
        if finding is not None and finding.blocks:
            rows.append(
                dict(
                    base,
                    verdict=UNGUARDED,
                    source="static check",
                    reason=finding.reason,
                )
            )
            counts[UNGUARDED] += 1
            blocked.append(
                cx.Entry(
                    name=str(k["kernel_name"]),
                    reason=finding.reason,
                    kernel_id=kid,
                    signature=ki.signature_label(k),
                    source="correctness gate (static check)",
                )
            )
            continue
        main, tiny_rec = known.get(sid), known_tiny.get(sid)
        verdict, decided = kernel_verdict(main, tiny_rec)
        if main is None and tiny_rec is None:
            rows.append(dict(base, verdict="unchecked", source=None))
            counts["unchecked"] += 1
            continue
        own = decided or main or tiny_rec or {}
        source = "gate" if sid in settled or sid in settled_tiny else "cache"
        row = dict(
            base,
            verdict=verdict,
            source=source,
            reason=str(own.get("reason") or "") if decided is not None else "",
            evidence=(main or {}).get("evidence"),
        )
        if tiny_rec is not None:
            row["tiny"] = {
                "verdict": tiny_rec["verdict"],
                "source": "gate" if sid in settled_tiny else "cache",
                "reason": tiny_rec.get("reason") or "",
                "evidence": tiny_rec.get("evidence"),
            }
        rows.append(row)
        counts[verdict] += 1
        if verdict in BLOCKING_VERDICTS:
            blocked.append(
                cx.Entry(
                    name=str(k["kernel_name"]),
                    reason=str(own.get("reason") or "wrong results in the exact check"),
                    kernel_id=kid,
                    signature=ki.signature_label(k),
                    source="correctness gate",
                )
            )
    summary["verdicts"] = dict(sorted(counts.items()))
    summary["blocked"] = len(blocked)
    summary["seconds"] = round(time.time() - t0, 1)
    head = {
        "library_stem": spec.library_stem,
        "library_digest": identity.digest,
        "device": dkey,
        "protocol": proto,
        "tiny_protocol": tiny_proto,
    }
    _write(out_dir / VERDICTS_FILE, dict(head, kernels=rows))
    cx.write_blocklist(out_dir / cx.BLOCKLIST_FILE, cx.Blocklist(blocked), **head)
    _write(out_dir / SUMMARY_FILE, summary)
    return summary


def _write(path: Path, doc: Any) -> None:
    from .fs import atomic_write_text

    atomic_write_text(path, json.dumps(doc, indent=1) + "\n")
