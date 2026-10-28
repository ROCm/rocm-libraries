# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward build schema: a ``Spec`` names one backward object (gfx950 or a generic gfx11 target).

No FlyDSL import: the string grammar, the key, the hash and the validation are pure Python,
shared by ``_sdpa_bwd_cdna_build`` (which turns a Spec into an object) and its callers.

Canonical string (``Spec.key()``, parsed by ``Spec.parse``):

    kernel,dtype,tile,cls,dform,dropout,padded,floor,hdv,hints[;knob=value;...]

  kernel  : prep | dkdv | dq | dbias_reduce | dsink_reduce | merged
            (prep: dtype, tile, padded and the ``sink`` knob. dsink_reduce and dbias_reduce: dtype only, spelled
            ``dbias_reduce,<dtype>,32,plain,bk,0,0,tail,0,1``: the tile is the ladder's first rung as a placeholder,
            every other field its default. merged: dK/dV + dQ + dB in one launch, no prep; both roles take their
            policy-default schedules, and the dQ role is built at the dK/dV role's wave count)
  dtype   : f16 | bf16 | fp32   (fp32: the native f32 MFMA family; tiles 32..256 and 512 (no 384); kernels prep,
            dkdv, dq and dbias_reduce only, no ``sink``, ``merged`` or ``dsink_reduce``; its perf knobs are
            listed in ``FP32_KNOBS``)
  tile    : head-dim tile, one of LADDERS[dtype]
  cls     : plain | causal | bias
  dform   : rt (dropout is a runtime branch on ``rt_flags`` bit 1; the compiled dropout flag is 1)
            bk (dropout baked into the object; ``dropout`` = 0 or 1)
  dropout : compiled dropout flag (rt => 1)
  padded  : 0 exact tile | 1 padded head compiled (the head dim itself is a launch value)
  floor   : g1 (pinned tile, floor 0) | tail (rung below the tile) | <int>
  hdv     : V head dim carried in meta (0 = tile; the dQ convention is ``hdv_of`` of the builder)
  hints   : 0|1  apply the kernel module's _COMPILE_HINTS
  knobs   : perf knobs, listed only when set, in KNOBS order. Unset = the FlyDSL policy value.
            dkdv: num_waves mfma_rows dkv_shards block_q waves_per_eu tight_registers num_stream_buffers gqa_pack
            dq  : num_waves block_m block_n mfma_rows waves_per_eu store_db bias_stage
            prep: sink (``sink=1``: the sink twin, a different object with a different ABI)
            merged: num_waves mfma_rows dkv_shards role_order store_db gqa_pack (num_waves, mfma_rows and
            dkv_shards are the dK/dV role's grid-fill knobs, with the split dK/dV meaning; the dQ role follows the
            wave count; role_order is concat | lpt)
            ``gqa_pack=1`` packs the GQA group into the q-tile row axis of the dK/dV role; ``store_db=0`` builds a bias
            dQ object without the dB store; ``bias_stage=direct`` loads the bias directly instead of staging it in LDS.

  arch    : not a field of the head: ``;arch=<arch>`` follows the head when the arch is not gfx950 (``gfx11-generic``,
            ``gfx12-generic``), before any knob, so every gfx950 key and hash is unchanged. A generic arch builds prep, dkdv,
            dq and dbias_reduce only; its tiles are ``GENERIC_LADDER`` for every dtype; its floor is ``g1`` for a padded dkdv
            or dq and ``tail`` for every other kernel; its knobs are ``KNOBS_GENERIC`` (fp32: ``KNOBS_GENERIC_F32``).
            dkdv: num_waves block_q waves_per_eu split_head_dim contraction_shards transposed_source sched_strategy dump rne
            dq  : num_waves block_m block_n waves_per_eu store_db bias_stage contraction_shards kt_lds_layout lpt_tile_order dump rne
            fp32 dkdv: num_waves block_q block_kv waves_per_eu sched_strategy dump
            fp32 dq  : num_waves block_m block_n waves_per_eu store_db lpt_tile_order dump
            A knob is listed only when it is not the policy value: ``split_head_dim=1``, ``contraction_shards=<2^n>``,
            ``transposed_source=derived``, ``sched_strategy=<non-default>``, ``lpt_tile_order=0``, ``kt_lds_layout=staged``,
            ``dump=1``. ``rne=1`` (bf16 dkdv and dq only) narrows P and dS to bf16 round-to-nearest-even instead of
            truncating; it is a spelled pin, so an RNE object and its truncating twin have different keys.

``Spec.hash()`` is the first 12 hex digits of sha256(key()).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)  # the 16-bit ladder
# fp32 rides tile 512 for head dims 257..512: no 384 rung, so ``rung_below(512, "fp32")`` is 256.
FP32_LADDER = tuple(t for t in LADDER if t != 384)
KERNELS = ("prep", "dkdv", "dq", "dbias_reduce", "dsink_reduce", "merged")
ARCHES = ("gfx950", "gfx11-generic", "gfx12-generic")
GFX950 = "gfx950"
# The generic arches build these kernels (from the gfx11 sources; gfx12-generic shares them).
GENERIC_KERNELS = ("prep", "dkdv", "dq", "dbias_reduce")
# The generic arches' native head-dim rungs: no 384 or 512, one ladder for every dtype.
GENERIC_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)
# dbias_reduce's placeholder tile: the gfx950 spelling on every arch (16 is the generic ladder's first rung).
REDUCE_TILE = 32
SCHED_STRATEGIES = (
    "max-ilp",
    "iterative-ilp",
    "iterative-minreg",
    "iterative-maxocc",
    "default",
)
ROLE_ORDERS = ("concat", "lpt")
DTYPES = ("f16", "bf16")  # the 16-bit dtypes
FP32 = "fp32"
ALL_DTYPES = DTYPES + (FP32,)
LADDERS = {"f16": LADDER, "bf16": LADDER, FP32: FP32_LADDER}
CLASSES = ("plain", "causal", "bias")
DFORMS = ("rt", "bk")
# Perf knobs per kernel, in canonical order. Values are ints; tight_registers is 0/1.
KNOBS = {
    "prep": ("sink",),
    "dbias_reduce": (),
    "dsink_reduce": (),
    "dkdv": (
        "num_waves",
        "mfma_rows",
        "dkv_shards",
        "block_q",
        "waves_per_eu",
        "tight_registers",
        "num_stream_buffers",
        "gqa_pack",
    ),
    "dq": (
        "num_waves",
        "block_m",
        "block_n",
        "mfma_rows",
        "waves_per_eu",
        "store_db",
        "bias_stage",
    ),
    "merged": (
        "num_waves",
        "mfma_rows",
        "dkv_shards",
        "role_order",
        "store_db",
        "gqa_pack",
    ),
}

# The generic arches' perf knobs per kernel, in canonical order (``KNOBS_GENERIC_F32`` for fp32).
KNOBS_GENERIC = {
    "prep": (),
    "dbias_reduce": (),
    "dkdv": (
        "num_waves",
        "block_q",
        "waves_per_eu",
        "split_head_dim",
        "contraction_shards",
        "transposed_source",
        "sched_strategy",
        "dump",
        "rne",
    ),
    "dq": (
        "num_waves",
        "block_m",
        "block_n",
        "waves_per_eu",
        "store_db",
        "bias_stage",
        "contraction_shards",
        "kt_lds_layout",
        "lpt_tile_order",
        "dump",
        "rne",
    ),
}
# The generic fp32 (VALU) family: block_q is dK/dV's BLOCK_M and block_kv its BLOCK_N, block_m and block_n dQ's.
KNOBS_GENERIC_F32 = {
    "prep": (),
    "dbias_reduce": (),
    "dkdv": (
        "num_waves",
        "block_q",
        "block_kv",
        "waves_per_eu",
        "sched_strategy",
        "dump",
    ),
    "dq": (
        "num_waves",
        "block_m",
        "block_n",
        "waves_per_eu",
        "store_db",
        "lpt_tile_order",
        "dump",
    ),
}

# The columns of a generic arch's instance-table row after the ten head fields (``tools/sdpa_bwd_tables_from_sweep.py``): the
# knobs of both dtype families, the 16-bit list first, then the fp32-only knobs.
GENERIC_TABLE_KNOBS = {
    kernel: tuple(dict.fromkeys(KNOBS_GENERIC[kernel] + KNOBS_GENERIC_F32[kernel]))
    for kernel in KNOBS_GENERIC
}

# The fp32-legal perf knobs and values per arch and kernel; anything else is rejected on fp32. On gfx950 the dK/dV
# schedule (per-wave or D-split) and the wave count are per-tile entries of the fp32 tuning table, not Spec keys.
FP32_KNOBS = {
    GFX950: {
        "prep": {},
        "dbias_reduce": {},
        "dkdv": {"mfma_rows": (16,), "block_q": (16, 32), "num_stream_buffers": (1,)},
        "dq": {"block_m": (16, 32), "block_n": (16,)},
    },
    "gfx11-generic": {
        "prep": {},
        "dbias_reduce": {},
        "dkdv": {
            "num_waves": (4, 8),
            "block_q": (16, 32),
            "block_kv": (16, 32),
            "waves_per_eu": (1, 2, 3, 4),
            "sched_strategy": SCHED_STRATEGIES,
            "dump": (1,),
        },
        "dq": {
            "num_waves": (4, 8),
            "block_m": (16, 32),
            "block_n": (16, 32),
            "waves_per_eu": (1, 2, 3, 4),
            "store_db": (0,),
            "lpt_tile_order": (0,),
            "dump": (1,),
        },
    },
}
FP32_KNOBS["gfx12-generic"] = FP32_KNOBS["gfx11-generic"]
# The arches with an fp32 family: gfx12-generic has none.
FP32_ARCHES = (GFX950, "gfx11-generic")
# Knob values only the gfx950 fp32 family has (the gfx950 16-bit policies never take them): rejected on a gfx950 16-bit Spec.
FP32_ONLY_VALUES = {
    "dkdv": {"block_q": (16,)},
    "dq": {"block_m": (16,), "block_n": (16,)},
}

ALL_KNOBS = tuple(dict.fromkeys(k for ks in KNOBS.values() for k in ks))
# Every knob field of a Spec, either arch.
KNOB_FIELDS = tuple(
    dict.fromkeys(
        ALL_KNOBS
        + tuple(k for ks in KNOBS_GENERIC.values() for k in ks)
        + tuple(k for ks in KNOBS_GENERIC_F32.values() for k in ks)
    )
)
STR_KNOBS = {
    "bias_stage": ("direct",),
    "role_order": ROLE_ORDERS,
    "transposed_source": ("derived",),
    "kt_lds_layout": ("staged",),
    "sched_strategy": SCHED_STRATEGIES,
}  # knobs whose value is one of these words rather than an int
# Boolean knobs, spelled ``<knob>=1`` only.
ONE_KNOBS = ("gqa_pack", "sink", "split_head_dim", "dump", "rne")


def knobs_of(kernel: str, arch: str = GFX950, dtype: str | None = None) -> tuple:
    """The knobs @p kernel takes on @p arch, in canonical order (a generic arch's fp32 family has its own list)."""
    if arch == GFX950:
        return KNOBS.get(kernel, ())
    return (KNOBS_GENERIC_F32 if dtype == FP32 else KNOBS_GENERIC).get(kernel, ())


def ladder(arch: str = GFX950, dtype: str | None = None) -> tuple:
    """The head-dim tiles of @p arch for @p dtype (None: the 16-bit ladder); a generic arch has one ladder for every dtype."""
    return LADDERS.get(dtype, LADDER) if arch == GFX950 else GENERIC_LADDER


def rung_below(t, dtype=None, arch=GFX950):
    """The rung below tile ``t`` on the ladder of ``arch`` and ``dtype`` (None: the 16-bit ladder); gfx950 fp32 tile 512 gives 256."""
    return max([r for r in ladder(arch, dtype) if r < t], default=0)


@dataclass(frozen=True)
class Spec:
    kernel: str
    dtype: str
    tile: int
    cls: str = "plain"
    dform: str = "bk"
    dropout: int = 0
    padded: int = 0
    floor: str = "tail"
    hdv: int = 0
    hints: int = 1
    num_waves: int | None = None
    mfma_rows: int | None = None
    dkv_shards: int | None = None
    block_q: int | None = None
    block_m: int | None = None
    block_n: int | None = None
    waves_per_eu: int | None = None
    tight_registers: int | None = None
    num_stream_buffers: int | None = None
    store_db: int | None = None
    gqa_pack: int | None = None
    bias_stage: str | None = None
    role_order: str | None = None
    sink: int | None = None
    split_head_dim: int | None = None
    contraction_shards: int | None = None
    transposed_source: str | None = None
    sched_strategy: str | None = None
    dump: int | None = None
    kt_lds_layout: str | None = None
    lpt_tile_order: int | None = None
    block_kv: int | None = None
    rne: int | None = None
    arch: str = GFX950

    def __post_init__(self):
        def bad(msg):
            raise ValueError(f"{msg}: {self!r}")

        if self.arch not in ARCHES:
            bad("arch")
        generic = self.arch != GFX950
        if self.kernel not in KERNELS:
            bad("kernel")
        if generic and self.kernel not in GENERIC_KERNELS:
            bad(f"{self.kernel} is not built on {self.arch}")
        if self.dtype not in ALL_DTYPES:
            bad("dtype")
        if self.dtype == FP32 and self.arch not in FP32_ARCHES:
            bad(f"fp32 is not built on {self.arch}")
        if self.tile not in ladder(self.arch, self.dtype):
            bad(
                f"tile (the {self.dtype} ladder of {self.arch} is {ladder(self.arch, self.dtype)})"
            )
        if self.dtype == FP32 and self.kernel not in FP32_KNOBS[self.arch]:
            bad(f"kernel {self.kernel} has no fp32 family")
        if self.cls not in CLASSES:
            bad("cls")
        if self.dform not in DFORMS:
            bad("dform")
        if self.dropout not in (0, 1):
            bad("dropout")
        if self.padded not in (0, 1):
            bad("padded")
        if self.hints not in (0, 1):
            bad("hints")
        if self.floor not in ("g1", "tail"):
            int(self.floor)
        if self.dform == "rt" and self.dropout != 1:
            bad("dform=rt needs dropout=1 (the compiled flag is the enabled value)")
        if self.hdv == self.tile:
            object.__setattr__(self, "hdv", 0)  # one spelling per object
        if self.kernel == "prep" and (
            self.cls != "plain" or self.dform != "bk" or self.dropout
        ):
            bad("prep takes only dtype, tile, padded")
        if generic:
            if self.kernel in ("dkdv", "dq"):
                if self.floor != ("g1" if self.padded else "tail"):
                    bad(
                        f"on {self.arch} a padded {self.kernel} carries floor=g1 (floor 0, the one padded form) "
                        "and an exact one the inert floor=tail"
                    )
            elif self.floor != "tail":
                bad(f"{self.kernel} on {self.arch} carries floor=tail")
        if self.kernel in ("dbias_reduce", "dsink_reduce") and (
            self.tile != REDUCE_TILE
            or self.cls != "plain"
            or self.dform != "bk"
            or self.dropout
            or self.padded
            or self.floor != "tail"
            or self.hdv
            or self.hints != 1
        ):
            bad(
                f"{self.kernel} takes only dtype (tile is the placeholder {REDUCE_TILE}, every other field its default)"
            )
        allowed = knobs_of(self.kernel, self.arch, self.dtype)
        for k in KNOB_FIELDS:
            v = getattr(self, k)
            if v is None:
                continue
            if k not in allowed:
                bad(f"knob {k} does not apply to {self.kernel} on {self.arch}")
            if self.dtype == FP32:
                if v not in FP32_KNOBS[self.arch][self.kernel].get(k, ()):
                    bad(
                        f"knob {k}={v} is not fp32-legal for {self.kernel} on {self.arch} "
                        f"(fp32: {FP32_KNOBS[self.arch][self.kernel]})"
                    )
            elif not generic and v in FP32_ONLY_VALUES.get(self.kernel, {}).get(k, ()):
                bad(f"knob {k}={v} is an fp32 knob value, not legal on {self.dtype}")
            if k in STR_KNOBS:
                if v not in STR_KNOBS[k]:
                    bad(f"knob {k} must be one of {STR_KNOBS[k]}")
                if k == "bias_stage" and (self.cls != "bias" or self.kernel != "dq"):
                    bad("bias_stage applies to bias dQ objects only")
                continue
            if k == "tight_registers":
                object.__setattr__(self, k, int(bool(v)))
            elif k == "store_db":
                if v != 0 or self.cls != "bias":
                    bad(
                        "store_db is spelled store_db=0 and applies to bias dQ objects only"
                    )
            elif k in ONE_KNOBS:
                if v != 1:
                    bad(f"{k} is spelled {k}=1")
                if k == "rne" and self.dtype != "bf16":
                    bad("rne applies to bf16 dkdv/dq objects only")
            elif k == "lpt_tile_order":
                if v != 0:
                    bad(
                        "lpt_tile_order is spelled lpt_tile_order=0 (on is the policy value)"
                    )
            elif not isinstance(v, int) or v <= 0:
                bad(f"knob {k} must be a positive int")
            elif k == "contraction_shards" and v & (v - 1):
                bad("contraction_shards must be a power of two")
        if self.dtype == FP32 and generic:
            bq, bk = (
                (self.block_q, self.block_kv)
                if self.kernel == "dkdv"
                else (self.block_m, self.block_n)
            )
            if bq is not None and bk is not None and bq < bk:
                bad("fp32 tiles have BLOCK_M >= BLOCK_N")
        if generic and self.kernel in ("dkdv", "dq"):
            shards = self.contraction_shards
            if self.kernel == "dkdv":
                if shards is not None and not self.split_head_dim:
                    bad("contraction_shards needs split_head_dim=1")
                team = 2 * (shards or 1) if self.split_head_dim else 1
            else:
                team = shards or 1
            if shards is not None and (
                self.tile % shards or (self.tile // shards) % 16
            ):
                bad(
                    f"per-wave column slice {self.tile}/{shards} must be a multiple of 16"
                )
            if self.num_waves is not None and self.num_waves % team:
                bad(f"num_waves must be a multiple of the team size {team}")

    @property
    def knobs(self):
        """The set perf knobs, in canonical order."""
        return {
            k: getattr(self, k)
            for k in knobs_of(self.kernel, self.arch, self.dtype)
            if getattr(self, k) is not None
        }

    @property
    def functional(self):
        """The functional key: everything but the perf knobs (the arch is the head's last element when it is not gfx950)."""
        head = (
            self.kernel,
            self.dtype,
            self.tile,
            self.cls,
            self.dform,
            self.dropout,
            self.padded,
            self.floor,
            self.hdv,
            self.hints,
        )
        return head if self.arch == GFX950 else head + (self.arch,)

    def key(self):
        head = ",".join(str(v) for v in self.functional[:10])
        if self.arch != GFX950:
            head += f";arch={self.arch}"
        return head + "".join(f";{k}={v}" for k, v in self.knobs.items())

    def hash(self):
        return hashlib.sha256(self.key().encode()).hexdigest()[:12]

    @staticmethod
    def parse(s):
        head, *kv = s.split(";")
        kernel, dtype, tile, cls, dform, dropout, padded, floor, hdv, hints = (
            head.split(",")
        )
        knobs, arch = {}, GFX950
        for item in kv:
            n, v = item.split("=")
            if n == "arch":
                if v == GFX950 or arch != GFX950:
                    raise ValueError(
                        f"arch is spelled once, and only when it is not gfx950: {s!r}"
                    )
                arch = v
                continue
            if n not in KNOB_FIELDS:
                raise ValueError(f"unknown knob {n!r} in {s!r}")
            knobs[n] = v if n in STR_KNOBS else int(v)
        return Spec(
            kernel,
            dtype,
            int(tile),
            cls,
            dform,
            int(dropout),
            int(padded),
            floor,
            int(hdv),
            int(hints),
            arch=arch,
            **knobs,
        )


def dbias_reduce_spec(dtype: str, arch: str = GFX950) -> Spec:
    """The standalone dBias reduce object of @p dtype on @p arch: the pin a combined bias object's reduce kernel is gated against."""
    return Spec("dbias_reduce", dtype, REDUCE_TILE, arch=arch)


# The instance-row spelling of an unset knob: -1 for the int knobs (0 is legal for
# ``tight_registers`` and ``store_db``), "auto" for the word knobs.
UNSET_INT = -1
UNSET_WORD = "auto"

# Row fields of an ``_instances.Instance`` that carry the functional key, besides ``dtype``.
ROW_FIELDS = (
    "head_dim",
    "head_dim_v",
    "causal",
    "has_bias",
    "dropout",
    "padded",
    "hints",
    "floor",
    "dform",
)

# Row fields a kernel's rows carry that are no part of its Spec: the object's content is
# named by the key, these only describe how it is shipped. The merged and dq rows carry
# the symbol of the dBias reduce kernel their bias objects hold (`""` on the others).
NON_KEY_FIELDS = {"merged": ("reduce_symbol",), "dq": ("reduce_symbol",)}


def to_spec(instance) -> Spec:
    """The Spec an ``_instances.Instance`` row of a ``sdpa_bwd_<kernel>`` op names.

    ``head_dim_v`` is the resolved V head dim (the tile when it equals it); the Spec
    spells that 0. ``causal`` and ``has_bias`` are the exclusive class.
    """
    kernel = instance.op.removeprefix("sdpa_bwd_")
    if kernel not in KERNELS:
        raise ValueError(f"{instance.op!r} is not a backward op")
    knobs = instance.knobs
    if knobs["causal"] and knobs["has_bias"]:
        raise ValueError(f"causal and has_bias are exclusive: {instance!r}")
    cls = "bias" if knobs["has_bias"] else "causal" if knobs["causal"] else "plain"
    set_knobs = {}
    for name in KNOBS[kernel]:
        value = knobs[name]
        if value not in (UNSET_INT, UNSET_WORD):
            set_knobs[name] = value
    extra = (
        set(knobs)
        - set(ROW_FIELDS)
        - set(KNOBS[kernel])
        - set(NON_KEY_FIELDS.get(kernel, ()))
    )
    if extra:
        raise ValueError(f"{instance.op}: knobs {sorted(extra)} do not belong to it")
    return Spec(
        kernel,
        instance.dtype,
        knobs["head_dim"],
        cls,
        knobs["dform"],
        knobs["dropout"],
        knobs["padded"],
        knobs["floor"],
        knobs["head_dim_v"],
        knobs["hints"],
        **set_knobs,
    )
