# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward build schema: a ``Spec`` names one gfx950 backward object.

No FlyDSL import: the string grammar, the key, the hash and the validation are pure Python,
shared by ``_sdpa_bwd_cdna_build`` (which turns a Spec into an object) and its callers.

Canonical string (``Spec.key()``, parsed by ``Spec.parse``):

    kernel,dtype,tile,cls,dform,dropout,padded,floor,hdv,hints[;knob=value;...]

  kernel  : prep | dkdv | dq | dbias_reduce | dsink_reduce | merged
            (prep: dtype, tile, padded and the ``sink`` knob. dsink_reduce and dbias_reduce: dtype only, spelled
            ``dbias_reduce,<dtype>,32,plain,bk,0,0,tail,0,1``: the tile is the ladder's first rung as a placeholder,
            every other field its default. merged: dK/dV + dQ + dB in one launch, no prep; both roles take their
            policy-default schedules, and the dQ role is built at the dK/dV role's wave count)
  dtype   : f16 | bf16
  tile    : head-dim tile, one of LADDER
  cls     : plain | causal | bias
  dform   : rt (dropout is a runtime branch on ``rt_flags`` bit 1; the compiled dropout flag is 1)
            bk (dropout baked into the object; ``dropout`` = 0 or 1)
  dropout : compiled dropout flag (rt => 1)
  padded  : 0 exact tile | 1 padded head compiled (the head dim itself is a launch value)
  floor   : g1 (pinned tile, floor 0) | tail (rung below the tile) | <int>
  hdv     : V head dim carried in meta (0 = tile)
  hints   : 0|1  apply the kernel module's _COMPILE_HINTS
  knobs   : perf knobs, listed only when set, in KNOBS order. Unset = the FlyDSL policy value.
            dkdv: num_waves mfma_rows dkv_shards block_q waves_per_eu tight_registers num_stream_buffers gqa_pack
            dq  : num_waves block_m block_n mfma_rows waves_per_eu store_db bias_stage
            prep: sink (``sink=1``: the sink twin, a different object with a different ABI)
            merged: role_order (concat | lpt), store_db, gqa_pack
            ``gqa_pack=1`` packs the GQA group into the q-tile row axis of the dK/dV role; ``store_db=0`` builds a bias
            dQ object without the dB store; ``bias_stage=direct`` loads the bias directly instead of staging it in LDS.

``Spec.hash()`` is the first 12 hex digits of sha256(key()).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)
KERNELS = ("prep", "dkdv", "dq", "dbias_reduce", "dsink_reduce", "merged")
ROLE_ORDERS = ("concat", "lpt")
DTYPES = ("f16", "bf16")
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
    "merged": ("role_order", "store_db", "gqa_pack"),
}

ALL_KNOBS = tuple(dict.fromkeys(k for ks in KNOBS.values() for k in ks))
STR_KNOBS = {
    "bias_stage": ("direct",),
    "role_order": ROLE_ORDERS,
}  # knobs whose value is one of these words rather than an int


def rung_below(t):
    return max([r for r in LADDER if r < t], default=0)


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

    def __post_init__(self):
        def bad(msg):
            raise ValueError(f"{msg}: {self!r}")

        if self.kernel not in KERNELS:
            bad("kernel")
        if self.dtype not in DTYPES:
            bad("dtype")
        if self.tile not in LADDER:
            bad("tile")
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
        if self.kernel == "dbias_reduce" and (
            self.tile != LADDER[0]
            or self.cls != "plain"
            or self.dform != "bk"
            or self.dropout
            or self.padded
            or self.floor != "tail"
            or self.hdv
            or self.hints != 1
        ):
            bad(
                "dbias_reduce takes only dtype (tile is the placeholder 32, every other field its default)"
            )
        if self.kernel == "dsink_reduce" and (
            self.tile != LADDER[0]
            or self.cls != "plain"
            or self.dform != "bk"
            or self.dropout
            or self.padded
            or self.floor != "tail"
            or self.hdv
            or self.hints != 1
        ):
            bad(
                "dsink_reduce takes only dtype (tile is the placeholder 32, every other field its default)"
            )
        for k in ALL_KNOBS:
            v = getattr(self, k)
            if v is None:
                continue
            if k not in KNOBS[self.kernel]:
                bad(f"knob {k} does not apply to {self.kernel}")
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
            elif k == "gqa_pack":
                if v != 1:
                    bad("gqa_pack is spelled gqa_pack=1")
            elif k == "sink":
                if v != 1:
                    bad("sink is spelled sink=1")
            elif not isinstance(v, int) or v <= 0:
                bad(f"knob {k} must be a positive int")

    @property
    def knobs(self):
        """The set perf knobs, in canonical order."""
        return {
            k: getattr(self, k)
            for k in KNOBS[self.kernel]
            if getattr(self, k) is not None
        }

    @property
    def functional(self):
        """The functional key: everything but the perf knobs."""
        return (
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

    def key(self):
        head = ",".join(str(v) for v in self.functional)
        return head + "".join(f";{k}={v}" for k, v in self.knobs.items())

    def hash(self):
        return hashlib.sha256(self.key().encode()).hexdigest()[:12]

    @staticmethod
    def parse(s):
        head, *kv = s.split(";")
        kernel, dtype, tile, cls, dform, dropout, padded, floor, hdv, hints = (
            head.split(",")
        )
        knobs = {}
        for item in kv:
            n, v = item.split("=")
            if n not in ALL_KNOBS:
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
            **knobs,
        )


def dbias_reduce_spec(dtype: str) -> Spec:
    """The standalone dBias reduce object of @p dtype: the pin a merged bias object's reduce kernel is gated against."""
    return Spec("dbias_reduce", dtype, LADDER[0])


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
# named by the key, these only describe how it is shipped. The merged rows carry the
# symbol of the dBias reduce kernel their bias objects hold (`""` on the others).
NON_KEY_FIELDS = {"merged": ("reduce_symbol",)}


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
