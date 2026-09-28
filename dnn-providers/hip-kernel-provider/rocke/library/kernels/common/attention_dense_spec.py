# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture-neutral spec contract for dense attention kernels.

The base type contains only problem fields, geometry, and policies implemented by
both gfx942 and gfx950. Architecture-specific codegen knobs belong on concrete
subclasses in the owning kernel modules.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as _dataclass_fields
from types import MappingProxyType

from rocke.core.ir import BF16, F16
from rocke.helpers.spec import kernel_name_join


_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# Shared query/KV geometry only. LDS layout choices are architecture-specific.
DENSE_TILE_GEOMETRIES = MappingProxyType(
    {
        "default": MappingProxyType({"block_m": 256, "block_n": 64}),
        "bm128": MappingProxyType({"block_m": 128, "block_n": 64}),
    }
)
DEFAULT_DENSE_TILE_GEOMETRY = DENSE_TILE_GEOMETRIES["default"]

_COMMON_PERSIST_DECODES = frozenset(
    # bt_hkv_minor is the digit order BVGQ -- batch fastest, then kv head, gqa
    # lane, query block -- carrying the same causal fold as hkv_minor. It is
    # deliberately the SAME NAME as the non-persistent grid order of the same
    # digit order: one mapping, named once, reached two ways.
    # qb_major_fold is qb_major's digit order (BGVQ) with the causal FOLD on
    # the query-block digit instead of the plain ascending walk. Same mapping of
    # work items to (batch, head), different traversal -- so it is a traversal
    # variant of qb_major, not a new locality class, and it is named to say so.
    #
    # swz_head_first[_rev|_fold]: the persistent-path port of the non-persistent
    # grid order of the same name (Zhang et al., arXiv 2511.02132, Fig. 11) --
    # SAME NAME, same head-axis contiguous-band split, reached as a work-index
    # decode instead of a 3-D grid. Unlike bt_hkv_minor/qb_major_fold, this
    # decode is EXPLICIT-ONLY and never auto-selected -- see
    # platform/dsl_docs/architecture/attention_thread_block_mapping.md for why.
    {"auto", "qb_major", "qb_major_fold", "hkv_major", "hkv_minor",
     "bt_hkv_minor", "swz_head_first", "swz_head_first_rev",
     "swz_head_first_fold"}
)

# XCD count every CDNA3/CDNA4 part this spec targets exposes. Only ``hkv_minor``
# reads it: that decode's whole point is that the hardware's round-robin
# ``xcd = linear_wgid % num_xcds`` lands on the kv-head index, which requires the
# persistent CTA count to be a multiple of it. A 6-XCD part changes the modulus
# and silently turns the decode into an arbitrary permutation, which is why the
# legality check below is a hard reject rather than a best-effort.
_PERSIST_XCD_MODULUS = 8


def xcd_partitionable(num_kv_heads: int, modulus: int = _PERSIST_XCD_MODULUS) -> bool:
    """Whether ``hkv_minor``'s kv-head -> XCD map confines each kv-head to ONE XCD.

    The decode emits ``hkv = wi % num_kv_heads`` while the hardware assigns
    ``xcd = wi % modulus``, so the two indices agree only when one of the moduli
    divides the other:

    * ``modulus % num_kv_heads == 0`` (1, 2, 4, 8): ``wi % modulus`` determines
      ``wi % num_kv_heads``, so each XCD sees exactly one kv-head. The head is
      replicated across ``modulus // num_kv_heads`` XCDs, which costs L2 capacity
      but keeps every XCD's working set to a single head.
    * ``num_kv_heads % modulus == 0`` (8, 16, 24, 32, 40, 48, ...): ``wi %
      num_kv_heads`` determines ``wi % modulus``, so each kv-head lands on exactly
      one XCD and each XCD holds ``num_kv_heads // modulus`` of them.

    Anything else (12, 20, 28, 10, 6, ...) has ``1 < gcd < min(...)``, which splits
    a single kv-head across ``num_kv_heads // gcd`` XCDs -- the case this rejects.

    NOTE this supersedes an earlier power-of-2 test. Power-of-2 is sufficient but
    NOT necessary: it wrongly rejected 24/40/48, where ``modulus`` divides the head
    count and the identity holds exactly. It never wrongly accepted anything, so the
    old guard was conservative rather than unsound -- but it excluded real shapes.
    """
    if num_kv_heads <= 0:
        return False
    return num_kv_heads % modulus == 0 or modulus % num_kv_heads == 0


# --- EXPERIMENTAL: generalized work-index ordering ---------------------------
#
# Every shipped thread-block mapping is one point in a single space: a mixed-radix
# decomposition of the linear work index over four digits. Naming them lets the
# whole space be swept instead of hand-writing one decode per point.
#
#   Q  blk  query-block counter   radix NQB
#   B  bt   batch element         radix batch
#   V  hkv  kv head               radix num_kv_heads
#   G  hql  query head within gqa radix gqa            (hq = hkv*gqa + hql)
#
# An order string lists the digits FASTEST FIRST, so "VBGQ" means
# ``wi = ((blk*gqa + hql)*B + bt)*Hkv + hkv`` -- the shipped ``hkv_minor``.
# The shipped decodes in this notation:
#
#   hkv_minor  VBGQ        hkv_major  BGQV        qb_major   BGVQ
#
# Note qb_major and hkv_major differ ONLY in their two SLOWEST digits, which makes
# them a built-in test of "do the slow digits matter at all".
DIGIT_LETTERS = ("Q", "B", "V", "G")

# Canonical order strings for the shipped decodes; the sweep uses these to assert
# the generic path reproduces the hand-written ones.
SHIPPED_DIGIT_ORDERS = MappingProxyType(
    {"hkv_minor": "VBGQ", "hkv_major": "BGQV", "qb_major": "BGVQ"}
)

QB_TRAVERSALS = ("asc", "rev", "fold")

# EXPERIMENTAL kv-phase split orders. A SEPARATE FAMILY, not extra permutations.
#
# The motivation: at B>1 the distinct K/V data is indexed by the PAIR (bt, hkv),
# so there are B*Hkv distinct tensors and the ideal schedule gives each XCD one
# of them at a time, sweeping B*Hkv/num_xcds phases. Expressing that needs the
# fused identity F = bt*Hkv + hkv to appear in TWO positions at once -- its low
# log2(num_xcds) bits fastest, so xcd = wi % num_xcds selects the tensor, and
# its high bits slowest, so the machine sweeps phases. A permutation of four
# digits cannot do that, which is why this is a family rather than an axis.
#
# Notation: lowercase 'x' is F's low part (radix num_xcds) and must be FASTEST
# -- that is the whole point, since it is what xcd = wi % num_xcds reads.
# Lowercase 'y' is F's high part; 'G' and 'Q' are the ordinary gqa-lane and
# query-block digits. The three non-x characters are free, so the family has
# 3! = 6 members.
#
# y does NOT have to be slowest. Putting it slowest (xGQy, xQGy) maximises the
# phase effect the idea targets: one K/V tensor per chiplet, swept over
# B*Hkv/num_xcds phases. Putting Q slowest instead (xGyQ, xQyG) gives up some of
# that to buy the property every measured winner has -- a query block that never
# wraps, so each CTA sweeps the whole causal cost range instead of being pinned
# to one block. The two are a real trade, not a preference, which is why both
# are in the family.
#
# IMPORTANT -- the split is a relabelling, not a new mapping, when Hkv == 8
# exactly: then F % 8 == hkv and F // 8 == bt, so x and y ARE the V and B
# digits and every split order equals a permutation (xGQy == VGQB,
# xQGy == VQGB, xGyQ == VGBQ, xQyG == VQBG). It is genuinely new only when
# Hkv != num_xcds -- Hkv in {4, 10, 32, 40} among the shapes in use -- because
# only then does the fused identity cut across a digit boundary. Measuring the
# family on Hkv == 8 shapes measures permutations already swept.
KV_SPLIT_ORDERS = tuple(
    "x" + "".join(p) for p in __import__("itertools").permutations("yGQ")
)

# Public alias: builders reconstruct the fused identity as x + MODULUS*y and
# must use the same modulus the split plan did.
KV_SPLIT_MODULUS = _PERSIST_XCD_MODULUS


def is_kv_split(order: str) -> bool:
    """Whether ``order`` names the kv-phase split family rather than one of the
    24 digit permutations. Case-sensitive on purpose: the split letters are
    lowercase precisely so an order string cannot be mistaken for a permutation
    after an ``.upper()`` somewhere in a name path."""
    return order in KV_SPLIT_ORDERS


def kv_split_steps(order: str, radices: dict) -> tuple[tuple, ...]:
    """``((name, radix, is_tail), ...)`` fastest-first for a kv-split order.

    Radix-1 steps are dropped, exactly as :func:`decode_plan` elides them: at
    B*Hkv == num_xcds there is a single phase and ``y`` disappears; at gqa == 1
    the ``G`` step does.
    """
    if order not in KV_SPLIT_ORDERS:
        raise ValueError(f"not a kv-split order: {order!r}")
    fused = radices["B"] * radices["V"]
    if fused % _PERSIST_XCD_MODULUS:
        # Deliberately a hard error, not a silent fallback. Without it the decode
        # produces batch indices >= B -- measured 48 of 320 work items at
        # 40/10 B=1 -- which reads out of bounds rather than merely mis-ordering.
        raise ValueError(
            f"kv-split needs batch*num_kv_heads ({fused}) divisible by "
            f"{_PERSIST_XCD_MODULUS}"
        )
    rad = {"x": _PERSIST_XCD_MODULUS,
           "y": fused // _PERSIST_XCD_MODULUS,
           "G": radices["G"], "Q": radices["Q"]}
    steps = [(c, rad[c]) for c in order]
    live = [(n, r) for n, r in steps if r > 1]
    return tuple((n, r, i + 1 >= len(live)) for i, (n, r) in enumerate(live))


def qb_rotation_period(order: str, radices: dict) -> int:
    """Radix product up to and INCLUDING the query-block digit, or 0 if the
    query block is the slowest live digit.

    ``phase = wi // period`` counts how many times the query-block digit has
    wrapped, so rotating the block by it hands each CTA a different block on
    each pass. A period of 0 means the rotation is provably inert: the query
    block is already the slowest digit, so it never wraps and every CTA already
    sweeps the whole range. The six permutations ending in Q are exactly that
    case -- and they are the ones that measured best, which is the same fact
    seen from the other side.
    """
    if is_kv_split(order):
        names = list(order)
        rad = dict(radices)
        rad["x"] = _PERSIST_XCD_MODULUS
        rad["y"] = (radices["B"] * radices["V"]) // _PERSIST_XCD_MODULUS
    else:
        names = list(parse_digit_order(order))
        rad = radices
    live = [n for n in names if rad[n] > 1]
    if "Q" not in live or live[-1] == "Q":
        return 0
    period = 1
    for n in live:
        period *= rad[n]
        if n == "Q":
            break
    return period


def parse_digit_order(order: str) -> tuple[str, ...]:
    """Validate an order string and return it as a tuple, fastest digit first.

    Raises rather than returning a flag: an unvalidated order silently produces a
    non-bijective decode, which does not crash -- it writes some query rows twice
    and leaves others unwritten, i.e. a wrong-answer bug that looks like a
    tolerance failure.
    """
    if is_kv_split(order):
        raise ValueError(
            f"{order!r} is a kv-split order, not a digit permutation; it is a "
            "separate family and has no four-digit letter tuple"
        )
    letters = tuple(order.upper())
    if len(letters) != len(DIGIT_LETTERS) or set(letters) != set(DIGIT_LETTERS):
        raise ValueError(
            f"digit_order must be a permutation of {''.join(DIGIT_LETTERS)!r}, "
            f"got {order!r}"
        )
    return letters


def digit_radices(nqb: int, batch: int, num_kv_heads: int, gqa: int) -> dict:
    """Radix of each digit for a concrete shape."""
    return {"Q": int(nqb), "B": int(batch), "V": int(num_kv_heads), "G": int(gqa)}


def decode_plan(order: str, radices: dict) -> tuple[tuple, ...]:
    """The emit plan for one order: a tuple of ``(kind, digits, radix)`` steps.

    Three optimizations are applied here rather than in each builder, because
    without them the generic decode emits more integer ops than the hand-written
    decode it is being compared against -- and the comparison, not the codegen, is
    what the sweep is for:

    * ``elide``  -- a radix-1 digit is constant 0 and costs nothing. This is also
      why the space collapses to 6 orders at ``B == 1`` and 2 for MHA.
    * ``tail``   -- the slowest live digit needs no ``mod``; the remaining
      quotient is already in range.
    * ``fuse``   -- adjacent ``G`` then ``V`` (hql faster than hkv) decode as ONE
      digit of radix ``Hq``, since ``hq = hkv*gqa + hql`` by definition. This is
      exactly what ``qb_major`` does by hand: two ops, not four.
    """
    letters = parse_digit_order(order)
    live = [d for d in letters if radices[d] > 1]
    steps, i = [], 0
    while i < len(live):
        d = live[i]
        nxt = live[i + 1] if i + 1 < len(live) else None
        # Fuse only G-then-V, and only if they were adjacent in the FULL order
        # too -- an elided digit between them would make the fused value wrong.
        adjacent = nxt is not None and abs(letters.index(d) - letters.index(nxt)) == 1
        if d == "G" and nxt == "V" and adjacent:
            radix = radices["G"] * radices["V"]
            steps.append(("fuse", ("G", "V"), radix, i + 2 >= len(live)))
            i += 2
            continue
        steps.append(("digit", (d,), radices[d], i + 1 >= len(live)))
        i += 1
    for d in letters:
        if radices[d] <= 1:
            steps.append(("elide", (d,), 1, False))
    return tuple(steps)


def split_fused_to_digits(fused: int, radices: dict, bt_minor: bool):
    """Split the fused K/V identity back into ``(bt, hkv)``.

    Two conventions, and the choice is NOT cosmetic -- it decides where the
    family collapses onto an ordinary permutation, because the collapse happens
    exactly when the fused index lines up with a digit boundary:

      bt_minor=False  F = bt*Hkv + hkv   degenerate when Hkv == num_xcds
      bt_minor=True   F = hkv*B  + bt    degenerate when B   == num_xcds

    The structural metrics are identical either way -- both fuse the same pair
    into the same 8-way residue structure, so only WHICH tensor lands on which
    chiplet differs, not how many. What does differ is the address distribution:
    with K laid out [B, S, Hkv, D], the 8 co-resident tensors are stride-D
    interleaved under bt_minor=False and a whole batch element apart under
    bt_minor=True. That is a DRAM channel question, not an L2 one.
    """
    if bt_minor:
        return fused % radices["B"], fused // radices["B"]
    return fused // radices["V"], fused % radices["V"]


def decode_reference(wi: int, order: str, radices: dict,
                     phase_rotate: bool = False,
                     bt_minor: bool = False) -> dict:
    """Pure-Python oracle for the generic decode -- the same arithmetic a builder
    emits, used to prove equivalence with the hand-written decodes without a GPU.

    Covers both families and the optional query-block phase rotation, so one
    oracle checks everything the builders can emit.
    """
    if is_kv_split(order):
        out = {d: 0 for d in DIGIT_LETTERS}
        vals, cur = {}, wi
        for name, radix, is_tail in kv_split_steps(order, radices):
            vals[name] = cur if is_tail else cur % radix
            if not is_tail:
                cur //= radix
        fused = (vals.get("x", 0)
                 + _PERSIST_XCD_MODULUS * vals.get("y", 0))
        out["B"], out["V"] = split_fused_to_digits(fused, radices, bt_minor)
        out["G"] = vals.get("G", 0)
        out["Q"] = vals.get("Q", 0)
        _apply_rotation(out, wi, order, radices, phase_rotate)
        return out
    out = {d: 0 for d in DIGIT_LETTERS}
    cur = wi
    for kind, digits, radix, is_tail in decode_plan(order, radices):
        if kind == "elide":
            continue
        val = cur if is_tail else cur % radix
        if kind == "fuse":
            out["G"] = val % radices["G"]
            out["V"] = val // radices["G"]
        else:
            out[digits[0]] = val
        if not is_tail:
            cur //= radix
    _apply_rotation(out, wi, order, radices, phase_rotate)
    return out


def _apply_rotation(out: dict, wi: int, order: str, radices: dict,
                    enabled: bool) -> None:
    """Rotate the decoded query block by the pass index, in place."""
    if not enabled:
        return
    period = qb_rotation_period(order, radices)
    if period:
        out["Q"] = (out["Q"] + wi // period) % radices["Q"]


def traverse_qb(blk: int, nqb: int, traversal: str) -> int:
    """Map a block counter in ``[0, NQB)`` to a query-block index.

    ``asc`` leaves the most expensive block last; ``rev`` leaves the cheapest last
    (LPT on a dispatch queue); ``fold`` pairs a cheap and an expensive block so a
    CTA striding both halves has constant causal cost.
    """
    if traversal not in QB_TRAVERSALS:
        raise ValueError(f"qb_traversal must be one of {QB_TRAVERSALS}, got {traversal!r}")
    if nqb <= 1 or traversal == "asc":
        return blk
    if traversal == "rev":
        return nqb - 1 - blk
    half = nqb // 2
    return blk if blk < half else nqb - 1 + half - blk


# Signed 32-bit ceiling for tensor extents. See ``check_dense_spec_preflight``
# check 4 for why the SIGNED bound binds even though the buffer-resource
# num_records field is unsigned in hardware.
INT32_LIMIT = 2**31


@dataclass(frozen=True)
class AttentionDenseSpec:
    """Shared compile-time problem and geometry for dense attention."""

    # Problem shape and semantics.
    batch: int
    seqlen_q: int
    seqlen_kv: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    causal: bool = True
    dtype: str = "bf16"
    sliding_window: int = 0
    ragged: bool = False
    varlen: bool = False

    # Geometry and common implementation policy.
    block_m: int = DEFAULT_DENSE_TILE_GEOMETRY["block_m"]
    block_n: int = DEFAULT_DENSE_TILE_GEOMETRY["block_n"]
    waves_per_eu: int = 2
    lds_k_group_pad: int = 8
    persistent: bool = False
    num_persistent: int = 256
    interleave: bool = False
    persist_decode: str = "auto"
    # Historical shared naming/behavior flag. Kept in the base for compatibility;
    # architecture-specific migration can move it independently in a follow-up.
    lazy_rescale: bool = True

    # force_baked_shape: MEASUREMENT CONTROL, not a tuning knob. Forces
    # ``runtime_shape`` off, so the body bakes batch/seqlen instead of reading
    # them as kernargs.
    #
    # It exists because ``runtime_shape`` is not a free variable: it switches
    # shape reads between kernargs and constants, and on gfx950 it also flips
    # the body to a baked k-tile trip count. Comparing two mappings that differ
    # in it measures the flag, not the mapping. The generic ``digit_order`` path
    # bakes every radix and so is always runtime_shape=False; without this field
    # a generic-vs-named comparison on a runtime_shape path is confounded by
    # construction, which is exactly how one such comparison was already
    # misread.
    #
    # Setting it cannot cause a cache collision: flipping runtime_shape OFF puts
    # sq/sk/b{batch} BACK into the kernel name, so a forced spec and an unforced
    # one at the same shape are already distinct names. Where runtime_shape is
    # False anyway (persistent, ragged, paged, sliding-window) this is inert --
    # same name, same IR.
    force_baked_shape: bool = False

    # Problem modes currently implemented only by a subset of architectures.
    # They remain shared semantic fields so supports_* can reject unsupported
    # requests explicitly; unlike codegen knobs, they never silently no-op.
    paged: bool = False
    block_size: int = 0
    num_kv_blocks: int = 0
    use_sinks: bool = False

    def supported_persist_decodes(self) -> frozenset[str]:
        """Decode values the concrete kernel type can actually emit."""
        return _COMMON_PERSIST_DECODES

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_IR:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_IR)}, got {self.dtype}"
            )
        if self.block_m <= 0:
            raise ValueError(f"block_m must be positive, got {self.block_m}")
        if self.block_n <= 0 or self.block_n % 32 != 0:
            raise ValueError(
                f"block_n must be a positive multiple of 32, got {self.block_n}"
            )
        if self.head_size not in (64, 128):
            raise ValueError(f"head_size must be 64 or 128, got {self.head_size}")
        if self.lds_k_group_pad < 0 or self.lds_k_group_pad % 8 != 0:
            raise ValueError(
                "lds_k_group_pad must be a non-negative multiple of 8 bf16 "
                "elements (16 bytes) so the K group pitch stays "
                f"ds_read_b128-aligned, got {self.lds_k_group_pad}"
            )

        if self.ragged:
            if self.seqlen_q <= 0 or self.seqlen_kv <= 0:
                raise ValueError("ragged requires positive seqlen_q/seqlen_kv")
            if self.seqlen_q != self.seqlen_kv:
                raise ValueError(
                    "ragged is self-attention only (seqlen_q == seqlen_kv), got "
                    f"{self.seqlen_q} != {self.seqlen_kv}"
                )
            if self.varlen:
                raise ValueError("ragged is not supported with varlen")
            if self.sliding_window > 0:
                raise ValueError("ragged is not supported with sliding_window")
        else:
            if self.seqlen_q % self.block_m != 0:
                raise ValueError(
                    f"seqlen_q must be a multiple of block_m={self.block_m}, "
                    f"got {self.seqlen_q}"
                )
            if self.seqlen_kv % self.block_n != 0:
                raise ValueError(
                    f"seqlen_kv must be a multiple of block_n={self.block_n}, "
                    f"got {self.seqlen_kv}"
                )

        if self.num_kv_heads == 0 or self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be a positive "
                f"multiple of num_kv_heads ({self.num_kv_heads})"
            )
        if self.persistent and self.num_persistent <= 0:
            raise ValueError(
                f"num_persistent must be positive, got {self.num_persistent}"
            )
        if self.persist_decode not in self.supported_persist_decodes():
            raise ValueError(
                f"persist_decode must be one of "
                f"{sorted(self.supported_persist_decodes())}, "
                f"got {self.persist_decode!r}"
            )
        if self.persist_decode == "hkv_minor":
            # hkv_minor exists to make the HARDWARE's workgroup->XCD round-robin
            # land on the kv-head index: it places hkv in the LOW digit of the
            # work index so that ``xcd == wi % num_xcds == hkv``. Each condition
            # below is load-bearing for that identity, and a violated one does
            # not produce a wrong answer -- it produces a CORRECT answer with the
            # locality silently gone, which is the failure mode that reads as "the
            # optimization does not work" instead of "the spec was illegal".
            if not self.persistent:
                raise ValueError("persist_decode='hkv_minor' requires persistent=True")
            if not self.causal:
                # The query-block fold carried over from hkv_major pairs a cheap
                # and an expensive qb; with uniform (non-causal) cost that fold is
                # a no-op reordering and the decode has no reason to exist.
                raise ValueError("persist_decode='hkv_minor' requires causal=True")
            # NOTE two former guards were REMOVED here, deliberately:
            #   num_persistent % num_xcds != 0
            #   not xcd_partitionable(num_kv_heads)
            # Neither was a correctness condition. The decode is a mixed-radix
            # decomposition, hence a bijection for ANY radix set; both conditions
            # only decide whether xcd = wi % num_xcds happens to coincide with
            # the kv-head index, which is a SPEED property.
            #
            # Measured: the same mapping (digit order VBGQ) ran on Hkv=10
            # (gcd(Hkv, num_xcds) == 2), exactly the geometry the second guard
            # rejected -- 161 cells, zero correctness failures, max abs error
            # ~3e-4. It is simply slower there, a median ~4% off the best variant
            # on that path, and ties the best on some shapes. For the first
            # guard: the shipped dispatch always passes a CU count, which is a
            # multiple of num_xcds, so other values are legal but pointless and
            # yield a worse mapping rather than a wrong answer.
            #
            # Enforcing a performance property as a construction error had a real
            # cost: it made legal shapes UNBUILDABLE, which is why 40/10 appeared
            # as a skip in every sweep. This is the second relaxation of the same
            # guard (it was pow2-only before H4), so the mechanism was fixed
            # rather than the threshold moved again. ``xcd_partitionable`` stays
            # as the PREDICTOR the auto policy keys on -- see
            # ``AttentionDenseSpec.resolved_persist_decode`` -- because Hkv vs
            # num_xcds is the one shape variable that moves the policy's regret.
            if self.ragged or self.varlen or self.paged:
                raise ValueError(
                    "persist_decode='hkv_minor' is validated only for aligned "
                    "dense attention (not ragged/varlen/paged)"
                )
        if self.persist_decode.startswith("swz_head_first"):
            # Unlike hkv_minor's relaxed guards above, this ONE condition is a
            # real bijection requirement, not a locality predictor. The decode
            # splits the query-head axis into a fast band `a` (radix
            # num_xcds) and a slow residual `c` (radix Hq // num_xcds):
            #   hq = a * (Hq // num_xcds) + c
            # If Hq % num_xcds != 0, `Hq // num_xcds` floors down and `hq`
            # ranges over only `num_xcds * (Hq // num_xcds) < Hq` values --
            # some query heads are never written and others get written twice.
            # Same requirement already enforced for the non-persistent grid
            # order of the same name.
            if not self.persistent:
                raise ValueError(
                    f"persist_decode={self.persist_decode!r} requires "
                    "persistent=True"
                )
            if self.num_query_heads % _PERSIST_XCD_MODULUS:
                raise ValueError(
                    f"persist_decode={self.persist_decode!r} needs "
                    f"num_query_heads ({self.num_query_heads}) divisible by "
                    f"{_PERSIST_XCD_MODULUS}"
                )
        if self.sliding_window < 0:
            raise ValueError(f"sliding_window must be >= 0, got {self.sliding_window}")
        if self.sliding_window > 0:
            if not self.causal:
                raise ValueError("sliding_window>0 requires causal=True")
            if self.sliding_window % self.block_n:
                raise ValueError(
                    f"sliding_window ({self.sliding_window}) must be a multiple "
                    f"of block_n={self.block_n}"
                )
        if self.varlen:
            if self.persistent:
                raise ValueError("varlen is not supported with persistent=True")
            if not self.causal:
                raise ValueError("varlen requires causal=True")
        if not 1 <= self.waves_per_eu <= 8:
            raise ValueError(f"waves_per_eu must be in [1, 8], got {self.waves_per_eu}")

        if self.paged:
            if self.block_size <= 0:
                raise ValueError("paged=True requires block_size > 0")
            if self.block_size & (self.block_size - 1):
                raise ValueError(
                    f"paged block_size ({self.block_size}) must be a power of two"
                )
            if self.block_n % self.block_size:
                raise ValueError(
                    f"block_n ({self.block_n}) must be a multiple of page "
                    f"block_size ({self.block_size})"
                )
            rows_per_wave = self.block_n // self.num_waves
            if self.block_size < rows_per_wave or self.block_size % rows_per_wave:
                raise ValueError(
                    f"paged block_size ({self.block_size}) must be >= and a "
                    f"multiple of ROWS_PER_WAVE ({rows_per_wave})"
                )
            if self.num_kv_blocks <= 0:
                raise ValueError("paged=True requires num_kv_blocks > 0")
            cache_bytes = (
                self.num_kv_blocks
                * self.block_size
                * self.num_kv_heads
                * self.head_size
                * 2
            )
            if cache_bytes > 2**31 - 1:
                raise ValueError(
                    f"paged cache {cache_bytes} B exceeds i32 addressing (2 GiB)"
                )
            if self.batch != 1:
                raise ValueError("paged multi-sequence (batch>1) not yet implemented")
            if self.varlen:
                raise ValueError("paged varlen not yet implemented (single-seq only)")
            if self.persistent:
                raise ValueError(
                    "paged + persistent not yet implemented "
                    "(persistent builder is contiguous-only)"
                )
            if self.head_size != 128:
                raise ValueError("paged not yet implemented for head_size != 128")
            if self.dtype not in ("fp16", "bf16"):
                raise ValueError(f"paged not yet implemented for dtype={self.dtype}")
            if self.sliding_window <= 0:
                raise ValueError(
                    "paged not yet implemented for plain-causal "
                    "(sliding_window>0 only)"
                )
        if self.use_sinks and self.paged:
            raise ValueError("use_sinks is not yet supported with paged KV")
        if self.use_sinks and self.varlen:
            raise ValueError("use_sinks is not yet supported with varlen")

    @property
    def num_waves(self) -> int:
        return self.block_m // 32

    @property
    def dtype_ir(self):
        return _DTYPE_IR[self.dtype]

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def resolved_persist_decode(self) -> str:
        """Resolve the common auto policy to hkv-major or qb-major.

        ``xcd_partitionable`` is consulted here as a PREDICTOR, which is the role
        it was demoted to when the hkv_minor construction guards were removed
        (see ``__post_init__``). A kv head that does not divide or get divided by
        the XCD count spreads across several chiplets, so the kv-head-locality
        decodes have nothing to exploit -- that is a reason to not CHOOSE them,
        not a reason to refuse to build them.
        """
        if self.persist_decode != "auto":
            return self.persist_decode
        # ARCH-DEPENDENT, and deliberately not chosen here. The base stays on
        # the conservative choice so a NEW arch does not inherit another arch's
        # result. Both shipped arches override it -- gfx942 and gfx950 each
        # return `_batch_conditional_auto_decode()` -- so this base value is
        # reached only by an arch that has not been measured yet.
        #
        # CORRECTION. This comment used to record bt_hkv_minor as "-1.5% on
        # gfx950, where the hand-written qb_major decode is fastest". That
        # measurement was an artifact: on gfx950 ONLY, bt_hkv_minor emitted no
        # decode tag -- an arch override of _persist_decode_name_part restated
        # the shared map instead of extending it -- so it shared qb_major's
        # symbol, and _DENSE_LAUNCHER_CACHE is keyed on that symbol. The A/B
        # timed ONE binary twice; the -1.5% was noise between two runs of
        # qb_major. Re-measured with distinct symbols, over 12 geometries
        # including non-pow2 seqlens, MHA and non-pow2 Hkv, two passes:
        # bt_hkv_minor is a clear WIN on gfx950 as well, by a larger margin than
        # on gfx942, and its generic twin (digit order BVGQ + fold) agrees with
        # it to within the noise floor on both arches.
        #
        # That re-measurement has since landed: gfx950 no longer selects
        # qb_major here. Its `_aligned_causal_auto_decode` override returns the
        # same B-conditional rule as gfx942, and because this arch enters the
        # persistent path whenever there is enough work to fill the grid, BOTH
        # arms of that rule are live on its default path. Do not re-derive the
        # old number from this comment.
        #
        # Why the sweep did not catch any of it: qb_major is the digit order
        # BGVQ with the ASCENDING traversal, and `asc` was pruned after the
        # screen -- so the shipped decode's exact configuration was never in the
        # full sweep, and no persistent qb_major reference column was carried to
        # catch the omission. The prune was sound for the generic decode path it
        # was measured on; it did not transfer to a hand-written decode that
        # skips the fold entirely.
        #
        # Gated on ALIGNED DENSE CAUSAL, which is exactly the envelope the sweep
        # covered. causal, because the decode carries the causal fold: with every
        # query block costing the same there is nothing to balance and the
        # reordering is pure churn. The rest -- ragged / varlen / paged /
        # sliding-window -- were never measured, and each changes either the
        # per-item cost profile the fold assumes (sliding window bounds the k
        # range, so cost stops being triangular) or the work-item space itself.
        # Selecting a new decode for them would be extrapolation, so they keep
        # the previous fallback.
        if (self.causal and not self.ragged and not self.varlen
                and not self.paged and self.sliding_window == 0):
            return self._aligned_causal_auto_decode()
        return "qb_major"

    def _aligned_causal_auto_decode(self) -> str:
        """Auto decode for aligned dense causal. Overridden per arch."""
        return "qb_major"

    def _batch_conditional_auto_decode(self) -> str:
        """The measured B-conditional persistent rule, shared by both arches.

        Both candidates put ``bt`` in the fastest digit and differ only in the
        SECOND: bt_hkv_minor (BVGQ) puts the kv head there, qb_major_fold (BGVQ)
        the gqa lane. The hardware assigns ``xcd = wi % num_xcds`` and
        ``wi = bt + B*X``, so how much that second digit reaches the chiplet map
        is decided entirely by ``gcd(B, num_xcds)``:

            B=1  -> 3 bits: the second digit alone picks the XCD
            B=2  -> 2 bits
            B=4  -> 1 bit
            8|B  -> 0 bits: the two orders place every work item on the SAME XCD

        So they are furthest apart at B=1 -- where BVGQ gives each chiplet
        exactly ONE kv head's K/V and BGVQ gives it four -- and their XCD
        placement becomes identical once the batch digit alone saturates the
        round robin. Past that point they still differ in WHICH kv head each
        work item carries, and there the ranking REVERSES: over the resident
        wave BGVQ holds fewer distinct kv heads per chiplet, and it measured
        ahead on 9/10 gfx942 and 5/10 gfx950 configs at B>=16, by up to 10%.

        Hence the split at num_xcds rather than at some fitted threshold: it is
        where the mechanism changes, not where a curve happened to cross.
        """
        return ("bt_hkv_minor" if self.batch < _PERSIST_XCD_MODULUS
                else "qb_major_fold")

    @property
    def runtime_param_fields(self) -> tuple[str, ...]:
        """Spec fields this kernel reads as runtime kernel params instead of
        baking into the body, so they must NOT split cache identity -- one
        compiled kernel serves every value of them.

        Empty here: the base contract is that the whole problem shape is baked.
        A subclass whose body emits a field as a param declares it by overriding
        this, and ``attention_dense_cache_key`` excludes it. Declaring a field
        that the body still bakes is a cache collision, which is why the
        declaration lives on the spec that owns the body rather than in the
        shared key function.

        NOTE: this does not drive the symbol name -- each spec curates its own
        name parts by hand, and the two must be kept in sync deliberately.

        Getting that wrong is not cosmetic: a field dropped from the key but kept
        in the name gives two specs that share ONE cache slot two DIFFERENT
        symbols, and ``run_attention_dense_torch``'s ``assert art.kernel_name ==
        spec.kernel_name()`` then fires on the second one served from that slot.
        ``Gfx942AttentionDenseSpec`` is the live example: it appends ``_b{batch}``
        (batch sized its buffer-resource extents), so declaring ``batch`` here
        obliged it to drop that token in the same change.

        Deriving the name FROM this tuple would make the two correct by
        construction. It does not today, which is the blocker for AOT packaging
        and per-batch specialization, where the symbol IS the identity.
        """
        return ()

    def _layout_name_parts(self) -> tuple[str, ...]:
        return ()

    def _shape_name_parts(self) -> tuple[str, ...]:
        """Name tokens for the baked problem shape. A subclass whose kernel takes
        the shape as runtime params (so one kernel serves every batch/seqlen)
        overrides this to drop sq/sk from the symbol name."""
        return (f"sq{self.seqlen_q}", f"sk{self.seqlen_kv}")

    def _algorithm_name_parts(self) -> tuple[str, ...]:
        return ("lazyrs",) if self.lazy_rescale else ()

    def _persist_decode_name_part(self) -> str:
        """Symbol tag for the resolved decode -- EVERY non-qb_major decode must
        return a distinct non-empty string.

        ``qb_major`` is the untagged baseline, so a decode that emits different IR
        and returns ``""`` collides with it in ``_DENSE_LAUNCHER_CACHE``, which is
        keyed on ``kernel_name`` and whose ``assert art.kernel_name == key`` PASSES
        on a collision -- serving the stale binary. An A/B against the baseline then
        times the same kernel twice and reports ~1.000x, i.e. the knob looks inert
        rather than broken. This kernel has shipped that exact bug twice
        (``batch``, then ``waves_per_eu``); see the Gfx942AttentionDenseSpec
        docstring.
        """
        return {
            "hkv_major": "hkvmaj",
            "hkv_minor": "hkvmin",
            "bt_hkv_minor": "bthkvmin",
            "qb_major_fold": "qbmajfold",
            "swz_head_first": "swzhf",
            "swz_head_first_rev": "swzhfrev",
            "swz_head_first_fold": "swzhffold",
        }.get(self.resolved_persist_decode, "")

    def kernel_name(self) -> str:
        parts = [
            "rocke_attention_dense",
            f"d{self.head_size}",
            f"hq{self.num_query_heads}",
            f"kv{self.num_kv_heads}",
            f"bn{self.block_n}",
            self.dtype,
        ]
        if self.block_m != DEFAULT_DENSE_TILE_GEOMETRY["block_m"]:
            parts.append(f"bm{self.block_m}")
        if 128 // self.head_size > 1:
            parts.append(f"kpad{self.lds_k_group_pad}")
        parts.extend(self._layout_name_parts())
        parts.extend(self._shape_name_parts())
        parts.append("causal" if self.causal else "full")
        if self.ragged:
            parts.append("ragged")
        if self.sliding_window > 0:
            parts.append(f"swa{self.sliding_window}")
        if self.use_sinks:
            parts.append("sinks")
        if self.varlen:
            parts.append("varlen")
        if self.paged:
            parts.extend((f"pgd{self.block_size}", f"nb{self.num_kv_blocks}"))
        parts.extend(self._algorithm_name_parts())
        if self.persistent:
            parts.append(f"persist{self.num_persistent}")
            decode = self._persist_decode_name_part()
            if decode:
                parts.append(decode)
            if self.interleave:
                parts.append("intl")
        return kernel_name_join(*parts)


def attention_dense_cache_key(spec: AttentionDenseSpec, *, arch: str) -> tuple:
    """Cache identity: the arch, the concrete spec type, and every spec field
    that affects codegen.

    Fields the spec declares in ``runtime_param_fields`` are excluded -- its
    kernel reads them at runtime, so one compiled binary serves every value and
    they must not split identity. This is what collapses the AOT
    batch x seqlen instance explosion on the arches that opt in. Specs that
    declare nothing (the default) keep every field, so their identity is
    unchanged.

    The shape is the same on both paths, so callers never branch on it. The
    class object rather than its name keeps subclass identity
    (``Gfx942AttentionDenseSpec`` is not ``Gfx950AttentionDenseSpec``) without
    stringly-typing it; if a stable on-disk identity is ever needed, that wants
    ``__qualname__`` or a digest instead.
    """
    if not arch:
        raise ValueError("attention dense cache identity requires an explicit arch")
    skip = frozenset(spec.runtime_param_fields)
    rest = tuple(
        (f.name, getattr(spec, f.name))
        for f in _dataclass_fields(spec)
        if f.name not in skip
    )
    return (arch, type(spec), rest)


def check_dense_spec_preflight(spec: AttentionDenseSpec) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the checks every dense body shares.

    Each arch's ``supports_attention_dense`` calls this and then adds its own scope
    (dtype / head-size sets, deferred modes, private knobs, LDS budget, the tile
    divisibility its own DMA imposes). A check belongs HERE only if it reads
    base-spec fields only AND its verdict is the same for every dense body -- which
    is why the LDS budget stays per-arch (it needs the arch capacity and the body's
    tile math) and the mode rejections stay per-arch (gfx942 defers varlen, gfx950
    ships it).

    The four checks, in a load-bearing order:

    1. Re-run the dataclass validators, so a hand-built spec (or one smuggled past
       the frozen ctor) is rejected with a structured reason instead of an exception
       escaping this ``(bool, str)`` API. The CONCRETE type is reconstructed, not
       the base, so an arch's private knob validators run too; ``ZeroDivisionError``
       is caught alongside ``ValueError`` because ``__post_init__`` evaluates
       ``seqlen_kv % block_n`` BEFORE it validates ``block_n > 0``.
    2. Positive extents. Every dataclass validator is a divisibility test and
       Python's ``%`` is sign-following (``-256 % 256 == 0``, ``8 % -1 == 0``), so
       zero and negative shapes pass all of them. ``num_query_heads == 0`` is the
       worst: ``gqa = Hq // Hkv == 0`` emits ``sdiv i32 %hq, 0`` into the kernel.
    3. ``block_n`` divides the query tile. The causal KV clamp uses
       ``n_per = block_m // block_n``, a FLOOR, so an indivisible ``block_n``
       silently drops every key past the last whole sub-tile, and
       ``block_n > block_m`` makes ``n_per`` 0 -> zero-trip loop -> ``l == 0`` ->
       ``rcp(0)`` -> NaN. Neither fails loudly.
    4. 32-bit addressing. Offsets are built from IRBuilder add/mul, which lower to
       ``add nsw`` / ``mul nsw`` i32 -- signed overflow is UB, not a wrap, so LLVM
       may poison the whole address chain rather than merely read the wrong place.
       The buffer-resource ``num_records`` field is unsigned in hardware, but it is
       emitted through ``const_i32`` (no range check) and the voffset feeding it is
       signed i32 arithmetic, so the signed bound is the binding one on both paths.

    Order matters between 2 and 4: a negative extent makes the products in 4
    vacuously true, so the sign check has to run first.

    On a kernel that takes the shape as runtime params (see
    ``runtime_param_fields``), check 4 is the ONLY defense. There the extent is a
    device-side ``mul`` of two kernargs rather than a constant folded at emission,
    so there is nothing in the IR to inspect -- and because those fields no longer
    split the launcher-cache key, the check has to run per launch rather than per
    compile. It does: every caller reaches ``supports`` before the cache lookup.
    """
    try:
        type(spec)(**{f.name: getattr(spec, f.name) for f in _dataclass_fields(spec)})
    except (ValueError, ZeroDivisionError) as e:
        return False, f"invalid {type(spec).__name__}: {e}"

    for name in (
        "batch",
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
    ):
        value = getattr(spec, name)
        if value <= 0:
            return False, f"{name} must be positive, got {value}"

    if spec.block_m % spec.block_n != 0:
        return False, (
            f"block_n must divide the {spec.block_m}-row query tile (got "
            f"block_n={spec.block_n}; the spec also requires block_n % 32 == 0, so "
            f"use 32, 64, 128 or 256). Load-bearing for causal=True, where "
            f"n_per = {spec.block_m} // block_n floors and drops keys"
        )

    kv_bytes = spec.batch * spec.seqlen_kv * spec.num_kv_heads * spec.head_size * 2
    if kv_bytes >= INT32_LIMIT:
        return False, (
            f"K/V extent is {kv_bytes} B, at or past the 32-bit buffer-resource "
            f"limit ({INT32_LIMIT} B)"
        )
    qo_elems = spec.batch * spec.seqlen_q * spec.num_query_heads * spec.head_size
    if qo_elems >= INT32_LIMIT:
        return False, (
            f"Q/O extent is {qo_elems} elements, at or past the 32-bit addressing "
            f"limit ({INT32_LIMIT})"
        )
    return True, ""


__all__ = [
    "AttentionDenseSpec",
    "xcd_partitionable",
    "DIGIT_LETTERS",
    "QB_TRAVERSALS",
    "SHIPPED_DIGIT_ORDERS",
    "decode_plan",
    "decode_reference",
    "digit_radices",
    "parse_digit_order",
    "traverse_qb",
    "DEFAULT_DENSE_TILE_GEOMETRY",
    "DENSE_TILE_GEOMETRIES",
    "INT32_LIMIT",
    "attention_dense_cache_key",
    "check_dense_spec_preflight",
]
