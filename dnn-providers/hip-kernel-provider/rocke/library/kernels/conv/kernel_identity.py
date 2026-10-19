# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The identity of an AOT convolution kernel and the shapes it can serve.

* :class:`KernelIdentity` -- the compile-time tuple that uniquely identifies
  one HSACO. Only configuration and capability fields belong here; no problem
  extents. See the class docstring for why the distinction matters.
* :func:`identity_supports_problem` -- whether a binary with a given identity
  can be launched on a concrete problem.

Both live with the kernels rather than with the benchmark tooling that first
needed them, because two layers consume them: the AOT cache
(``benchmarks.common.kernel_cache``) files binaries under their identity and
offers them to shapes, and the grouped-convolution dispatcher
(``dispatch.grouped_convolution``) registers its instances by identity and
admits a request with the same predicate -- one rule, so the dispatcher can
never select an instance the cache would refuse to launch, or the reverse.
Names are spelled by :mod:`kernels.conv.kernel_label`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

__all__ = [
    "KernelIdentity",
    "current_llvm_flavor",
    "dgrad_sub_gemms",
    "identity_supports_problem",
]


# The per-axis workgroup-count cap the validators enforce (gridDim y/z on
# AMD; x is capped the same way for symmetry with is_valid_spec_for_problem).
_MAX_GRID_DIM = 65535


def current_llvm_flavor() -> str:
    """LLVM IR flavor this process lowers to (the one ``compile_kernel`` uses).

    The platform's resolver owns the choice -- the ``ROCKE_LLVM_FLAVOR``
    override and COMGR detection -- and caches the loaded compiler's evidence,
    so this neither reads the environment itself nor memoises. Host tooling
    (the AOT cache, ``--compile-all``) asks it; dispatch never does: a
    request carries its own ``llvm_flavor``.
    """
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


# ---------------------------------------------------------------------------
# KernelIdentity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelIdentity:
    """Compile-time tuple that uniquely identifies an AOT kernel.

    No problem *extents* here — that is the whole point of AOT. Capability
    fields that genuinely constrain which shapes the binary accepts do belong,
    and are checked against the problem by :func:`identity_supports_problem`.

    Two kinds of field live here, and the distinction matters:

    * **Configuration** — tile sizes, pipeline, epilogue, vector widths. These
      change the ISA but not what the kernel can be *launched on*.
    * **Capability** — ``is_3d``, ``filter_h/w``, ``stride``, ``pad``,
      ``cpg``/``kpg`` for direct conv, ``max_sub_gemms`` for dgrad. These bound
      the set of problems the binary can serve, and
      :func:`identity_supports_problem` checks them before a kernel is offered
      for a shape.

    Every field that changes the emitted ISA must be here. A field that is
    missing does not merely weaken the cache key — two different kernels hash to
    the same name and silently overwrite each other on disk.
    """

    arch: str
    direction: str
    algorithm: str
    dtype_a: str
    dtype_b: str
    dtype_d: str
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    pipeline: str
    epilogue: str
    wave_size: int
    vector_size_a: int
    vector_size_b: int
    vector_size_c: int
    # ---- configuration that changes the emitted ISA ----
    async_dma: bool = False
    unroll_k: bool = False
    chiplet_swizzle: bool = False
    lds_layout: str = "default"
    lds_k_pad: int = 0
    lds_k_outer: bool = False
    waves_per_eu: int | None = None
    acc_epilogue: str = "none"
    split_k: int = 1
    two_stage: bool = False
    # Two-stage scratch slabs per group (WgradConvSpec.ws_replicas). The host
    # sizes the scratch and builds Stage 2 from it, so it has to come from the
    # binary's identity rather than a default the two could drift apart on.
    # 0 = no scratch (not two-stage).
    ws_replicas: int = 0
    # wgrad stream-K (WgradConvSpec.streamk / streamk_reduction): the launch
    # mode ("dp_sk" or "persistent") and the partial-sum reduction are baked
    # into the ISA; the CTA pool is a kernarg, so it is not here. "off" / ""
    # for every other kernel.
    streamk: str = "off"
    streamk_reduction: str = ""
    group_merge: int = 1
    # wgrad only (WgradConvSpec.prefetch_addr): carried K-loop byte offsets.
    # Left out of the hash while False, like ws_replicas, so identities that
    # predate the field keep their hash and their cache entries.
    prefetch_addr: bool = False
    # dgrad only (DgradConvSpec.uniform_k_decode): the per-tile reduction
    # decode K loop. Left out of the hash while False, like prefetch_addr.
    uniform_k_decode: bool = False
    num_load_waves: int = 0
    cshuffle_no_alias: bool = False
    # ---- capability: bounds which problems this binary can serve ----
    is_3d: bool = False
    is_pointwise: bool = False
    # Direct conv bakes the filter geometry and per-group channel counts into
    # the unrolled MFMA chain and the LDS row layout; implicit GEMM leaves
    # them 0 (runtime).
    filter_h: int = 0
    filter_w: int = 0
    filter_d: int = 0
    stride_h: int = 0
    stride_w: int = 0
    dilation_h: int = 0
    dilation_w: int = 0
    pad_h: int = 0
    pad_w: int = 0
    cpg: int = 0
    kpg: int = 0
    # Grouped convolution takes a different code path in every direction (the
    # contraction index only spans one group, so the channel decode gains an
    # embed and the epilogue gains a per-group k_out fold). A binary built one
    # way cannot serve the other, and the difference is invisible at launch --
    # it produces wrong numbers rather than an error -- so it is a capability.
    grouped: bool = False
    # Direct conv only: rows per block (the H-loop trip count).
    block_h: int = 0
    # Dgrad only: the largest tilde sub-GEMM count the record table (and the
    # binary search's runtime trip count) was built to support.
    max_sub_gemms: int = 0
    # async_dma only: elements per DRAM->LDS chunk of the A and B loaders. The
    # width is picked from the build-time cpg so a chunk never straddles a
    # filter position, and is baked into the ISA -- a binary is only correct
    # for problems whose contiguous run it divides.
    async_chunk_a: int = 0
    async_chunk_b: int = 0
    # Direct conv only: the spec's tuning kwargs (block_q, block_w, waves...)
    # as canonical JSON. Direct kernels have no GEMM tile, so their knobs do
    # not fit the tile/warp fields above; ``algorithm`` names the kernel
    # variant and this string carries the rest.
    knobs: str = ""
    # ---- provenance: the LLVM flavor the binary was lowered for ----
    # It changes the HSACO without showing up in the configuration, so a
    # binary for another flavor has to miss. Defaults to this process's
    # flavor, so a lookup only matches binaries lowered the same way.
    #
    # The emitter version is deliberately NOT part of the identity: an
    # identity names a kernel configuration, and the same configuration keeps
    # its entry across emitter changes. Whether its binary is still current is
    # tracked per entry (the build-time emitter digest and the binary's
    # content key in its metadata), which is what lets --compile-all rebuild
    # only the kernels whose code changed.
    llvm_flavor: str = field(default_factory=current_llvm_flavor)

    @property
    def is_direct(self) -> bool:
        """Direct-conv kernels: every capability field is baked, even a 0."""
        return self.algorithm.startswith("direct")

    def stable_hash(self) -> str:
        """Deterministic SHA-1 hex digest (40 chars) of the identity.

        Suitable as a filesystem-safe filename component. Deterministic
        across Python versions (sorted JSON keys, no randomization).
        """
        fields = asdict(self)
        # Left out at its "no scratch" value so identities that predate the
        # field keep their hash (and their cache entries).
        if not fields["ws_replicas"]:
            del fields["ws_replicas"]
        if not fields["prefetch_addr"]:
            del fields["prefetch_addr"]
        if not fields["uniform_k_decode"]:
            del fields["uniform_k_decode"]
        # Same for the stream-K pair at "no stream-K".
        if fields["streamk"] == "off" and not fields["streamk_reduction"]:
            del fields["streamk"], fields["streamk_reduction"]
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()

    def label(self) -> str:
        """The kernel's name in logs, benchmark tables and instance lists.

        Lossless: it spells every field, so :meth:`from_label` turns it back
        into an equal identity and ``reproduce_kernel.py`` rebuilds the kernel
        from it. Format: :mod:`kernels.conv.kernel_label`. Not the
        HSACO's entry point -- the AOT cache records that in its metadata under
        ``kernel_name``.
        """
        from kernels.conv.kernel_label import encode

        return encode(self)

    @classmethod
    def from_label(cls, name: str) -> "KernelIdentity":
        """The identity :meth:`label` named ``name``.

        Launch parameters a benchmark table appends after whitespace
        (``@split_k=4``) are ignored. Raises ``ValueError`` on a malformed
        name.
        """
        from kernels.conv.kernel_label import decode

        return decode(name, cls)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KernelIdentity":
        """Rebuild from ``to_dict`` output, tolerating older cache entries.

        Unknown keys are dropped rather than raising: a cache written by an
        older build should be ignorable, not fatal. Entries missing a field
        that has since been added fall back to its default, which is also why
        the hash changes and they simply never match again.
        """
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        # An entry that predates the provenance field was lowered for an
        # unknown flavor; "" never equals the current one, so it never matches.
        kw.setdefault("llvm_flavor", "")
        return cls(**kw)


# ---------------------------------------------------------------------------
# Which problems an identity can serve
# ---------------------------------------------------------------------------

# ``DgradConvSpec.max_sub_gemms``'s default; identities written before the
# field existed recorded 0 and were built with it.
_DGRAD_DEFAULT_MAX_SUB_GEMMS = 64


def dgrad_sub_gemms(identity: KernelIdentity, problem: object) -> list:
    """The tilde sub-GEMMs ``problem`` decomposes into for this binary's tile.

    An implicit-GEMM dgrad binary reads them at run time, as the record table
    :func:`~kernels.common.conv_implicit_gemm_dgrad.pack_sub_gemm_buffer`
    packs; its grid is their total tile count.
    """
    from kernels.common.conv_implicit_gemm_dgrad import (
        compute_tilde,
        enumerate_sub_gemms,
    )

    return enumerate_sub_gemms(
        problem,
        compute_tilde(problem),
        identity.tile_m,
        identity.tile_n,
        tile_k=identity.tile_k,
        split_k=max(1, identity.split_k),
    )


def _group_merge_fits(
    identity: KernelIdentity, problem: object, gm: int
) -> tuple[bool, str]:
    """Can a binary merging ``gm`` groups serve ``problem``?

    Mirrors the shape rules of ``wgrad_group_merge_available`` /
    ``fwd_group_merge_available``: merging is depthwise only, ``gm`` must
    divide the group count, and the merged GEMM must fit one tile -- the
    filter extent is a runtime value, so it is checked here, per problem.
    wgrad's merged GEMM is ``kpg*Gm`` x ``[Z*]Y*X*cpg*Gm``; fwd's only
    merged free axis is GEMM-N, ``kpg*Gm``.
    """
    cpg = int(getattr(problem, "cpg", 0))
    kpg = int(getattr(problem, "kpg", 0))
    groups = int(getattr(problem, "groups", 1))
    if cpg != 1 or kpg != 1:
        return False, "group-merged kernel needs depthwise (cpg=kpg=1)"
    if groups % gm:
        return False, f"group_merge {gm} does not divide groups {groups}"
    if identity.direction == "fwd":
        if kpg * gm > identity.tile_n:
            return False, f"merged GEMM-N {kpg * gm} exceeds tile_n {identity.tile_n}"
        return True, "ok"
    spatial = int(getattr(problem, "Y", 1)) * int(getattr(problem, "X", 1))
    if bool(getattr(problem, "is_3d", False)):
        spatial *= int(getattr(problem, "Z", 1))
    if kpg * gm > identity.tile_m:
        return False, f"merged GEMM-M {kpg * gm} exceeds tile_m {identity.tile_m}"
    if spatial * cpg * gm > identity.tile_n:
        return False, (
            f"merged GEMM-N {spatial * cpg * gm} exceeds tile_n {identity.tile_n}"
        )
    return True, "ok"


def _grid_fits(identity: KernelIdentity, problem: object) -> tuple[bool, str]:
    """Does the launch grid for ``problem`` fit the hardware limits?

    The tile counts are runtime values of an AOT kernel, so a small tile on
    a large problem can need more workgroups along y (or x) than the
    65535 cap -- the launch then fails or leaves tiles unwritten. Mirrors
    the grid check of ``is_valid_spec_for_problem``; split-K degrees, the
    third axis of wgrad, are capped where the benchmark picks them.
    """
    cap = _MAX_GRID_DIM
    groups = max(1, int(getattr(problem, "groups", 1)))
    cpg = int(getattr(problem, "cpg", 0))
    kpg = int(getattr(problem, "kpg", 0))
    tm, tn = max(1, identity.tile_m), max(1, identity.tile_n)
    if identity.direction == "fwd":
        gm = max(1, identity.group_merge)
        gy = -(-int(getattr(problem, "M", 0)) // tm)
        gx = -(-(kpg * gm) // tn)
        axes = (
            ("y", gy, f"M={getattr(problem, 'M', 0)} tile_m={tm}"),
            ("x", gx, f"kpg*Gm={kpg * gm} tile_n={tn}"),
            ("z", groups // gm, f"groups/Gm={groups // gm}"),
        )
    elif identity.direction == "wgrad":
        gm = max(1, identity.group_merge)
        spatial = int(getattr(problem, "Y", 1)) * int(getattr(problem, "X", 1))
        if bool(getattr(problem, "is_3d", False)):
            spatial *= int(getattr(problem, "Z", 1))
        gy = -(-(kpg * gm) // tm)
        gx = -(-(spatial * cpg * gm) // tn)
        axes = (
            ("y", gy, f"wg_M={kpg * gm} tile_m={tm}"),
            ("x", gx, f"wg_N={spatial * cpg * gm} tile_n={tn}"),
            ("z", groups // gm, f"groups/Gm={groups // gm}"),
        )
    else:
        axes = (("y", groups, f"groups={groups}"),)
    for axis, extent, what in axes:
        if extent > cap:
            return False, f"grid {axis} {extent} > {cap} (hardware cap): {what}"
    return True, "ok"


def identity_supports_problem(
    identity: KernelIdentity, problem: object, llvm_flavor: str | None = None
) -> tuple[bool, str]:
    """Can a binary built for ``identity`` be launched on ``problem``?

    ``problem`` is a ``ConvProblem`` for the implicit-GEMM identities and a
    ``DirectConvProblem`` for the direct ones; both are read by attribute.
    Returns ``(ok, reason)``; the reason makes an empty candidate list
    diagnosable instead of just "no compatible kernels".

    ``llvm_flavor`` is the flavor to check provenance against; ``None`` (the
    default) falls back to :func:`current_llvm_flavor` (environment/COMGR
    detection), preserving this function's standalone behavior. Callers that
    have a flavor on hand -- a dispatch request, a test -- should pass it
    explicitly instead of relying on process environment.

    Provenance is checked first: a binary lowered for another LLVM flavor is
    never offered. (Emitter changes are not checked here: most leave most
    kernels' code alone, and only a ``--compile-all`` can tell which, by
    re-emitting and comparing content keys.) Then two classes of constraint:

    * **Vector alignment.** The load/store widths are baked into the ISA,
      so each operand's contiguous per-group extent has to stay divisible
      by its width. Which extent that is depends on the direction: fwd
      loads X and W along ``cpg`` and stores Y along ``kpg``; wgrad and
      dgrad load dY along ``kpg`` and run their B operand (X / W) and
      output (dW / dX) along ``cpg``.
    * **Capability.** A kernel that unrolled a 3x3 filter, or that baked
      ``cpg``/``kpg`` into its MFMA chain, simply cannot run another
      geometry. The identity records those, so a mismatch is a hard no
      rather than a silent wrong answer.

    The direct kernels' tuning knobs carry shape rules of their own (groups
    divisible by block_groups, ...), which this capability check does not
    see: :func:`kernels.conv.direct_variants.direct_spec_for_problem`
    re-validates those against the real shape.
    """
    flavor = current_llvm_flavor() if llvm_flavor is None else llvm_flavor
    if identity.llvm_flavor != flavor:
        return False, (
            f"built for LLVM flavor {identity.llvm_flavor or '<unknown>'}, "
            f"this process lowers to {flavor}"
        )

    cpg = int(getattr(problem, "cpg", 0))
    kpg = int(getattr(problem, "kpg", 0))
    # The contiguous extent of A / B / D. fwd: X (cpg), W (cpg), Y (kpg).
    # wgrad: dY (kpg), X (cpg), dW (cpg). dgrad: dY (kpg), W (cpg),
    # dX (cpg). Direct conv has no vector-width fields (they are 0); its
    # load widths follow the baked cpg/kpg, which are checked exactly below.
    if identity.direction in ("wgrad", "dgrad"):
        extents = ((kpg, "kpg"), (cpg, "cpg"), (cpg, "cpg"))
    else:
        extents = ((cpg, "cpg"), (cpg, "cpg"), (kpg, "kpg"))
    gm = max(1, int(identity.group_merge))
    if gm > 1:
        ok, why = _group_merge_fits(identity, problem, gm)
        if not ok:
            return False, why
        if identity.direction == "fwd":
            # A group-merged fwd kernel (depthwise only) loads X and stores
            # Y along the Gm merged channels; W stays per group (its merged
            # load is per-element masked).
            extents = ((cpg * gm, "cpg*Gm"), (cpg, "cpg"), (kpg * gm, "kpg*Gm"))
        else:
            # A group-merged wgrad kernel (depthwise only) loads dY and X
            # along the Gm merged channels; dW stays per group.
            extents = ((kpg * gm, "kpg*Gm"), (cpg * gm, "cpg*Gm"), (cpg, "cpg"))
    for name, (extent, extent_name) in zip(
        ("vector_size_a", "vector_size_b", "vector_size_c"), extents
    ):
        vec = getattr(identity, name)
        if vec and extent and extent % vec != 0:
            return False, f"{extent_name}={extent} not divisible by {name}={vec}"

    if identity.is_3d != bool(getattr(problem, "is_3d", False)):
        return False, "3-D capability mismatch"
    if identity.is_pointwise and not bool(getattr(problem, "is_pointwise", False)):
        return False, "kernel is pointwise-only"

    # Grouped convolution is a different code path in every direction. An
    # ungrouped binary on a grouped problem gives wrong numbers with no
    # error, so it is checked before anything else that could mask it. The
    # grouped path computes groups == 1 too (group 0, cpg == C), so a
    # grouped binary serves both; the AOT grid only builds those, and
    # ungrouped entries of older caches still serve ungrouped problems.
    # Direct kernels take the group count as a kernarg and index channels
    # per group in every variant, so they carry no grouped/ungrouped split.
    problem_grouped = int(getattr(problem, "groups", 1)) > 1
    if not identity.is_direct and problem_grouped and not identity.grouped:
        return False, "kernel is ungrouped, problem is grouped"

    # Split-K wgrad without two-stage atomic-adds straight into dW. For a
    # 16-bit dW that is a packed <2 x dtype> atomic, which needs an even
    # dW row (wg_N = [Z*]Y*X*cpg) and store width; on an odd row it
    # misaddresses silently. Ask the same predicate the spec validator uses.
    if (
        identity.direction == "wgrad"
        and not identity.is_direct
        and identity.split_k > 1
        and not identity.two_stage
        and hasattr(problem, "Y")
    ):
        from kernels.common.conv_implicit_gemm_wgrad import (
            wgrad_atomic_epilogue_available,
        )

        ok, why = wgrad_atomic_epilogue_available(
            problem, identity.dtype_d, identity.vector_size_c or None
        )
        if not ok:
            return False, why

    # Baked filter geometry (direct conv), and the stride/dilation older
    # implicit-GEMM dgrad caches recorded per binary (current ones record
    # 0: the tilde kernel takes them at run time). For implicit GEMM 0
    # means "runtime"; direct conv bakes all of them, and there 0 is a real
    # value (PAD=0 for a 1x1 filter) that has to match too.
    for name, attrs in (
        ("filter_h", ("KH", "Y")),
        ("filter_w", ("KW", "X")),
        ("stride_h", ("stride", "sH")),
        ("stride_w", ("stride", "sW")),
        ("dilation_h", ("dilation", "dH", "dil_h")),
        ("dilation_w", ("dilation", "dW", "dil_w")),
        ("pad_h", ("PAD", "pH")),
        ("pad_w", ("PAD", "pW")),
    ):
        baked = getattr(identity, name)
        if not baked and not identity.is_direct:
            continue
        actual = next((getattr(problem, a) for a in attrs if hasattr(problem, a)), None)
        if actual is not None and int(actual) != baked:
            return False, f"{name}={baked} but problem has {actual}"

    # 0 means runtime here, for direct conv too: the non-grouped direct
    # kernel takes its channel counts as kernargs (cpg == 0 is never a
    # real shape, unlike PAD == 0 above).
    if identity.cpg and identity.cpg != cpg:
        return False, f"kernel baked cpg={identity.cpg}, problem has {cpg}"
    if identity.kpg and identity.kpg != kpg:
        return False, f"kernel baked kpg={identity.kpg}, problem has {kpg}"

    # The async loaders fetch fixed-width chunks along a run that is only
    # contiguous over the per-group channels; a chunk that does not divide
    # it silently reads across filter positions.
    if identity.async_dma:
        if not (identity.async_chunk_a and identity.async_chunk_b):
            return False, "async_dma binary without a recorded chunk width"
        # (A, B) contiguous runs: fwd reads NHWC and KYXC along c; wgrad
        # reads dY along k_out and X along c.
        runs = {"wgrad": (kpg, cpg)}.get(identity.direction, (cpg, cpg))
        for chunk, run, operand in (
            (identity.async_chunk_a, runs[0], "A"),
            (identity.async_chunk_b, runs[1], "B"),
        ):
            if run % chunk != 0:
                return False, (
                    f"async {operand} chunk of {chunk} elements does not "
                    f"divide the problem's contiguous run of {run}"
                )

    # Implicit-GEMM dgrad dispatches CTAs with a binary search whose trip
    # count is runtime, but the sub-GEMM record table the search walks is
    # sized at build time for ``max_sub_gemms``; a problem that decomposes
    # into more would be rejected only at launch.
    if identity.direction == "dgrad" and not identity.is_direct:
        n_sub = len(dgrad_sub_gemms(identity, problem))
        bound = identity.max_sub_gemms or _DGRAD_DEFAULT_MAX_SUB_GEMMS
        if n_sub > bound:
            return False, (
                f"problem needs {n_sub} tilde sub-GEMMs, kernel's record "
                f"table supports {bound} (max_sub_gemms)"
            )
        # Mirrors is_valid_dgrad_spec: the grouped implicit-GEMM dgrad has
        # no depthwise (cpg == 1) path, and the grouped binary is offered
        # to every grouped problem otherwise.
        if int(getattr(problem, "groups", 1)) > 1 and cpg == 1:
            return False, (
                "depthwise dgrad (cpg == 1) is not supported by the "
                "implicit-GEMM grouped path"
            )

    if not identity.is_direct:
        ok, why = _grid_fits(identity, problem)
        if not ok:
            return False, why

    return True, "ok"
