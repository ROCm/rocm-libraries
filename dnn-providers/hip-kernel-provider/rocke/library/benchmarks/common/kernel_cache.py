# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AOT kernel identity and disk cache for shape-generic convolution kernels.

Combines two tightly coupled pieces:

* :class:`KernelIdentity` — the compile-time tuple that uniquely identifies
  one HSACO. Only configuration and capability fields belong here; no problem
  extents. See the class docstring for why the distinction matters.

* :class:`KernelCache` — reads, writes and lists HSACO blobs keyed by identity
  from a directory tree under ``<root>/<arch>/``.

Both classes live in ``library/`` (not in the installable ``rocke`` wheel)
because they are specific to this library's conv kernel families and their
sweep tooling.  :mod:`benchmarks.common.kernel_sweep` is the primary consumer;
``library/tests/`` uses both for the ABI regression suite.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Build provenance
# ---------------------------------------------------------------------------


def current_llvm_flavor() -> str:
    """LLVM IR flavor this process lowers to (the one ``compile_kernel`` uses)."""
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


@functools.lru_cache(maxsize=None)
def current_emitter_digest() -> str:
    """SHA-1 over the emitter sources: the ``rocke`` and ``kernels`` packages.

    The C++ engine build-id would not do: it reads ``unknown`` whenever the
    binding is not built, and it does not move when only the Python emitter
    changes -- which is the engine the sweep compiles with. Hashing the
    sources the IR comes from means any emitter change invalidates the cache.
    """
    import kernels
    import rocke

    h = hashlib.sha1()
    for pkg in (rocke, kernels):
        root = Path(pkg.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            h.update(path.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# KernelIdentity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelIdentity:
    """Compile-time tuple that uniquely identifies an AOT kernel.

    No problem *extents* here — that is the whole point of AOT. Capability
    fields that genuinely constrain which shapes the binary accepts do belong,
    and are checked against the problem at cache-lookup time.

    Two kinds of field live here, and the distinction matters:

    * **Configuration** — tile sizes, pipeline, epilogue, vector widths. These
      change the ISA but not what the kernel can be *launched on*.
    * **Capability** — ``is_3d``, ``filter_h/w``, ``stride``, ``pad``,
      ``cpg``/``kpg`` for direct conv, ``max_sub_gemms`` for dgrad. These bound
      the set of problems the binary can serve, and
      :meth:`KernelCache.supports_problem` checks them before a cached kernel is
      offered for a shape.

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
    waves_per_eu: Optional[int] = None
    acc_epilogue: str = "none"
    split_k: int = 1
    two_stage: bool = False
    group_merge: int = 1
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
    # Dgrad only: the largest tilde sub-GEMM count the CTA dispatch search
    # was unrolled for.
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
    # ---- provenance: which emitter and LLVM flavor produced the binary ----
    # Neither shows up in the configuration, yet either changes the HSACO: a
    # binary from an older emitter or another flavor has to miss, not be
    # silently reused. They default to this process's values, so an identity
    # built for a lookup only ever matches binaries built the same way.
    llvm_flavor: str = field(default_factory=current_llvm_flavor)
    emitter_digest: str = field(default_factory=current_emitter_digest)

    @property
    def is_direct(self) -> bool:
        """Direct-conv kernels: every capability field is baked, even a 0."""
        return self.algorithm.startswith("direct")

    def stable_hash(self) -> str:
        """Deterministic SHA-1 hex digest (40 chars) of the identity.

        Suitable as a filesystem-safe filename component. Deterministic
        across Python versions (sorted JSON keys, no randomization).
        """
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()

    def short_label(self) -> str:
        """Human-readable label for logs and tables.

        Not a kernel symbol name — the HSACO's entry point is recorded in the
        cache metadata under ``kernel_name``.
        """
        if self.is_direct:
            knobs = json.loads(self.knobs) if self.knobs else {}
            bits = [
                self.algorithm,
                f"f{self.filter_h}x{self.filter_w}",
                f"p{self.pad_h}",
                f"s{self.stride_h}",
                f"c{self.cpg}k{self.kpg}",
            ]
            bits += [
                f"{k}{int(v) if isinstance(v, bool) else v}"
                for k, v in sorted(knobs.items())
            ]
            return "_".join(bits)
        bits = [
            f"{self.direction}_{self.algorithm}",
            f"{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warp_m}x{self.warp_n}",
            f"a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"v{self.vector_size_a}{self.vector_size_b}{self.vector_size_c}",
            self.pipeline,
            self.epilogue,
        ]
        if self.unroll_k:
            bits.append("unroll")
        if self.async_dma:
            bits.append("async")
        if self.split_k != 1:
            bits.append(f"sk{self.split_k}")
        if self.two_stage:
            bits.append("2stage")
        if self.filter_h:
            bits.append(f"f{self.filter_h}x{self.filter_w}")
        if self.cpg:
            bits.append(f"c{self.cpg}k{self.kpg}")
        if self.grouped:
            bits.append("grp")
        if self.stride_h or self.stride_w:
            bits.append(f"s{self.stride_h}x{self.stride_w}")
        if self.dilation_h or self.dilation_w:
            bits.append(f"d{self.dilation_h}x{self.dilation_w}")
        return "_".join(bits)

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
        # An entry that predates the provenance fields was built by an unknown
        # emitter; "" never equals the current values, so it never matches.
        kw.setdefault("llvm_flavor", "")
        kw.setdefault("emitter_digest", "")
        return cls(**kw)


# ``DgradConvSpec.max_sub_gemms``'s default; identities written before the
# field existed recorded 0 and were built with it.
_DGRAD_DEFAULT_MAX_SUB_GEMMS = 64


def _dgrad_sub_gemm_count(identity: KernelIdentity, problem: object) -> int:
    """Tilde sub-GEMM count ``problem`` decomposes into for this binary's tile."""
    # Deferred so listing and hashing identities stays free of the kernel
    # builders; this is a downward (benchmarks -> kernels) import.
    from kernels.common.conv_implicit_gemm_dgrad import (
        compute_tilde,
        enumerate_sub_gemms,
    )

    return len(
        enumerate_sub_gemms(
            problem,
            compute_tilde(problem),
            identity.tile_m,
            identity.tile_n,
            tile_k=identity.tile_k,
            split_k=max(1, identity.split_k),
        )
    )


# ---------------------------------------------------------------------------
# KernelCache
# ---------------------------------------------------------------------------

# Direction → subdirectory name.
_DIRECTION_DIRS: Dict[str, str] = {
    "fwd": "conv_fwd",
    "wgrad": "conv_wgrad",
    "dgrad": "conv_dgrad",
    "direct_fwd": "conv_direct",
    "direct_dgrad": "conv_direct_dgrad",
    # Weight-transform kernels of the MFMA direct dgrad pipeline. They are
    # looked up by exact identity from the pipeline's main entry and never
    # offered as candidates on their own.
    "direct_dgrad_helper": "conv_direct_dgrad_helpers",
}


class KernelCache:
    """Read/write/list AOT HSACO blobs on disk.

    Layout::

        <root>/
            <arch>/
                conv_fwd/
                    <sha1>.hsaco
                    <sha1>.meta.json
                conv_wgrad/
                conv_dgrad/
                conv_direct/

    The cache is per-arch. Callers pass an ``arch`` at construction time and
    the cache scopes all operations under ``<root>/<arch>/``.
    """

    def __init__(self, root: Path, arch: str) -> None:
        self._root = Path(root)
        self._arch = arch
        self._base = self._root / arch

    def _dir_for(self, identity: KernelIdentity) -> Path:
        subdir = _DIRECTION_DIRS.get(identity.direction, identity.direction)
        return self._base / subdir

    def _paths(self, identity: KernelIdentity) -> Tuple[Path, Path]:
        d = self._dir_for(identity)
        h = identity.stable_hash()
        return d / f"{h}.hsaco", d / f"{h}.meta.json"

    def put(
        self,
        identity: KernelIdentity,
        hsaco: bytes,
        meta: Optional[dict] = None,
    ) -> Path:
        """Write an HSACO + metadata to the cache. Returns the HSACO path."""
        hsaco_path, meta_path = self._paths(identity)
        hsaco_path.parent.mkdir(parents=True, exist_ok=True)
        hsaco_path.write_bytes(hsaco)
        full_meta = {
            "identity": identity.to_dict(),
            "hsaco_bytes": len(hsaco),
        }
        if meta:
            full_meta.update(meta)
        meta_path.write_text(
            json.dumps(full_meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        return hsaco_path

    def get(self, identity: KernelIdentity) -> Optional[Tuple[bytes, dict]]:
        """Load an HSACO + metadata from cache, or ``None`` on miss."""
        hsaco_path, meta_path = self._paths(identity)
        if not hsaco_path.exists() or hsaco_path.stat().st_size == 0:
            return None
        hsaco = hsaco_path.read_bytes()
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return hsaco, meta

    def hsaco_path(self, identity: KernelIdentity) -> Path:
        """Where the identity's HSACO lives (whether or not it exists yet)."""
        return self._paths(identity)[0]

    def has(self, identity: KernelIdentity) -> bool:
        """Check whether the cache has a valid entry for the identity."""
        hsaco_path, _ = self._paths(identity)
        return hsaco_path.exists() and hsaco_path.stat().st_size > 0

    def list_all(
        self, direction: Optional[str] = None
    ) -> List[Tuple[KernelIdentity, Path]]:
        """List all cached identities, optionally filtered by direction.

        Returns ``(identity, hsaco_path)`` pairs.
        """
        results: List[Tuple[KernelIdentity, Path]] = []
        if direction is not None:
            subdirs = [_DIRECTION_DIRS.get(direction, direction)]
        else:
            subdirs = list(_DIRECTION_DIRS.values())
        for subdir in subdirs:
            d = self._base / subdir
            if not d.is_dir():
                continue
            for meta_path in sorted(d.glob("*.meta.json")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ident = KernelIdentity.from_dict(meta["identity"])
                    hsaco_path = meta_path.with_suffix("").with_suffix(".hsaco")
                    if hsaco_path.exists() and hsaco_path.stat().st_size > 0:
                        results.append((ident, hsaco_path))
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        return results

    def supports_problem(
        self, identity: KernelIdentity, problem: object
    ) -> Tuple[bool, str]:
        """Can this cached binary be launched on ``problem``?

        Returns ``(ok, reason)``; the reason makes an empty candidate list
        diagnosable instead of just "no compatible kernels".

        Provenance is checked first: a binary from another emitter version or
        LLVM flavor is never offered. Then two classes of constraint:

        * **Vector alignment.** The load/store widths are baked into the ISA,
          so the per-group channel counts have to stay divisible by them.
          ``vector_size_b`` matters as well as ``a``/``c``: the B operand's
          innermost extent is ``cpg`` too.
        * **Capability.** A kernel that unrolled a 3x3 filter, or that baked
          ``cpg``/``kpg`` into its MFMA chain, simply cannot run another
          geometry. The identity records those, so a mismatch is a hard no
          rather than a silent wrong answer.
        """
        # A binary from another emitter or LLVM flavor is stale, whatever it
        # was configured for.
        if identity.llvm_flavor != current_llvm_flavor():
            return False, (
                f"built for LLVM flavor {identity.llvm_flavor or '<unknown>'}, "
                f"this process lowers to {current_llvm_flavor()}"
            )
        if identity.emitter_digest != current_emitter_digest():
            return False, "built by a different emitter version (stale entry)"

        cpg = int(getattr(problem, "cpg", 0))
        kpg = int(getattr(problem, "kpg", 0))
        # Direct conv has no vector-width fields (they are 0); its load widths
        # follow the baked cpg/kpg, which are checked exactly below.
        for field, extent, name in (
            ("vector_size_a", cpg, "cpg"),
            ("vector_size_b", cpg, "cpg"),
            ("vector_size_c", kpg, "kpg"),
        ):
            vec = getattr(identity, field)
            if vec and extent and extent % vec != 0:
                return False, f"{name}={extent} not divisible by {field}={vec}"

        if identity.is_3d != bool(getattr(problem, "is_3d", False)):
            return False, "3-D capability mismatch"
        if identity.is_pointwise and not bool(getattr(problem, "is_pointwise", False)):
            return False, "kernel is pointwise-only"

        # Grouped convolution is a different code path in every direction, and
        # the mismatch is silent at launch (wrong numbers, no error), so it is
        # checked before anything else that could mask it.
        # Direct kernels take the group count as a kernarg and index channels
        # per group in every variant, so they carry no grouped/ungrouped split.
        problem_grouped = int(getattr(problem, "groups", 1)) > 1
        if not identity.is_direct and identity.grouped != problem_grouped:
            want = "grouped" if problem_grouped else "ungrouped"
            have = "grouped" if identity.grouped else "ungrouped"
            return False, f"kernel is {have}, problem is {want}"

        # Baked filter geometry (direct conv) and the stride/dilation that
        # implicit-GEMM dgrad folds into its tilde decomposition. For implicit
        # GEMM 0 means "runtime"; direct conv bakes all of them, and there 0 is
        # a real value (PAD=0 for a 1x1 filter) that has to match too.
        for field, attrs in (
            ("filter_h", ("KH", "Y")),
            ("filter_w", ("KW", "X")),
            ("stride_h", ("stride", "sH")),
            ("stride_w", ("stride", "sW")),
            ("dilation_h", ("dilation", "dH")),
            ("dilation_w", ("dilation", "dW")),
            ("pad_h", ("PAD", "pH")),
            ("pad_w", ("PAD", "pW")),
        ):
            baked = getattr(identity, field)
            if not baked and not identity.is_direct:
                continue
            actual = next(
                (getattr(problem, a) for a in attrs if hasattr(problem, a)), None
            )
            if actual is not None and int(actual) != baked:
                return False, f"{field}={baked} but problem has {actual}"

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
            for chunk, run, name in (
                (identity.async_chunk_a, runs[0], "A"),
                (identity.async_chunk_b, runs[1], "B"),
            ):
                if run % chunk != 0:
                    return False, (
                        f"async {name} chunk of {chunk} elements does not "
                        f"divide the problem's contiguous run of {run}"
                    )

        # Implicit-GEMM dgrad dispatches CTAs with a binary search over the
        # tilde sub-GEMM table whose depth is baked for ``max_sub_gemms``; a
        # problem that decomposes into more would be rejected only at launch.
        if identity.direction == "dgrad" and not identity.is_direct:
            n_sub = _dgrad_sub_gemm_count(identity, problem)
            bound = identity.max_sub_gemms or _DGRAD_DEFAULT_MAX_SUB_GEMMS
            if n_sub > bound:
                return False, (
                    f"problem needs {n_sub} tilde sub-GEMMs, kernel unrolled "
                    f"for {bound}"
                )

        return True, "ok"

    def compatible(
        self, problem: object, *, direction: Optional[str] = None
    ) -> List[Tuple[KernelIdentity, Path, dict]]:
        """Every cached kernel that can run ``problem``, with its metadata.

        The metadata is returned because the caller needs ``kernel_name`` from
        it: the HSACO's entry-point symbol is not derivable from the identity.
        """
        out: List[Tuple[KernelIdentity, Path, dict]] = []
        for identity, hsaco_path in self.list_all(direction):
            ok, _ = self.supports_problem(identity, problem)
            if not ok:
                continue
            meta_path = hsaco_path.with_suffix(".meta.json")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            out.append((identity, hsaco_path, meta))
        return out
