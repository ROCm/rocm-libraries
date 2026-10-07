# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AOT kernel identity and disk cache for shape-generic convolution kernels.

Combines two tightly coupled pieces:

* :class:`KernelIdentity` — the compile-time tuple that uniquely identifies
  one HSACO. Only configuration and capability fields belong here; no problem
  extents. See the class docstring for why the distinction matters.

* :class:`KernelCache` — reads, writes and lists HSACO blobs from a directory
  tree under ``<root>/<arch>/``. Binaries are content-addressed by the
  compiler input that produced them (see :func:`comgr_input_key`), so
  identities that emit the same code share one binary and a rebuild after an
  emitter change only recompiles the kernels whose code actually changed.

Both classes live in ``library/`` (not in the installable ``rocke`` wheel)
because they are specific to this library's conv kernel families and their
sweep tooling.  :mod:`benchmarks.common.kernel_sweep` is the primary consumer;
``library/tests/`` uses both for the ABI regression suite.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Build provenance
# ---------------------------------------------------------------------------


# The per-axis workgroup-count cap the validators enforce (gridDim y/z on
# AMD; x is capped the same way for symmetry with is_valid_spec_for_problem).
_MAX_GRID_DIM = 65535


def current_llvm_flavor() -> str:
    """LLVM IR flavor this process lowers to (the one ``compile_kernel`` uses).

    Memoised per ``ROCKE_LLVM_FLAVOR`` value: without the override the flavor
    is detected from the COMGR library, and resolving that path globs the
    filesystem -- per cache entry, it was most of a --run-from-cache startup.
    The library does not change within a process.
    """
    return _llvm_flavor_for(os.environ.get("ROCKE_LLVM_FLAVOR", ""))


@functools.lru_cache(maxsize=None)
def _llvm_flavor_for(env_override: str) -> str:
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


@functools.lru_cache(maxsize=None)
def current_emitter_digest() -> str:
    """SHA-1 over the emitter: the ``rocke`` and ``kernels`` sources, plus the
    C++ engine's build-id when the ``rocke_engine`` binding is loaded.

    The sweep lowers through the default backend -- the C++ engine when its
    binding imports, the Python emitter otherwise -- so either may have
    produced a cached binary. The sources cover the Python emitter (and the
    kernel definitions both engines build from); the build-id covers a
    rebuilt or stale ``.so`` whose sources here did not change. It reads
    ``unknown`` without the binding, and is then left out.

    This digest is only a staleness hint (see :meth:`KernelCache.stale_entries`):
    whether a kernel is rebuilt is decided by its content key, the hash of
    the LLVM IR the engine actually emitted.
    """
    import kernels
    import rocke
    from rocke.helpers.manifest import engine_build_id

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
    build_id = engine_build_id()
    if build_id != "unknown":
        h.update(b"engine:" + build_id.encode())
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def current_comgr_id() -> str:
    """Identity of the COMGR library this process compiles with.

    The same LLVM IR compiled by another COMGR is another binary, so the
    binary key has to change when the library does. Path plus size and
    modification time is enough to tell installs apart without loading it.
    """
    from rocke.runtime.comgr import resolved_lib_path

    path = resolved_lib_path()
    if not path:
        return "<no-comgr>"
    real = os.path.realpath(path)
    try:
        st = os.stat(real)
    except OSError:
        return real
    return f"{real}:{st.st_size}:{st.st_mtime_ns}"


def comgr_input_key(comgr_input) -> str:
    """Content key of the binary a :class:`rocke.helpers.compile.ComgrInput`
    compiles to.

    The HSACO is a function of the LLVM IR, the ISA, the COMGR options and the
    COMGR library. The kernel's own name is normalised out of the IR, so two
    identities whose code differs only in its name (an alias pipeline, a
    split-K degree that rides a kernarg) share one binary -- each keeps its
    own entry, which records the symbol to launch.
    """
    h = hashlib.sha256()
    h.update(comgr_input.llvm_text.replace(comgr_input.kernel_name, "@K@").encode())
    for part in (comgr_input.isa, *comgr_input.options, current_comgr_id()):
        h.update(b"\0")
        h.update(str(part).encode())
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
    # Two-stage scratch slabs per group (WgradConvSpec.ws_replicas). The host
    # sizes the scratch and builds Stage 2 from it, so it has to come from the
    # binary's identity rather than a default the two could drift apart on.
    # 0 = no scratch (not two-stage).
    ws_replicas: int = 0
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
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
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
        if self.group_merge > 1:
            bits.append(f"gm{self.group_merge}")
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
        # An entry that predates the provenance field was lowered for an
        # unknown flavor; "" never equals the current one, so it never matches.
        kw.setdefault("llvm_flavor", "")
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
                blobs/
                    <content key>.hsaco      one per distinct compiled binary
                conv_fwd/
                    <identity sha1>.meta.json   one per kernel identity
                conv_wgrad/
                conv_dgrad/
                conv_direct/

    An identity's metadata names its binary (``"blob"``, a
    :func:`comgr_input_key`) and the symbol to launch (``"kernel_name"``).
    Binaries are content-addressed, so identities that emit the same code
    share one file, and entries are kept per identity, so an identity keeps
    its entry when the emitter changes and only its ``blob`` moves if its code
    did. Entries written before blobs existed (an ``.hsaco`` next to the
    metadata) are still read.

    The cache is per-arch. Callers pass an ``arch`` at construction time and
    the cache scopes all operations under ``<root>/<arch>/``.
    """

    def __init__(self, root: Path, arch: str) -> None:
        self._root = Path(root)
        self._arch = arch
        self._base = self._root / arch
        # Entries read so far, per direction subdirectory: a run reads the
        # whole cache several times (describe, stale check, compatible), and
        # each pass parses every metadata file. Dropped on every write.
        self._loaded: Dict[str, list] = {}

    def _dir_for(self, identity: KernelIdentity) -> Path:
        subdir = _DIRECTION_DIRS.get(identity.direction, identity.direction)
        return self._base / subdir

    def _meta_path(self, identity: KernelIdentity) -> Path:
        return self._dir_for(identity) / f"{identity.stable_hash()}.meta.json"

    def blob_path(self, key: str) -> Path:
        """Where the binary with content key ``key`` lives."""
        return self._base / "blobs" / f"{key}.hsaco"

    def has_blob(self, key: str) -> bool:
        p = self.blob_path(key)
        return p.exists() and p.stat().st_size > 0

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        # Several processes may finish the same binary; a rename never exposes
        # a half-written file to a reader.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def put_blob(self, key: str, hsaco: bytes, kernel_name: str) -> Path:
        """Store a binary under its content key (no-op if already present).

        ``kernel_name`` is the entry-point symbol inside it -- the name of the
        identity it was compiled for. Every identity later linked to the same
        binary launches that symbol, so it is recorded next to the binary.
        """
        path = self.blob_path(key)
        if not self.has_blob(key):
            self._write_atomic(
                path.with_suffix(".json"),
                json.dumps({"kernel_name": kernel_name}).encode("utf-8"),
            )
            self._write_atomic(path, hsaco)
        return path

    def blob_kernel_name(self, key: str) -> str:
        """Entry-point symbol of the stored binary ``key``."""
        side = self.blob_path(key).with_suffix(".json")
        return json.loads(side.read_text(encoding="utf-8"))["kernel_name"]

    def link(self, identity: KernelIdentity, key: str, meta: dict) -> None:
        """Point ``identity`` at the stored binary ``key``."""
        full_meta = {
            "identity": identity.to_dict(),
            "blob": key,
            "hsaco_bytes": self.blob_path(key).stat().st_size,
        }
        full_meta.update(meta)
        # The symbol to launch is the binary's, whichever identity compiled it.
        full_meta["kernel_name"] = self.blob_kernel_name(key)
        self._write_atomic(
            self._meta_path(identity),
            json.dumps(full_meta, indent=2, sort_keys=True).encode("utf-8"),
        )
        self._loaded.clear()

    def put(
        self,
        identity: KernelIdentity,
        hsaco: bytes,
        meta: Optional[dict] = None,
    ) -> Path:
        """Store ``hsaco`` for ``identity``. Returns the binary's path.

        Keyed by the bytes themselves; the sweep uses :meth:`put_blob` and
        :meth:`link` with the compiler-input key instead, which is known
        before anything is compiled.
        """
        meta = dict(meta or {})
        key = hashlib.sha256(hsaco).hexdigest()
        path = self.put_blob(key, hsaco, meta.get("kernel_name", ""))
        self.link(identity, key, meta)
        return path

    def index(self, directions: Optional[Iterable[str]] = None) -> Dict[str, dict]:
        """``{identity sha1: metadata}`` for every entry with a binary.

        ``directions`` limits the walk to those directions' entries; every
        direction is parsed per entry, so on a large cache reading only the
        ones needed is the difference between a few files and ~10^6.
        """
        if directions is None:
            entries: Iterable = self._entries(None)
        else:
            entries = itertools.chain.from_iterable(
                self._entries(d) for d in sorted(set(directions))
            )
        return {ident.stable_hash(): meta for ident, _, meta in entries}

    def meta(self, identity: KernelIdentity) -> Optional[dict]:
        """The identity's metadata, or ``None`` when it has no entry."""
        meta_path = self._meta_path(identity)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _resolve(self, meta_path: Path, meta: dict) -> Path:
        if "blob" in meta:
            return self.blob_path(meta["blob"])
        # Pre-blob layout: the binary sits next to its metadata.
        return meta_path.with_suffix("").with_suffix(".hsaco")

    def get(self, identity: KernelIdentity) -> Optional[Tuple[bytes, dict]]:
        """Load an HSACO + metadata from cache, or ``None`` on miss."""
        meta = self.meta(identity)
        if meta is None:
            return None
        path = self._resolve(self._meta_path(identity), meta)
        if not path.exists() or path.stat().st_size == 0:
            return None
        return path.read_bytes(), meta

    def hsaco_path(self, identity: KernelIdentity) -> Path:
        """Where the identity's HSACO lives (whether or not it exists yet)."""
        meta = self.meta(identity) or {}
        return self._resolve(self._meta_path(identity), meta)

    def has(self, identity: KernelIdentity) -> bool:
        """Check whether the cache has a valid entry for the identity."""
        return self.get(identity) is not None

    def _entries(self, direction: Optional[str]):
        if direction is not None:
            subdirs = [_DIRECTION_DIRS.get(direction, direction)]
        else:
            subdirs = list(_DIRECTION_DIRS.values())
        for subdir in subdirs:
            if subdir not in self._loaded:
                self._loaded[subdir] = self._load(subdir)
            yield from self._loaded[subdir]

    def _load(self, subdir: str) -> list:
        """Parse one direction's entries: those whose binary is present."""
        d = self._base / subdir
        if not d.is_dir():
            return []
        out = []
        # Most entries share a binary with others; stat each one once.
        blob_ok: Dict[Path, bool] = {}
        for meta_path in sorted(d.glob("*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ident = KernelIdentity.from_dict(meta["identity"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            path = self._resolve(meta_path, meta)
            ok = blob_ok.get(path)
            if ok is None:
                ok = blob_ok[path] = path.exists() and path.stat().st_size > 0
            if ok:
                out.append((ident, path, meta))
        return out

    def list_all(
        self, direction: Optional[str] = None
    ) -> List[Tuple[KernelIdentity, Path]]:
        """List all cached identities, optionally filtered by direction.

        Returns ``(identity, hsaco_path)`` pairs.
        """
        return [(ident, path) for ident, path, _ in self._entries(direction)]

    def stale_entries(self, direction: Optional[str] = None) -> int:
        """Entries built from different emitter sources than this process's.

        Such a binary may still be current -- most emitter changes leave most
        kernels' code alone -- but only a ``--compile-all`` can tell, by
        re-emitting and comparing content keys. The run side reports the count
        rather than rejecting the entries.
        """
        digest = current_emitter_digest()
        return sum(
            1
            for _, _, meta in self._entries(direction)
            if meta.get("emitter_digest") != digest
        )

    @staticmethod
    def _group_merge_fits(
        identity: KernelIdentity, problem: object, gm: int
    ) -> Tuple[bool, str]:
        """Can a wgrad binary merging ``gm`` groups serve ``problem``?

        Mirrors the shape rules of ``wgrad_group_merge_available``: merging is
        depthwise only, ``gm`` must divide the group count, and the merged
        GEMM (``kpg*Gm`` x ``[Z*]Y*X*cpg*Gm``) must fit one tile -- the
        filter extent is a runtime value, so it is checked here, per problem.
        """
        cpg = int(getattr(problem, "cpg", 0))
        kpg = int(getattr(problem, "kpg", 0))
        groups = int(getattr(problem, "groups", 1))
        if cpg != 1 or kpg != 1:
            return False, "group-merged kernel needs depthwise (cpg=kpg=1)"
        if groups % gm:
            return False, f"group_merge {gm} does not divide groups {groups}"
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

    def supports_problem(
        self, identity: KernelIdentity, problem: object
    ) -> Tuple[bool, str]:
        """Can this cached binary be launched on ``problem``?

        Returns ``(ok, reason)``; the reason makes an empty candidate list
        diagnosable instead of just "no compatible kernels".

        Provenance is checked first: a binary lowered for another LLVM flavor
        is never offered. A binary built from other emitter sources *is*
        offered -- most emitter changes leave most kernels' code alone, and only
        a ``--compile-all`` can tell which, by re-emitting and comparing content
        keys; :meth:`stale_entries` counts them so the run side can warn. Then
        two classes of constraint:

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
        """
        # A binary lowered for another LLVM flavor is stale, whatever it was
        # configured for. (Emitter changes are not checked here: see
        # :meth:`stale_entries`.)
        if identity.llvm_flavor != current_llvm_flavor():
            return False, (
                f"built for LLVM flavor {identity.llvm_flavor or '<unknown>'}, "
                f"this process lowers to {current_llvm_flavor()}"
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
            # A group-merged wgrad kernel (depthwise only) loads dY and X along
            # the Gm merged channels; dW stays per group.
            ok, why = self._group_merge_fits(identity, problem, gm)
            if not ok:
                return False, why
            extents = ((kpg * gm, "kpg*Gm"), (cpg * gm, "cpg*Gm"), (cpg, "cpg"))
        for field, (extent, name) in zip(
            ("vector_size_a", "vector_size_b", "vector_size_c"), extents
        ):
            vec = getattr(identity, field)
            if vec and extent and extent % vec != 0:
                return False, f"{name}={extent} not divisible by {field}={vec}"

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
            # Mirrors is_valid_dgrad_spec: the grouped implicit-GEMM dgrad has
            # no depthwise (cpg == 1) path, and the grouped binary is offered
            # to every grouped problem otherwise.
            if int(getattr(problem, "groups", 1)) > 1 and cpg == 1:
                return False, (
                    "depthwise dgrad (cpg == 1) is not supported by the "
                    "implicit-GEMM grouped path"
                )

        if not identity.is_direct:
            ok, why = self._grid_fits(identity, problem)
            if not ok:
                return False, why

        return True, "ok"

    @staticmethod
    def _grid_fits(identity: KernelIdentity, problem: object) -> Tuple[bool, str]:
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
            gy = -(-int(getattr(problem, "M", 0)) // tm)
            gx = -(-kpg // tn)
            axes = (
                ("y", gy, f"M={getattr(problem, 'M', 0)} tile_m={tm}"),
                ("x", gx, f"kpg={kpg} tile_n={tn}"),
                ("z", groups, f"groups={groups}"),
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

    def compatible(
        self, problem: object, *, direction: Optional[str] = None
    ) -> List[Tuple[KernelIdentity, Path, dict]]:
        """Every cached kernel that can run ``problem``, with its metadata.

        The metadata is returned because the caller needs ``kernel_name`` from
        it: the HSACO's entry-point symbol is not derivable from the identity.
        """
        out: List[Tuple[KernelIdentity, Path, dict]] = []
        for identity, hsaco_path, meta in self._entries(direction):
            ok, _ = self.supports_problem(identity, problem)
            if ok:
                out.append((identity, hsaco_path, meta))
        return out
