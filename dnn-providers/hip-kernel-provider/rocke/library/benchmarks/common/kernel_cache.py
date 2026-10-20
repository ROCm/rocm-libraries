# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AOT disk cache for shape-generic convolution kernels.

:class:`KernelCache` reads, writes and lists HSACO blobs from a directory tree
under ``<root>/<arch>/``. Binaries are content-addressed by the compiler input
that produced them (see :func:`comgr_input_key`), so identities that emit the
same code share one binary and a rebuild after an emitter change only
recompiles the kernels whose code actually changed.

Entries are keyed by :class:`~kernels.conv.kernel_identity.KernelIdentity`,
the compile-time tuple that uniquely identifies one HSACO. It lives with the
kernels because the grouped-convolution dispatcher registers its instances by
the same identity; it is re-exported here, next to the cache that files
binaries under it.

The cache lives in ``library/`` (not in the installable ``rocke`` wheel)
because it is specific to this library's conv kernel families and their sweep
tooling. :mod:`benchmarks.common.kernel_sweep` is the primary consumer;
``library/tests/`` uses it for the ABI regression suite.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from kernels.conv.kernel_identity import (  # noqa: F401 (re-exported)
    KernelIdentity,
    current_llvm_flavor,
    identity_supports_problem,
)


# ---------------------------------------------------------------------------
# Build provenance
# ---------------------------------------------------------------------------


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
    def supports_problem(identity: KernelIdentity, problem: object) -> Tuple[bool, str]:
        """Can this cached binary be launched on ``problem``?

        :func:`kernels.conv.kernel_identity.identity_supports_problem`;
        the grouped-convolution dispatcher admits its instances with the same
        rule, so the two never disagree on what a binary serves.
        """
        return identity_supports_problem(identity, problem)

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
