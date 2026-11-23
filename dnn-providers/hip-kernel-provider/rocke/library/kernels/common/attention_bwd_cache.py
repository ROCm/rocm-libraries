# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-disk code-object cache of the attention backward kernels.

One cache serves every layer: the run glue
(:func:`kernels.common.attention_bwd_run.attn_bwd_kernels`, opt-in through
``ROCKE_BWD_HSACO_CACHE``), the backward tests' session pre-build, and the
sweep and timing harness (``benchmarks/common/attention_bwd_compile.py``,
which re-exports this module and adds the batch compiler). An entry written by
one layer is read by every other one: the key is the same everywhere.

* :func:`toolchain_identity` names the toolchain this process compiles with:
  the LLVM IR flavor of the lowering (``ROCKE_LLVM_FLAVOR`` or the autodetected
  one), the ``libamd_comgr`` rocKE resolves (path, ROCm vintage and a content
  digest of the library file) and the environment switches that change the
  emitted IR (``ROCKE_DEBUG_LOC``). It never imports torch: it pins the
  production comgr (``attention_bwd_comgr``) and names the one the next
  compile in this process would load.
* :func:`backward_source_hash` is a content hash of every Python source the
  backward kernels can depend on under the Python lowering backend: the whole
  ``rocke`` package and every module under ``library/kernels``. Line endings
  are normalised, so a synced copy and the working tree hash alike. Any change
  invalidates every entry.
* :class:`CompileCache` stores one code object per (kernel stage, resolved
  spec, arch, toolchain, source hash), written atomically, re-checked against
  its metadata and its digest on every read.
* :func:`env_cache` is the cache named by ``ROCKE_BWD_HSACO_CACHE`` for this
  process's toolchain (``None`` when the variable is unset: the default).

The key holds the spec, not the builder: a code object built by another
builder than the shipped one (a test-only body switch) is cached only under a
``variant`` naming that builder; ``attn_bwd_kernels`` bypasses the cache for a
swapped builder that declares none.

The key also assumes the live emission code is the hashed source. A negative
control that monkeypatches an emitter (a method of a block GEMM, a function
of the prefetch ring) builds another kernel from the same spec;
:func:`patched_emission` names such patches, and ``attn_bwd_kernels`` neither
reads nor writes the disk while one is active (else the control would load
the shipped kernel, and a cold cache would store the mutant under the
shipped kernel's key).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import site
import sys
import sysconfig
import time
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CACHE_SCHEMA",
    "EMISSION_PACKAGES",
    "EMIT_ENV",
    "HSACO_CACHE_ENV",
    "NOT_EMISSION",
    "STAGES",
    "CacheEntry",
    "CachedArtifact",
    "CompileCache",
    "Toolchain",
    "backward_source_hash",
    "comgr_digests",
    "emission_env",
    "entry_key",
    "env_cache",
    "env_cache_stats",
    "patched_emission",
    "process_source_hash",
    "seed_comgr_digests",
    "source_files",
    "spec_identity",
    "toolchain_identity",
]

STAGES = ("main", "prep", "convert", "dq", "reduce", "dense", "dense_dq")
CACHE_SCHEMA = "rocke_bwd_compile_cache/v1"
# Directory of the opt-in cache of the run glue (unset: no disk cache).
HSACO_CACHE_ENV = "ROCKE_BWD_HSACO_CACHE"
# Environment switches that change the emitted IR of a kernel built from the
# same spec; their values are part of the toolchain identity.
EMIT_ENV = ("ROCKE_DEBUG_LOC",)

_ROOT = Path(__file__).resolve().parents[3]  # .../rocke


# --------------------------------------------------------------------------- identity
def _sha256_file(path: str | os.PathLike, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


_FILE_SHA: dict[tuple, str] = {}


def _comgr_sha(path: str) -> str:
    """Content digest of the comgr library, memoised per (path, size, mtime)."""
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _FILE_SHA:
        _FILE_SHA[key] = _sha256_file(path, 12)
    return _FILE_SHA[key]


def comgr_digests() -> dict[tuple, str]:
    """This process's comgr digest memo (to seed a worker process with)."""
    return dict(_FILE_SHA)


def seed_comgr_digests(memo: Mapping[tuple, str]) -> None:
    """Take another process's digests: the key holds the path, size and
    mtime, so a file changed since is digested again. Saves every pool
    worker re-reading the comgr library (a large file on a network home)."""
    for key, sha in memo.items():
        if isinstance(key, tuple) and len(key) == 3 and isinstance(sha, str):
            _FILE_SHA.setdefault(key, sha)


def emission_env() -> str:
    """The set :data:`EMIT_ENV` switches, as ``NAME=value`` (``""``: none)."""
    vals = []
    for name in EMIT_ENV:
        v = os.environ.get(name, "")
        if v not in ("", "0"):
            vals.append(f"{name}={v}")
    return ";".join(vals)


@dataclass(frozen=True)
class Toolchain:
    """The LLVM flavor and the comgr library a process compiles with."""

    flavor: str
    comgr_path: str | None
    comgr_rocm: str | None  # ROCm vintage of the resolved library, "major.minor"
    comgr_sha: str | None  # content digest of the resolved library file
    emit: str = ""  # emission-changing environment (see :data:`EMIT_ENV`)

    @property
    def comgr(self) -> str:
        """Compact comgr identity used in cache keys and records."""
        name = Path(self.comgr_path).name if self.comgr_path else "none"
        return f"{name}@rocm{self.comgr_rocm or '?'}#{self.comgr_sha or '?'}"

    def record(self) -> dict:
        return {
            "rocke_llvm_flavor": self.flavor,
            "comgr": self.comgr,
            "comgr_path": self.comgr_path,
            "comgr_rocm": self.comgr_rocm,
            "emit": self.emit,
        }


def toolchain_identity() -> Toolchain:
    """Flavor, comgr and emission switches the next compile here would use.

    Never imports torch. It first pins the production comgr (torch's bundled
    library, found without importing torch; see
    :func:`kernels.common.attention_bwd_comgr.ensure_production_comgr`), so a
    torch-free process names the same library, vintage and digest as one that
    imported torch. The library digest is memoised per file version.
    """
    from kernels.common.attention_bwd_comgr import ensure_production_comgr

    ensure_production_comgr()
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR")
    if not flavor:
        try:
            from rocke.core.lower_llvm import _detect_llvm_flavor

            flavor = _detect_llvm_flavor()
        except Exception:  # noqa: BLE001
            flavor = "unknown"
    path = rocm = sha = None
    try:
        from rocke.runtime.comgr import resolved_lib_path, resolved_lib_rocm_version

        path = resolved_lib_path()
        v = resolved_lib_rocm_version()
        rocm = f"{v[0]}.{v[1]}" if v else None
        if path and os.path.exists(path):
            path = os.path.realpath(path)
            sha = _comgr_sha(path)
    except Exception:  # noqa: BLE001 - recorded as unknown
        pass
    return Toolchain(flavor, path, rocm, sha, emission_env())


def source_files(root: Path | None = None) -> list[Path]:
    """Every source the backward kernels can depend on (see module doc)."""
    base = Path(root) if root is not None else _ROOT
    out = []
    for sub in ("platform/python/rocke", "library/kernels"):
        d = base / sub
        if d.is_dir():
            out.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out, key=lambda p: p.relative_to(base).as_posix())


def backward_source_hash(root: Path | None = None) -> str:
    """Content hash of :func:`source_files` (paths and LF-normalised bytes)."""
    base = Path(root) if root is not None else _ROOT
    h = hashlib.sha256()
    for p in source_files(base):
        h.update(p.relative_to(base).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:16]


_SOURCE_HASH: dict[Path, str] = {}


def process_source_hash() -> str:
    """:func:`backward_source_hash` of this tree, computed once per process
    (a process does not see source edits made after it imported the kernels)."""
    if _ROOT not in _SOURCE_HASH:
        _SOURCE_HASH[_ROOT] = backward_source_hash(_ROOT)
    return _SOURCE_HASH[_ROOT]


# --------------------------------------------------------------------------- live code
# The emission code: the packages that turn a resolved spec into kernel IR (the
# kernel builders and emitters, the IR builder, its helpers and the lowering).
# Not emission: the run glue and this cache (they pick and load code objects),
# the compile driver and the comgr / HIP runtime (the toolchain, keyed by
# :func:`toolchain_identity`; the cache's own tests stub them).
EMISSION_PACKAGES = ("kernels", "rocke.core", "rocke.helpers", "rocke.instances")
NOT_EMISSION = frozenset(
    (
        "kernels.common.attention_bwd_cache",
        "kernels.common.attention_bwd_run",
        "kernels.common.attention_bwd_unified_run",
        "rocke.helpers.compile",
    )
)


def _norm_dir(p) -> str:
    return os.path.normcase(os.path.realpath(str(p))).rstrip("\\/") + os.sep


@functools.lru_cache(maxsize=1)
def _code_dirs() -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    """``(hashed source roots, the rocke tree, the Python installation's
    library directories)``, normalised with a trailing separator."""
    home = tuple(
        _norm_dir(_ROOT / s) for s in ("platform/python/rocke", "library/kernels")
    )
    py = set()
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        if paths.get(key):
            py.add(_norm_dir(paths[key]))
    try:
        py.update(_norm_dir(p) for p in site.getsitepackages())
    except AttributeError:  # an interpreter without site.getsitepackages
        pass
    user = site.getusersitepackages()
    if user:
        py.add(_norm_dir(user))
    return home, _norm_dir(_ROOT), tuple(sorted(py))


_FOREIGN_FILE: dict[str, bool] = {}


def _foreign_file(filename: str) -> bool:
    """True for code that is neither hashed source nor installed library code:
    the rocke tree outside the hashed roots (tests, benchmarks) or any file
    outside the Python installation. Generated code (``<string>``, frozen
    modules) is not foreign."""
    hit = _FOREIGN_FILE.get(filename)
    if hit is None:
        if filename.startswith("<"):
            hit = False
        else:
            home, root, py = _code_dirs()
            p = os.path.normcase(os.path.realpath(filename))
            if p.startswith(home):
                hit = False
            elif p.startswith(root):
                hit = True
            else:
                hit = not p.startswith(py)
        _FOREIGN_FILE[filename] = hit
    return hit


def _foreign(obj: Any) -> bool:
    """A callable that cannot be the shipped emission code: a mock, a partial,
    or a function (also behind a static / class method or a property) whose
    code is foreign (:func:`_foreign_file`)."""
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        return any(_foreign(f) for f in (obj.fget, obj.fset, obj.fdel) if f is not None)
    if isinstance(obj, (functools.partial, functools.partialmethod)):
        return True
    if type(obj).__module__ == "unittest.mock":
        return True
    if isinstance(obj, types.FunctionType):
        return _foreign_file(obj.__code__.co_filename)
    return False


def _emission_modules():
    home = _code_dirs()[0]
    for name, mod in list(sys.modules.items()):
        if mod is None or name in NOT_EMISSION:
            continue
        if not any(name == p or name.startswith(p + ".") for p in EMISSION_PACKAGES):
            continue
        f = getattr(mod, "__file__", None)
        if f and os.path.normcase(os.path.realpath(f)).startswith(home):
            yield name, mod


def _scan(owner: str, ns: Mapping, out: list, depth: int) -> None:
    for attr, value in list(ns.items()):
        if attr == "__builtins__":
            continue
        where = f"{owner}.{attr}"
        if _foreign(value):
            out.append(where)
        elif depth and isinstance(value, Mapping):
            _scan(where, value, out, 0)
        elif depth and isinstance(value, (list, tuple)):
            _scan(where, dict(enumerate(value)), out, 0)


def patched_emission() -> list[str]:
    """Where the live emission code is not the hashed source: every function
    or method of an emission module (:data:`EMISSION_PACKAGES` minus
    :data:`NOT_EMISSION`), of a class defined there, or held one level deep in
    a module-level table, that is bound to a mock, a partial or foreign code
    (a test's monkeypatch); ``[]`` for the shipped code.

    A patch that swaps in another shipped function or changes plain data is
    not seen; a swapped ``attention_bwd_run._BUILDERS`` entry is handled by
    its ``cache_variant``.
    """
    out: list[str] = []
    for name, mod in _emission_modules():
        ns = vars(mod)
        _scan(name, ns, out, 1)
        for value in list(ns.values()):
            if isinstance(value, type) and value.__module__ == name:
                _scan(f"{name}.{value.__qualname__}", vars(value), out, 0)
    return sorted(set(out))


# --------------------------------------------------------------------------- keys
def spec_identity(stage: str, spec: Any) -> str:
    """Stable text of a resolved spec (dataclass repr is field-ordered)."""
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} not in {STAGES}")
    return f"{stage}|{type(spec).__name__}|{spec!r}"


def entry_key(
    stage: str,
    spec: Any,
    *,
    arch: str,
    toolchain: Toolchain,
    source_hash: str,
    variant: str = "",
) -> str:
    """Cache key: spec, arch, flavor, comgr, emission switches, source hash
    and the builder variant (``""``: the shipped builder)."""
    parts = [
        spec_identity(stage, spec),
        arch,
        toolchain.flavor,
        toolchain.comgr,
        source_hash,
    ]
    if toolchain.emit:  # absent when unset: keys of plain builds are unchanged
        parts.append(toolchain.emit)
    if variant:
        parts.append(f"variant={variant}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


# --------------------------------------------------------------------------- cache
def _replace(src: Path, dst: Path, tries: int = 50) -> None:
    """``os.replace``, retried while the destination is held open by another
    process (Windows refuses the rename then; POSIX never does)."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.01)


@dataclass
class CacheEntry:
    key: str
    kernel_name: str
    hsaco: bytes
    meta: dict

    @property
    def hsaco_sha(self) -> str:
        return hashlib.sha256(self.hsaco).hexdigest()[:16]


class CachedArtifact:
    """Stand-in for a ``KernelArtifact`` of a cached code object."""

    def __init__(self, entry: CacheEntry) -> None:
        self.kernel_name = entry.kernel_name
        self.hsaco = entry.hsaco
        self.cache_key = entry.key
        self.hsaco_sha = entry.hsaco_sha


class CompileCache:
    """Code objects on disk: ``<dir>/<key[:2]>/<key>.{hsaco,json}``.

    Writes are atomic per file (a temporary file with a process- and
    call-unique name, then a rename), so concurrent writers (pytest-xdist
    workers, shards on one node, jobs sharing a directory) never expose a
    partial file. ``read`` returns ``None`` for a missing entry and for an
    entry whose metadata names another schema, flavor, comgr, emission
    switches, arch, source hash, spec or builder variant than the caller's, or
    whose code object does not match the recorded digest (``rejected`` counts
    those; the next ``write`` replaces them). ``variant`` names a builder other
    than the shipped one that builds a stage from the same spec (a test-only
    body switch); it is part of the key.
    """

    def __init__(
        self, root: str | os.PathLike, *, toolchain: Toolchain, source_hash: str
    ) -> None:
        self.root = Path(root)
        self.toolchain = toolchain
        self.source_hash = source_hash
        self.hits = 0
        self.misses = 0
        self.rejected = 0
        self.writes = 0
        self.bypassed = 0  # built with the disk bypassed (patched emission)

    def key(self, stage: str, spec: Any, arch: str, variant: str = "") -> str:
        return entry_key(
            stage,
            spec,
            arch=arch,
            toolchain=self.toolchain,
            source_hash=self.source_hash,
            variant=variant,
        )

    def _paths(self, key: str) -> tuple[Path, Path]:
        d = self.root / key[:2]
        return d / f"{key}.hsaco", d / f"{key}.json"

    def _want(self, stage: str, spec: Any, arch: str, variant: str) -> dict:
        return {
            "schema": CACHE_SCHEMA,
            "arch": arch,
            "flavor": self.toolchain.flavor,
            "comgr": self.toolchain.comgr,
            "emit": self.toolchain.emit,
            "source_hash": self.source_hash,
            "spec": spec_identity(stage, spec),
            "variant": variant,
        }

    def read(
        self, stage: str, spec: Any, arch: str, *, variant: str = ""
    ) -> CacheEntry | None:
        key = self.key(stage, spec, arch, variant)
        hp, mp = self._paths(key)
        if not (hp.exists() and mp.exists()):
            self.misses += 1
            return None
        try:
            text = mp.read_text(encoding="utf-8")
            blob = hp.read_bytes()
        except OSError:  # vanished or held by a concurrent replace
            self.misses += 1
            return None
        try:
            meta = json.loads(text)
        except ValueError:
            self.rejected += 1
            return None
        if not isinstance(meta, dict):
            self.rejected += 1
            return None
        want = self._want(stage, spec, arch, variant)
        # entries written before "emit" / "variant" existed hold the defaults
        got = {k: meta.get(k, "" if k in ("emit", "variant") else None) for k in want}
        if (
            got != want
            or not isinstance(meta.get("kernel_name"), str)
            or hashlib.sha256(blob).hexdigest()[:16] != meta.get("hsaco_sha")
        ):
            self.rejected += 1
            return None
        self.hits += 1
        return CacheEntry(key, meta["kernel_name"], blob, meta)

    def write(
        self,
        stage: str,
        spec: Any,
        arch: str,
        *,
        kernel_name: str,
        hsaco: bytes,
        extra: Mapping[str, Any] | None = None,
        variant: str = "",
    ) -> CacheEntry:
        key = self.key(stage, spec, arch, variant)
        hp, mp = self._paths(key)
        hp.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            **self._want(stage, spec, arch, variant),
            "kernel_name": kernel_name,
            "hsaco_sha": hashlib.sha256(hsaco).hexdigest()[:16],
            "written": time.time(),
            **dict(extra or {}),
        }
        tag = f".{os.getpid()}.{time.time_ns()}.{os.urandom(4).hex()}.tmp"
        th, tm = hp.with_name(hp.name + tag), mp.with_name(mp.name + tag)
        try:
            th.write_bytes(hsaco)
            tm.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
            # code object first: a reader that sees the new metadata also sees
            # the code object it names (an older one fails the digest check)
            _replace(th, hp)
            _replace(tm, mp)
        finally:
            for t in (th, tm):
                try:
                    t.unlink()
                except OSError:
                    pass
        self.writes += 1
        return CacheEntry(key, kernel_name, hsaco, meta)


# --------------------------------------------------------------------------- env cache
_ENV_CACHES: dict[tuple, CompileCache] = {}


def env_cache() -> CompileCache | None:
    """The cache in ``$ROCKE_BWD_HSACO_CACHE`` for this process's toolchain.

    ``None`` when the variable is unset or empty (the default: no disk cache).
    One object per (directory, toolchain), so its counters cover the process.
    """
    root = os.environ.get(HSACO_CACHE_ENV, "").strip()
    if not root:
        return None
    tc = toolchain_identity()
    path = Path(root).expanduser()
    k = (str(path), tc)
    if k not in _ENV_CACHES:
        _ENV_CACHES[k] = CompileCache(
            path, toolchain=tc, source_hash=process_source_hash()
        )
    return _ENV_CACHES[k]


def env_cache_stats() -> dict[str, int]:
    """Counters of every :func:`env_cache` object of this process: ``hits``
    (code objects loaded from disk), ``misses`` (compiled here), ``rejected``,
    ``writes`` and ``bypassed`` (built without the disk: patched emission)."""
    out = {"hits": 0, "misses": 0, "rejected": 0, "writes": 0, "bypassed": 0}
    for c in _ENV_CACHES.values():
        for k in out:
            out[k] += getattr(c, k)
    return out
