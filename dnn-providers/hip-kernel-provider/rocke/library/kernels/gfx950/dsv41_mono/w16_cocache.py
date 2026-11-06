# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W16 I1: persistent caches for the dsv41_mono rocKE builds (fast iteration, PLAN decision 12).

Two caches, each keyed on what the step reads, each with a FRESH-COMPILE CONTROL mode that proves a hit is the bytes
a compile in this process would give (the compile-order effect: ``iglp_opt(1)`` code depends on earlier compiles in
the process, so a cache hit is checked against fresh compiles in fresh processes, not assumed).

1. **HSACO cache** (``install(dir, mode)`` / ``ROCKE_W16_COCACHE=<dir>``): every rocKE code-object compile of the
   process goes through it.

   * COMGR (``rocke.helpers.compile.build_hsaco_from_llvm_ir``): key = sha256 of the LLVM IR text (the IR hash),
     the ISA, the COMGR options, the fresh-process flag and the toolchain (sha256 of the COMGR library FILE that
     compiles, its LLVM version, the ``ROCKE_LLVM_FLAVOR``);
   * Rust direct (``rocke.runtime.code_object.build_direct_rocke_ir_hsaco``): key = sha256 of the serialized rocKE
     IR (``ck.dsl.ir/v1``), the ISA, the direct levers in force (``ROCKE_DIRECT_*``) and the toolchain (sha256 of
     the ``rocke_codeobj_ffi`` library file).

   Modes (``ROCKE_W16_COCACHE_MODE``): ``hit`` (default; a miss compiles and stores), ``verify`` (always compiles;
   a stored entry must be byte-identical, else ``DIFF`` is logged and, with ``ROCKE_W16_COCACHE_STRICT=1``,
   raised), ``fresh`` (always compiles, stores a missing entry), ``ro`` (reads only). ``ROCKE_W16_COCACHE_LOG``:
   one JSON line per event.

2. **Stage-function generation cache** (``python -m kernels.gfx950.dsv41_mono.w16_cocache gen``): a stage
   generator's output directory (``.ll`` + ``.json`` sidecars) keyed on the generator's command line (without its
   ``--out``), the codegen environment (``ROCKE_*``) and the sha256 of every rocKE source the generators import
   (``platform/python/rocke`` and ``library/kernels/gfx950/{dsv41_mono,qwen38_mono}``, plus an ``--atom-root``
   tree's ``atom/models/deepseek_v41`` when the command checks ATOM's constants). A hit unpacks the stored files;
   ``verify`` regenerates and compares every file byte for byte.

        python -m kernels.gfx950.dsv41_mono.w16_cocache gen --cache C [--mode hit|verify|fresh|ro] --out D -- \
            kernels.gfx950.dsv41_mono.stage_moe --stages down --knobs r_max=1 ...

   The ``.ll`` text of a stage function is exactly what the ATOM code-object cache then hashes (as a linked file),
   so the two caches chain: same sources and knobs -> same ``.ll`` -> same linked kernel -> its stored code object.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ENV = "ROCKE_W16_COCACHE"
MODE_ENV = "ROCKE_W16_COCACHE_MODE"
LOG_ENV = "ROCKE_W16_COCACHE_LOG"
STRICT_ENV = "ROCKE_W16_COCACHE_STRICT"
MODES = ("hit", "verify", "fresh", "ro")
VERSION = 1
_STATE: dict = {"dir": None, "mode": None, "orig": {}, "stats": {}, "tc": {}}


def _sha(b) -> str:
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _log(rec: dict) -> None:
    st = _STATE["stats"]
    st[rec["event"]] = st.get(rec["event"], 0) + 1
    for k in ("compile_s", "load_s"):
        if k in rec:
            st[k] = round(st.get(k, 0.0) + rec[k], 4)
    path = os.environ.get(LOG_ENV)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "pid": os.getpid(), **rec}, sort_keys=True) + "\n")


def stats() -> dict:
    return dict(_STATE["stats"])


def _mode() -> str:
    m = _STATE["mode"] or os.environ.get(MODE_ENV, "hit") or "hit"
    if m not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {m!r}")
    return m


# ------------------------------------------------------------------ toolchains
def comgr_toolchain() -> str:
    """sha256 of the COMGR library file this process compiles with, its LLVM version and the IR flavor."""
    if "comgr" not in _STATE["tc"]:
        from rocke.runtime import comgr

        comgr._resolve_lib()
        path = comgr.resolved_lib_path() or ""
        real = os.path.realpath(path) if path else ""
        info = comgr.loaded_compiler_info()
        llvm = ".".join(str(x) for x in (getattr(info, "llvm_version", None) or ())) or "?"
        _STATE["tc"]["comgr"] = _sha(json.dumps({"lib": _file_sha(real) if real and os.path.isfile(real) else real,
                                                 "llvm": llvm, "flavor": os.environ.get("ROCKE_LLVM_FLAVOR", "")}))  # fmt: skip
    return _STATE["tc"]["comgr"]


def direct_toolchain() -> str:
    """sha256 of the Rust direct compiler library file (``rocke_codeobj_ffi``) this process loads."""
    from rocke.runtime import code_object as CO

    path = CO._require_direct_library()
    key = ("direct", path)
    if key not in _STATE["tc"]:
        _STATE["tc"][key] = _sha(json.dumps({"lib": _file_sha(os.path.realpath(path))}))
    return _STATE["tc"][key]


def _direct_env() -> list:
    return sorted((k, v) for k, v in os.environ.items() if k.startswith("ROCKE_DIRECT"))


# ------------------------------------------------------------------ store
def _paths(d: str, k: str):
    sub = os.path.join(d, "hsaco", k[:2])
    return os.path.join(sub, k + ".pkl"), os.path.join(sub, k + ".json")


def _load(d: str, k: str):
    pk, _ = _paths(d, k)
    try:
        with open(pk, "rb") as fh:
            ent = pickle.load(fh)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return None
    if ent.get("hsaco_sha256") != _sha(ent.get("hsaco", b"")):
        return None
    return ent


def _store(d: str, k: str, ent: dict, meta: dict) -> None:
    pk, js = _paths(d, k)
    _atomic_write(pk, pickle.dumps(ent, protocol=4))
    _atomic_write(js, json.dumps({**meta, "hsaco_sha256": ent["hsaco_sha256"], "host": socket.gethostname(),
                                  "pid": os.getpid(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                 indent=1, sort_keys=True).encode())  # fmt: skip


def _cached(kind: str, key_parts: dict, fresh, pack, unpack):
    """The cache in front of one compile. ``fresh()`` compiles; ``pack(result)`` -> (hsaco, extra) to store;
    ``unpack(entry)`` -> the result from an entry."""
    d, m = _STATE["dir"], _mode()
    k = _sha(json.dumps({"v": VERSION, "kind": kind, **key_parts}, sort_keys=True))
    rec = {"kind": kind, "key": k, "mode": m}
    got = _load(d, k) if m in ("hit", "ro", "verify") else None
    if got is not None and m in ("hit", "ro"):
        t0 = time.perf_counter()
        out = unpack(got)
        _log({**rec, "event": "hit", "hsaco": got["hsaco_sha256"], "load_s": round(time.perf_counter() - t0, 4)})
        return out
    t0 = time.perf_counter()
    out = fresh()
    dt = time.perf_counter() - t0
    hsaco, extra = pack(out)
    hs = _sha(hsaco)
    rec.update(compile_s=round(dt, 4), hsaco=hs)
    if got is not None:
        ev = "verify_SAME" if got["hsaco_sha256"] == hs else "verify_DIFF"
        _log({**rec, "event": ev, "stored": got["hsaco_sha256"]})
        if ev == "verify_DIFF" and os.environ.get(STRICT_ENV, "0") == "1":
            raise RuntimeError(f"{ENV}: fresh {kind} compile differs from entry {k}")
    elif m != "ro":
        _store(d, k, {"hsaco": hsaco, "hsaco_sha256": hs, "extra": extra}, {"key": k, "kind": kind,
               **{a: b for a, b in key_parts.items() if a != "ir"}, "ir_sha256": key_parts.get("ir"),
               "compile_s": round(dt, 3)})  # fmt: skip
        _log({**rec, "event": "miss_store"})
    else:
        _log({**rec, "event": "miss_ro"})
    return out


def _wrap_comgr(orig):
    def build_hsaco_from_llvm_ir(ir_text, *, isa="amdgcn-amd-amdhsa--gfx950", options=None, fresh_process=False):
        from rocke.runtime.comgr import ComgrTimings

        parts = {"ir": _sha(ir_text), "isa": isa, "options": list(options or []), "fresh_process": bool(fresh_process),
                 "toolchain": comgr_toolchain()}  # fmt: skip
        return _cached(
            "comgr", parts,
            lambda: orig(ir_text, isa=isa, options=options, fresh_process=fresh_process),
            lambda r: (r[0], None),
            lambda e: (e["hsaco"], ComgrTimings()),
        )  # fmt: skip

    return build_hsaco_from_llvm_ir


def _wrap_direct(orig):
    def build_direct_rocke_ir_hsaco(serialized_ir, *, isa=None):
        from rocke.runtime import code_object as CO

        isa = isa or CO._DIRECT_ISA
        b = serialized_ir.encode("utf-8") if isinstance(serialized_ir, str) else bytes(serialized_ir)
        parts = {"ir": _sha(b), "isa": isa, "direct_env": _direct_env(), "toolchain": direct_toolchain()}
        return _cached(
            "direct", parts,
            lambda: orig(serialized_ir, isa=isa),
            lambda r: (r[0], {"resources": r[3]}),
            lambda e: (e["hsaco"], CO.CodeObjectTimings(), CO._direct_rocke_compiler_info(CO._require_direct_library()),
                       (e.get("extra") or {}).get("resources", "")),
        )  # fmt: skip

    return build_direct_rocke_ir_hsaco


def install(directory: str | None = None, mode: str | None = None) -> bool:
    """Route rocKE's COMGR and Rust direct compiles through the cache (``directory`` or ``ROCKE_W16_COCACHE``).
    Idempotent; no directory: nothing is patched."""
    d = directory or os.environ.get(ENV)
    if not d:
        return False
    _STATE["dir"], _STATE["mode"] = d, mode
    _mode()
    if _STATE["orig"]:
        return True
    from rocke.helpers import compile as HC
    from rocke.runtime import code_object as CO

    _STATE["orig"]["comgr"] = HC.build_hsaco_from_llvm_ir
    HC.build_hsaco_from_llvm_ir = _wrap_comgr(HC.build_hsaco_from_llvm_ir)
    _STATE["orig"]["direct"] = CO.build_direct_rocke_ir_hsaco
    CO.build_direct_rocke_ir_hsaco = _wrap_direct(CO.build_direct_rocke_ir_hsaco)
    os.makedirs(d, exist_ok=True)
    return True


# ------------------------------------------------------------------ generation cache
def _rocke_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # .../library/kernels/gfx950/dsv41_mono
    return os.path.normpath(os.path.join(here, "..", "..", "..", ".."))


def source_digest(atom_root: str | None = None) -> str:
    """sha256 over every Python source the stage generators can import (rocKE platform + the two gfx950 families),
    plus ATOM's V4.1 tree when the generator reads ATOM's constants."""
    root = _rocke_root()
    roots = [os.path.join(root, "platform", "python", "rocke"),
             os.path.join(root, "library", "kernels", "gfx950", "dsv41_mono"),
             os.path.join(root, "library", "kernels", "gfx950", "qwen38_mono"),
             os.path.join(root, "library", "kernels", "__init__.py"),
             os.path.join(root, "library", "kernels", "gfx950", "__init__.py")]  # fmt: skip
    if atom_root:
        roots.append(os.path.join(atom_root, "atom", "models", "deepseek_v41"))
    h = hashlib.sha256()
    for r in roots:
        files = [r] if os.path.isfile(r) else sorted(
            os.path.join(dp, f) for dp, dn, fs in os.walk(r) for f in fs
            if f.endswith(".py") and "__pycache__" not in dp)  # fmt: skip
        for f in files:
            rel = os.path.relpath(f, os.path.dirname(r)).replace("\\", "/")
            data = open(f, "rb").read().replace(b"\r\n", b"\n")
            h.update(f"{rel}\0{_sha(data)}\n".encode())
    return h.hexdigest()


def _gen_key(module: str, args: list, atom_root: str | None) -> tuple[str, dict]:
    env = sorted((k, v) for k, v in os.environ.items()
                 if k.startswith("ROCKE_") and not k.startswith("ROCKE_W16_") and k not in ("ROCKE_COMGR_LIB",))  # fmt: skip
    parts = {"v": VERSION, "module": module, "args": args, "env": env, "src": source_digest(atom_root)}
    return _sha(json.dumps(parts, sort_keys=True)), parts


def _tar_dir(d: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                ti = tarfile.TarInfo(f)
                data = open(p, "rb").read()
                ti.size, ti.mtime = len(data), 0
                tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _files_of_tar(blob: bytes) -> dict:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            out[m.name] = tf.extractfile(m).read()
    return out


def gen(cache: str, out: str, module: str, args: list, mode: str = "hit") -> dict:
    """Generate one stage-function directory through the cache (see the module docstring)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    atom_root = None
    if "--atom-root" in args:
        atom_root = args[args.index("--atom-root") + 1]
    k, parts = _gen_key(module, args, atom_root)
    blob_path = os.path.join(cache, "gen", k[:2], k + ".tar")
    rec = {"kind": "gen", "key": k, "module": module, "args": args, "mode": mode, "out": out}
    os.makedirs(out, exist_ok=True)
    have = os.path.isfile(blob_path)
    if have and mode in ("hit", "ro"):
        t0 = time.perf_counter()
        files = _files_of_tar(open(blob_path, "rb").read())
        for name, data in files.items():
            _atomic_write(os.path.join(out, name), data)
        rec.update(event="hit", files=len(files), load_s=round(time.perf_counter() - t0, 3))
        _log(rec)
        return rec
    if mode == "ro":
        rec.update(event="miss_ro")
        _log(rec)
        return rec
    t0 = time.perf_counter()
    # a fresh work dir under the cache, kept (nothing here deletes files)
    os.makedirs(os.path.join(cache, "work"), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"gen-{k[:12]}-", dir=os.path.join(cache, "work"))
    cmd = [sys.executable, "-m", module, "--out", tmp, *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        rec.update(event="gen_fail", rc=p.returncode, stderr=p.stderr[-2000:])
        _log(rec)
        raise RuntimeError(f"generator failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    blob = _tar_dir(tmp)
    dt = time.perf_counter() - t0
    files = _files_of_tar(blob)
    rec.update(compile_s=round(dt, 3), files=len(files), tar_sha256=_sha(blob))
    if have:  # verify: regenerated files against the stored ones
        stored = _files_of_tar(open(blob_path, "rb").read())
        same = stored.keys() == files.keys() and all(stored[n] == files[n] for n in files)
        rec.update(event="verify_SAME" if same else "verify_DIFF",
                   differ=sorted(n for n in files if stored.get(n) != files[n])[:20])  # fmt: skip
        if not same and os.environ.get(STRICT_ENV, "0") == "1":
            raise RuntimeError(f"regenerated {module} differs from entry {k}")
    else:
        _atomic_write(blob_path, blob)
        _atomic_write(blob_path[:-4] + ".json", json.dumps({**parts, "files": len(files), "gen_s": round(dt, 3),
                                                            "host": socket.gethostname()}, indent=1).encode())  # fmt: skip
        rec["event"] = "miss_store"
    for name, data in files.items():
        _atomic_write(os.path.join(out, name), data)
    _log(rec)
    return rec


def main(argv=None) -> int:
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "gen":
        print(__doc__)
        return 2
    if "--" not in argv:
        raise SystemExit("gen: expected '--' before the generator module")
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="w16_cocache gen")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="hit", choices=MODES)
    a = ap.parse_args(argv[1:i])
    rest = argv[i + 1:]
    rec = gen(a.cache, a.out, rest[0], rest[1:], a.mode)
    print("GEN", json.dumps(rec, sort_keys=True))
    return 0 if rec["event"] != "gen_fail" else 1


if __name__ == "__main__":
    sys.exit(main())
