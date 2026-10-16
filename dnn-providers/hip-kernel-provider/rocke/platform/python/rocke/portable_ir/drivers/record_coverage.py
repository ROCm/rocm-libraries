#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# record_coverage.py -- prove the RecordingIRBuilder records the WHOLE production
# kernel surface, not just a handful (scaling-plan P1: "record is universal").
#
# It reuses the spec values already encoded in the byte-identity parity emitters
# (platform/tests/instances/parity/*_emit.py): for each, it pulls `_spec(0)` and the
# production `build_*` function, records the build via record_kernel(), and
# asserts the live recording matches an independent post-hoc walk of the same
# KernelDef. Pure Python -- no device / comgr.
#
# Per kernel: PASS (recorded faithfully), SKIP (emitter not in the reusable
# spec+build shape), or FAIL (recorder gap -> needs attention).
#
#   python3 -m rocke.portable_ir.drivers.record_coverage [--verbose]

import argparse
import importlib.util
import inspect
import os
import sys
import traceback

from rocke.portable_ir.src.recording_builder import kernel_to_recipe, record_kernel


def _parity_dirs():
    """The parity-emitter trees, in scan order.

    rocke splits them across the two source roots: the platform emitters live in
    platform/tests/instances/parity, while the whole attention family lives in
    the library tree (library/tests/parity). The library dir is optional so a
    platform-only checkout still runs -- it just covers fewer kernels.
    """
    here = os.path.dirname(__file__)
    platform = os.path.normpath(
        os.path.join(here, "..", "..", "..", "..", "tests", "instances", "parity")
    )
    library = os.path.normpath(
        os.path.join(here, "..", "..", "..", "..", "..", "library", "tests", "parity")
    )
    return [d for d in (platform, library) if os.path.isdir(d)]


_PARITY_DIRS = _parity_dirs()


def _load_module(path):
    name = "rocke_parity_" + os.path.splitext(os.path.basename(path))[0]
    # Parity emitters import their shared helper as `from _emit_common import ...`
    # so the parity dirs must be importable (mirrors export_parity.py).
    for d in _PARITY_DIRS:
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_build_fn(mod):
    """The production builder a parity emitter imported (build_* defined in
    rocke.instances for platform kernels, or kernels.* for library kernels)."""
    cands = []
    for k, v in vars(mod).items():
        if not callable(v) or not k.startswith("build"):
            continue
        origin = getattr(v, "__module__", "")
        if origin.startswith("rocke.instances") or origin.startswith("kernels."):
            cands.append((k, v))
    # Prefer a unique candidate; otherwise the shortest name (the top-level entry).
    if not cands:
        return None
    cands.sort(key=lambda kv: len(kv[0]))
    return cands[0][1]


def _count(prog):
    n = 0
    for i in prog:
        if i.get("op") in ("emit", "scf_for", "scf_if", "ret"):
            n += 1
        for key in ("body", "then", "else"):
            if key in i:
                n += _count(i[key])
    return n


def _kernel_thunks(mod):
    """Yield zero-arg callables that each build one production KernelDef, across
    the emitter shapes seen in the parity tree: a direct ``_kernel(idx)`` factory;
    ``_spec(idx)`` returning a spec or a ``(spec, arch)`` tuple; or ``_specs()``
    returning a list. Configs are scanned (some idx 0 are intentionally invalid)."""
    # Direct kernel factory (e.g. attention_unified, tiled attention emitters).
    for fac in ("_kernel", "_make_kernel", "_build_kernel", "_build"):
        fn = getattr(mod, fac, None)
        if callable(fn):
            for i in range(12):
                yield (lambda i=i, fn=fn: fn(i))
            return

    build_fn = _find_build_fn(mod)
    if build_fn is None:
        return
    wants_arch = "arch" in inspect.signature(build_fn).parameters

    # Arch candidates, label-inferred first (some kernels are arch-restricted,
    # e.g. WMMA -> gfx1151/gfx1201, matmul_nbits -> gfx1151/gfx1201).
    label = getattr(mod, "__name__", "").replace("rocke_parity_", "")
    arch_cands = []
    for a in ("gfx1151", "gfx1201", "gfx942", "gfx950"):
        if label.startswith(a):
            arch_cands.append(a)
    for a in ("gfx950", "gfx942", "gfx1151", "gfx1201"):
        if a not in arch_cands:
            arch_cands.append(a)

    def specs():
        if hasattr(mod, "_spec"):
            for i in range(12):
                try:
                    yield mod._spec(i)
                except SystemExit:
                    return
                except Exception:  # noqa: BLE001
                    return
        elif hasattr(mod, "_specs"):
            try:
                for s in mod._specs():
                    yield s
            except Exception:  # noqa: BLE001
                return

    for s in specs():
        if isinstance(s, tuple):
            # Heterogeneous (spec, arch, ...) tuples: try the call forms seen in
            # the parity tree until one binds.
            strs = [x for x in s if isinstance(x, str)]
            yield (lambda s=s: build_fn(*s))
            if wants_arch and strs:
                yield (lambda s=s, a=strs[-1]: build_fn(s[0], arch=a))
            yield (lambda s=s: build_fn(s[0]))
            continue
        if wants_arch:
            for a in arch_cands:
                yield (lambda spec=s, a=a: build_fn(spec, arch=a))
        else:
            yield (lambda spec=s: build_fn(spec))


def _record_one(mod):
    """-> (status, detail). status in {PASS, SKIP, FAIL}."""
    thunks = list(_kernel_thunks(mod))
    if not thunks:
        return "SKIP", "no recognized spec/build entry"
    last = None
    for thunk in thunks:
        try:
            kernel, recorded = record_kernel(thunk)
        except NotImplementedError as e:
            return "FAIL", f"recorder gap: {e}"
        except SystemExit as e:
            last = f"SystemExit: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - a bad config, try the next one
            last = f"{type(e).__name__}: {e}"
            continue
        if recorded != kernel_to_recipe(kernel):
            return "FAIL", "live recording != post-hoc walk"
        nops = _count([i for i in recorded["program"] if i.get("op") != "param"])
        return "PASS", f"{nops} ops"
    return "SKIP", f"no buildable config (last: {last})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    paths = sorted(
        os.path.join(d, f)
        for d in _PARITY_DIRS
        for f in os.listdir(d)
        if f.endswith("_emit.py")
    )
    results = []
    for path in paths:
        label = os.path.basename(path)[: -len("_emit.py")]
        try:
            mod = _load_module(path)
        except Exception as e:  # noqa: BLE001
            results.append((label, "SKIP", f"import failed: {e}"))
            continue
        try:
            status, detail = _record_one(mod)
        except Exception:  # noqa: BLE001
            status, detail = (
                "FAIL",
                "unexpected: " + traceback.format_exc().splitlines()[-1],
            )
        results.append((label, status, detail))

    npass = sum(1 for _, s, _ in results if s == "PASS")
    nskip = sum(1 for _, s, _ in results if s == "SKIP")
    nfail = sum(1 for _, s, _ in results if s == "FAIL")

    for label, status, detail in results:
        if status == "PASS" and not args.verbose:
            continue
        print(f"  [{status}] {label:<38} {detail}")
    print("-" * 70)
    print(
        f"recorded OK: {npass}   skipped: {nskip}   FAILED: {nfail}   "
        f"(of {len(results)} parity emitters)"
    )
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
