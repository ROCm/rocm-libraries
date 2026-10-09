# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rebuild an AOT convolution kernel from the name a benchmark printed.

Every name in a ``--run-from-cache`` table (or a ``--compile-all`` log) is a
lossless :meth:`~benchmarks.common.kernel_cache.KernelIdentity.label`, so it
fixes the kernel completely: direction, algorithm, dtypes, 2-D/3-D, tile,
pipeline, capabilities, tuning knobs, arch and LLVM flavor. This tool turns
it back into the identity, rebuilds the job ``--compile-all`` would build for
it, lowers it and (unless ``--ir-only``) compiles it::

    python -m benchmarks.common.reproduce_kernel \\
        fwd-implicit_gemm-bf16-2d-t256x256x64-w8x2-a32x32x16-v1x1x1-compv4-cshuffle-grp-unroll-wave64-gfx950-llvm22 \\
        --cache ./kernel_cache/

and writes ``<name>.ll``, ``<name>.hsaco`` and ``<name>.json`` (identity, spec
kwargs, content key, provenance) to ``--out-dir`` -- a name too long for a file
name is shortened to a prefix plus a hash (:func:`artifact_stem`), the full
name kept in the ``.json``. Every implicit-GEMM
(fwd/wgrad/dgrad) and direct-conv (fwd/dgrad, and the MFMA dgrad's weight
transforms) kernel is covered.

The name fixes the *configuration*. The binary is also a function of the
emitter sources and the COMGR that compiled it, which no name carries: with
``--cache`` the rebuilt binary is compared with the cached one, and a
difference is reported together with the emitter digest and COMGR each was
built with -- a cache entry from older sources needs that checkout to be
reproduced bit for bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from benchmarks.common.kernel_cache import (
    KernelCache,
    KernelIdentity,
    comgr_input_key,
    current_comgr_id,
    current_emitter_digest,
    current_llvm_flavor,
)
from benchmarks.common.kernel_label import strip_launch_suffix

# Longest artifact stem written to --out-dir: a file name is capped at 255
# bytes (NAME_MAX) and a lossless label can exceed it, so a longer name is cut
# and suffixed with a hash of the whole name; the full name stays in the .json
# record.
_MAX_STEM = 255 - len(".hsaco")
_STEM_HASH_LEN = 16


def artifact_stem(name: str) -> str:
    """File stem for ``name``'s artifacts: ``name`` itself when it fits in
    :data:`_MAX_STEM` bytes, else a readable prefix plus a stable hash."""
    if len(name.encode("utf-8")) <= _MAX_STEM:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_STEM_HASH_LEN]
    prefix = name.encode("utf-8")[: _MAX_STEM - _STEM_HASH_LEN - 1]
    return f"{prefix.decode('utf-8', 'ignore')}-{digest}"


def job_and_builder(identity: KernelIdentity):
    """``(job, build)``: the job ``--compile-all`` builds ``identity`` from
    and the family's ``build(job, arch, dtype)``."""
    if identity.is_direct:
        from benchmarks.common.direct_kernel_sweep import (
            build_direct_job,
            direct_job_for_identity,
        )

        return direct_job_for_identity(identity), build_direct_job
    from benchmarks.common.kernel_sweep import build_kernel, job_for_identity

    return job_for_identity(identity), build_kernel


def reproduce(
    name: str,
    *,
    out_dir: Optional[Path],
    compile_hsaco: bool = True,
    cache: Optional[Path] = None,
    log=print,
) -> bool:
    """Rebuild the kernel ``name``; ``True`` when it was rebuilt (and, with a
    ``cache``, matches the cached binary)."""
    from rocke.helpers.compile import lower_kernel_for_comgr

    name = strip_launch_suffix(name)
    log(f"\n{name}")
    try:
        identity = KernelIdentity.from_label(name)
    except ValueError as e:
        log(f"  [error] {e}")
        return False
    flavor = current_llvm_flavor()
    if identity.llvm_flavor != flavor:
        log(
            f"  [error] built for LLVM flavor {identity.llvm_flavor!r}, this "
            f"process lowers {flavor!r}: rerun with "
            f"ROCKE_LLVM_FLAVOR={identity.llvm_flavor}"
        )
        return False
    try:
        job, build = job_and_builder(identity)
    except LookupError as e:
        log(f"  [error] {e}")
        return False

    arch, dtype = identity.arch, identity.dtype_a
    try:
        kernel, meta = build(job, arch, dtype)
        comgr_input = lower_kernel_for_comgr(kernel, arch=arch)
    except Exception as e:  # noqa: BLE001 - reported per kernel
        log(f"  [error] build failed: {type(e).__name__}: {e}")
        return False
    key = comgr_input_key(comgr_input)
    log(f"  kernel symbol   {comgr_input.kernel_name}")
    log(f"  spec kwargs     {json.dumps(job.spec_kwargs, sort_keys=True)}")
    log(f"  capabilities    {json.dumps(job.caps, sort_keys=True)}")
    log(f"  content key     {key}")

    hsaco = None
    if compile_hsaco:
        from rocke.runtime.comgr import build_hsaco_from_llvm_ir

        try:
            hsaco, _ = build_hsaco_from_llvm_ir(
                comgr_input.llvm_text,
                isa=comgr_input.isa,
                options=list(comgr_input.options),
            )
        except Exception as e:  # noqa: BLE001 - reported per kernel
            log(f"  [error] compile failed: {type(e).__name__}: {e}")
            return False
        log(f"  hsaco           {len(hsaco)} bytes")

    ok = True
    if cache is not None:
        entry = KernelCache(cache, arch).get(identity)
        if entry is None:
            log(f"  cache           no entry for this kernel in {cache}")
        else:
            cached, cached_meta = entry
            same_key = cached_meta.get("blob") == key
            # Identities whose code differs only in its name share one binary
            # (comgr_input_key normalises the name out), and that binary
            # carries the symbol of whichever compiled it first: its bytes
            # only match a rebuild under the same symbol.
            shared_with = cached_meta.get("kernel_name")
            if shared_with == comgr_input.kernel_name:
                shared_with = None
            same_bytes = hsaco is None or shared_with is not None or hsaco == cached
            if same_key and same_bytes:
                log("  cache           IDENTICAL to the cached binary")
                if shared_with is not None:
                    log(
                        f"                  (same code; the cached binary is "
                        f"shared and carries the symbol {shared_with})"
                    )
            else:
                ok = False
                log("  cache           DIFFERS from the cached binary")
                for what, cached_v, now_v in (
                    (
                        "emitter",
                        cached_meta.get("emitter_digest"),
                        current_emitter_digest(),
                    ),
                    ("comgr", cached_meta.get("comgr_id"), current_comgr_id()),
                ):
                    note = "same" if cached_v == now_v else "changed"
                    log(f"    {what:8s} cached {cached_v}  now {now_v}  ({note})")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = artifact_stem(name)
        (out_dir / f"{stem}.ll").write_text(comgr_input.llvm_text, encoding="utf-8")
        if hsaco is not None:
            (out_dir / f"{stem}.hsaco").write_bytes(hsaco)
        record = dict(
            name=name,
            identity=identity.to_dict(),
            spec_kwargs=job.spec_kwargs,
            caps=job.caps,
            kernel_name=comgr_input.kernel_name,
            isa=comgr_input.isa,
            options=list(comgr_input.options),
            content_key=key,
            emitter_digest=current_emitter_digest(),
            comgr_id=current_comgr_id(),
            **{k: v for k, v in meta.items() if k not in ("kernel_name",)},
        )
        (out_dir / f"{stem}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        log(
            f"  written         {out_dir / stem}.{{ll,json{',hsaco' if hsaco else ''}}}"
        )
    return ok


def _names(args) -> List[str]:
    names = list(args.names)
    for path in args.from_file or ():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild AOT conv kernels from the names benchmarks print."
    )
    parser.add_argument("names", nargs="*", help="kernel names (KernelIdentity.label)")
    parser.add_argument(
        "--from-file",
        action="append",
        metavar="FILE",
        help="read names from FILE, one per line ('#' comments allowed)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced_kernels"),
        help="where to write <name>.ll/.hsaco/.json (default: %(default)s)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="rebuild and report, write nothing"
    )
    parser.add_argument(
        "--ir-only", action="store_true", help="lower to LLVM IR, do not compile"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        metavar="DIR",
        help="compare each rebuilt binary with its entry in this AOT cache",
    )
    args = parser.parse_args(argv)
    names = _names(args)
    if not names:
        parser.error("no kernel names given")

    results: List[Tuple[str, bool]] = []
    for name in names:
        ok = reproduce(
            name,
            out_dir=None if args.no_write else args.out_dir,
            compile_hsaco=not args.ir_only,
            cache=args.cache,
        )
        results.append((name, ok))
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} reproduced", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
