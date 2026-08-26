#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared scaffolding for the block-scale quant GEMM dispatcher bridges.

The five ``gemm_<op>_utils.py`` bridges (aquant / abquant / bquant /
rowcolquant / tensor_quant) were ~85% mechanically duplicated.  This module
folds the genuinely-shared, mechanical parts into one place so they cannot
drift, while every op-specific / correctness-load-bearing part stays in its own
file.  It builds on :mod:`dispatcher_common` (path + tool helpers) rather than
re-deriving them.

What lives here (verified byte-identical across the copies it replaces):

  * :func:`find_ck_include_dir` -- the ``_get_ck_include_dir`` include-probe.
  * :func:`install_dispatcher_lib_api` -- the ctypes ``_setup`` scaffold for the
    ``dispatcher_initialize`` / ``get_kernel_name`` / ``get_kernel_count`` /
    ``cleanup`` symbols, plus the per-op ``dispatcher_run_<op>_gemm`` argtypes
    driven from an ARGSPEC the caller passes in.
  * :func:`generate_kernel` -- the codegen subprocess (``_generate_<op>_kernel``).
  * :func:`build_dispatchers` -- the dedupe-by-name + ThreadPoolExecutor +
    "fill duplicates" orchestration (``setup_multiple_<op>_dispatchers``),
    parameterized by a per-op ``compile_fn`` (each op keeps its own hipcc flag /
    arch-define / static-lib / timeout choices, which genuinely diverge).

What deliberately does NOT live here (kept per-op -- see the report / each file):

  * ``_detect_gpu_arch`` -- five distinct implementations (different supported-arch
    sets, messages, and validation; tensor_quant's even differs subtly).
  * the ``_compile_<op>_kernel`` flag/define/timeout/static-lib bodies.
  * the fp8/bf8 encode helpers -- the surrounding tensor packing differs per op.
    Every op agrees on the codec itself: OCP fp8 is ``float8_e4m3fn`` and OCP bf8
    is ``float8_e5m2``; FNUZ is ``float8_e4m3fnuz`` / ``float8_e5m2fnuz``.
  * ``default_*_config`` / ``KernelConfig`` / ``run()``.

No GPU / hipcc is required to import or exercise the codegen path here.
"""

import concurrent.futures
import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from quant_bridge_flags import flags_cache_tag  # noqa: E402


# ============================================================================
# Include-directory probe (canonical _get_ck_include_dir)
# ============================================================================


def find_ck_include_dir() -> Optional[Path]:
    """Locate the CK include directory relative to this file (or None).

    Byte-for-byte the ``_get_ck_include_dir`` every bridge carried: walk up from
    ``dispatcher/python/`` and return the first ancestor ``include/`` that
    contains ``ck_tile/``.
    """
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


# ============================================================================
# ctypes DispatcherLib scaffold
# ============================================================================


def install_dispatcher_lib_api(
    lib: "ctypes.CDLL",
    run_symbol: str,
    run_argtypes: Sequence["object"],
) -> None:
    """Wire the restype/argtypes on a loaded quant-bridge ``.so``.

    Every bridge's ``DispatcherLib._setup`` declared the identical
    ``dispatcher_initialize`` / ``dispatcher_get_kernel_name`` /
    ``dispatcher_get_kernel_count`` / ``dispatcher_cleanup`` signatures; only the
    ``dispatcher_run_<op>_gemm`` argtypes differed per op.  Pass that op's
    ``run_symbol`` name and ``run_argtypes`` list (its ARGSPEC) and this installs
    all of them exactly as the copies did.
    """
    lib.dispatcher_initialize.restype = ctypes.c_int
    lib.dispatcher_initialize.argtypes = []

    run_fn = getattr(lib, run_symbol)
    run_fn.restype = ctypes.c_int
    run_fn.argtypes = list(run_argtypes)

    lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p
    lib.dispatcher_get_kernel_name.argtypes = []

    lib.dispatcher_get_kernel_count.restype = ctypes.c_int
    lib.dispatcher_get_kernel_count.argtypes = []

    lib.dispatcher_cleanup.restype = None
    lib.dispatcher_cleanup.argtypes = []

    # Optional: older .so builds predate the timing knobs.
    set_timing = getattr(lib, "dispatcher_set_timing_config", None)
    if set_timing is not None:
        set_timing.restype = ctypes.c_int
        set_timing.argtypes = [ctypes.c_int] * 4


class DispatcherLibBase:
    """Common ctypes wrapper for a compiled quant-bridge ``.so``.

    Subclasses declare two class attributes and (optionally) their own ``run``:

      * ``_NOT_FOUND_LABEL`` -- op label used in the FileNotFoundError message
        (e.g. ``"AQuant"``), preserving each bridge's original wording.
      * ``_RUN_SYMBOL`` / ``_RUN_ARGTYPES`` -- the ``dispatcher_run_<op>_gemm``
        symbol name and its ctypes argtypes list (the per-op ARGSPEC).

    ``__init__`` / ``get_kernel_name`` / ``get_kernel_count`` / ``cleanup`` /
    ``__del__`` were byte-identical across the five bridges (modulo the op label),
    so they live here once.
    """

    _NOT_FOUND_LABEL: str = "quant"
    _RUN_SYMBOL: str = ""
    _RUN_ARGTYPES: Sequence["object"] = ()

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(
                f"{self._NOT_FOUND_LABEL} .so not found: {self.so_path}"
            )
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        install_dispatcher_lib_api(
            self._lib, self._RUN_SYMBOL, self._RUN_ARGTYPES
        )

    def get_kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else ""

    def get_kernel_count(self) -> int:
        return self._lib.dispatcher_get_kernel_count()

    def set_timing_config(
        self,
        flush_cache: Optional[bool] = None,
        rotating_count: Optional[int] = None,
        cold_niters: Optional[int] = None,
        nrepeat: Optional[int] = None,
    ) -> int:
        """Configure the measured launch: warmup iterations and repeat count.

        The bridge is benchmarked against Old-TE's ``gemm_quant``, which defaults
        to ``cold_niters``/``nrepeat`` the caller can pick and to
        ``flush_cache=true`` / ``rotating_count=1000``.  The bridge hardcoded all
        four and nothing on the Python side could change any of them.
        ``cold_niters`` and ``nrepeat`` are now settable; ``None`` leaves a field
        at its current value.  The environment variables ``CK_BRIDGE_COLD_NITERS``
        and ``CK_BRIDGE_NREPEAT`` set the same fields at first use.

        ``flush_cache`` and ``rotating_count`` are **refused**, not stored.  The
        generated ``launch()`` goes through ``ck_tile::launch_kernel``, which
        ignores ``stream_config::flush_cache_`` and ``rotating_count_`` entirely
        -- Old-TE implements the rotating-buffer flush in its own invoker.
        Accepting them would make the bridge report a cache-flushed measurement
        it never performed, which is worse than not offering the knob.  So this
        remains a **disclosed asymmetry** against the Old-TE baseline: closing it
        needs a flush-cache launch overload in the generated header, not Python
        plumbing.

        Returns the library status: 0 on success; -1 if the loaded ``.so``
        predates the knobs; -2 if ``flush_cache``/``rotating_count`` were
        requested, in which case **nothing** is applied -- call again with only
        the supported fields.
        """
        fn = getattr(self._lib, "dispatcher_set_timing_config", None)
        if fn is None:
            return -1
        return int(fn(
            -1 if flush_cache is None else int(bool(flush_cache)),
            -1 if rotating_count is None else int(rotating_count),
            -1 if cold_niters is None else int(cold_niters),
            -1 if nrepeat is None else int(nrepeat),
        ))

    def cleanup(self):
        self._lib.dispatcher_cleanup()

    def __del__(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:
            pass


# ============================================================================
# Codegen subprocess (canonical _generate_<op>_kernel)
# ============================================================================


def generate_kernel(
    config,
    output_dir: Path,
    codegen_script: Path,
    timeout: int = 120,
) -> Optional[Path]:
    """Run a unified ``*_codegen.py`` for one config; return the ``.hpp`` or None.

    Identical to every bridge's ``_generate_<op>_kernel``: serialize
    ``config.to_codegen_config()`` to JSON, invoke the op's codegen script, and
    return ``<output_dir>/<config.name>.hpp`` if it materialized.
    """
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    cmd = [
        sys.executable,
        str(codegen_script),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.error("Codegen failed for %s:\n%s", config.name, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("Codegen timed out for %s", config.name)
        return None

    hpp = output_dir / f"{config.name}.hpp"
    if not hpp.exists():
        log.error("Codegen succeeded but %s not found", hpp)
        return None

    return hpp


# ============================================================================
# Build orchestration (canonical setup_multiple_<op>_dispatchers)
# ============================================================================


def build_dispatchers(
    configs: List,
    arch: str,
    tmp_prefix: str,
    log_label: str,
    generate_fn: Callable[[object, Path], Optional[Path]],
    compile_fn: Callable[[Path, Path, str], bool],
    output_dir: Optional[Path] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    hipcc: str = "hipcc",
    flags_tag: Optional[str] = None,
) -> List[Optional[Path]]:
    """codegen -> compile -> ``.so`` for each config, deduped by name, in parallel.

    This is the ``setup_multiple_<op>_dispatchers`` body every bridge copied
    verbatim -- dedupe-by-name, ThreadPoolExecutor fan-out, ``[cached]`` skip,
    and the "fill duplicates" pass -- lifted out once.  The op-specific pieces
    are injected:

      * ``arch``           already resolved by the caller (each op has its own
                           ``_detect_gpu_arch`` / ``_validate_arch`` policy and
                           any early guards, e.g. bquant's MX-arch check).
      * ``tmp_prefix``     ``tempfile.mkdtemp`` prefix (e.g. ``"aquant_dispatcher_"``).
      * ``log_label``      human label in the log lines (e.g. ``"AQuant"``).
      * ``generate_fn``    ``lambda cfg, hdr_dir -> Optional[Path]`` (the op's
                           ``_generate_<op>_kernel``).
      * ``compile_fn``     ``lambda hpp, so, arch -> bool`` (the op's
                           ``_compile_<op>_kernel``; keeps its flags/defines).

    Returns a list parallel to ``configs`` (Path or None per entry).  No GPU is
    required to call this.
    """
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix=tmp_prefix))
    base_dir.mkdir(parents=True, exist_ok=True)

    # The .so cache key must include the compile flags, not just name + arch:
    # otherwise CK_BRIDGE_NO_TE_FLAGS (or a toolchain whose coerce probe answers
    # differently) silently reuses a .so built with the other flag set.
    if flags_tag is None:
        flags_tag = flags_cache_tag(hipcc)

    headers_dir = base_dir / "generated_kernels"
    so_dir = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info(
        "Building %d %s kernel(s) for %s into %s",
        len(configs), log_label, arch, base_dir,
    )

    # Deduplicate by name so we don't build the same kernel twice.
    seen: Dict[str, int] = {}          # name -> index of first occurrence
    deduped: List[Tuple[int, object]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg) -> Tuple[int, Optional[Path]]:
        hpp = generate_fn(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}_{flags_tag}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = compile_fn(hpp, so, arch)
        return idx, so if ok else None

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_one, idx, cfg): (idx, cfg) for idx, cfg in deduped}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, so_path = fut.result()
                    results[idx] = so_path
                    if so_path:
                        log.info("  built %s", so_path.name)
                    else:
                        _, cfg = futures[fut]
                        log.error("  FAILED %s", cfg.name)
                except Exception as e:
                    _, cfg = futures[fut]
                    log.error("  EXCEPTION for %s: %s", cfg.name, e)
    else:
        for idx, cfg in deduped:
            _, so_path = _build_one(idx, cfg)
            results[idx] = so_path

    # Fill in duplicates.
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d %s kernels", built, len(configs), log_label)
    return results


# ============================================================================
# QDataType / ADataType host-buffer contract (shared by both bquant bridges)
# ============================================================================
#
# The ctypes libs reinterpret the raw BQ and A bytes as the kernel's
# compile-time QDataType / ADataType.  Hand them the wrong element width and
# the .so reads a different number of bytes than the caller allocated: a
# float32 BQ fed to an fp8i4 kernel (QDataType == fp8_t, 1 byte) is read 4x
# short and the result is NaN, silently.
#
# The non-grouped gemm_bquant bridge learned this and encoded here; the grouped
# twin did not, and shipped two default configs (fp8i4 / bf8i4) that returned a
# 3.5%-non-finite C on gfx950.  One implementation, both bridges.

# variant_key -> tag for the QDataType the kernel was compiled with.
# "float32" means "no re-encode".
BQ_QDTYPE_BY_VARIANT: Dict[str, str] = {
    "fp8":         "float32",
    "bf8":         "float32",
    "fp8i4":       "fp8",
    "bf8i4":       "bf8",
    "mx_bf16bf16": "e8m0",
    "mx_bf16bf8":  "e8m0",
    "mx_bf16fp4":  "e8m0",
}

# variant_key -> numpy element size (bytes) of the kernel's ADataType.
ADTYPE_ELEMSIZE_BY_VARIANT: Dict[str, int] = {
    "fp8":         1,
    "bf8":         1,
    "fp8i4":       1,
    "bf8i4":       1,
    "mx_bf16bf16": 2,
    "mx_bf16bf8":  2,
    "mx_bf16fp4":  2,
}


def variant_from_kernel_name(name: str, name_prefix: str) -> Optional[str]:
    """Extract the variant_key from a KERNEL_NAME of the form ``<prefix>_<variant>_...``.

    Longest-token-first, so ``mx_bf16bf16`` is matched before any shorter
    prefix would swallow it.
    """
    if not name or not name.startswith(name_prefix + "_"):
        return None
    rest = name[len(name_prefix) + 1:]
    for v in sorted(BQ_QDTYPE_BY_VARIANT, key=len, reverse=True):
        if rest.startswith(v + "_"):
            return v
    return None


def uses_ocp_fp8(gfx_arch: Optional[str]) -> bool:
    """True when the kernel is compiled with OCP fp8 (not FNUZ) for ``gfx_arch``.

    gfx950 and gfx12* build with -DCK_TILE_USE_OCP_FP8 so ``ck_tile::fp8_t`` is
    OCP e4m3 / e5m2; every other arch (notably gfx942 / gfx90a) falls back to
    the FNUZ encodings.  Encoding host bytes in the wrong flavour makes gfx942
    read NaN.  Unknown arch -> assume OCP, the historical gfx950 default.
    """
    if not gfx_arch:
        return True
    return ("gfx950" in gfx_arch) or ("gfx12" in gfx_arch)


def ml_fp8_dtype(dtype: str, gfx_arch: Optional[str]):
    """The ml_dtypes fp8 type matching this arch's ``ck_tile::fp8_t`` / ``bf8_t``."""
    import ml_dtypes
    if uses_ocp_fp8(gfx_arch):
        return ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
    return ml_dtypes.float8_e4m3fnuz if dtype == "fp8" else ml_dtypes.float8_e5m2fnuz


def encode_fp8_bytes(arr, dtype: str, gfx_arch: Optional[str] = None):
    """float32 -> fp8/bf8 raw bytes (uint8), arch-aware (FNUZ on gfx942)."""
    import numpy as np
    a = np.asarray(arr, dtype=np.float32)
    try:
        return a.astype(ml_fp8_dtype(dtype, gfx_arch)).view(np.uint8)
    except ImportError:
        # Deterministic fallback so CPU-only unit tests without ml_dtypes still
        # exercise the byte-width contract.
        return (np.clip(a, -2.0, 2.0) * 64).astype(np.int8).view(np.uint8)


def encode_e8m0(arr):
    """float32 scale -> e8m0 uint8 (block-scale exponent; byte b == 2^(b-127))."""
    import numpy as np
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, np.float32(2.0 ** 127))
    out = np.zeros(a.shape, dtype=np.uint8)
    nonzero = a > 0.0
    out[nonzero] = np.clip(
        np.floor(np.log2(a[nonzero])).astype(np.int32) + 127, 0, 254
    ).astype(np.uint8)
    return out


def encode_bq_for_variant(BQ, variant_key: Optional[str],
                          gfx_arch: Optional[str] = None):
    """Return BQ in the kernel's QDataType bytes for ``variant_key``.

    fp8/bf8 -> float32 (unchanged); fp8i4 -> fp8 bytes, bf8i4 -> bf8 bytes
    (arch-aware); mx_* -> e8m0 uint8.  Already-encoded 1-byte input is passed
    through so a caller that pre-encoded is never double-encoded.  A None or
    unknown variant is passed through unchanged.
    """
    import numpy as np
    if variant_key is None:
        return BQ
    tag = BQ_QDTYPE_BY_VARIANT.get(variant_key, "float32")
    arr = np.asarray(BQ)
    if tag == "float32":
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    if arr.dtype == np.uint8 or arr.dtype == np.int8:
        return arr
    if tag == "e8m0":
        return encode_e8m0(arr)
    if tag in ("fp8", "bf8"):
        return encode_fp8_bytes(arr, tag, gfx_arch=gfx_arch)
    return arr


def coerce_a_for_variant(A, variant_key: Optional[str]):
    """Return A whose element byte-width matches the kernel's ADataType.

    The ctypes lib copies ``M*K * sizeof(ADataType)`` bytes from the host A
    pointer, so A must be exactly that wide or the copy reads out of bounds --
    harmless slack at small M*K, a host-pin failure and SEGFAULT at large M*K.
    """
    import numpy as np
    if variant_key is None:
        return A
    want = ADTYPE_ELEMSIZE_BY_VARIANT.get(variant_key)
    if want is None:
        return A
    arr = np.ascontiguousarray(A)
    have = arr.dtype.itemsize
    if have == want:
        return arr
    if want == 2:
        try:
            import ml_dtypes
            bf16 = ml_dtypes.bfloat16
        except Exception:  # pragma: no cover - ml_dtypes is present on GPU nodes
            bf16 = None
        if have == 1:
            vals = arr.astype(np.float32)
            return vals.astype(bf16) if bf16 is not None else vals.astype(np.float16)
        return arr.astype(bf16) if bf16 is not None else arr.astype(np.float16)
    if have != 1:
        return arr.astype(np.uint8)
    return arr


_ARCH_IN_SO_NAME = None


def arch_from_so_path(so_path) -> Optional[str]:
    """Recover the gfx arch from a built ``.so`` filename.

    Two shapes are in use and both must parse::

        libgemm_bquant_fp8i4_..._gfx942.so             (pre-flags-tag)
        libgemm_bquant_fp8i4_..._gfx942_9bf231b3.so    (with the flag digest)

    A regex anchored at the end of the stem silently returns None for the
    second, and None means "assume OCP fp8" -- correct on gfx950 and wrong on
    gfx942, where ``ck_tile::fp8_t`` is FNUZ.  That is a silent wrong-answer
    path, so parse both.
    """
    import re
    from pathlib import Path as _Path
    global _ARCH_IN_SO_NAME
    if _ARCH_IN_SO_NAME is None:
        _ARCH_IN_SO_NAME = re.compile(r"_(gfx[0-9a-zA-Z]+?)(?:_[0-9a-f]{6,16})?$")
    m = _ARCH_IN_SO_NAME.search(_Path(so_path).stem)
    return m.group(1) if m else None
