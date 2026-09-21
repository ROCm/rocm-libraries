# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest schema emitters for the Python runner and C++ launcher.

`make_gemm_manifest(...)` and `make_conv_manifest(...)` produce the
JSON object the `rocke.run_manifest` runner (and the legacy C++
launcher) consume. Keeping the schema in one place means every
example/sweep emits the same fields and any change can be reviewed
once.

Schema (version `ck.dsl.example.manifest/v1`):

    {
      "schema": "ck.dsl.example.manifest/v1",
      "kind": "gemm_fp16" | "conv_fp16" | "conv_bf16" | "conv_fp32",
      "kernel_name": <str>,
      "hsaco": <basename of the .hsaco file next to this manifest>,
      "block_m": <int>, "block_n": <int>, "block_k": <int>,
      "threads_per_block": <int>,
      "default_shape": [M, N, K],            // gemm only
      "conv": [N, H, W, C, K, R, S, sH, sW, pH, pW, dH, dW],  // conv only
      "groups": <int>,                       // conv only
      "cpg": <int>, "kpg": <int>,            // conv only
      "grid_explicit": [gx, gy, gz],         // optional, overrides grid_order
      "grid_order": "MN" | "NM",             // optional
      "args_signature": [
        {"name": ..., "type": "ptr<f16,global>" | "i32", "size_bytes": ...},
        ...
      ],
      "sig_has_bytes": 0 | 1,                // 1 if A_bytes/B_bytes/D_bytes are kernel args
      "warmup_iters": <int>,
      "timed_iters": <int>,
      "timing_ms": {... codegen breakdown ...},
      "hsaco_bytes": <int>,
      "notes": <str>,
      "ck_dependency": false,
      "ir_authored": true,
      "engine_build_id": <str>,   // content hash of the engine that produced this
      "engine_version": <str>     // engine version that produced this
    }

The `engine_build_id` / `engine_version` fields stamp the engine provenance so a
consumer (e.g. the rocke-provider) can fail loud on a stale/mixed bundle whose
engine does not match the engine it is linked against, rather than silently
mixing artifacts.

The runner cares about `kind`, `kernel_name`, `hsaco`, `block_m/n/k`,
`threads_per_block`, the shape-providing field (`default_shape` or
`conv`), the grid hint, the args layout, and the iteration counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .compile import KernelArtifact


MANIFEST_SCHEMA = "ck.dsl.example.manifest/v1"


# Re-exports for examples to use.
__all__ = [
    "MANIFEST_SCHEMA",
    "attention_args_signature",
    "conv_args_signature",
    "engine_build_id",
    "engine_version",
    "gemm_args_signature",
    "make_attention_manifest",
    "make_conv_manifest",
    "make_gemm_manifest",
    "make_simple_op_manifest",
    "write_artifact",
]


# ---------------------------------------------------------------------
# Engine freshness / provenance stamp
# ---------------------------------------------------------------------
#
# Every manifest records the engine build-id + version it was produced with, so
# a consumer (e.g. the rocke-provider) can fail loud when a shipped bundle was
# built by a different engine than the one it is linked against, instead of
# silently mixing stale artifacts. The values come from the C++ engine's
# rocke_build_id()/rocke_engine_version() (exposed via the rocke_engine binding) when
# that module is importable; otherwise they fall back to "unknown" so manifest
# emission never hard-depends on the binding being built. These are artifact
# stamps only -- they never enter the emitted IR.


def _engine_provenance() -> tuple:
    """(build_id, version) from the rocke_engine binding, or ("unknown", ...)."""
    try:
        import rocke_engine  # noqa: PLC0415

        return rocke_engine.build_id(), rocke_engine.engine_version()
    except Exception:  # noqa: BLE001
        return "unknown", "unknown"


def engine_build_id() -> str:
    """Engine source content hash recorded in manifests (or 'unknown')."""
    return _engine_provenance()[0]


def engine_version() -> str:
    """Engine version recorded in manifests (or 'unknown')."""
    return _engine_provenance()[1]


def _provenance_fields() -> Dict[str, str]:
    bid, ver = _engine_provenance()
    return {"engine_build_id": bid, "engine_version": ver}


# ---------------------------------------------------------------------
# Args signature helpers
# ---------------------------------------------------------------------


def gemm_args_signature(*, with_bytes: bool = False) -> List[Dict[str, Any]]:
    """Standard GEMM kernel args signature: A, B, C ptrs + M, N, K i32s.

    `with_bytes=True` adds A_bytes/B_bytes/C_bytes args before the
    M, N, K dimensions (this is the implicit-GEMM convolution
    signature; the universal GEMM doesn't need them since it doesn't
    use buffer_rsrc).
    """
    sig: List[Dict[str, Any]] = [
        {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "B", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "C", "type": "ptr<f16, global>", "size_bytes": 8},
    ]
    if with_bytes:
        sig += [
            {"name": "A_bytes", "type": "i32", "size_bytes": 4},
            {"name": "B_bytes", "type": "i32", "size_bytes": 4},
            {"name": "C_bytes", "type": "i32", "size_bytes": 4},
        ]
    else:
        sig += [
            {"name": "M", "type": "i32", "size_bytes": 4},
            {"name": "N", "type": "i32", "size_bytes": 4},
            {"name": "K", "type": "i32", "size_bytes": 4},
        ]
    return sig


def conv_args_signature(dtype: str = "fp16") -> List[Dict[str, Any]]:
    """Conv kernel args signature: A, B, D ptrs + A_bytes/B_bytes/D_bytes."""
    _dtype_map = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}
    ir_type = _dtype_map.get(dtype, dtype)
    ptr_type = f"ptr<{ir_type}, global>"
    return [
        {"name": "A", "type": ptr_type, "size_bytes": 8},
        {"name": "B", "type": ptr_type, "size_bytes": 8},
        {"name": "D", "type": ptr_type, "size_bytes": 8},
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "B_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
    ]


def attention_args_signature(*, path: str = "2d") -> List[Dict[str, Any]]:
    """Standard unified-attention torch launch signature.

    The 2D and 3D kernels share most arguments; the reduce kernel uses a
    smaller segment-workspace signature and is represented by `path="reduce"`.
    """
    if path == "reduce":
        return [
            {"name": "output_ptr", "type": "ptr<f16, global>", "size_bytes": 8},
            {"name": "segm_output_ptr", "type": "ptr<f32, global>", "size_bytes": 8},
            {"name": "segm_max_ptr", "type": "ptr<f32, global>", "size_bytes": 8},
            {"name": "segm_expsum_ptr", "type": "ptr<f32, global>", "size_bytes": 8},
            {"name": "seq_lens_ptr", "type": "ptr<i32, global>", "size_bytes": 8},
            {
                "name": "query_start_len_ptr",
                "type": "ptr<i32, global>",
                "size_bytes": 8,
            },
        ]
    return [
        {"name": "output_ptr", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "query_ptr", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "key_cache_ptr", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "value_cache_ptr", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "sink_ptr", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "block_tables_ptr", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "seq_lens_ptr", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "alibi_slopes_ptr", "type": "ptr<f32, global>", "size_bytes": 8},
        {"name": "qq_bias_ptr", "type": "ptr<f32, global>", "size_bytes": 8},
        {"name": "query_start_len_ptr", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "scale", "type": "f32", "size_bytes": 4},
        {"name": "k_scale", "type": "f32", "size_bytes": 4},
        {"name": "v_scale", "type": "f32", "size_bytes": 4},
        {"name": "out_scale", "type": "f32", "size_bytes": 4},
        {"name": "softcap", "type": "f32", "size_bytes": 4},
        {"name": "num_seqs", "type": "i32", "size_bytes": 4},
    ]


# ---------------------------------------------------------------------
# Manifest emitters
# ---------------------------------------------------------------------


def make_simple_op_manifest(
    *,
    artifact: KernelArtifact,
    kind: str,
    op: str,
    dtype: str,
    threads_per_block: int,
    default_shape: Sequence[int],
    args_signature: List[Dict[str, Any]],
    elems_per_block: Optional[int] = None,
    is_binary: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    grid_explicit: Optional[Sequence[int]] = None,
    eps: Optional[float] = None,
    warmup_iters: int = 5,
    timed_iters: int = 100,
    atoms: Iterable[str] = (),
    notes: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Generic manifest emitter for the ``simple_op`` family.

    Handles ``elementwise_fp16``, ``reduce_fp16``, ``layernorm_fp16``,
    ``rmsnorm_fp16``, and ``transpose_fp16`` -- every "X(input)→output"
    op that the :mod:`rocke.run_manifest` runner can dispatch via
    ``_simple_op_problem``. Each example's ``gen.py`` builds one of
    these by passing in the artifact + the per-op shape / signature.

    ``default_shape`` is the canonical shape the runner uses:
      * elementwise: ``[N]``
      * reduce / layernorm / rmsnorm / transpose: ``[M, N]``

    ``args_signature`` mirrors what the kernel emitted via
    ``SignatureBuilder`` (one entry per kernel param). The runner
    packs these into the AMDGPU kernarg buffer in declaration order.
    """
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "kind": kind,
        "op": op,
        "dtype": dtype,
        "kernel_name": artifact.kernel_name,
        "hsaco": f"{artifact.kernel_name}.hsaco",
        "threads_per_block": int(threads_per_block),
        "default_shape": [int(x) for x in default_shape],
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "args_signature": list(args_signature),
        "sig_has_bytes": 0,
        "timing_ms": dict(artifact.timings),
        "hsaco_bytes": artifact.hsaco_bytes,
        "atoms": list(atoms),
        "notes": notes,
        "ck_dependency": False,
        "ir_authored": True,
        "is_binary": bool(is_binary),
        **_provenance_fields(),
    }
    if elems_per_block is not None:
        manifest["elems_per_block"] = int(elems_per_block)
    if block_m is not None:
        manifest["block_m"] = int(block_m)
    if block_n is not None:
        manifest["block_n"] = int(block_n)
    if grid_explicit is not None:
        manifest["grid_explicit"] = [int(x) for x in grid_explicit]
    if eps is not None:
        manifest["eps"] = float(eps)
    if extra:
        manifest.update(dict(extra))
    return manifest


def make_gemm_manifest(
    *,
    artifact: KernelArtifact,
    block_m: int,
    block_n: int,
    block_k: int,
    threads_per_block: int,
    default_shape: Sequence[int] = (3328, 4096, 4096),
    warmup_iters: int = 5,
    timed_iters: int = 100,
    grid_order: str = "NM",
    args_signature: Optional[List[Dict[str, Any]]] = None,
    atoms: Iterable[str] = (),
    notes: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the v1 manifest JSON object for one GEMM kernel.

    `grid_order` chooses how the runner translates `(M_tiles, N_tiles)`
    to `(gx, gy)`:
      - `"MN"`: `gx = ceil(M/block_m)`, `gy = ceil(N/block_n)`
      - `"NM"` (default): `gx = ceil(N/block_n)`, `gy = ceil(M/block_m)`
    Match what your kernel reads from `block_id.x` and `block_id.y`.
    The default `"NM"` (block.x->N) is the universal-GEMM / host-launcher
    convention; the gfx1151 WMMA GEMM passes `"MN"` to flip to block.x->M.

    `extra` lets you splice in kernel-specific fields (e.g. an MLIR
    config dump, a transform-DAG JSON, dispatcher metadata).
    """
    args = args_signature if args_signature is not None else gemm_args_signature()
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "kind": "gemm_fp16",
        "kernel_name": artifact.kernel_name,
        "hsaco": f"{artifact.kernel_name}.hsaco",
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "threads_per_block": int(threads_per_block),
        "default_shape": [int(x) for x in default_shape],
        "grid_order": grid_order,
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "args_signature": args,
        "sig_has_bytes": int(any(a["name"].endswith("_bytes") for a in args)),
        "timing_ms": dict(artifact.timings),
        "hsaco_bytes": artifact.hsaco_bytes,
        "atoms": list(atoms),
        "notes": notes,
        "ck_dependency": False,
        "ir_authored": True,
        **_provenance_fields(),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def make_conv_manifest(
    *,
    artifact: KernelArtifact,
    block_m: int,
    block_n: int,
    block_k: int,
    threads_per_block: int,
    conv: Sequence[int],
    groups: int,
    cpg: int,
    kpg: int,
    dtype: str = "fp16",
    grid_explicit: Optional[Sequence[int]] = None,
    grid_order: Optional[str] = None,
    conv_layout: str = "implicit_gemm",
    warmup_iters: int = 5,
    timed_iters: int = 100,
    atoms: Iterable[str] = (),
    notes: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the v1 manifest JSON object for one convolution kernel.

    `conv` is `[N, H, W, C, K, R, S, sH, sW, pH, pW, dH, dW]` (13
    ints). `groups` / `cpg` / `kpg` describe the grouping; for dense
    conv pass `groups=1, cpg=C, kpg=K`.

    `dtype` is `"fp16"` (default), `"bf16"`, or `"fp32"`. It sets the
    manifest `kind` (e.g. `"conv_bf16"`) and the pointer types in the
    args signature so the runner allocates tensors of the right dtype.

    Pass `grid_explicit=[gx, gy, gz]` to bypass the runner's automatic
    grid derivation (this is what the direct conv kernels use; the
    Q-tile axis isn't simply `ceil(K/block_n)`).
    """
    _valid_dtypes = {"fp16", "bf16", "fp32"}
    if dtype not in _valid_dtypes:
        raise ValueError(f"dtype must be one of {_valid_dtypes}, got {dtype!r}")
    if len(list(conv)) != 13:
        raise ValueError(f"conv expects 13 ints (got {len(list(conv))})")

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "kind": f"conv_{dtype}",
        "dtype": dtype,
        "conv_layout": conv_layout,
        "kernel_name": artifact.kernel_name,
        "hsaco": f"{artifact.kernel_name}.hsaco",
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "threads_per_block": int(threads_per_block),
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "conv": [int(x) for x in conv],
        "groups": int(groups),
        "cpg": int(cpg),
        "kpg": int(kpg),
        "args_signature": conv_args_signature(dtype),
        "sig_has_bytes": 1,
        "timing_ms": dict(artifact.timings),
        "hsaco_bytes": artifact.hsaco_bytes,
        "atoms": list(atoms),
        "notes": notes,
        "ck_dependency": False,
        "ir_authored": True,
        **_provenance_fields(),
    }
    if grid_explicit is not None:
        manifest["grid_explicit"] = [int(x) for x in grid_explicit]
    if grid_order is not None:
        manifest["grid_order"] = grid_order
    if extra:
        manifest.update(dict(extra))
    return manifest


def make_attention_manifest(
    *,
    artifact: KernelArtifact,
    path: str,
    grid: Sequence[int],
    block: Sequence[int],
    config: Mapping[str, Any],
    warmup_iters: int = 5,
    timed_iters: int = 100,
    notes: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a manifest object for a torch-launched attention kernel."""
    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "kind": "attention_unified",
        "attention_path": path,
        "kernel_name": artifact.kernel_name,
        "hsaco": f"{artifact.kernel_name}.hsaco",
        "grid_explicit": [int(x) for x in grid],
        "block_explicit": [int(x) for x in block],
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "attention_config": dict(config),
        "args_signature": attention_args_signature(path=path),
        "timing_ms": dict(artifact.timings),
        "hsaco_bytes": artifact.hsaco_bytes,
        "notes": notes,
        "ck_dependency": False,
        "ir_authored": True,
        **_provenance_fields(),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def write_artifact(
    artifact: KernelArtifact,
    out_dir: Path,
    manifest: Dict[str, Any],
    *,
    write_ir_text: bool = True,
    write_llvm_text: bool = True,
) -> Dict[str, Path]:
    """Write `kernel.hsaco`, `kernel.ir.txt`, `kernel.ll`, `manifest.json`.

    Returns the paths written, keyed by short name (`hsaco`, `ir`,
    `ll`, `manifest`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = artifact.kernel_name
    paths: Dict[str, Path] = {}

    hsaco_path = out_dir / f"{name}.hsaco"
    hsaco_path.write_bytes(artifact.hsaco)
    paths["hsaco"] = hsaco_path

    if write_ir_text and artifact.ir_text:
        ir_path = out_dir / f"{name}.ir.txt"
        ir_path.write_text(artifact.ir_text + "\n", encoding="utf-8")
        paths["ir"] = ir_path
    if write_llvm_text:
        ll_path = out_dir / f"{name}.ll"
        ll_path.write_text(artifact.llvm_text, encoding="utf-8")
        paths["ll"] = ll_path

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths
