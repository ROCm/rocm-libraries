# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest schema emitters for the Python runner and C++ launcher.

`make_gemm_manifest(...)` and `make_conv_manifest(...)` produce the
JSON object the `ck_dsl.run_manifest` runner (and the legacy C++
launcher) consume. Keeping the schema in one place means every
example/sweep emits the same fields and any change can be reviewed
once.

Schema (version `ck.dsl.example.manifest/v1`):

    {
      "schema": "ck.dsl.example.manifest/v1",
      "kind": "gemm_fp16" | "conv_fp16",
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
      "ir_authored": true
    }

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


def conv_args_signature() -> List[Dict[str, Any]]:
    """Conv kernel args signature: A, B, D ptrs + A_bytes/B_bytes/D_bytes."""
    return [
        {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "B", "type": "ptr<f16, global>", "size_bytes": 8},
        {"name": "D", "type": "ptr<f16, global>", "size_bytes": 8},
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
    grid_order: str = "MN",
    args_signature: Optional[List[Dict[str, Any]]] = None,
    atoms: Iterable[str] = (),
    notes: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the v1 manifest JSON object for one GEMM kernel.

    `grid_order` chooses how the runner translates `(M_tiles, N_tiles)`
    to `(gx, gy)`:
      - `"MN"`: `gx = ceil(M/block_m)`, `gy = ceil(N/block_n)`
      - `"NM"`: `gx = ceil(N/block_n)`, `gy = ceil(M/block_m)`
    Match what your kernel reads from `block_id.x` and `block_id.y`.

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

    Pass `grid_explicit=[gx, gy, gz]` to bypass the runner's automatic
    grid derivation (this is what the direct conv kernels use; the
    Q-tile axis isn't simply `ceil(K/block_n)`).
    """
    if len(list(conv)) != 13:
        raise ValueError(f"conv expects 13 ints (got {len(list(conv))})")

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "kind": "conv_fp16",
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
        "args_signature": conv_args_signature(),
        "sig_has_bytes": 1,
        "timing_ms": dict(artifact.timings),
        "hsaco_bytes": artifact.hsaco_bytes,
        "atoms": list(atoms),
        "notes": notes,
        "ck_dependency": False,
        "ir_authored": True,
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
