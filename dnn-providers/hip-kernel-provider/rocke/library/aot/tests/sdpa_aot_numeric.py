# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Provider-local numeric verifier for checked-in rocKE SDPA AOT artifacts.

The test consumes loose build-tree artifacts: one copied ``aot_list.json``, and
per instance one matching ``<name>.sidecar.json`` plus one HSACO named by the
sidecar. It
intentionally uses the runtime-loaded HSACO path and skips with CTest code 77
unless the visible HIP device exactly matches ``--arch``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

SKIP_RETURN_CODE = 77
DEFAULT_ATOL = 2e-2
DEFAULT_RTOL = 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_aot_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty JSON array")
    return value


def _device_arch() -> str | None:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch()
    except Exception as exc:  # HIP runtime may be absent on CPU-only machines.
        print(f"SKIP: unable to query HIP device arch: {exc}")
        return None


def _ref_attention(q, k, v):
    import numpy as np

    d = q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", q.astype(np.float32), k.astype(np.float32))
    scores /= math.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("ihj,jhd->ihd", probs, v.astype(np.float32))


def _eval_grid_formula(
    formula: dict[str, Any], values: dict[str, Any]
) -> tuple[int, int, int]:
    def eval_axis(axis: Any) -> int:
        if isinstance(axis, int):
            return axis
        if isinstance(axis, str):
            return int(values[axis])
        if isinstance(axis, dict) and "ceil_div" in axis:
            args = axis["ceil_div"]
            if not isinstance(args, list) or len(args) != 2:
                raise ValueError(f"invalid ceil_div grid formula axis: {axis!r}")
            numerator, denominator = args
            n = int(values[numerator] if isinstance(numerator, str) else numerator)
            d = int(
                values[denominator] if isinstance(denominator, str) else denominator
            )
            return (n + d - 1) // d
        raise ValueError(f"unsupported grid formula axis: {axis!r}")

    return (eval_axis(formula["x"]), eval_axis(formula["y"]), eval_axis(formula["z"]))


def _host_buffer(array):
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return (ctypes.c_uint8 * int(contiguous.nbytes)).from_buffer(contiguous)


def _pack_args(signature: list[dict[str, Any]], values: dict[str, Any]) -> bytes:
    packed = bytearray()
    offset = 0
    for arg in signature:
        name = arg["name"]
        size = int(arg["size_bytes"])
        alignment = int(arg.get("alignment", size))
        padding = (-offset) % alignment
        if padding:
            packed.extend(b"\0" * padding)
            offset += padding

        ty = arg["type"]
        kind = arg["kind"]
        value = values[name]
        if kind == "pointer":
            if size != 8:
                raise ValueError(f"pointer arg {name} must be 8 bytes, got {size}")
            chunk = struct.pack("<Q", int(value))
        elif kind == "scalar" and ty == "f32" and size == 4:
            chunk = struct.pack("<f", float(value))
        elif kind == "scalar" and ty == "i32" and size == 4:
            chunk = struct.pack("<i", int(value))
        elif kind == "scalar" and ty == "i64" and size == 8:
            chunk = struct.pack("<q", int(value))
        else:
            raise ValueError(f"unsupported arg signature entry: {arg!r}")
        packed.extend(chunk)
        offset += len(chunk)
    return bytes(packed)


def _verify_profile(
    instance: dict[str, Any], sidecar: dict[str, Any], hsaco: bytes, batch: int
) -> bool:
    import numpy as np
    from rocke.runtime.hip_module import Runtime

    compile_spec = instance["compile_spec"]
    seqlen_q = int(compile_spec["seqlen_q"])
    seqlen_k = int(compile_spec["seqlen_k"])
    num_query_heads = int(compile_spec["num_query_heads"])
    num_kv_heads = int(compile_spec["num_kv_heads"])
    head_size = int(compile_spec["head_size"])
    if compile_spec.get("mask_mode") != "none":
        raise ValueError("only mask_mode='none' is supported by this verifier")

    rng = np.random.default_rng(0xA07F00D)
    q = (
        rng.standard_normal((batch, seqlen_q, num_query_heads, head_size)) * 0.3
    ).astype(np.float16)
    k = (rng.standard_normal((batch, seqlen_k, num_kv_heads, head_size)) * 0.3).astype(
        np.float16
    )
    v = (rng.standard_normal((batch, seqlen_k, num_kv_heads, head_size)) * 0.3).astype(
        np.float16
    )
    out = np.zeros((batch, seqlen_q, num_query_heads, head_size), dtype=np.float16)

    stride_q_token = num_query_heads * head_size
    stride_q_head = head_size
    stride_k_token = num_kv_heads * head_size
    stride_k_head = head_size
    stride_v_token = num_kv_heads * head_size
    stride_v_head = head_size
    stride_o_token = num_query_heads * head_size
    stride_o_head = head_size
    scale_log2 = float(1.0 / math.sqrt(head_size) * math.log2(math.e))

    grid_values = {**compile_spec, "batch": batch}
    grid = _eval_grid_formula(sidecar["launch"]["grid_formula"], grid_values)
    block = tuple(int(x) for x in sidecar["launch"]["block"])
    shared_mem = int(sidecar["launch"].get("shared_mem_bytes", 0))

    runtime = Runtime()
    module = runtime.load_module(hsaco)
    device_ptrs: list[int] = []
    try:
        function = module.get_function(sidecar["artifact"]["symbol"])
        qd = runtime.alloc(q.nbytes)
        device_ptrs.append(qd)
        kd = runtime.alloc(k.nbytes)
        device_ptrs.append(kd)
        vd = runtime.alloc(v.nbytes)
        device_ptrs.append(vd)
        od = runtime.alloc(out.nbytes)
        device_ptrs.append(od)

        runtime.memcpy_h2d(qd, _host_buffer(q), q.nbytes)
        runtime.memcpy_h2d(kd, _host_buffer(k), k.nbytes)
        runtime.memcpy_h2d(vd, _host_buffer(v), v.nbytes)
        runtime.memset(od, 0, out.nbytes)

        arg_values = {
            "Q": qd,
            "K": kd,
            "V": vd,
            "O": od,
            "scale_log2": scale_log2,
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "stride_q_token": stride_q_token,
            "stride_q_head": stride_q_head,
            "stride_k_token": stride_k_token,
            "stride_k_head": stride_k_head,
            "stride_v_token": stride_v_token,
            "stride_v_head": stride_v_head,
            "stride_o_token": stride_o_token,
            "stride_o_head": stride_o_head,
        }
        packed = _pack_args(sidecar["args_signature"], arg_values)
        runtime.launch(function, grid, block, packed, shared_bytes=shared_mem)
        runtime.sync()
        runtime.memcpy_d2h(_host_buffer(out), od, out.nbytes)
    finally:
        for ptr in device_ptrs:
            runtime.free(ptr)
        module.unload()

    reference = np.empty(out.shape, dtype=np.float32)
    for batch_index in range(batch):
        if num_kv_heads != num_query_heads:
            repeat = num_query_heads // num_kv_heads
            kb = np.repeat(k[batch_index], repeat, axis=1)
            vb = np.repeat(v[batch_index], repeat, axis=1)
        else:
            kb = k[batch_index]
            vb = v[batch_index]
        reference[batch_index] = _ref_attention(q[batch_index], kb, vb)

    actual = out.astype(np.float32)
    expected = reference.astype(np.float32)
    close = np.isclose(actual, expected, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    diff = np.abs(actual - expected)
    max_abs = float(diff.max())
    bad = int(np.count_nonzero(~close))
    ok = bool(np.allclose(actual, expected, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL))
    tag = "PASS" if ok else "FAIL"
    print(
        f"{tag}: batch={batch} Sq={seqlen_q} Sk={seqlen_k} Hq={num_query_heads} "
        f"Hkv={num_kv_heads} D={head_size} max_abs={max_abs:.3e} "
        f"bad={bad}/{out.size} rtol={DEFAULT_RTOL:.0e} atol={DEFAULT_ATOL:.0e}"
    )
    return ok


def _verify_instance(instance: dict[str, Any], artifact_dir: Path) -> bool:
    name = instance.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"instance in {artifact_dir} is missing a name")
    sidecar_path = artifact_dir / f"{name}.sidecar.json"
    sidecar = _load_json(sidecar_path)
    hsaco_path = artifact_dir / sidecar["artifact"]["hsaco_filename"]
    hsaco = hsaco_path.read_bytes()

    expected_sha = sidecar["artifact"]["hsaco_sha256"]
    expected_size = int(sidecar["artifact"]["hsaco_size"])
    actual_sha = hashlib.sha256(hsaco).hexdigest()
    if actual_sha != expected_sha or len(hsaco) != expected_size:
        raise ValueError(
            f"HSACO digest/size mismatch for {hsaco_path}: "
            f"sha {actual_sha} size {len(hsaco)}"
        )

    profiles = instance.get("test_profiles", [])
    if not profiles:
        print(f"SKIP instance without test profiles: {name}")
        return True

    ok = True
    for profile in profiles:
        batch = int(profile["batch"])
        ok = _verify_profile(instance, sidecar, hsaco, batch) and ok
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    device_arch = _device_arch()
    if device_arch != args.arch:
        print(
            f"SKIP: device arch {device_arch!r} does not match requested {args.arch!r}"
        )
        return SKIP_RETURN_CODE

    aot_list_path = args.artifact_dir / "aot_list.json"
    if not aot_list_path.is_file():
        raise SystemExit(f"no aot_list.json found in {args.artifact_dir}")

    ok = True
    for instance in _load_aot_list(aot_list_path):
        ok = _verify_instance(instance, args.artifact_dir) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
