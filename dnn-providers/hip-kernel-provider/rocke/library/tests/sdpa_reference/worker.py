# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Isolated HIP worker, frozen with the old runtime in a qualified bundle.

Source mode exercises the same dispatch and public launch entry as the existing
SDPA test. Replay mode loads the already compiled old HSACO and frozen ABI.
Neither mode computes independent reference answers or adjusts error budgets.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

from .contract import Case, array_digest, decode, file_digest, write_json


def run(request: dict, output: Path) -> None:
    """Execute a single declared case, poisoning every output before each run."""
    import rocke
    from rocke.runtime import DeviceMem, KernelLauncher, LaunchConfig, Runtime
    from rocke.runtime.hip_module import get_device_arch, get_device_target_id

    root = Path(request["platform_root"]).resolve()
    if not Path(rocke.__file__).resolve().is_relative_to(root):
        raise RuntimeError("worker imported rocKE outside its selected source/runtime")
    if get_device_arch() != "gfx942":
        raise RuntimeError("SDPA correctness requires a HIP-visible gfx942 device")
    case = Case(**request["case"])
    with np.load(request["inputs"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("q", "k", "v")}
    expected_shapes = {
        "q": case.shape,
        "k": (*case.shape[:2], case.kv_heads, case.head_dim),
        "v": (*case.shape[:2], case.kv_heads, case.head_dim),
    }
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"wrong input shape for {name}: {array.shape}")
        if not np.isfinite(decode(array, case.dtype)).all():
            raise ValueError(f"non-finite input: {name}")
        if array_digest(array) != request["input_digests"][name]:
            raise ValueError(f"input digest mismatch: {name}")

    rt = Runtime()
    storage = np.dtype("<f2" if case.dtype == "fp16" else "<u2")
    host_out = np.empty(case.shape, dtype=storage)
    buffers = {name: DeviceMem(array.nbytes) for name, array in arrays.items()}
    buffers["out"] = DeviceMem(host_out.nbytes)
    try:
        for name, array in arrays.items():
            rt.memcpy_h2d(
                buffers[name].ptr(), ctypes.c_void_p(array.ctypes.data), array.nbytes
            )

        if request["mode"] == "source":
            import kernels
            from dispatch.attention import AttentionRequest
            from dispatch.attention.gfx942 import _dense_spec
            from kernels.common.attention_dense_spec import attention_dense_cache_key
            from kernels.gfx942.attention_dense import (
                _DENSE_LAUNCHER_CACHE,
                _as_gfx942_spec,
                attention_dense_block,
                attention_dense_grid,
                run_attention_dense_torch,
            )

            if (
                not Path(kernels.__file__)
                .resolve()
                .is_relative_to(Path(request["library_root"]).resolve())
            ):
                raise RuntimeError(
                    "worker imported kernels outside the selected library"
                )
            spec = _as_gfx942_spec(
                _dense_spec(
                    AttentionRequest(
                        batch=case.batch,
                        nhead_q=case.query_heads,
                        nhead_k=case.kv_heads,
                        seqlen_q=case.sequence_length,
                        seqlen_k=case.sequence_length,
                        hdim_q=case.head_dim,
                        hdim_v=case.head_dim,
                        arch="gfx942",
                        mask_type=1 if case.causal else 0,
                        dtype=case.dtype,
                        algorithm="attention_dense",
                        dense_persistent="on" if case.persistent else "off",
                    )
                )
            )
        else:
            metadata = request["kernel"]
            hsaco = Path(request["hsaco"])
            if file_digest(hsaco) != metadata["sha256"]:
                raise ValueError("pinned HSACO digest mismatch")
            launcher = KernelLauncher(
                hsaco=hsaco.read_bytes(),
                kernel_name=metadata["name"],
                signature=metadata["signature"],
            )

        launches = 0
        launch_call = KernelLauncher.__call__

        def counted_launch(self, *args, **kwargs):
            nonlocal launches
            result = launch_call(self, *args, **kwargs)
            launches += result.launches
            return result

        KernelLauncher.__call__ = counted_launch
        results = {}
        try:
            for repetition in range(request["repetitions"]):
                # 0xffff is a NaN in both storage types. A missing or partial
                # launch must never inherit a plausible answer from allocation.
                rt.memset(buffers["out"].ptr(), 0xFF, host_out.nbytes)
                if request["mode"] == "source":
                    run_attention_dense_torch(
                        spec=spec,
                        q=buffers["q"],
                        k=buffers["k"],
                        v=buffers["v"],
                        out=buffers["out"],
                        scale=case.scale,
                    )
                else:
                    values = {
                        "q_ptr": buffers["q"],
                        "k_ptr": buffers["k"],
                        "v_ptr": buffers["v"],
                        "o_ptr": buffers["out"],
                        "scale": case.scale,
                    }
                    if metadata["runtime_shape"]:
                        values.update(
                            batch=case.batch,
                            seqlen_q=case.sequence_length,
                            seqlen_kv=case.sequence_length,
                        )
                    launcher(
                        values,
                        config=LaunchConfig(
                            grid=tuple(metadata["grid"]),
                            block=tuple(metadata["block"]),
                        ),
                    )
                rt.sync()
                rt.memcpy_d2h(
                    ctypes.c_void_p(host_out.ctypes.data),
                    buffers["out"].ptr(),
                    host_out.nbytes,
                )
                if not np.isfinite(decode(host_out, case.dtype)).all():
                    raise ValueError("SDPA produced non-finite or unwritten output")
                results[f"out_{repetition}"] = host_out.copy()
        finally:
            KernelLauncher.__call__ = launch_call

        if launches != request["repetitions"]:
            raise RuntimeError(
                f"expected {request['repetitions']} GPU launches, got {launches}"
            )
        for name, array in arrays.items():
            returned = np.empty_like(array)
            rt.memcpy_d2h(
                ctypes.c_void_p(returned.ctypes.data), buffers[name].ptr(), array.nbytes
            )
            if array_digest(returned) != request["input_digests"][name]:
                raise AssertionError(f"SDPA modified read-only input: {name}")

        report = {
            "launches": launches,
            "device_target": get_device_target_id(),
            "torch_imported": "torch" in sys.modules,
        }
        if request["mode"] == "source" and request.get("export"):
            from rocke.core.lower_llvm import _resolve_llvm_flavor
            from rocke.runtime.comgr import _resolve_lib

            compiler = Path(str(_resolve_lib()._name))
            report["compiler"] = {
                "library": compiler.name,
                "sha256": file_digest(compiler) if compiler.is_file() else None,
                "llvm_flavor": _resolve_llvm_flavor(),
            }
            launcher = _DENSE_LAUNCHER_CACHE[
                attention_dense_cache_key(spec, arch="gfx942")
            ]
            hsaco = Path(request["export"])
            hsaco.write_bytes(launcher._hsaco)
            report["kernel"] = {
                "name": launcher.kernel_name,
                "signature": launcher.signature,
                "grid": attention_dense_grid(spec),
                "block": attention_dense_block(spec),
                "runtime_shape": spec.runtime_shape,
                "sha256": file_digest(hsaco),
            }
        np.savez(output / "outputs.npz", **results)
        write_json(output / "report.json", report)
    finally:
        rt.sync()
        for buffer in buffers.values():
            buffer.realloc(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()), args.output)


if __name__ == "__main__":
    main()
