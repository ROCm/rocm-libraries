# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The isolated HIP execution loop that every operation's worker runs.

Qualification freezes this module with the old runtime and the operation's
worker in each bundle. Source mode exercises the operation's current dispatch
and public launch entry. Replay mode loads the already compiled old HSACO and
frozen ABI. Neither mode computes independent reference answers or adjusts
error budgets.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .numeric import array_digest, decode, file_digest, write_json
from .targets import get_device_processor


@dataclass(frozen=True)
class WorkerOperation:
    """What an operation's worker binds around the shared execution loop.

    ``case`` rebuilds the requested case, whose ``input_shapes`` name the
    read-only operands and whose ``output_shape`` is the single output.
    ``launch(target, prepared, buffers, case)`` runs the current kernel that the
    architecture adapter prepared. ``replay_values(metadata, buffers, case)``
    binds live buffers to a frozen kernel's recorded arguments.
    """

    label: str
    case: type
    launch: Callable[[ModuleType, Any, dict, Any], None]
    replay_values: Callable[[dict, dict, Any], dict]


def run_case(
    operation: WorkerOperation, target: ModuleType, request: dict, output: Path
) -> None:
    """Execute a single declared case, poisoning every output before each run."""
    import rocke
    from rocke.runtime import DeviceMem, KernelLauncher, LaunchConfig, Runtime
    from rocke.runtime.hip_module import get_device_target_id

    root = Path(request["platform_root"]).resolve()
    if not Path(rocke.__file__).resolve().is_relative_to(root):
        raise RuntimeError("worker imported rocKE outside its selected source/runtime")
    if get_device_processor() != target.NAME:
        raise RuntimeError(
            f"{operation.label} correctness requires a HIP-visible {target.NAME} device"
        )
    if request["mode"] not in ("source", "replay") or request["repetitions"] < 1:
        raise ValueError(f"invalid {operation.label} worker request")
    case = operation.case(**request["case"])
    with np.load(request["inputs"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in case.input_shapes}
    expected_shapes = case.input_shapes
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"wrong input shape for {name}: {array.shape}")
        if not np.isfinite(decode(array, case.dtype)).all():
            raise ValueError(f"non-finite input: {name}")
        if array_digest(array) != request["input_digests"][name]:
            raise ValueError(f"input digest mismatch: {name}")

    rt = Runtime()
    storage = np.dtype("<f2" if case.dtype == "fp16" else "<u2")
    host_out = np.empty(case.output_shape, dtype=storage)
    buffers = {name: DeviceMem(array.nbytes) for name, array in arrays.items()}
    buffers["out"] = DeviceMem(host_out.nbytes)
    try:
        for name, array in arrays.items():
            rt.memcpy_h2d(
                buffers[name].ptr(), ctypes.c_void_p(array.ctypes.data), array.nbytes
            )

        if request["mode"] == "source":
            spec = target.prepare(case, request["library_root"])
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
                    operation.launch(target, spec, buffers, case)
                else:
                    launcher(
                        operation.replay_values(metadata, buffers, case),
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
                    raise ValueError(
                        f"{operation.label} produced non-finite or unwritten output"
                    )
                results[f"out_{repetition}"] = host_out.copy()
        finally:
            KernelLauncher.__call__ = launch_call

        if launches != request["repetitions"]:
            raise RuntimeError(
                f"{operation.label} {request['mode']} expected "
                f"{request['repetitions']} GPU launches, one per repetition, "
                f"got {launches}"
            )
        for name, array in arrays.items():
            returned = np.empty_like(array)
            rt.memcpy_d2h(
                ctypes.c_void_p(returned.ctypes.data), buffers[name].ptr(), array.nbytes
            )
            if array_digest(returned) != request["input_digests"][name]:
                raise AssertionError(
                    f"{operation.label} modified read-only input: {name}"
                )

        report = {
            "launches": launches,
            "device_target": get_device_target_id(),
            "torch_imported": "torch" in sys.modules,
        }
        if request["mode"] == "source" and request.get("export"):
            from rocke.core.lower_llvm import _resolve_llvm_flavor
            from rocke.runtime.comgr import _resolve_lib

            code, kernel = target.exported_kernel(spec)
            # Instances selected by a registered identity were lowered with
            # its IR flavor; others use the worker's resolved flavor.
            identity = kernel.get("dispatch_spec", {}).get("identity", {})
            compiler = Path(str(_resolve_lib()._name))
            report["compiler"] = {
                "library": compiler.name,
                "sha256": file_digest(compiler) if compiler.is_file() else None,
                "llvm_flavor": identity.get("llvm_flavor") or _resolve_llvm_flavor(),
            }
            hsaco = Path(request["export"])
            hsaco.write_bytes(code)
            report["kernel"] = dict(kernel, sha256=file_digest(hsaco))
        np.savez(output / "outputs.npz", **results)
        write_json(output / "report.json", report)
    finally:
        rt.sync()
        for buffer in buffers.values():
            buffer.realloc(0)


def main(run: Callable[[dict, Path], None]) -> None:
    """Run one request file, as the runner does without a reused interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()), args.output)
