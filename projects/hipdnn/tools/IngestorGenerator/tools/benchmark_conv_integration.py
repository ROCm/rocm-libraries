#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate and measure the packaged gfx950 convolution against direct rocKE.

Run once per dtype and mode, writing reports and logs to a private directory.
The caller supplies the installed plugin, the compiler used for packaging, and
the cache directory; this tool neither packages kernels nor deletes cache data.

For each dtype, run ``--tile-k 64`` and ``--tile-k 128`` for forced correctness
and timing. Then use a fresh cache with ``--mode auto`` and start another Python
process with ``--mode reuse`` and the same cache. Auto requires an uncached graph;
reuse requires an existing complete ranking and fails if another search occurs.
Every mode checks the exact engine, selected descriptor UUID, and tuning metadata.

Use ``--request-file`` with ``--tile-k`` for a different packaged convolution.
The file contains one ConvGroupedRequest JSON object, optionally including the
catalog's underscore-prefixed provenance annotations. These runs require forced
mode and default to an independent PyTorch FP32 GPU reference; the smoke modes
retain their CPU reference and two-candidate selection/cache checks. Override
the reference location with ``--reference-device cpu|gpu``. Every run validates
both integrated and direct outputs before timing; correctness cannot be skipped.

Example (all paths are caller-supplied)::

    python benchmark_conv_integration.py --plugin-path "$PLUGIN" \\
        --descriptor-root "$DESCRIPTORS" --backend-library "$BACKEND" \\
        --comgr-library "$COMGR" --cache-dir "$PRIVATE_CACHE" \\
        --log-file "$PRIVATE_LOG" --output "$PRIVATE_REPORT" \\
        --dtype fp16 --mode auto --timing-mode graph

The default timing captures repeated executions in a HIP graph and measures its
replays with GPU events. This removes Python submission gaps from the kernel
comparison. ``--timing-mode events`` measures a stream of ordinary submissions,
including gaps caused by host enqueue overhead. Both modes exclude compilation,
plan creation, first execution/search, correctness checks, capture, and warmup.
Timing also removes the synchronous selection-log callback and sets the backend
global log level to OFF through its public API. Both are restored before checking
recreated-plan evidence. The report records and verifies these logging states.
Dependencies are imported only when running; ``--help`` works without ROCm.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

ENGINE_NAME = "hipkernel:Gfx950ConvFwd"
BENCHMARK_KNOB = "global.benchmarking"
LOG_LEVELS = ("INFO", "WARN", "ERROR", "FATAL", "OFF")
UUID_PATTERN = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
SMOKE = {
    "N": 2,
    "Hi": 14,
    "Wi": 14,
    "C": 32,
    "K": 32,
    "Y": 3,
    "X": 3,
    "sH": 1,
    "sW": 1,
    "pH": 1,
    "pW": 1,
    "dH": 1,
    "dW": 1,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check(error, operation: str) -> None:
    require(error.is_good(), f"{operation}: {error.get_message()}")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plugin-path",
        type=Path,
        required=True,
        help="Installed provider library or plugin search directory.",
    )
    parser.add_argument(
        "--descriptor-root",
        type=Path,
        help="Installed descriptor tree, read only for UUID/metadata verification.",
    )
    parser.add_argument(
        "--backend-library",
        type=Path,
        help="The loaded libhipdnn_backend library (for its public log callback API).",
    )
    parser.add_argument(
        "--comgr-library",
        type=Path,
        default=os.environ.get("ROCKE_COMGR_LIB"),
        help="Same libamd_comgr used by hkp_pack; defaults to ROCKE_COMGR_LIB.",
    )
    parser.add_argument(
        "--llvm-flavor",
        choices=("llvm20", "llvm22"),
        default=os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22"),
        help="Same lowering flavor used for packaging; defaults to ROCKE_LLVM_FLAVOR or llvm22.",
    )
    parser.add_argument(
        "--rocke-root",
        type=Path,
        help="Source rocke/ tree; otherwise discover it beside this tool.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--request-file",
        type=Path,
        help="One ConvGroupedRequest JSON object; auto/reuse need both packaged tiles.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        help="Default: fp16 for smoke, otherwise the request dtype; must agree with the file.",
    )
    parser.add_argument(
        "--reference-device",
        choices=("cpu", "gpu"),
        help="Independent PyTorch FP32 reference; defaults to CPU for smoke, GPU for requests.",
    )
    parser.add_argument("--tile-k", type=int, choices=(64, 128))
    parser.add_argument(
        "--mode",
        choices=("forced", "auto", "reuse"),
        help="Default: forced when tile-k is given, otherwise auto.",
    )
    parser.add_argument("--timing-mode", choices=("graph", "events"), default="graph")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    args.mode = args.mode or ("forced" if args.tile_k is not None else "auto")
    if (args.mode == "forced") != (args.tile_k is not None):
        parser.error(
            "--mode forced requires --tile-k; auto/reuse require it to be omitted"
        )
    if args.request_file is not None and args.request_file.resolve() in {
        args.log_file.resolve(),
        args.output.resolve(),
    }:
        parser.error("--request-file must differ from --log-file and --output")
    if args.request_file is None:
        args.dtype = args.dtype or "fp16"
    args.reference_device = args.reference_device or (
        "gpu" if args.request_file is not None else "cpu"
    )
    if args.iterations < 1 or args.samples < 1 or args.warmup < 1 or args.device < 0:
        parser.error(
            "iterations, samples and warmup must be positive; device must be nonnegative"
        )
    if args.comgr_library is None:
        parser.error(
            "pass --comgr-library or ROCKE_COMGR_LIB to pin the packaging compiler"
        )
    if args.llvm_flavor not in ("llvm20", "llvm22"):
        parser.error("ROCKE_LLVM_FLAVOR must be llvm20 or llvm22")
    return args


def configure(args: argparse.Namespace) -> None:
    if args.rocke_root is None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "dnn-providers" / "hip-kernel-provider" / "rocke"
            if (candidate / "platform" / "python" / "rocke").is_dir():
                args.rocke_root = candidate
                break
    if args.rocke_root is not None:
        root = args.rocke_root.resolve()
        for path in (root / "library", root / "platform" / "python"):
            require(path.is_dir(), f"Missing rocKE source directory: {path}")
            sys.path.insert(0, str(path))
    args.plugin_path = args.plugin_path.resolve()
    require(
        args.plugin_path.exists(), f"Plugin path does not exist: {args.plugin_path}"
    )
    if args.descriptor_root is None:
        parent = (
            args.plugin_path if args.plugin_path.is_dir() else args.plugin_path.parent
        )
        args.descriptor_root = parent / "arch_content" / "hip-kernel-provider"
    args.descriptor_root = args.descriptor_root.resolve()
    require(
        args.descriptor_root.is_dir(),
        f"Missing descriptor tree: {args.descriptor_root}",
    )
    require(
        not os.environ.get("HIPDNN_DESCRIPTOR_DIR"),
        "Unset HIPDNN_DESCRIPTOR_DIR so the installed plugin resolves its own descriptors",
    )
    args.comgr_library = args.comgr_library.resolve()
    require(
        args.comgr_library.is_file(), f"Missing COMGR library: {args.comgr_library}"
    )
    for name in ("cache_dir", "log_file", "output"):
        setattr(args, name, getattr(args, name).resolve())
    require(args.log_file != args.output, "Log and report paths must differ")
    if args.request_file is not None:
        args.request_file = args.request_file.resolve()
        require(
            args.request_file.is_file(), f"Missing request file: {args.request_file}"
        )
        require(
            args.request_file not in (args.log_file, args.output),
            "Request, log, and report paths must differ",
        )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text("", encoding="utf-8")
    os.environ["HIPDNN_CACHE_DIR"] = str(args.cache_dir)
    os.environ["HIPDNN_DISABLE_CACHE"] = "0"
    os.environ["HIPDNN_FORCE_BENCHMARKING"] = "0" if args.mode == "forced" else "1"
    os.environ["HIPDNN_LOG_LEVEL"] = "info"
    os.environ["HIPDNN_LOG_FILE"] = str(args.log_file)
    os.environ["ROCKE_COMGR_LIB"] = str(args.comgr_library)
    os.environ["ROCKE_LLVM_FLAVOR"] = args.llvm_flavor
    os.environ["ROCKE_BACKEND"] = "python"


class LogCapture:
    """Synchronous public backend callback; the normal file sink is asynchronous.

    The Python bindings do not yet expose this API. Keep the ctypes callback and
    user token alive until explicit unregistration. Only retain matching strings;
    file I/O remains in hipDNN's normal asynchronous sink outside this callback.
    """

    def __init__(self, library: Path | None):
        name = (
            str(library.resolve())
            if library
            else (
                ctypes.util.find_library("hipdnn_backend")
                or ("hipdnn_backend.dll" if os.name == "nt" else "libhipdnn_backend.so")
            )
        )
        self.library = ctypes.CDLL(name)
        self.stage = "build"
        self.messages: list[dict] = []
        self.received_count = 0
        self._callback_registered = False
        callback_type = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
        )
        self.callback = callback_type(self._receive)
        self.token = ctypes.c_int(0)
        self.pointer = ctypes.cast(ctypes.pointer(self.token), ctypes.c_void_p)
        self.setter = self.library.hipdnnSetUserLogCallback_ext
        self.setter.argtypes = [
            callback_type,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.setter.restype = ctypes.c_int
        self.level_setter = self.library.hipdnnBackendSetGlobalLogLevel_ext
        self.level_setter.argtypes = [ctypes.c_int]
        self.level_setter.restype = ctypes.c_int
        self.level_getter = self.library.hipdnnBackendGetGlobalLogLevel_ext
        self.level_getter.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.level_getter.restype = ctypes.c_int

    def _receive(self, _user, severity, message):
        self.received_count += 1
        text = message.decode("utf-8", errors="replace") if message else ""
        if "ingestor:" in text:
            self.messages.append(
                {"stage": self.stage, "severity": severity, "message": text}
            )

    def _set_callback(self, enabled: bool) -> None:
        if enabled == self._callback_registered:
            return
        # Mode SYNC=0, INFO=0, OFF=4, SUCCESS=0 (public ABI).
        # OFF waits for in-flight callbacks before removing this registration.
        require(
            self.setter(self.callback, 0 if enabled else 4, 0, self.pointer) == 0,
            f"Failed to {'register' if enabled else 'unregister'} the synchronous log callback",
        )
        self._callback_registered = enabled

    def _global_level(self) -> int:
        level = ctypes.c_int()
        require(
            self.level_getter(ctypes.byref(level)) == 0,
            "Failed to query the backend's global log level",
        )
        require(level.value in range(len(LOG_LEVELS)), "Unknown backend log level")
        return level.value

    def _set_global_level(self, level: int) -> None:
        require(
            self.level_setter(level) == 0,
            f"Failed to set the backend's global log level to {LOG_LEVELS[level]}",
        )

    @contextmanager
    def suspend_for_timing(self, state: dict):
        require(
            self._callback_registered,
            "Timing log suspension requires an active selection-log capture",
        )
        previous_level = self._global_level()
        state.update(
            {
                "scope": "warmup, graph capture/replays, and event sampling",
                "control_api": [
                    "hipdnnSetUserLogCallback_ext",
                    "hipdnnBackendSetGlobalLogLevel_ext",
                    "hipdnnBackendGetGlobalLogLevel_ext",
                ],
                "backend_global_level_before": LOG_LEVELS[previous_level],
                "sync_callback_registered_before": True,
                "restored": False,
                "frontend_logging": (
                    "The frontend has a separate severity cache without a Python setter; "
                    "its INFO message may be formatted but is dropped by the backend OFF gate."
                ),
            }
        )
        self._set_callback(False)
        callback_count = self.received_count
        try:
            # The public setter updates the backend and all loaded engine plugins.
            # Changing HIPDNN_LOG_LEVEL here would not update their initialized caches.
            self._set_global_level(4)
            during_level = self._global_level()
            state["backend_global_level_during"] = LOG_LEVELS[during_level]
            state["sync_callback_registered_during"] = self._callback_registered
            require(during_level == 4, "Backend logging remained enabled for timing")
            yield
            require(
                self.received_count == callback_count,
                "The selection-log callback ran during timing",
            )
        finally:
            pending_error = sys.exc_info()[1]
            state["callbacks_during_timing"] = self.received_count - callback_count
            restoration_errors = []
            try:
                self._set_global_level(previous_level)
                restored_level = self._global_level()
                state["backend_global_level_after"] = LOG_LEVELS[restored_level]
                require(
                    restored_level == previous_level,
                    "The backend's global log level was not restored",
                )
            except Exception as error:  # noqa: BLE001 - Preserve the timing failure.
                restoration_errors.append(
                    f"Global logging: {type(error).__name__}: {error}"
                )
            try:
                self._set_callback(True)
                state["sync_callback_registered_after"] = self._callback_registered
            except Exception as error:  # noqa: BLE001 - Preserve the timing failure.
                restoration_errors.append(
                    f"Selection callback: {type(error).__name__}: {error}"
                )
            state["restored"] = not restoration_errors
            if restoration_errors:
                state["restoration_errors"] = restoration_errors
                if pending_error is None:
                    raise RuntimeError("; ".join(restoration_errors))

    def __enter__(self):
        self._set_callback(True)
        return self

    def __exit__(self, _kind, error, _traceback):
        # OFF=4 atomically unregisters; after this returns, Python may release callback state.
        try:
            self._set_callback(False)
        except Exception as cleanup_error:  # noqa: BLE001 - Preserve the failure.
            message = f"Failed to unregister the backend log callback: {cleanup_error}"
            self.messages.append(
                {"stage": "cleanup", "severity": 3, "message": message}
            )
            if error is None:
                raise RuntimeError(message)

    def text(self, stage: str | None = None) -> str:
        return "\n".join(
            entry["message"]
            for entry in self.messages
            if stage is None or entry["stage"] == stage
        )


def kernel_objects(value):
    if isinstance(value, dict):
        if "kernel_source" in value and "metadata" in value and "id" in value:
            yield value
        for child in value.values():
            yield from kernel_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from kernel_objects(child)


def packaged_candidates(root: Path, specs: dict) -> dict:
    found: dict[int, dict] = {}
    for path in sorted(root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for kernel in kernel_objects(value):
            metadata = kernel["metadata"]
            tile_k = metadata.get("tile_k")
            if tile_k not in specs or any(
                metadata.get(k) != v for k, v in asdict(specs[tile_k]).items()
            ):
                continue
            source = kernel["kernel_source"]
            require(
                source.get("kind") == "kpack", f"Descriptor is not packaged: {path}"
            )
            require(
                source.get("symbol") == specs[tile_k].kernel_name(),
                f"Packaged symbol disagrees with the adapter for tile_k={tile_k}",
            )
            item = {
                "id": kernel["id"].lower(),
                "name": kernel.get("name"),
                "metadata": metadata,
                "source": source,
                "descriptor": str(path),
                "descriptor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            require(
                tile_k not in found or found[tile_k]["id"] == item["id"],
                f"Multiple packaged descriptors have the same requested configuration: {tile_k}",
            )
            found[tile_k] = item
    require(
        set(found) == set(specs),
        f"Installed catalog is missing requested tile_k variants: {sorted(set(specs) - set(found))}",
    )
    require(
        len({item["id"] for item in found.values()}) == len(found),
        "Variants must have distinct descriptor UUIDs",
    )
    return found


def cached_ranking(cache_dir: Path, candidate_ids: set[str]) -> dict | None:
    result = None
    for path in sorted(cache_dir.rglob("winners.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not isinstance(
                record.get("entries"), list
            ):
                continue
            ids = {entry.get("kernel_id", "").lower() for entry in record["entries"]}
            if candidate_ids.issubset(ids):
                result = {
                    "path": str(path),
                    "record": record,
                    "record_sha256": hashlib.sha256(line.encode()).hexdigest(),
                }
    return result


def load_request(args: argparse.Namespace, request_type):
    if args.request_file is None:
        return request_type(
            N=2,
            C=32,
            K=32,
            Hi=14,
            Wi=14,
            Y=3,
            X=3,
            pad_h=1,
            pad_w=1,
            dtype=args.dtype,
            arch="gfx950",
        ), {"kind": "smoke"}

    raw = args.request_file.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), "Request file must contain one JSON object")
    parameters = {key: item for key, item in value.items() if not key.startswith("_")}
    request = request_type(**parameters)
    request = request_type(**request.normalized())
    require(
        args.dtype is None or args.dtype == request.dtype,
        f"Explicit dtype {args.dtype!r} disagrees with request dtype {request.dtype!r}",
    )
    args.dtype = request.dtype
    return request, {
        "kind": "request_file",
        "path": str(args.request_file.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": value.get("_provenance"),
        "annotations": {
            key: item
            for key, item in value.items()
            if key.startswith("_") and key != "_provenance"
        },
    }


def tensor_geometry(problem) -> dict:
    geometry = {
        "x_dims": [problem.N, problem.C, problem.Hi, problem.Wi],
        "w_dims": [problem.K, problem.C, problem.Y, problem.X],
        "y_dims": [problem.N, problem.K, problem.Ho, problem.Wo],
        "stride": [problem.sH, problem.sW],
        "padding": [problem.pH, problem.pW],
        "dilation": [problem.dH, problem.dW],
    }
    for operand in ("x", "w", "y"):
        _, channels, height, width = geometry[f"{operand}_dims"]
        geometry[f"{operand}_strides"] = [
            height * width * channels,
            1,
            width * channels,
            channels,
        ]
    return geometry


def build_graph(
    hipdnn,
    handle,
    dtype,
    capture,
    stage: str,
    tile_k: int | None,
    geometry: dict,
    *,
    smoke: bool,
):
    capture.stage = stage
    graph = hipdnn.Graph()
    graph.set_name("gfx950_conv_fwd_probe")
    graph.set_io_data_type(dtype)
    graph.set_intermediate_data_type(dtype)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    x = hipdnn.Tensor.create(geometry["x_dims"], dtype)
    x.set_uid(1)
    x.set_stride(geometry["x_strides"])
    w = hipdnn.Tensor.create(geometry["w_dims"], dtype)
    w.set_uid(2)
    w.set_stride(geometry["w_strides"])
    attrs = hipdnn.ConvFpropAttributes()
    attrs.set_padding(geometry["padding"])
    attrs.set_stride(geometry["stride"])
    attrs.set_dilation(geometry["dilation"])
    attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
    attrs.set_convolution_mode(hipdnn.ConvolutionMode.CROSS_CORRELATION)
    y = graph.conv_fprop(x, w, attrs)
    y.set_uid(3)
    y.set_output(True)
    y.set_data_type(dtype)
    if not smoke:
        # A unit input channel makes NCHW/NHWC order ambiguous to stride inference.
        y.set_stride(geometry["y_strides"])
    check(graph.validate(), "graph validation")
    check(graph.build_operation_graph(handle), "operation graph build")
    engine_ids = graph.get_ranked_engine_ids()
    ids = [
        engine_id
        for engine_id in engine_ids
        if handle.engine_id_to_name(engine_id) == ENGINE_NAME
    ]
    require(
        len(ids) == 1,
        f"Exact engine {ENGINE_NAME!r} is not uniquely offered: {engine_ids}",
    )
    engine_id = ids[0]
    knobs = {knob.knob_id: knob for knob in graph.get_knobs_for_engine(engine_id)}
    require(
        "tile_k" in knobs and BENCHMARK_KNOB in knobs,
        "Required engine knobs are missing",
    )
    choices = set(knobs["tile_k"].constraint.valid_values)
    if smoke:
        require(
            choices == {64, 128},
            "Catalog does not expose exactly the two packaged tile_k choices",
        )
    elif tile_k is None:
        require(
            choices == {64, 128},
            f"Auto/reuse need both packaged tile_k variants; this request offers {sorted(choices)}",
        )
    else:
        require(
            tile_k in choices, f"Requested tile_k={tile_k} is not offered: {choices}"
        )
    settings = [hipdnn.KnobSetting(BENCHMARK_KNOB, 0 if tile_k is not None else 1)]
    if tile_k is not None:
        setting = hipdnn.KnobSetting("tile_k", tile_k)
        check(knobs["tile_k"].validate(setting), "tile_k validation")
        settings.append(setting)
    check(
        graph.create_execution_plan_ext(engine_id, settings),
        "exact engine/knob selection",
    )
    check(graph.check_support(), "plan support")
    check(graph.build_plans(), "plan build")
    require(
        graph.get_execution_plan_engine_id() == engine_id,
        "Plan selected a different engine",
    )
    require(
        graph.get_plan_name(handle) == ENGINE_NAME,
        "Plan reports a different engine name",
    )
    require(
        graph.get_workspace_size() == 0,
        "This packaged convolution must not need workspace",
    )
    require(
        y.get_dim() == geometry["y_dims"] and y.get_stride() == geometry["y_strides"],
        "Frontend inferred a different output shape or layout",
    )
    return graph, {
        "engine_id": engine_id,
        "engine_name": ENGINE_NAME,
        "requested_knobs": {s.knob_id: s.value for s in settings},
        "tile_k_default": knobs["tile_k"].default_value,
        "tile_k_choices": sorted(choices),
    }


def make_reference(torch, torch_nn, x, w, geometry: dict):
    kwargs = {
        "stride": geometry["stride"],
        "padding": geometry["padding"],
        "dilation": geometry["dilation"],
        "groups": 1,
    }
    info = {
        "function": "torch.nn.functional.conv2d",
        "device": str(x.device),
        "accumulation_dtype": "float32",
        "rounded_to_dtype": str(x.dtype),
    }
    if x.device.type == "cuda":
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(
                enabled=True, benchmark=False, allow_tf32=False
            ):
                result = torch_nn.conv2d(x.float(), w.float(), **kwargs)
                info.update(
                    {
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "cudnn_enabled": torch.backends.cudnn.enabled,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    }
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    else:
        result = torch_nn.conv2d(x.float(), w.float(), **kwargs)
    return result.to(x.dtype).float(), info


def validate_output(torch, output, reference, dtype: str) -> dict:
    # Large workloads stay on the GPU; smoke retains its independent CPU comparison.
    actual = output.detach().to(device=reference.device, dtype=torch.float32)
    require(
        bool(torch.isfinite(actual).all()),
        "Output contains NaN/Inf or an unwritten element",
    )
    atol, rtol = 0.001, (0.002 if dtype == "fp16" else 0.016)
    difference = (actual - reference).abs()
    reference_location = "HIP GPU" if reference.device.type == "cuda" else "CPU"
    require(
        bool((difference <= atol + rtol * reference.abs()).all()),
        f"Convolution differs from independent torch {reference_location} FP32 reference; max_abs={difference.max().item()}",
    )
    return {
        "passed": True,
        "reference": f"torch.nn.functional.conv2d on {reference_location}, FP32 accumulation",
        "max_abs_error": difference.max().item(),
        "atol": atol,
        "rtol": rtol,
    }


def measure(torch, launch, stream, args, release) -> dict:
    for _ in range(args.warmup):
        launch()
    stream.synchronize()
    release(int(stream.cuda_stream))
    replay = launch
    divisor = args.iterations
    graph = None
    if args.timing_mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.iterations):
                launch()
        replay = graph.replay
        for _ in range(2):
            replay()
        stream.synchronize()
    samples = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if graph is None:
            for _ in range(args.iterations):
                replay()
        else:
            replay()
        stop.record(stream)
        stop.synchronize()
        elapsed = start.elapsed_time(stop) / divisor
        require(
            math.isfinite(elapsed) and elapsed > 0,
            "GPU event timer returned an invalid duration",
        )
        samples.append(elapsed)
        release(int(stream.cuda_stream))
    return {
        "method": (
            "HIP graph replay with GPU events"
            if graph is not None
            else "GPU events around Python submissions (includes host enqueue gaps)"
        ),
        "iterations_per_sample": args.iterations,
        "samples_ms_per_call": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "warmup": args.warmup,
        "first_execution_and_search_excluded": True,
    }


def selected_uuid(text: str, *, benchmark: bool = False) -> str:
    prefix = (
        "benchmarking selected kernel "
        if benchmark
        else re.escape(f"engine '{ENGINE_NAME}' ") + r"(?:selected|served) kernel "
    )
    matches = re.findall(prefix + f"({UUID_PATTERN})", text)
    require(len(matches) == 1, f"Expected one selected kernel UUID in {text!r}")
    return matches[0].lower()


def run(args: argparse.Namespace, report: dict) -> dict:
    # Import torch before rocKE so both use the same initialized HIP runtime.
    # isort: off
    import torch
    import torch.nn.functional as torch_nn

    # isort: on

    import hipdnn_frontend as hipdnn
    from builders.common.convolution_forward import (
        build_gfx950_conv_fwd,
        gfx950_conv_fwd_spec_for_request,
    )
    from dispatch.grouped_convolution import ConvGroupedRequest
    from rocke import compile_kernel
    from rocke.core.ir_print import print_ir
    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
    from rocke.runtime.comgr import resolved_lib_path
    from rocke.runtime.launcher import (
        KernelLauncher,
        LaunchConfig,
        release_retained_for_stream,
    )

    require(
        torch.version.hip is not None and torch.cuda.is_available(),
        "A ROCm PyTorch GPU is required",
    )
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    arch = getattr(properties, "gcnArchName", "")
    require(arch.split(":")[0] == "gfx950", f"Expected gfx950, got {arch!r}")
    request, source = load_request(args, ConvGroupedRequest)
    report.update(
        {
            "dtype": args.dtype,
            "request": request.normalized(),
            "request_source": source,
            "reference_device": args.reference_device,
            "seed": args.seed,
        }
    )
    smoke = args.request_file is None
    # A forced request builds only its own spec; auto/reuse rank every packaged variant.
    tiles = (64, 128) if (smoke or args.tile_k is None) else (args.tile_k,)
    specs = {
        tile: gfx950_conv_fwd_spec_for_request(request, tile_k=tile) for tile in tiles
    }
    report["requested_specs"] = {tile: asdict(spec) for tile, spec in specs.items()}
    problem = next(iter(specs.values())).to_problem()
    geometry = tensor_geometry(problem)
    candidates = packaged_candidates(args.descriptor_root, specs)
    candidate_ids = {candidate["id"] for candidate in candidates.values()}
    before = cached_ranking(args.cache_dir, candidate_ids)
    report.update(
        {
            "engine_name": ENGINE_NAME,
            "mode": args.mode,
            "dtype": args.dtype,
            "pid": os.getpid(),
            "arch": arch,
            "device": args.device,
            "gpu_name": properties.name,
            "torch_version": torch.__version__,
            "rocm_version": torch.version.hip,
            "problem": {key: getattr(problem, key) for key in SMOKE},
            "tensor_geometry": geometry,
            "cache_before": before,
            "catalog": candidates,
            "toolchain": {
                "lowering_backend": "python",
                "llvm_flavor": args.llvm_flavor,
                "requested_comgr": str(args.comgr_library),
            },
            "plugin_path": str(args.plugin_path),
            "descriptor_root": str(args.descriptor_root),
            "cache_dir": str(args.cache_dir),
            "log_file": str(args.log_file),
        }
    )
    require(
        args.mode != "auto" or before is None,
        "Auto mode requires a fresh cache for this graph",
    )
    require(
        args.mode != "reuse" or before is not None,
        "Reuse mode requires a complete earlier ranking",
    )
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    frontend_dtype = (
        hipdnn.DataType.HALF if args.dtype == "fp16" else hipdnn.DataType.BFLOAT16
    )
    generator = torch.Generator().manual_seed(args.seed)
    x_cpu = (
        torch.empty(geometry["x_dims"]).uniform_(-1, 1, generator=generator).to(dtype)
    )
    w_cpu = (
        torch.empty(geometry["w_dims"]).uniform_(-1, 1, generator=generator).to(dtype)
    )
    reference = None
    if args.reference_device == "cpu":
        reference, report["reference"] = make_reference(
            torch, torch_nn, x_cpu, w_cpu, geometry
        )
    stream = torch.cuda.Stream(device=args.device)
    hipdnn.set_engine_plugin_paths(
        [str(args.plugin_path)], hipdnn.PluginLoadingMode.ABSOLUTE
    )
    handle = hipdnn.create_handle()
    graph = recreated = launcher = None
    try:
        hipdnn.set_stream(handle, int(stream.cuda_stream))
        with torch.cuda.stream(stream), LogCapture(args.backend_library) as capture:
            # Keep this list in the shared report even if a later check fails.
            report["selection_evidence"] = capture.messages
            x = x_cpu.to(device="cuda", memory_format=torch.channels_last)
            w = w_cpu.to(device="cuda", memory_format=torch.channels_last)
            if args.reference_device == "gpu":
                reference, report["reference"] = make_reference(
                    torch, torch_nn, x, w, geometry
                )
            output = torch.empty(
                geometry["y_dims"],
                device="cuda",
                dtype=dtype,
                memory_format=torch.channels_last,
            )
            output.fill_(float("nan"))
            buffers = {1: x.data_ptr(), 2: w.data_ptr(), 3: output.data_ptr()}
            build_start = time.perf_counter()
            graph, plan = build_graph(
                hipdnn,
                handle,
                frontend_dtype,
                capture,
                "build",
                args.tile_k,
                geometry,
                smoke=smoke,
            )
            report["plan_build_wall_ms"] = (time.perf_counter() - build_start) * 1000
            report["plan"] = plan

            def integrated():
                check(graph.execute(handle, buffers, 0), "hipDNN execute")

            stream.synchronize()
            capture.stage = "first_execute"
            first_start = time.perf_counter()
            integrated()
            stream.synchronize()
            report["first_execute_wall_ms"] = (time.perf_counter() - first_start) * 1000
            report["integrated_correctness"] = validate_output(
                torch, output, reference, args.dtype
            )
            if args.mode == "auto":
                require(
                    "will benchmark 2 candidate(s)" in capture.text("build"),
                    "The first plan did not prepare exactly two benchmark candidates",
                )
                winner = selected_uuid(capture.text("first_execute"), benchmark=True)
            else:
                winner = selected_uuid(capture.text("build"))
                require(
                    "benchmarking selected kernel" not in capture.text(),
                    "Unexpected new search",
                )
            matching = [
                tile
                for tile, candidate in candidates.items()
                if candidate["id"] == winner
            ]
            require(
                len(matching) == 1,
                "Selected UUID does not identify an installed requested variant",
            )
            winner_tile = matching[0]
            require(
                args.tile_k is None or winner_tile == args.tile_k,
                "The selected descriptor does not implement the forced tile_k",
            )
            report["selected_kernel_id"] = winner
            report["selected_tile_k"] = winner_tile
            after = cached_ranking(args.cache_dir, candidate_ids)
            if args.mode != "forced":
                require(after is not None, "No complete measured ranking was persisted")
                entries = after["record"]["entries"]
                require(
                    len(entries) == 2 and entries[0]["kernel_id"].lower() == winner,
                    "Selected kernel differs from the cached winner",
                )
                times = [entry["time_ms"] for entry in entries]
                require(
                    all(math.isfinite(t) and t > 0 for t in times)
                    and times == sorted(times),
                    "The winner is not the fastest measured valid candidate",
                )
                if args.mode == "reuse":
                    require(
                        after["record_sha256"] == before["record_sha256"],
                        "Reuse changed the persisted measurements",
                    )
                    require(
                        "from a benchmarked record" in capture.text("build"),
                        "The restarted process did not reuse its stored winner",
                    )
            report["cache_after"] = after
            capture.stage = "repeat_execute"
            output.fill_(float("nan"))
            integrated()
            stream.synchronize()
            report["repeat_correctness"] = validate_output(
                torch, output, reference, args.dtype
            )
            require(
                "benchmarking selected kernel" not in capture.text("repeat_execute"),
                "The same plan benchmarked again on its second execution",
            )
            capture.stage = "direct_compile"
            spec = specs[winner_tile]
            direct_kernel = build_implicit_gemm_conv(
                spec.to_instance_spec(), arch="gfx950"
            )
            adapter_ir = print_ir(build_gfx950_conv_fwd(spec, arch="gfx950"))
            require(
                print_ir(direct_kernel) == adapter_ir,
                "Adapter changed the original builder's IR",
            )
            artifact = compile_kernel(direct_kernel, arch="gfx950", backend="python")
            report["toolchain"]["resolved_comgr"] = resolved_lib_path()
            require(
                Path(resolved_lib_path()).resolve() == args.comgr_library,
                "Direct rocKE used a different COMGR library than requested",
            )
            launcher = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=conv_args_signature(args.dtype),
            )
            direct_output = torch.empty_like(output)
            direct_output.fill_(float("nan"))
            values = {
                "A": x,
                "B": w,
                "D": direct_output,
                "A_bytes": x.numel() * x.element_size(),
                "B_bytes": w.numel() * w.element_size(),
                "D_bytes": direct_output.numel() * direct_output.element_size(),
            }
            config = LaunchConfig(
                grid=spec.grid(),
                block=spec.block(),
                stream=int(stream.cuda_stream),
                fence=False,
            )

            def direct():
                launcher(values, config=config)

            direct()
            done = torch.cuda.Event()
            done.record(stream)
            done.synchronize()
            release_retained_for_stream(int(stream.cuda_stream))
            report["direct_correctness"] = validate_output(
                torch, direct_output, reference, args.dtype
            )
            report["direct"] = {
                "spec": asdict(spec),
                "symbol": artifact.kernel_name,
                "adapter_ir_matches_original": True,
                "llvm_sha256": hashlib.sha256(artifact.llvm_text.encode()).hexdigest(),
                "hsaco_sha256": hashlib.sha256(artifact.hsaco).hexdigest(),
                "grid": spec.grid(),
                "block": spec.block(),
            }
            # Finish both compilation paths and correctness checks before timing either.
            capture.stage = "steady_state"
            report["timing_logging"] = {}
            with capture.suspend_for_timing(report["timing_logging"]):
                report["integrated_timing"] = measure(
                    torch, integrated, stream, args, release_retained_for_stream
                )
                report["direct_timing"] = measure(
                    torch, direct, stream, args, release_retained_for_stream
                )
            integrated_ms = report["integrated_timing"]["median_ms"]
            direct_ms = report["direct_timing"]["median_ms"]
            report["comparison"] = {
                "same_spec": True,
                "same_gpu": True,
                "integrated_over_direct": integrated_ms / direct_ms,
                "integrated_minus_direct_ms": integrated_ms - direct_ms,
                "order": ["integrated", "direct"],
                "compilation_before_timing": True,
                "timing_mode": args.timing_mode,
            }

            if args.mode != "forced":
                recreated, recreated_plan = build_graph(
                    hipdnn,
                    handle,
                    frontend_dtype,
                    capture,
                    "recreated_plan",
                    None,
                    geometry,
                    smoke=smoke,
                )
                require(
                    selected_uuid(capture.text("recreated_plan")) == winner,
                    "A recreated plan selected a different winner",
                )
                require(
                    "from a benchmarked record" in capture.text("recreated_plan")
                    and "will benchmark" not in capture.text("recreated_plan"),
                    "A recreated plan did not reuse the complete ranking",
                )
                output.fill_(float("nan"))
                check(recreated.execute(handle, buffers, 0), "recreated-plan execute")
                stream.synchronize()
                report["recreated_plan"] = recreated_plan
                report["recreated_correctness"] = validate_output(
                    torch, output, reference, args.dtype
                )
    finally:
        pending_error = sys.exc_info()[1]
        cleanup_errors = []
        try:
            stream.synchronize()
            release_retained_for_stream(int(stream.cuda_stream))
        except Exception as error:  # noqa: BLE001 - Preserve failures during cleanup.
            cleanup_errors.append(f"Stream cleanup: {type(error).__name__}: {error}")
        # Drop plan and launcher references before destroying their backend handle.
        graph = recreated = launcher = None
        try:
            hipdnn.destroy_handle(handle)
        except Exception as error:  # noqa: BLE001 - Preserve failures during cleanup.
            cleanup_errors.append(f"Handle cleanup: {type(error).__name__}: {error}")
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            if pending_error is None:
                raise RuntimeError("; ".join(cleanup_errors))
    report["status"] = "passed"
    return report


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = {
        "status": "failed",
        "mode": args.mode,
        "dtype": args.dtype,
        "requested_tile_k": args.tile_k,
        "pid": os.getpid(),
    }
    exit_code = 1
    try:
        configure(args)
        run(args, report)
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - Write failures to the report.
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{report['status']}: report written to {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
