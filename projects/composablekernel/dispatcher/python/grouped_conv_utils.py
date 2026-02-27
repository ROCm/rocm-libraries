#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Grouped Convolution Dispatcher Utilities

Typed Python API for grouped convolution kernels, matching the patterns from
the old conv_utils.py and the GEMM ctypes_utils.py.

Classes:
    GroupedConvKernelConfig  - Kernel configuration (tile, wave, pipeline, arch)
    GroupedConvProblem       - Runtime problem specification (N,C,K,H,W,etc.)
    GroupedConvProblemC      - ctypes struct matching C++ ConvProblemC
    GroupedConvDispatcherLib - Wrapper for libdispatcher_conv_lib.so
    GpuGroupedConvRunner    - High-level GPU execution runner
    GroupedConvResult        - Result of GPU execution (output, time, tflops)
    GroupedConvRegistry      - Collection of kernel configs with JSON export

Usage:
    from grouped_conv_utils import (
        GroupedConvKernelConfig,
        GroupedConvProblem,
        GpuGroupedConvRunner,
    )

    config = GroupedConvKernelConfig(variant="forward", ndim_spatial=2)
    problem = GroupedConvProblem(N=1, C=64, K=128, Hi=28, Wi=28, Y=3, X=3,
                                 stride_h=1, pad_h=1, direction="forward")
    runner = GpuGroupedConvRunner()
    if runner.is_available():
        result = runner.run(input_np, weight_np, problem)
        print(f"Time: {result.time_ms:.4f} ms, TFLOPS: {result.tflops:.2f}")
"""

import ctypes
import json
import copy
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dispatcher_common import (
    ValidationResultBase,
    auto_correct_trait,
    auto_correct_wave,
    get_arch_filter_data,
    validate_trait_combo,
    validate_wave_config,
    validate_warp_tile_config,
)


# =============================================================================
# Constants
# =============================================================================

VALID_VARIANTS = ("forward", "bwd_data", "bwd_weight")
VALID_NDIM_SPATIAL = (1, 2, 3)
BACKWARD_VARIANTS = ("bwd_data", "bwd_weight")
BACKWARD_PIPELINES = ("compv3", "mem")

VARIANT_ALIASES = {
    "2d_fwd": "forward",
    "2d_bwdd": "bwd_data",
    "2d_bwdw": "bwd_weight",
    "fwd": "forward",
    "bwdd": "bwd_data",
    "bwdw": "bwd_weight",
}

DIRECTION_MAP = {"forward": 0, "bwd_data": 1, "bwd_weight": 2}


def _resolve_variant(v: str) -> str:
    return VARIANT_ALIASES.get(v, v)


# =============================================================================
# GroupedConvDataType
# =============================================================================


class GroupedConvDataType(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP8 = "fp8"
    BF8 = "bf8"
    INT8 = "int8"


# =============================================================================
# GroupedConvKernelConfig
# =============================================================================


@dataclass
class GroupedConvKernelConfig:
    """Complete kernel configuration for grouped convolution.

    Captures all parameters needed to identify and run a specific kernel.
    Mirrors the C++ GroupedConvSignature + GroupedConvAlgorithm.
    """

    # What: signature
    variant: str = "forward"
    ndim_spatial: int = 2
    dtype: str = "fp16"
    layout: str = "nhwgc"
    arch: str = "gfx942"

    # How: algorithm - tile shape
    tile_m: int = 1
    tile_n: int = 128
    tile_k: int = 128

    # How: wave config
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1

    # How: warp tile
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    # How: pipeline traits
    pipeline: str = "compv4"
    epilogue: str = "cshuffle"
    scheduler: str = "intrawave"

    # Padding (enables arbitrary problem sizes)
    pad_m: bool = True
    pad_n: bool = True
    pad_k: bool = True

    def __post_init__(self):
        self.variant = _resolve_variant(self.variant)
        if self.variant in BACKWARD_VARIANTS and self.pipeline not in BACKWARD_PIPELINES:
            self.pipeline = "compv3"

    @property
    def tile_str(self) -> str:
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}"

    @property
    def wave_str(self) -> str:
        return f"{self.wave_m}x{self.wave_n}x{self.wave_k}"

    @property
    def warp_str(self) -> str:
        return f"{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}"

    @property
    def name(self) -> str:
        return (f"grouped_conv_{self.variant}_{self.dtype}_{self.ndim_spatial}d_"
                f"{self.tile_str}_{self.pipeline}")

    def to_dict(self) -> dict:
        """Convert to legacy dict format for codegen compatibility."""
        return {
            "tile_config": {
                "tile_m": [self.tile_m], "tile_n": [self.tile_n], "tile_k": [self.tile_k],
                "wave_m": [self.wave_m], "wave_n": [self.wave_n], "wave_k": [self.wave_k],
                "warp_tile_m": [self.warp_tile_m], "warp_tile_n": [self.warp_tile_n],
                "warp_tile_k": [self.warp_tile_k],
            },
            "trait_config": {
                "pipeline": [self.pipeline], "epilogue": [self.epilogue],
                "scheduler": [self.scheduler],
                "pad_m": [self.pad_m], "pad_n": [self.pad_n], "pad_k": [self.pad_k],
            },
            "variant": self.variant, "ndim_spatial": self.ndim_spatial,
            "arch": self.arch, "layout": self.layout, "dtype": self.dtype,
        }

    def to_json_obj(self) -> dict:
        """Serializable dict for JSON export."""
        return {
            "name": self.name,
            "signature": {
                "variant": self.variant, "dtype": self.dtype,
                "ndim_spatial": self.ndim_spatial, "layout": self.layout,
            },
            "algorithm": {
                "tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "wave": self.wave_str, "warp": self.warp_str,
                "pipeline": self.pipeline, "epilogue": self.epilogue,
                "scheduler": self.scheduler,
            },
            "arch": self.arch,
        }

    def print_config(self, indent: str = "  "):
        print(f"{indent}GroupedConvKernelConfig:")
        print(f"{indent}  Variant:  {self.variant} {self.ndim_spatial}D")
        print(f"{indent}  Dtype:    {self.dtype}")
        print(f"{indent}  Layout:   {self.layout}")
        print(f"{indent}  Arch:     {self.arch}")
        print(f"{indent}  Tile:     {self.tile_str}")
        print(f"{indent}  Wave:     {self.wave_str}")
        print(f"{indent}  Warp:     {self.warp_str}")
        print(f"{indent}  Pipeline: {self.pipeline}/{self.scheduler}/{self.epilogue}")


# =============================================================================
# GroupedConvProblem
# =============================================================================


@dataclass
class GroupedConvProblem:
    """Runtime convolution problem specification.

    Describes the actual sizes of a convolution to be computed.
    Matches the old ConvProblem from conv_utils.py.
    """

    N: int = 1
    C: int = 64
    K: int = 128
    G: int = 1

    Hi: int = 28
    Wi: int = 28
    Di: int = 1

    Y: int = 3
    X: int = 3
    Z: int = 1

    stride_h: int = 1
    stride_w: int = 1
    stride_d: int = 1

    pad_h: int = 0
    pad_w: int = 0
    pad_d: int = 0

    dilation_h: int = 1
    dilation_w: int = 1
    dilation_d: int = 1

    direction: str = "forward"

    @property
    def Ho(self) -> int:
        eff_y = (self.Y - 1) * self.dilation_h + 1
        return (self.Hi + 2 * self.pad_h - eff_y) // self.stride_h + 1

    @property
    def Wo(self) -> int:
        eff_x = (self.X - 1) * self.dilation_w + 1
        return (self.Wi + 2 * self.pad_w - eff_x) // self.stride_w + 1

    @property
    def Do(self) -> int:
        eff_z = (self.Z - 1) * self.dilation_d + 1
        return (self.Di + 2 * self.pad_d - eff_z) // self.stride_d + 1

    @property
    def is_3d(self) -> bool:
        return self.Di > 1 or self.Z > 1

    @property
    def ndim_spatial(self) -> int:
        return 3 if self.is_3d else 2

    @property
    def flops(self) -> float:
        """Total FLOPs for this convolution (any direction, same count)."""
        c_per_group = self.C // self.G
        if self.is_3d:
            return (2.0 * self.N * self.K * self.Do * self.Ho * self.Wo
                    * c_per_group * self.Z * self.Y * self.X)
        return 2.0 * self.N * self.K * self.Ho * self.Wo * c_per_group * self.Y * self.X

    @property
    def gflops(self) -> float:
        return self.flops / 1e9

    def input_shape(self) -> tuple:
        """NHWGC or NDHWGC layout."""
        c_per_g = self.C // self.G
        if self.is_3d:
            return (self.N, self.Di, self.Hi, self.Wi, self.G, c_per_g)
        return (self.N, self.Hi, self.Wi, self.G, c_per_g)

    def weight_shape(self) -> tuple:
        """GKYXC or GKZYXC layout."""
        c_per_g = self.C // self.G
        k_per_g = self.K // self.G
        if self.is_3d:
            return (self.G, k_per_g, self.Z, self.Y, self.X, c_per_g)
        return (self.G, k_per_g, self.Y, self.X, c_per_g)

    def output_shape(self) -> tuple:
        """NHWGK or NDHWGK layout."""
        k_per_g = self.K // self.G
        if self.is_3d:
            return (self.N, self.Do, self.Ho, self.Wo, self.G, k_per_g)
        return (self.N, self.Ho, self.Wo, self.G, k_per_g)

    def print_problem(self, indent: str = "  "):
        dim_str = "3D" if self.is_3d else "2D"
        print(f"{indent}GroupedConvProblem ({dim_str} {self.direction}):")
        print(f"{indent}  Batch: N={self.N}, G={self.G}")
        print(f"{indent}  Channels: C={self.C}, K={self.K}")
        if self.is_3d:
            print(f"{indent}  Input:  Di={self.Di}, Hi={self.Hi}, Wi={self.Wi}")
            print(f"{indent}  Filter: Z={self.Z}, Y={self.Y}, X={self.X}")
            print(f"{indent}  Output: Do={self.Do}, Ho={self.Ho}, Wo={self.Wo}")
        else:
            print(f"{indent}  Input:  Hi={self.Hi}, Wi={self.Wi}")
            print(f"{indent}  Filter: Y={self.Y}, X={self.X}")
            print(f"{indent}  Output: Ho={self.Ho}, Wo={self.Wo}")
        print(f"{indent}  GFLOPs: {self.gflops:.2f}")


# =============================================================================
# GroupedConvProblemC (ctypes struct matching C++)
# =============================================================================


class GroupedConvProblemC(ctypes.Structure):
    """C structure matching ConvProblemC in conv_ctypes_lib.cpp."""

    _fields_ = [
        ("N", ctypes.c_int), ("G", ctypes.c_int),
        ("C", ctypes.c_int), ("K", ctypes.c_int),
        ("input_d", ctypes.c_int), ("input_h", ctypes.c_int), ("input_w", ctypes.c_int),
        ("filter_z", ctypes.c_int), ("filter_y", ctypes.c_int), ("filter_x", ctypes.c_int),
        ("stride_d", ctypes.c_int), ("stride_h", ctypes.c_int), ("stride_w", ctypes.c_int),
        ("pad_d", ctypes.c_int), ("pad_h", ctypes.c_int), ("pad_w", ctypes.c_int),
        ("dilation_d", ctypes.c_int), ("dilation_h", ctypes.c_int), ("dilation_w", ctypes.c_int),
        ("direction", ctypes.c_int),
    ]

    @classmethod
    def from_problem(cls, p: GroupedConvProblem) -> "GroupedConvProblemC":
        c = cls()
        c.N, c.G, c.C, c.K = p.N, p.G, p.C, p.K
        c.input_d, c.input_h, c.input_w = p.Di, p.Hi, p.Wi
        c.filter_z, c.filter_y, c.filter_x = p.Z, p.Y, p.X
        c.stride_d, c.stride_h, c.stride_w = p.stride_d, p.stride_h, p.stride_w
        c.pad_d, c.pad_h, c.pad_w = p.pad_d, p.pad_h, p.pad_w
        c.dilation_d, c.dilation_h, c.dilation_w = p.dilation_d, p.dilation_h, p.dilation_w
        c.direction = DIRECTION_MAP.get(p.direction, 0)
        return c


# =============================================================================
# GroupedConvResult
# =============================================================================


@dataclass
class GroupedConvResult:
    """Result of GPU convolution execution."""

    success: bool = False
    time_ms: float = 0.0
    tflops: float = 0.0
    output: Optional[np.ndarray] = None
    error: str = ""


# =============================================================================
# GroupedConvDispatcherLib
# =============================================================================


class GroupedConvDispatcherLib:
    """Wrapper for the compiled convolution dispatcher library.

    Provides Python interface to the C API in conv_ctypes_lib.cpp.
    """

    SEARCH_PATHS = [
        "build/examples/libdispatcher_conv_lib.so",
        "build/bindings/libdispatcher_conv_lib.so",
        "build/lib/libdispatcher_conv_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
        self._setup_functions()

    def _setup_functions(self):
        self._lib.conv_dispatcher_init.argtypes = []
        self._lib.conv_dispatcher_init.restype = ctypes.c_int
        self._lib.conv_dispatcher_cleanup.argtypes = []
        self._lib.conv_dispatcher_cleanup.restype = ctypes.c_int
        self._lib.conv_dispatcher_version.argtypes = []
        self._lib.conv_dispatcher_version.restype = ctypes.c_char_p
        self._lib.conv_dispatcher_has_kernels.argtypes = []
        self._lib.conv_dispatcher_has_kernels.restype = ctypes.c_int
        self._lib.conv_dispatcher_has_bwd_data.argtypes = []
        self._lib.conv_dispatcher_has_bwd_data.restype = ctypes.c_int
        self._lib.conv_dispatcher_has_bwd_weight.argtypes = []
        self._lib.conv_dispatcher_has_bwd_weight.restype = ctypes.c_int
        self._lib.conv_dispatcher_get_kernel_count.argtypes = []
        self._lib.conv_dispatcher_get_kernel_count.restype = ctypes.c_int
        self._lib.conv_dispatcher_get_kernel_name.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.conv_dispatcher_get_kernel_name.restype = ctypes.c_int
        self._lib.conv_dispatcher_is_supported.argtypes = [
            ctypes.POINTER(GroupedConvProblemC),
        ]
        self._lib.conv_dispatcher_is_supported.restype = ctypes.c_int
        self._lib.conv_dispatcher_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(GroupedConvProblemC), ctypes.c_void_p,
        ]
        self._lib.conv_dispatcher_run.restype = ctypes.c_float

    @classmethod
    def find(cls) -> Optional["GroupedConvDispatcherLib"]:
        """Search standard paths for the conv library."""
        root = Path(__file__).parent.parent
        for rel in cls.SEARCH_PATHS:
            path = root / rel
            if path.exists():
                try:
                    lib = ctypes.CDLL(str(path))
                    return cls(lib, path)
                except OSError:
                    continue
        return None

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self):
        self._lib.conv_dispatcher_init()

    def cleanup(self):
        self._lib.conv_dispatcher_cleanup()

    def version(self) -> str:
        return self._lib.conv_dispatcher_version().decode()

    def has_forward(self) -> bool:
        return self._lib.conv_dispatcher_has_kernels() != 0

    def has_bwd_data(self) -> bool:
        return self._lib.conv_dispatcher_has_bwd_data() != 0

    def has_bwd_weight(self) -> bool:
        return self._lib.conv_dispatcher_has_bwd_weight() != 0

    def kernel_count(self) -> int:
        return self._lib.conv_dispatcher_get_kernel_count()

    def kernel_names(self) -> List[str]:
        names = []
        for i in range(self.kernel_count()):
            buf = ctypes.create_string_buffer(256)
            if self._lib.conv_dispatcher_get_kernel_name(i, buf, 256) == 0:
                names.append(buf.value.decode())
        return names

    def is_supported(self, problem: GroupedConvProblem) -> bool:
        pc = GroupedConvProblemC.from_problem(problem)
        return self._lib.conv_dispatcher_is_supported(ctypes.byref(pc)) != 0

    def run(self, a_ptr: int, b_ptr: int, c_ptr: int,
            problem: GroupedConvProblem) -> float:
        """Run convolution. Returns time_ms (>0 success, <0 error)."""
        pc = GroupedConvProblemC.from_problem(problem)
        return self._lib.conv_dispatcher_run(a_ptr, b_ptr, c_ptr,
                                             ctypes.byref(pc), None)


# =============================================================================
# GpuGroupedConvRunner
# =============================================================================


class GpuGroupedConvRunner:
    """High-level GPU convolution runner.

    Handles library loading, HIP memory management, and kernel execution.
    Follows the same pattern as the old GpuConvRunner from conv_utils.py.

    Usage:
        runner = GpuGroupedConvRunner()
        if runner.is_available():
            result = runner.run(input_np, weight_np, problem)
            print(f"Time: {result.time_ms:.4f} ms, TFLOPS: {result.tflops:.2f}")
    """

    HIP_MEMCPY_H2D = 1
    HIP_MEMCPY_D2H = 2

    def __init__(self, lib_path: Optional[str] = None):
        self._dispatch_lib: Optional[GroupedConvDispatcherLib] = None
        self._hip = None
        self._initialized = False

        try:
            if lib_path:
                lib = ctypes.CDLL(lib_path)
                self._dispatch_lib = GroupedConvDispatcherLib(lib, Path(lib_path))
            else:
                self._dispatch_lib = GroupedConvDispatcherLib.find()

            if self._dispatch_lib is None:
                return

            self._hip = ctypes.CDLL("libamdhip64.so")
            self._hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self._hip.hipMalloc.restype = ctypes.c_int
            self._hip.hipFree.argtypes = [ctypes.c_void_p]
            self._hip.hipFree.restype = ctypes.c_int
            self._hip.hipMemcpy.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ]
            self._hip.hipMemcpy.restype = ctypes.c_int
            self._hip.hipDeviceSynchronize.argtypes = []
            self._hip.hipDeviceSynchronize.restype = ctypes.c_int

            self._dispatch_lib.initialize()
            self._initialized = True
        except Exception:
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self._dispatch_lib is not None

    @property
    def library_path(self) -> Optional[str]:
        if self._dispatch_lib:
            return str(self._dispatch_lib.path)
        return None

    @property
    def lib(self) -> Optional[GroupedConvDispatcherLib]:
        return self._dispatch_lib

    def run(self, input_np: np.ndarray, weight_np: np.ndarray,
            problem: GroupedConvProblem,
            output_np: Optional[np.ndarray] = None) -> GroupedConvResult:
        """Run convolution on GPU.

        Args:
            input_np:  For forward: X (NHWGC). For bwd_data: dY. For bwd_weight: X.
            weight_np: For forward: W (GKYXC). For bwd_data: W. For bwd_weight: dY.
            problem:   Problem specification.
            output_np: Optional pre-allocated output buffer.

        Returns:
            GroupedConvResult with success, time_ms, tflops, output.
        """
        if not self.is_available():
            return GroupedConvResult(error="GPU not available")

        try:
            # Determine output shape based on direction
            d = problem.direction
            if d == "bwd_data":
                out_shape = problem.input_shape()
            elif d == "bwd_weight":
                out_shape = problem.weight_shape()
            else:
                out_shape = problem.output_shape()

            if output_np is None:
                output_np = np.zeros(out_shape, dtype=input_np.dtype)

            output_size = output_np.nbytes

            # Allocate GPU memory
            d_a, d_b, d_c = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            self._hip.hipMalloc(ctypes.byref(d_a), input_np.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_b), weight_np.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_c), output_size)

            # Host → Device
            self._hip.hipMemcpy(d_a, input_np.ctypes.data, input_np.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_b, weight_np.ctypes.data, weight_np.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipDeviceSynchronize()

            # Launch kernel
            time_ms = self._dispatch_lib.run(d_a.value, d_b.value, d_c.value, problem)
            self._hip.hipDeviceSynchronize()

            result = GroupedConvResult()

            if time_ms > 0:
                # Device → Host
                self._hip.hipMemcpy(output_np.ctypes.data, d_c, output_size, self.HIP_MEMCPY_D2H)
                self._hip.hipDeviceSynchronize()
                result.success = True
                result.time_ms = time_ms
                result.tflops = problem.flops / (time_ms * 1e9)
                result.output = output_np
            else:
                result.error = (
                    "unsupported" if time_ms == -3.0
                    else "no kernel" if time_ms == -2.0
                    else f"error (code {time_ms})"
                )

            # Free GPU memory
            self._hip.hipFree(d_a)
            self._hip.hipFree(d_b)
            self._hip.hipFree(d_c)

            return result

        except Exception as e:
            return GroupedConvResult(error=str(e))

    def cleanup(self):
        if self._dispatch_lib:
            try:
                self._dispatch_lib.cleanup()
            except Exception:
                pass


# =============================================================================
# GroupedConvRegistry
# =============================================================================


class GroupedConvRegistry:
    """Collection of grouped conv kernel configs with JSON export/import."""

    def __init__(self, name: str = "default"):
        self.name = name
        self._kernels: List[GroupedConvKernelConfig] = []

    def add(self, config: GroupedConvKernelConfig):
        self._kernels.append(config)

    @property
    def kernels(self) -> List[GroupedConvKernelConfig]:
        return list(self._kernels)

    def __len__(self) -> int:
        return len(self._kernels)

    def filter_by_variant(self, variant: str) -> "GroupedConvRegistry":
        variant = _resolve_variant(variant)
        reg = GroupedConvRegistry(f"{self.name}_{variant}")
        for k in self._kernels:
            if k.variant == variant:
                reg.add(k)
        return reg

    def filter_by_arch(self, arch: str) -> "GroupedConvRegistry":
        reg = GroupedConvRegistry(f"{self.name}_{arch}")
        for k in self._kernels:
            if k.arch == arch:
                reg.add(k)
        return reg

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "name": self.name,
            "kernels": [k.to_json_obj() for k in self._kernels],
        }, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "GroupedConvRegistry":
        data = json.loads(json_str)
        reg = cls(data.get("name", "imported"))
        for kd in data.get("kernels", []):
            sig = kd.get("signature", {})
            algo = kd.get("algorithm", {})
            wave = algo.get("wave", "2x2x1").split("x")
            warp = algo.get("warp", "32x32x16").split("x")
            reg.add(GroupedConvKernelConfig(
                variant=sig.get("variant", "forward"),
                ndim_spatial=sig.get("ndim_spatial", 2),
                dtype=sig.get("dtype", "fp16"),
                layout=sig.get("layout", "nhwgc"),
                arch=kd.get("arch", "gfx942"),
                tile_m=algo.get("tile_m", 1),
                tile_n=algo.get("tile_n", 128),
                tile_k=algo.get("tile_k", 128),
                wave_m=int(wave[0]), wave_n=int(wave[1]), wave_k=int(wave[2]),
                warp_tile_m=int(warp[0]), warp_tile_n=int(warp[1]), warp_tile_k=int(warp[2]),
                pipeline=algo.get("pipeline", "compv3"),
                epilogue=algo.get("epilogue", "cshuffle"),
                scheduler=algo.get("scheduler", "intrawave"),
            ))
        return reg

    def print_registry(self, indent: str = "  "):
        print(f"{indent}Registry '{self.name}': {len(self)} kernels")
        for i, k in enumerate(self._kernels):
            print(f"{indent}  [{i}] {k.name} (valid={validate_grouped_conv_config(k.to_dict()).is_valid})")


# =============================================================================
# GroupedConvValidationResult
# =============================================================================


@dataclass
class GroupedConvValidationResult(ValidationResultBase):
    """Result of grouped conv kernel config validation."""

    variant: str = "forward"

    def __init__(self, is_valid=True, errors=None, warnings=None,
                 suggested_fixes=None, variant="forward"):
        super().__init__(
            is_valid=is_valid,
            errors=errors or [],
            warnings=warnings or [],
            suggested_fixes=suggested_fixes or {},
        )
        self.variant = variant


# =============================================================================
# Validation helpers (extracted from the original config extraction code)
# =============================================================================


def _first(val):
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return val


def _get_tile_config(config: dict) -> dict:
    return config.get("tile_config") or {}


def _get_trait_config(config: dict) -> dict:
    return config.get("trait_config") or {}


def _extract_wave_config(tile_config: dict) -> List[int]:
    wm = tile_config.get("wave_m") or tile_config.get("warp_m")
    wn = tile_config.get("wave_n") or tile_config.get("warp_n")
    wk = tile_config.get("wave_k") or tile_config.get("warp_k")
    if wm is not None and wn is not None and wk is not None:
        return [_first(wm), _first(wn), _first(wk)]
    return [2, 2, 1]


def _extract_warp_tile_config(tile_config: dict) -> List[int]:
    wtm = tile_config.get("warp_tile_m") or tile_config.get("warp_m")
    wtn = tile_config.get("warp_tile_n") or tile_config.get("warp_n")
    wtk = tile_config.get("warp_tile_k") or tile_config.get("warp_k")
    if wtm is not None and wtn is not None and wtk is not None:
        return [_first(wtm), _first(wtn), _first(wtk)]
    return [32, 32, 16]


def _extract_trait_values(trait_config: dict) -> Tuple[str, str, str]:
    p = _first(trait_config.get("pipeline", "compv4"))
    e = _first(trait_config.get("epilogue", "cshuffle"))
    s = _first(trait_config.get("scheduler", "intrawave"))
    if isinstance(p, list): p = p[0] if p else "compv4"
    if isinstance(e, list): e = e[0] if e else "cshuffle"
    if isinstance(s, list): s = s[0] if s else "intrawave"
    return (str(p), str(e), str(s))


# =============================================================================
# validate_grouped_conv_config / auto_correct_grouped_conv_config
# =============================================================================


def validate_grouped_conv_config(config: dict) -> GroupedConvValidationResult:
    """Validate a grouped conv kernel config dict.

    Accepts either a raw dict (legacy) or GroupedConvKernelConfig.to_dict() output.
    """
    errors: List[str] = []
    warnings: List[str] = []
    suggested_fixes: Dict[str, Any] = {}

    required = ("tile_config", "trait_config", "variant", "ndim_spatial", "arch", "layout")
    for key in required:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    if errors:
        return GroupedConvValidationResult(
            is_valid=False, errors=errors, warnings=warnings,
            suggested_fixes=suggested_fixes, variant=config.get("variant", "forward"),
        )

    tile_config = _get_tile_config(config)
    trait_config = _get_trait_config(config)
    variant = _first(config.get("variant", "forward"))
    if isinstance(variant, list):
        variant = variant[0] if variant else "forward"
    variant = _resolve_variant(str(variant))

    ndim_spatial = config.get("ndim_spatial")
    arch = config.get("arch", "gfx942")
    dtype = config.get("dtype", "fp16")

    if variant not in VALID_VARIANTS:
        errors.append(f"Invalid variant: {variant}. Valid: {', '.join(VALID_VARIANTS)}")
        suggested_fixes["variant"] = "forward"

    if ndim_spatial is not None:
        ndim = ndim_spatial
        if isinstance(ndim, list):
            ndim = ndim[0] if ndim else 2
        if ndim not in VALID_NDIM_SPATIAL:
            errors.append(f"Invalid ndim_spatial: {ndim}. Valid: {', '.join(map(str, VALID_NDIM_SPATIAL))}")
            suggested_fixes["ndim_spatial"] = 2

    pipeline, epilogue, scheduler = _extract_trait_values(trait_config)
    if variant in BACKWARD_VARIANTS and pipeline not in BACKWARD_PIPELINES:
        errors.append(f"Backward variant '{variant}' requires pipeline compv3 or mem, got {pipeline}")
        suggested_fixes["pipeline"] = "compv3"

    ok, msg = validate_trait_combo(pipeline, epilogue, scheduler)
    if not ok:
        errors.append(msg)
        suggested_fixes["scheduler"] = "intrawave"

    wave_cfg = _extract_wave_config(tile_config)
    ok, msg = validate_wave_config(wave_cfg, arch)
    if not ok:
        errors.append(msg)
        arch_data = get_arch_filter_data()
        valid_waves = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
        if valid_waves:
            suggested_fixes["wave_m"] = valid_waves[0][0]
            suggested_fixes["wave_n"] = valid_waves[0][1]
            suggested_fixes["wave_k"] = valid_waves[0][2]

    warp_cfg = _extract_warp_tile_config(tile_config)
    ok, msg = validate_warp_tile_config(warp_cfg, arch, dtype)
    if not ok:
        errors.append(msg)
        arch_data = get_arch_filter_data()
        acc = "int32" if dtype == "int8" else "fp32"
        dtype_key = f"{dtype}_{dtype}_{acc}"
        valid_tiles = (arch_data["warp_tile_combos"]
                       .get(arch, {}).get(dtype_key, [[32, 32, 16], [16, 16, 16]]))
        if valid_tiles:
            suggested_fixes["warp_tile_m"] = valid_tiles[0][0]
            suggested_fixes["warp_tile_n"] = valid_tiles[0][1]
            suggested_fixes["warp_tile_k"] = valid_tiles[0][2]

    arch_data = get_arch_filter_data()
    if arch not in arch_data["supported_archs"]:
        errors.append(f"Unsupported architecture: {arch}. Supported: {', '.join(arch_data['supported_archs'])}")

    return GroupedConvValidationResult(
        is_valid=len(errors) == 0, errors=errors, warnings=warnings,
        suggested_fixes=suggested_fixes, variant=variant,
    )


def auto_correct_grouped_conv_config(config: dict) -> Tuple[dict, GroupedConvValidationResult]:
    """Auto-correct invalid grouped conv config. Returns (corrected, result)."""
    result = validate_grouped_conv_config(config)
    corrected = copy.deepcopy(config)

    if result.is_valid:
        return corrected, result

    tile_config = corrected.setdefault("tile_config", {})
    trait_config = corrected.setdefault("trait_config", {})

    wave_cfg = _extract_wave_config(tile_config)
    arch = config.get("arch", "gfx942")
    fixed_wave = auto_correct_wave(wave_cfg, arch)
    tile_config["wave_m"] = fixed_wave[0]
    tile_config["wave_n"] = fixed_wave[1]
    tile_config["wave_k"] = fixed_wave[2]

    pipeline, epilogue, scheduler = _extract_trait_values(trait_config)
    fixed_pipeline, fixed_scheduler = auto_correct_trait(pipeline, scheduler)
    trait_config["pipeline"] = fixed_pipeline
    trait_config["scheduler"] = fixed_scheduler

    variant = _first(config.get("variant", "forward"))
    if isinstance(variant, list):
        variant = variant[0] if variant else "forward"
    variant = _resolve_variant(str(variant))
    if variant in BACKWARD_VARIANTS and fixed_pipeline not in BACKWARD_PIPELINES:
        trait_config["pipeline"] = "compv3"

    if "warp_tile_m" in result.suggested_fixes:
        tile_config["warp_tile_m"] = result.suggested_fixes["warp_tile_m"]
        tile_config["warp_tile_n"] = result.suggested_fixes["warp_tile_n"]
        tile_config["warp_tile_k"] = result.suggested_fixes["warp_tile_k"]

    result = validate_grouped_conv_config(corrected)
    return corrected, result


# =============================================================================
# Convenience functions
# =============================================================================


def get_grouped_conv_default_config(
    variant: str = "forward", ndim_spatial: int = 2,
    arch: str = "gfx942", dtype: str = "fp16",
) -> GroupedConvKernelConfig:
    """Return a valid default GroupedConvKernelConfig."""
    return GroupedConvKernelConfig(
        variant=variant, ndim_spatial=ndim_spatial, arch=arch, dtype=dtype,
    )


def format_grouped_conv_summary(config) -> str:
    """Format a config (dict or GroupedConvKernelConfig) into a human-readable string."""
    if isinstance(config, GroupedConvKernelConfig):
        lines = [
            f"Grouped Conv Config: {config.variant} {config.ndim_spatial}D",
            f"  Arch:    {config.arch}",
            f"  Layout:  {config.layout}",
            f"  Dtype:   {config.dtype}",
            f"  Tile:    {config.tile_str}",
            f"  Wave:    {config.wave_str}",
            f"  Warp:    {config.warp_str}",
            f"  Traits:  pipeline={config.pipeline} epilogue={config.epilogue} scheduler={config.scheduler}",
        ]
        return "\n".join(lines)

    # Legacy dict support
    tile_config = _get_tile_config(config) if isinstance(config, dict) else {}
    trait_config = _get_trait_config(config) if isinstance(config, dict) else {}
    variant = config.get("variant", "?") if isinstance(config, dict) else "?"
    ndim = config.get("ndim_spatial", "?") if isinstance(config, dict) else "?"
    arch = config.get("arch", "?") if isinstance(config, dict) else "?"
    layout = config.get("layout", "?") if isinstance(config, dict) else "?"
    dtype = config.get("dtype", "fp16") if isinstance(config, dict) else "fp16"

    lines = [f"Grouped Conv Config: {variant} {ndim}D"]
    lines.append(f"  Arch:    {arch}")
    lines.append(f"  Layout:  {layout}")
    lines.append(f"  Dtype:   {dtype}")

    if tile_config:
        wave = _extract_wave_config(tile_config)
        warp = _extract_warp_tile_config(tile_config)
        lines.append(f"  Tile:    M={_first(tile_config.get('tile_m', 1))} N={_first(tile_config.get('tile_n', 128))} K={_first(tile_config.get('tile_k', 128))}")
        lines.append(f"  Wave:    {wave[0]}x{wave[1]}x{wave[2]}")
        lines.append(f"  Warp:    {warp[0]}x{warp[1]}x{warp[2]}")

    if trait_config:
        pipeline = _first(trait_config.get("pipeline", "?"))
        epilogue = _first(trait_config.get("epilogue", "?"))
        scheduler = _first(trait_config.get("scheduler", "?"))
        lines.append(f"  Traits:  pipeline={pipeline} epilogue={epilogue} scheduler={scheduler}")

    return "\n".join(lines) if lines else "(empty config)"


def detect_gpu_arch() -> str:
    """Detect GPU architecture using rocminfo."""
    try:
        out = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, text=True)
        for line in out.split("\n"):
            if "gfx" in line.lower() and "name:" in line.lower():
                for part in line.split():
                    if part.startswith("gfx"):
                        return part
    except Exception:
        pass
    return "gfx942"
