# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host side of the direct-convolution GPU tests, without torch.

A direct kernel takes a ``DirectConvProblem`` (NHWC activations, KRSC filters,
symmetric padding per axis) and three buffers in ABI order:

=========  ==============  ==========
direction  inputs          output
=========  ==============  ==========
fwd        ``x``, ``w``    ``y``
dgrad      ``dy``, ``w``   ``dx``
wgrad      ``dy``, ``x``   ``dw``
=========  ==============  ==========

:func:`launch_direct_kernel` launches a compiled kernel over device arrays in
that order and :func:`launch_direct_conv` over host arrays;
:func:`poisoned` builds the initial output (a byte pattern an unwritten element
cannot pass as). The three references are the float64 ``ConvProblem``
references of :mod:`builders.common.conv_reference` on the problem's
``ConvGeometry`` (the one map from direct-conv names to the shared geometry),
fed the values the kernel reads (decode stored arrays with
:func:`rocke.numeric.storage.decode` first) and taking their operands by name.
:func:`direct_fwd_peak_error` composes them for the common forward check;
:func:`stored_uniform` draws operands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

from builders.common.conv_reference import (
    conv_dgrad_reference,
    conv_fwd_reference,
    conv_wgrad_reference,
)
from kernels.common.conv_abi import (
    conv_direct_args_signature,
    conv_direct_output_dtype,
)
from kernels.common.conv_args import ConvArgs, ConvGeometry
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.direct_aot_jobs import direct_storage_dtype

if TYPE_CHECKING:
    from rocke.helpers.compile import KernelArtifact

__all__ = [
    "direct_dgrad_reference",
    "direct_fwd_peak_error",
    "direct_fwd_reference",
    "direct_wgrad_reference",
    "launch_direct_conv",
    "launch_direct_kernel",
    "poisoned",
    "stored_uniform",
]


def direct_fwd_reference(
    problem: DirectConvProblem, x: npt.ArrayLike, w: npt.ArrayLike
) -> np.ndarray:
    """``y[N, Ho, Wo, K]`` of ``x[N, H, W, C]`` and ``w[K, KH, KW, cpg]``."""
    geometry = ConvGeometry.from_direct_problem(problem)
    return conv_fwd_reference(np.asarray(x), np.asarray(w), geometry)


def direct_dgrad_reference(
    problem: DirectConvProblem, dy: npt.ArrayLike, w: npt.ArrayLike
) -> np.ndarray:
    """``dx[N, H, W, C]`` of ``dy[N, Ho, Wo, K]`` and ``w[K, KH, KW, cpg]``."""
    geometry = ConvGeometry.from_direct_problem(problem)
    return conv_dgrad_reference(np.asarray(dy), np.asarray(w), geometry)


def direct_wgrad_reference(
    problem: DirectConvProblem, x: npt.ArrayLike, dy: npt.ArrayLike
) -> np.ndarray:
    """``dw[K, KH, KW, cpg]`` of ``x[N, H, W, C]`` and ``dy[N, Ho, Wo, K]``."""
    geometry = ConvGeometry.from_direct_problem(problem)
    return conv_wgrad_reference(np.asarray(x), np.asarray(dy), geometry)


def stored_uniform(
    rng: np.random.Generator, shape: Sequence[int], dtype: str, bound: float = 1.0
) -> np.ndarray:
    """U(-bound, bound) draws of ``shape``, stored as ``dtype``."""
    return storage.encode(rng.uniform(-bound, bound, shape), dtype)


def poisoned(shape: Sequence[int], dtype: str, byte: int = 0x7F) -> np.ndarray:
    """A ``dtype`` host array of ``shape`` with every byte ``byte`` (0x7F is a
    NaN in fp16, 0xFF one in fp16, bf16 and fp32)."""
    array_dtype = storage.host_dtype(dtype)
    raw = np.full((*shape, array_dtype.itemsize), byte, dtype=np.uint8)
    return raw.view(array_dtype).reshape(tuple(shape))


def launch_direct_kernel(
    artifact: KernelArtifact,
    problem: DirectConvProblem,
    buffers: tuple[DeviceArray, DeviceArray, DeviceArray],
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    direction: str = "fwd",
    dtype: str | None = None,
) -> None:
    """Launch ``artifact`` with ``problem``'s direct-conv ``direction``
    kernargs over the three device ``buffers`` (ABI order), and wait for it.

    ``dtype`` is the activation / filter element type of the launch signature;
    it defaults to ``problem``'s storage dtype (a spec that carries its own
    dtype, as the column-streamed depthwise one does, passes it)."""
    dtype = dtype or direct_storage_dtype(problem.dtype)
    values = ConvArgs.from_problem(problem, direction=direction).to_launch_values(
        *(buffer.data_ptr() for buffer in buffers),
        *(buffer.nbytes for buffer in buffers),
    )
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=conv_direct_args_signature(dtype, direction=direction),
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))


def launch_direct_conv(
    artifact: KernelArtifact,
    problem: DirectConvProblem,
    inputs: tuple[np.ndarray, np.ndarray],
    output: np.ndarray,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    direction: str = "fwd",
    dtype: str | None = None,
) -> np.ndarray:
    """Run ``artifact``'s ``direction`` kernel on ``problem`` over host arrays
    (:func:`launch_direct_kernel` on device copies) and return its output,
    read back as a host array like ``output``.

    ``inputs`` are the two stored input arrays in ABI order, in ``dtype``
    (default: ``problem``'s storage dtype), and ``output`` the output's initial
    contents (fp32 for wgrad, which accumulates through fp32 atomics; else
    ``dtype``)."""
    dtype = dtype or direct_storage_dtype(problem.dtype)
    first, second = (DeviceArray(array, dtype) for array in inputs)
    out = DeviceArray(output, conv_direct_output_dtype(dtype, direction))
    launch_direct_kernel(
        artifact,
        problem,
        (first, second, out),
        grid=grid,
        block=block,
        direction=direction,
        dtype=dtype,
    )
    return out.read()


def direct_fwd_peak_error(
    artifact: KernelArtifact,
    problem: DirectConvProblem,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    seed: int = 0,
) -> float:
    """``max|y - ref| / max|ref|`` of ``artifact``'s forward on ``problem``.

    ``x`` and ``w`` are U(-1, 1) draws from ``seed``, stored in ``problem``'s
    dtype; ``y`` starts poisoned, so an element the kernel does not write
    fails; ``ref`` is :func:`direct_fwd_reference` of the stored values."""
    dtype = direct_storage_dtype(problem.dtype)
    rng = np.random.default_rng(seed)
    x = stored_uniform(rng, (problem.N, problem.H, problem.W, problem.total_c), dtype)
    w = stored_uniform(
        rng, (problem.total_k, problem.KH, problem.KW, problem.cpg), dtype
    )
    y_shape = (problem.N, problem.Ho, problem.Wo, problem.total_k)
    y = launch_direct_conv(
        artifact, problem, (x, w), poisoned(y_shape, dtype), grid=grid, block=block
    )
    ref = direct_fwd_reference(
        problem, storage.decode(x, dtype), storage.decode(w, dtype)
    )
    return peak_relative_error(storage.decode(y, dtype), ref)
