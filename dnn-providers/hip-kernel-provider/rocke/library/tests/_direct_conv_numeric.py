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
cannot pass as). The three references compute each direction in float64 with
:mod:`rocke.numeric.conv`, from the values the kernel reads (decode stored
arrays with :func:`rocke.numeric.storage.decode` first), taking their operands
by name in that module's order;
:func:`direct_conv_geometry` is the one map from a problem's geometry to that
module's keywords. :func:`direct_fwd_peak_error` composes the three for the
common forward check.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rocke.numeric import storage
from rocke.numeric.conv import conv_dgrad, conv_fwd, conv_wgrad
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

from kernels.common.conv_abi import conv_direct_args_signature
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.direct_aot_jobs import direct_storage_dtype

if TYPE_CHECKING:
    from rocke.helpers.compile import KernelArtifact

__all__ = [
    "direct_conv_geometry",
    "direct_dgrad_reference",
    "direct_fwd_peak_error",
    "direct_fwd_reference",
    "direct_wgrad_reference",
    "launch_direct_conv",
    "launch_direct_kernel",
    "poisoned",
]


def direct_conv_geometry(problem: DirectConvProblem) -> dict[str, object]:
    """``problem``'s stride, per-axis padding, dilation and groups, as the
    :mod:`rocke.numeric.conv` keywords."""
    return {
        "stride": problem.stride,
        "padding": (problem.PAD, problem.PAD_W),
        "dilation": (problem.dil_h, problem.dil_w),
        "groups": problem.groups,
    }


def direct_fwd_reference(
    problem: DirectConvProblem, x: npt.ArrayLike, w: npt.ArrayLike
) -> np.ndarray:
    """``y[N, Ho, Wo, K]`` of ``x[N, H, W, C]`` and ``w[K, KH, KW, cpg]``."""
    return conv_fwd(x, w, **direct_conv_geometry(problem))


def direct_dgrad_reference(
    problem: DirectConvProblem, dy: npt.ArrayLike, w: npt.ArrayLike
) -> np.ndarray:
    """``dx[N, H, W, C]`` of ``dy[N, Ho, Wo, K]`` and ``w[K, KH, KW, cpg]``."""
    return conv_dgrad(
        dy, w, in_spatial=(problem.H, problem.W), **direct_conv_geometry(problem)
    )


def direct_wgrad_reference(
    problem: DirectConvProblem, x: npt.ArrayLike, dy: npt.ArrayLike
) -> np.ndarray:
    """``dw[K, KH, KW, cpg]`` of ``x[N, H, W, C]`` and ``dy[N, Ho, Wo, K]``."""
    return conv_wgrad(
        x,
        dy,
        filter_spatial=(problem.KH, problem.KW),
        **direct_conv_geometry(problem),
    )


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
    out = DeviceArray(output, "fp32" if direction == "wgrad" else dtype)
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
    x = storage.encode(
        rng.uniform(-1, 1, (problem.N, problem.H, problem.W, problem.total_c)), dtype
    )
    w = storage.encode(
        rng.uniform(-1, 1, (problem.total_k, problem.KH, problem.KW, problem.cpg)),
        dtype,
    )
    y_shape = (problem.N, problem.Ho, problem.Wo, problem.total_k)
    y = launch_direct_conv(
        artifact, problem, (x, w), poisoned(y_shape, dtype), grid=grid, block=block
    )
    ref = direct_fwd_reference(
        problem, storage.decode(x, dtype), storage.decode(w, dtype)
    )
    return float(np.abs(storage.decode(y, dtype) - ref).max() / np.abs(ref).max())
