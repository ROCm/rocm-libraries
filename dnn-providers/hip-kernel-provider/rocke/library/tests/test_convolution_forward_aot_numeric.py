# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A packaged gfx950 forward-convolution binary computes every problem it serves.

A packaged AOT instance is built once, for its probe problem, and the native
pack then launches that one binary on every runtime problem
``gfx950_conv_fwd_aot_serves`` accepts, with the arguments of
``Gfx950ConvFwdAotSpec.launch_values`` over the grid of ``launch_step``.
``test_convolution_forward_aot_packaging.py`` checks those verdicts and
arguments on the host; this file runs the binary on the GPU for several served
problems that differ from its build problem (dense, grouped, depthwise and
grouped pointwise for an implicit-GEMM instance; other images, batches and
group counts for a direct depthwise one) and compares each output with the
float64 NumPy convolution of the forward reference corpus
(``conv_reference.contract``), within that corpus's tolerance.

The output buffer starts as all-ones bytes (NaN in fp16 and bf16), so an
element the kernel does not write fails the check.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from benchmarks.common.kernel_sweep import current_llvm_flavor
from builders.common.convolution_forward import (
    Gfx950ConvFwdAotSpec,
    Gfx950ConvFwdSpec,
    build_gfx950_conv_fwd_aot,
    gfx950_conv_fwd_aot_serves,
    gfx950_conv_fwd_aot_spec_for_identity,
)
from kernels.conv.kernel_identity import KernelIdentity
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

from reference_common.numeric import decode

from .conv_reference.architectures.common import compile_reference_kernel
from .conv_reference.contract import (
    Case,
    independent_reference,
    make_inputs,
    normalized_distance,
    reference_scale,
)

pytestmark = pytest.mark.gpu

_ARCH = "gfx950"


def _instance(label: str) -> Gfx950ConvFwdAotSpec:
    """The packaged instance of the sweep identity ``label``, at the LLVM flavor
    of the loaded compiler."""
    identity = replace(
        KernelIdentity.from_label(label), llvm_flavor=current_llvm_flavor()
    )
    spec, reason = gfx950_conv_fwd_aot_spec_for_identity(identity)
    assert spec is not None, f"{label} is no longer packaged: {reason}"
    return spec


def _geometry(n, hi, wi, c, k, *, filt=3, stride=1, groups=1) -> dict:
    """A square-filter problem with "same" padding, as ``Case`` fields."""
    pad = filt // 2
    return {
        "N": n,
        "Hi": hi,
        "Wi": wi,
        "C": c,
        "K": k,
        "Y": filt,
        "X": filt,
        "sH": stride,
        "sW": stride,
        "pH": pad,
        "pW": pad,
        "groups": groups,
    }


# label -> (the instance's sweep identity, the served problems it runs). The problems are
# not the instance's build (probe) problem: they vary the batch, image, channel
# and group counts.
_INSTANCES = {
    # The smallest unvectorized implicit-GEMM tile: serves dense, grouped,
    # depthwise and grouped pointwise problems with odd channel counts.
    "implicit_gemm_fp16_t64x64_v1": (
        (
            "fwd-implicit_gemm-fp16-2d-t64x64x16-w1x1-a32x32x16-v1x1x1-mem-default-grp"
            "-wave64-gfx950-llvm20"
        ),
        [
            _geometry(2, 14, 14, 32, 32),
            _geometry(2, 11, 11, 12, 24, groups=4),
            _geometry(2, 14, 14, 32, 32, groups=32),
            _geometry(2, 8, 8, 32, 32, filt=1, groups=4),
        ],
    ),
    # A four-wave 128x128 tile, vectorized, with the async DRAM->LDS loaders,
    # on channel counts its 8-wide loads and stores divide. Each problem spans
    # two tiles in M or N; the last two end in partial tiles.
    "implicit_gemm_bf16_t128x128_v8_async": (
        (
            "fwd-implicit_gemm-bf16-2d-t128x128x16-w2x2-a32x32x16-v8x8x8-mem-cshuffle-grp"
            "-async-ac8x8-wave64-gfx950-llvm20"
        ),
        [
            _geometry(4, 16, 16, 64, 256, stride=2),
            _geometry(2, 9, 13, 32, 160),
            _geometry(3, 7, 7, 64, 128, filt=1, groups=2),
        ],
    ),
    # A direct depthwise kernel: the filter, stride and padding are compiled
    # in; the image, batch and group count are runtime values.
    "direct_depthwise_spatial_fp16": (
        (
            "fwd-direct_depthwise_spatial-fp16-2d-f3x3-p1x1-s1x1-d1x1-c1k1-block_waves=1"
            "-wave64-gfx950-llvm20"
        ),
        [
            _geometry(1, 64, 64, 8, 8, groups=8),
            _geometry(3, 9, 21, 5, 5, groups=5),
            _geometry(2, 14, 14, 32, 32, groups=32),
        ],
    ),
}


@pytest.fixture(scope="module")
def gpu_arch():
    return current_device_arch_in((_ARCH,), subject="the packaged forward conv")


def _run(artifact, spec: Gfx950ConvFwdAotSpec, problem: Gfx950ConvFwdSpec, case):
    """Launch ``spec``'s binary on ``problem`` with the corpus inputs of ``case``;
    return the output storage."""
    inputs = make_inputs(case)
    out = np.empty(case.output_shape, dtype=inputs["a"].dtype)
    out.view(np.uint8)[...] = 0xFF
    buffers = {"A": inputs["a"], "B": inputs["b"], "D": out}
    # The pointers are the launcher's own allocations; only the byte counts
    # and the problem block come from the packaged launch values.
    values = spec.launch_values(
        problem, 0, 0, 0, *(buffers[name].nbytes for name in ("A", "B", "D"))
    )
    launch_with_buffers(
        artifact,
        spec.launch_signature(),
        buffers,
        grid=tuple(spec.launch_step(problem).grid),
        block=spec.block(),
        scalars={name: value for name, value in values.items() if name not in buffers},
    )
    return inputs, out


@pytest.mark.parametrize("label", sorted(_INSTANCES))
def test_one_packaged_binary_computes_every_served_problem_on_gpu(gpu_arch, label):
    identity_label, problems = _INSTANCES[label]
    spec = _instance(identity_label)
    assert not spec.is_split_c, "this test launches one kernel per problem"
    artifact = compile_reference_kernel(
        build_gfx950_conv_fwd_aot(spec, arch=gpu_arch),
        gpu_arch,
        spec.identity().llvm_flavor,
    )
    for index, fields in enumerate(problems):
        problem = Gfx950ConvFwdSpec(dtype=spec.dtype, **fields)
        assert gfx950_conv_fwd_aot_serves(spec, problem) == (True, "ok"), problem
        case = Case(name=f"served-{index}", dtype=spec.dtype, **fields)
        inputs, out = _run(artifact, spec, problem, case)
        got = decode(out, spec.dtype)
        assert np.isfinite(got).all(), f"{label} {problem}: unwritten or non-finite"
        reference = independent_reference(case, inputs)
        error = normalized_distance(got, reference, reference_scale(reference))
        assert error <= case.tolerance, f"{label} {problem}: error {error:.3g}"
