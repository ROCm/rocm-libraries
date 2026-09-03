# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric checks for the GDN decode kernel (gfx950).

Compares the kernel against a whole-tensor fp32 reference that shares no
algebra with it -- no tiling, no warp structure, no cross-lane reductions -- so
agreement is evidence about the algorithm rather than a restatement of the
implementation.

Both results are compared. The kernel writes its output *and* mutates the
recurrent state in place; checking only the output would let a corrupted state
write ship silently, because nothing reads the state back until the next decode
step.

These lanes need a real gfx950 and ROCm torch, so they are marked ``gpu`` and
skipped elsewhere. The spec rules, emission and dispatch selection are covered
by CPU-only tests so this family still contributes coverage without a device.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return ARCH in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _device_is_gfx950(), reason=f"needs a {ARCH} device"
)


@pytest.fixture(scope="module")
def harness():
    from builders.gfx950.gdn.gdn_decode import TOL, check, launch, launcher_for
    from builders.gfx950.gdn.gdn_decode import make_inputs, prepare, ref_fp32

    return {
        "TOL": TOL,
        "check": check,
        "launch": launch,
        "launcher_for": launcher_for,
        "make_inputs": make_inputs,
        "prepare": prepare,
        "ref_fp32": ref_fp32,
    }


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
def test_matches_fp32_reference(harness, batch):
    """Output and updated state both agree with the reference."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    out_err, state_err = harness["check"](GdnDecodeSpec(), batch)
    assert out_err <= harness["TOL"], f"output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"state error {state_err:.3e}"


@requires_gfx950
def test_simple_reference_path_matches(harness):
    """The one-thread-per-row path is a correctness baseline; keep it working."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    out_err, state_err = harness["check"](GdnDecodeSpec(simple=True), 4)
    assert max(out_err, state_err) <= harness["TOL"]


@requires_gfx950
def test_every_dispatched_tile_is_correct(harness):
    """Every tile the dispatcher can select must be numerically sound.

    A tuning table that can route a request to a wrong kernel is worse than no
    tuning at all, so the whole table is exercised rather than the default only.
    """
    from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode

    for batch in (1, 16, 64, 256):
        spec = dispatch_gdn_decode(GdnDecodeRequest(batch=batch, arch=ARCH)).spec
        out_err, state_err = harness["check"](spec, batch)
        assert max(out_err, state_err) <= harness["TOL"], (
            f"batch {batch} tile "
            f"({spec.num_warps},{spec.warp_threads_k},{spec.blocks_per_v_dim}) "
            f"out={out_err:.3e} state={state_err:.3e}"
        )


@requires_gfx950
def test_padding_lanes_are_skipped_and_leave_state_untouched(harness):
    """A negative index means 'skip', and must not disturb that state slot.

    This is the continuous-batching contract: idle slots in a ragged request
    cost nothing and must come back bit-identical.
    """
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    spec = GdnDecodeSpec()
    batch = 8
    inp = harness["make_inputs"](spec, batch)
    # Mark the odd sequences inactive.
    inp["read_indices"][1::2] = -1
    inp["write_indices"][1::2] = -1

    before = inp["state"].clone()
    values, cfg = harness["prepare"](spec, inp, batch)
    harness["launch"](harness["launcher_for"](spec), values, cfg)
    torch.cuda.synchronize()

    inactive = torch.arange(1, batch, 2, device=values["state"].device)
    assert torch.equal(
        values["state"][inactive], before[inactive]
    ), "state of an inactive (negative-index) sequence was modified"


@requires_gfx950
def test_results_are_deterministic(harness):
    """Same inputs, same answer -- no dependence on scheduling or leftovers."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    spec = GdnDecodeSpec()
    first = harness["check"](spec, 16)
    second = harness["check"](spec, 16)
    assert first == second


@requires_gfx950
def test_state_dtype_variant_is_correct(harness):
    """An f16 recurrent state is a distinct kernel; it must be checked too."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    spec = dc.replace(GdnDecodeSpec(), state_dtype="f16")
    ok, why = is_valid_spec(spec, arch=ARCH)
    assert ok, why
    out_err, state_err = harness["check"](spec, 8)
    assert max(out_err, state_err) <= harness["TOL"]


@requires_gfx950
def test_end_to_end_through_the_dispatch_result(harness):
    """Drive a launch from the dispatch result alone, as a caller would.

    Every other lane reaches into the kernel module for the signature and grid.
    This one uses only what ``dispatch_gdn_decode`` hands back -- the built
    kernel, its signature, its grid and block -- so a disagreement between the
    dispatcher's launch contract and the kernel it selected shows up as wrong
    numbers rather than passing unnoticed.
    """
    from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    batch = 16
    result = dispatch_gdn_decode(GdnDecodeRequest(batch=batch, arch=ARCH))

    artifact = compile_kernel(result.build(), arch=ARCH)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=result.signature,
    )

    inp = harness["make_inputs"](result.spec, batch)
    ref_out, ref_state = harness["ref_fp32"](result.spec, inp)
    values, _ = harness["prepare"](result.spec, inp, batch)
    cfg = LaunchConfig(grid=result.grid, block=result.block, stream=0)
    with no_fence():
        launcher(values, config=cfg)
    torch.cuda.synchronize()

    written = inp["write_indices"].long()
    out_err = (values["out"].float() - ref_out).abs().max().item()
    state_err = (values["state"].float()[written] - ref_state).abs().max().item()
    assert max(out_err, state_err) <= harness["TOL"], (
        f"dispatch-driven launch disagreed with the reference: "
        f"out={out_err:.3e} state={state_err:.3e}"
    )
