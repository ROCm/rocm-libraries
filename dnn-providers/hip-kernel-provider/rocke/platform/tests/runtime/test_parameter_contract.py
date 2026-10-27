# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A kernel's parameter contract and the launch door that checks it.

A kernel states facts on its parameters, alignments on its pointers and may
state its exact workgroup size. ``parameter_contract`` reads them from the
``KernelDef`` as rows; ``KernelLauncher.from_artifact`` holds that contract and
checks every row before every launch, refusing with every failed row named.
``write_artifact``, which would drop the contract on disk, refuses a kernel that
states facts. The first group of tests needs no GPU; the ``test_gpu_`` tests
launch on the current GPU and skip without one.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from rocke.core.intexpr import product_of
from rocke.core.ir import F16, I16, I32, I64, IRBuilder, Param, PtrType
from rocke.helpers.compile import (
    KernelArtifact,
    KernelModule,
    compile_kernel,
    compile_kernel_via_hipcc,
    compile_module,
)
from rocke.helpers.manifest import write_artifact
from rocke.runtime.contract import LaunchRefused, parameter_contract
from rocke.runtime.launcher import (
    DeviceMem,
    KernelLauncher,
    LaunchConfig,
    PipelineLauncher,
    StreamConfig,
    first_supported,
    launch_kernel,
    make_kernel,
    synchronize_and_release,
)
from rocke.runtime.torch_interop import launch_torch_kernel
from rocke.tests.gpu import current_device_arch


def _gemm_like_kernel():
    """The facts a batched GEMM's C store states, plus one GPU-checked value."""
    b = IRBuilder("contract_probe")
    b.kernel.attrs["workgroup_size"] = 64
    b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    m = b.param("M", I32, ge=1, le=1 << 20, multiple_of=64)
    n = b.param("N", I32, ge=1)
    ldc = b.param("ldc", I32, ge=n)
    b.param("batch_stride_c", I32, ge=product_of(m, ldc))
    cu = b.param("cu", PtrType(I32, "global"))
    t = b.param("T", I32, ge=0, le=4096)
    loaded = b.readfirstlane(b.global_load(cu, b.block_id_x(), I32))
    b.gpu_checked_value(loaded, ge=0, le=t)
    b.ret()
    return b.kernel


GOOD = {
    "C": 0x7F3A00000000,
    "M": 1024,
    "N": 128,
    "ldc": 128,
    "batch_stride_c": 1024 * 128,
    "cu": 0,
    "T": 4096,
}


def test_the_contract_lists_host_rows_promises_and_gpu_checks():
    """Every stated fact is a host row; a side no word states shows its type's
    limit; noalias / writeonly are promises nothing checks; a gpu_checked_value
    is listed as a GPU check."""
    contract = parameter_contract(_gemm_like_kernel())
    assert [str(r) for r in contract.facts] == [
        "M ge 1",
        "M le 1048576",
        "M multiple_of 64",
        "N ge 1",
        "ldc ge N",
        "batch_stride_c ge product_of(M, ldc)",
        "T ge 0",
        "T le 4096",
    ]
    assert [str(r) for r in contract.alignments] == ["C: align 16"]
    assert contract.workgroup_size == 64
    assert contract.promises == (("C", "noalias"), ("C", "writeonly"))
    text = str(contract)
    assert text.splitlines()[:12] == [
        "parameter contract of contract_probe:",
        "  checked on the host before every launch:",
        "    M: ge 1, le 1048576, multiple_of 64",
        "    N: ge 1, le 2147483647 (I32 limit)",
        "    ldc: ge N, le 2147483647 (I32 limit)",
        "    batch_stride_c: ge product_of(M, ldc), le 2147483647 (I32 limit)",
        "    T: ge 0, le 4096",
        "    C: align 16",
        "    block: workgroup_size 64",
        "  trusted by the GPU, not checked:",
        "    C: noalias, writeonly",
        "  checked on the GPU (a false fact traps the kernel; the process can no "
        "longer use the GPU):",
    ]
    assert text.splitlines()[12].endswith(": ge 0, le T")


def test_a_failed_launch_names_every_row_it_breaks():
    """The refusal lists each failed row with its value, the declaration that
    states it, and -- for a relation -- the bound with the arguments put in."""
    contract = parameter_contract(_gemm_like_kernel())
    bad = dict(GOOD, C=0x7F3A00000002, M=1000, ldc=100, batch_stride_c=12800)
    support = contract.check(bad, block=(128, 1, 1), at_launch=True)
    assert not support.ok
    assert str(LaunchRefused.of("contract_probe", support.reasons)) == (
        "contract_probe refused this launch:\n"
        "  M = 1000: needs M multiple_of 64"
        '                            (b.param("M", ..., multiple_of=64))\n'
        "  ldc = 100: needs ldc ge N (N = 128)"
        '                         (b.param("ldc", ..., ge=N))\n'
        "  batch_stride_c = 12800: needs batch_stride_c ge product_of(M, ldc)"
        " (= 1000 * 100 = 100000)\n"
        "  C = 0x7f3a00000002: needs align 16\n"
        "  block (128, 1, 1) differs from the kernel's workgroup_size 64 -- "
        "pass (64, 1, 1) or leave block unset"
    )


def test_good_arguments_pass_and_an_unset_or_equal_block_is_accepted():
    contract = parameter_contract(_gemm_like_kernel())
    for block in (None, (64, 1, 1)):
        assert contract.check(GOOD, block=block, at_launch=True).ok


def test_an_applicability_check_leaves_alignment_to_the_launch():
    """Without pointers (a check made before any buffer exists), an alignment
    row is reported as checked at launch, not as passed; at launch a missing
    argument is a failure."""
    contract = parameter_contract(_gemm_like_kernel())
    ints = {k: v for k, v in GOOD.items() if k not in ("C", "cu")}
    support = contract.check(ints)
    assert support.ok
    assert support.checked_at_launch == ("C: align 16 -- checked at launch",)
    assert contract.check(ints, at_launch=True).reasons == ("C: no value given",)


def test_a_missing_or_out_of_range_integer_is_refused():
    contract = parameter_contract(_gemm_like_kernel())
    missing = {k: v for k, v in GOOD.items() if k != "N"}
    assert contract.check(missing).reasons == ("N: no value given",)
    too_big = dict(GOOD, T=1 << 40)
    assert contract.check(too_big).reasons[0] == (
        "T = 1099511627776: outside I32 (-2147483648 .. 2147483647)"
    )


def test_a_relation_that_cannot_be_evaluated_is_refused():
    """The one intexpr evaluator refuses an int64 overflow; the row fails and
    says why instead of comparing against a wrapped bound."""
    b = IRBuilder("overflow_probe")
    m = b.param("M", I64, ge=1)
    ld = b.param("ld", I64, ge=1)
    b.param("stride", I64, ge=product_of(m, ld))
    b.ret()
    contract = parameter_contract(b.kernel)
    reasons = contract.check({"M": 1 << 40, "ld": 1 << 40, "stride": 1}).reasons
    assert reasons == (
        (
            "stride = 1: needs stride ge product_of(M, ld), which cannot be evaluated: "
            f"intexpr mul overflows int64 ({1 << 40}, {1 << 40})"
        ),
    )


def test_a_declaration_the_verifier_refuses_has_no_contract():
    b = IRBuilder("bad_probe")
    b.ret()
    b.kernel.params.append(Param("M", I32, {"ge": 4, "gt": 3}))
    with pytest.raises(ValueError, match=r"param M: states two lower bounds"):
        parameter_contract(b.kernel)


def _artifact(kernel) -> KernelArtifact:
    """A compile artifact as compile_kernel returns it; writing one needs no code
    object, so these bytes are a placeholder."""
    return KernelArtifact(kernel=kernel, ir_text="", llvm_text="", hsaco=b"\0")


def test_write_artifact_refuses_a_kernel_that_states_facts(tmp_path):
    with pytest.raises(ValueError) as caught:
        write_artifact(_artifact(_gemm_like_kernel()), tmp_path, {})
    assert str(caught.value) == (
        "contract_probe: this kernel states parameter facts, and a written .hsaco "
        "does not carry them yet, so a launch from the file would go unchecked. "
        "Launch it with KernelLauncher.from_artifact(artifact)."
    )
    assert not list(tmp_path.iterdir())


def test_write_artifact_still_writes_a_kernel_that_states_only_alignment(tmp_path):
    """Shipped kernels state pointer alignment; their written code objects were
    always launched without checking it, so alignment alone is not refused."""
    b = IRBuilder("align_only")
    b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b.param("M", I32)
    b.ret()
    assert parameter_contract(b.kernel).alignments
    paths = write_artifact(_artifact(b.kernel), tmp_path, {})
    assert paths["hsaco"].read_bytes() == b"\0"


def test_from_artifact_refuses_what_it_cannot_launch_before_loading():
    """An argument type the packer cannot pack, an unknown kernel name or a
    non-artifact is refused before any code object is loaded."""
    b = IRBuilder("i16_probe")
    b.param("h", I16)
    b.ret()
    with pytest.raises(
        ValueError, match=r"parameter h has type i16, which the launcher"
    ):
        KernelLauncher.from_artifact(_artifact(b.kernel))
    with pytest.raises(ValueError, match=r"has no kernel 'other' \(it has i16_probe\)"):
        KernelLauncher.from_artifact(_artifact(b.kernel), "other")
    with pytest.raises(TypeError, match=r"got bytes"):
        KernelLauncher.from_artifact(b"\0")


# --- on the GPU ----------------------------------------------------------------------------------


def _fill_kernel(name: str = "fill_probe"):
    """``Y[tid] = M + tid``, with the facts the host checks: M a multiple of 4,
    Y 16-byte aligned, a block of exactly 64 threads."""
    b = IRBuilder(name)
    b.kernel.attrs["workgroup_size"] = 64
    y = b.param("Y", PtrType(I32, "global"), align=16)
    m = b.param("M", I32, ge=0, le=1 << 20, multiple_of=4)
    tid = b.thread_id_x()
    b.global_store(y, tid, b.add(m, tid))
    b.ret()
    return b.kernel


class _Buffer:
    """128 i32 slots on the device, zeroed; ``read()`` copies them back."""

    def __init__(self):
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.host_buffers import as_u8_buffer

        self._rt, self._u8 = Runtime(), as_u8_buffer
        self.host = np.zeros(128, dtype=np.int32)
        self.device = DeviceMem(self.host.nbytes)
        self._rt.memcpy_h2d(self.device.ptr(), self._u8(self.host), self.host.nbytes)

    def read(self) -> np.ndarray:
        synchronize_and_release()
        self._rt.memcpy_d2h(self._u8(self.host), self.device.ptr(), self.host.nbytes)
        return self.host.copy()


def _expected(m: int) -> np.ndarray:
    out = np.zeros(128, dtype=np.int32)
    out[:64] = m + np.arange(64, dtype=np.int32)
    return out


def _refusal(m: int) -> str:
    return (
        f"fill_probe refused this launch:\n  M = {m}: needs M multiple_of 4"
        + " " * (59 - len(f"M = {m}: needs M multiple_of 4") + 1)
        + '(b.param("M", ..., multiple_of=4))'
    )


def _check_door(launcher: KernelLauncher) -> None:
    """A launch that passes runs one 64-lane block (block left unset); a launch
    that breaks a row is refused before it reaches the GPU."""
    assert launcher.signature == (
        {"name": "Y", "type": "ptr<i32,global>"},
        {"name": "M", "type": "i32"},
    )
    buf = _Buffer()
    launcher({"Y": buf.device, "M": 8}, config=LaunchConfig(grid=(1, 1, 1)))
    np.testing.assert_array_equal(buf.read(), _expected(8))
    untouched = _Buffer()
    with pytest.raises(LaunchRefused) as caught:
        launcher({"Y": untouched.device, "M": 6}, config=LaunchConfig(grid=(1, 1, 1)))
    assert str(caught.value) == _refusal(6)
    np.testing.assert_array_equal(untouched.read(), np.zeros(128, dtype=np.int32))


def test_gpu_from_artifact_checks_a_kernel_compiled_through_llvm():
    arch = current_device_arch()
    _check_door(KernelLauncher.from_artifact(compile_kernel(_fill_kernel(), arch=arch)))


def test_gpu_from_artifact_checks_a_kernel_compiled_through_hipcc():
    """The contract is read from the KernelDef the artifact holds, so a kernel
    compiled through hipcc is checked exactly like one compiled through LLVM."""
    arch = current_device_arch()
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc is not on PATH")
    _check_door(
        KernelLauncher.from_artifact(
            compile_kernel_via_hipcc(_fill_kernel(), arch=arch)
        )
    )


def test_gpu_from_artifact_picks_a_kernel_of_a_module():
    arch = current_device_arch()
    companion = IRBuilder("fill_companion")
    companion.param("Y", PtrType(I32, "global"))
    companion.param("K", I32, eq=3)
    companion.ret()
    artifact = compile_module(
        KernelModule(_fill_kernel(), {"other": companion.kernel}), arch=arch
    )
    primary = KernelLauncher.from_artifact(artifact)
    assert primary.kernel_name == "fill_probe"
    _check_door(primary)
    other = KernelLauncher.from_artifact(artifact, "fill_companion")
    assert [str(r) for r in other.contract.facts] == ["K eq 3"]
    assert other.supports({"K": 4}).reasons[0].startswith("K = 4: needs K eq 3")


def test_gpu_bind_checks_once_when_it_packs():
    arch = current_device_arch()
    launcher = KernelLauncher.from_artifact(compile_kernel(_fill_kernel(), arch=arch))
    buf = _Buffer()
    run = launcher.bind({"Y": buf.device, "M": 12}, config=LaunchConfig(grid=(1, 1, 1)))
    run()
    np.testing.assert_array_equal(buf.read(), _expected(12))
    with pytest.raises(LaunchRefused):
        launcher.bind({"Y": buf.device, "M": 13}, config=LaunchConfig(grid=(1, 1, 1)))


def test_gpu_a_passed_block_must_equal_the_workgroup_size():
    """make_kernel and PipelineLauncher pass their block on; each launch
    accepts one equal to the kernel's workgroup_size and refuses another."""
    arch = current_device_arch()
    launcher = KernelLauncher.from_artifact(compile_kernel(_fill_kernel(), arch=arch))
    buf = _Buffer()
    values = {"Y": buf.device, "M": 16}
    launch_kernel(StreamConfig(), make_kernel(launcher, values, (1, 1, 1), (64, 1, 1)))
    np.testing.assert_array_equal(buf.read(), _expected(16))
    with pytest.raises(LaunchRefused, match=r"block \(128, 1, 1\) differs"):
        launch_kernel(
            StreamConfig(), make_kernel(launcher, values, (1, 1, 1), (128, 1, 1))
        )
    pipeline = PipelineLauncher([launcher])
    buf = _Buffer()
    values = {"Y": buf.device, "M": 20}
    pipeline([values], [LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1))])
    np.testing.assert_array_equal(buf.read(), _expected(20))
    with pytest.raises(LaunchRefused, match=r"block \(32, 1, 1\) differs"):
        pipeline([values], [LaunchConfig(grid=(1, 1, 1), block=(32, 1, 1))])


def test_gpu_first_supported_returns_the_first_launcher_that_accepts():
    arch = current_device_arch()
    aligned = KernelLauncher.from_artifact(compile_kernel(_fill_kernel(), arch=arch))
    loose = IRBuilder("fill_loose")
    y = loose.param("Y", PtrType(I32, "global"))
    m = loose.param("M", I32, ge=0)
    loose.global_store(y, loose.thread_id_x(), m)
    loose.ret()
    ragged = KernelLauncher.from_artifact(compile_kernel(loose.kernel, arch=arch))
    assert first_supported([aligned, ragged], {"M": 8}) is aligned
    assert first_supported([aligned, ragged], {"M": 6}) is ragged
    with pytest.raises(LaunchRefused) as caught:
        first_supported([aligned, ragged], {"M": -1})
    assert str(caught.value).startswith(
        "no candidate accepts this launch:\n" "  fill_probe:\n    M = -1: needs M ge 0"
    )
    assert "\n  fill_loose:\n    M = -1: needs M ge 0" in str(caught.value)


def test_gpu_a_launcher_built_from_bytes_checks_nothing():
    """The raw constructor sees only code-object bytes, which carry no contract:
    it has none, cannot answer supports, and launches as it always did, as does
    launch_torch_kernel, which builds one."""
    arch = current_device_arch()
    artifact = compile_kernel(_fill_kernel(), arch=arch)
    raw = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=[
            {"name": "Y", "type": "ptr<i32,global>"},
            {"name": "M", "type": "i32"},
        ],
    )
    assert raw.contract is None
    with pytest.raises(ValueError, match=r"built from code-object bytes"):
        raw.supports({"M": 8})
    buf = _Buffer()
    raw({"Y": buf.device, "M": 6}, config=LaunchConfig(grid=(1, 1, 1)))
    np.testing.assert_array_equal(buf.read(), _expected(6))
    buf = _Buffer()
    launch_torch_kernel(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=raw.signature,
        values={"Y": buf.device, "M": 10},
        grid=(1, 1, 1),
        block=(64, 1, 1),
        warmup=0,
        attempts=1,
    )
    np.testing.assert_array_equal(buf.read(), _expected(10))
