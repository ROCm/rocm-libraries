# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerics for stream-K implicit-GEMM backward-weight convolution.

Every case builds, compiles and launches a stream-K wgrad kernel, then checks
dW against a float32 ``torch.nn.grad.conv2d_weight`` reference computed on the
CPU. Before the tolerance check a row-coverage guard reports whole output
channels that were never written (a dropped group or tile reports as such
rather than as an opaque numeric miss).

The shapes are chosen to exercise the partitioner, not the GEMM: several
stream-K CTAs per tile (multi-round linear/tree fixups), a data-parallel plus
stream-K mix, a grouped problem whose last M tile is partial, and a small CTA
pool so the stream-K path is taken at these sizes. A shape large enough to
split under the default (per-CU) pool runs every reduction too.

The linear and tree fixups are also run repeatedly and must be bit-identical
across launches. Before every launch the partial slots and dW are poisoned
with NaN bytes (only the flags are zeroed, as the contract requires), so a
consumer that reads a slot before its producer's partial is visible sees NaN
rather than the previous launch's bit-identical partial. That catches a
missing *release* on the producer side. It does not catch a missing
*acquire*: the consumer's partial loads are data- and control-dependent on
the spin exit and no stale cache line exists to observe, so a relaxed spin
still reads the right bytes on this hardware. The acquire side is pinned by
codegen instead -- ``TestConvWgradStreamKFlagOrderingIR`` checks the lowered
``.ll`` (host-only), ``TestConvWgradStreamKFlagOrderingIsa`` the gfx942/gfx950
ISA (COMGR + ``llvm-objdump``, no GPU; skips when either is missing).

The numeric cases require a gfx942/gfx950 GPU and torch and skip cleanly
otherwise. Host-side coverage lives in ``test_conv_wgrad_streamk.py``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH not in ("gfx942", "gfx950"):
        return f"stream-K wgrad is gfx942/gfx950 only; got {GPU_ARCH}"
    return ""


_SKIP_REASON = _skip_reason()

# Relative max-error bound by dW dtype: the reference is fp32 over fp16/bf16
# inputs, so the bound is set by the output rounding.
_TOL = {"fp32": 1e-4, "fp16": 2e-3, "bf16": 2e-2}
_REDUCTIONS = (
    ("linear", "fp16"),
    ("tree", "fp16"),
    ("atomic", "fp32"),
    ("workspace", "bf16"),
)


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _reference(X, dY, p):
    import torch

    w = torch.nn.grad.conv2d_weight(
        X.permute(0, 3, 1, 2).contiguous(),
        weight_size=(p.K, p.C // p.groups, p.Y, p.X),
        grad_output=dY.permute(0, 3, 1, 2).contiguous(),
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return w.permute(0, 2, 3, 1).contiguous()


def _run(spec, *, launches: int = 1, problem=None, artifact=None):
    """Build, launch ``launches`` times, return (dW outputs, fp32 reference).

    ``problem`` is the launch shape (default ``spec.problem``): the kernel is
    AOT, so one binary -- passed in as ``artifact`` to reuse it -- serves any
    shape its spec admits.
    """
    import torch

    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from kernels.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
        wgrad_streamk_grid,
        wgrad_streamk_launch_values,
        wgrad_streamk_signature,
        wgrad_streamk_workspace_layout,
        wgrad_streamk_workspace_nbytes,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
        wgrad_reduce_grid,
        wgrad_reduce_signature,
    )

    arch = GPU_ARCH
    p = problem if problem is not None else spec.problem
    art = artifact or compile_kernel(
        build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
    )
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=wgrad_streamk_signature(spec),
    )
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch.manual_seed(0)
    X = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(dtypes[spec.data.dtype_b])
    dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1, 1).to(dtypes[spec.data.dtype_a])
    dW = torch.empty(p.K, p.Y, p.X, p.C // p.groups, dtype=dtypes[spec.data.dtype_d])
    ref = _reference(X.float(), dY.float(), p)

    rt = Runtime()
    ws_bytes = wgrad_streamk_workspace_nbytes(spec, arch=arch, problem=p)
    bufs = [rt.alloc(dY.nbytes), rt.alloc(X.nbytes), rt.alloc(dW.nbytes)]
    ws = rt.alloc(max(ws_bytes, 4))
    bufs.append(ws)
    rt.memcpy_h2d(bufs[0], _u8(dY), dY.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(X), X.nbytes)
    reduction = spec.streamk_reduction
    flags_bytes, partials_bytes = wgrad_streamk_workspace_layout(
        spec, arch=arch, problem=p
    )
    extra = {}
    if reduction == "workspace":
        extra = dict(ws_ptr=ws, ws_bytes=ws_bytes)
    elif reduction in ("linear", "tree"):
        extra = dict(sk_flags=ws, sk_partials=ws + flags_bytes)
    values = wgrad_streamk_launch_values(
        spec,
        bufs[0],
        bufs[1],
        bufs[2],
        dY.nbytes,
        X.nbytes,
        dW.nbytes,
        arch=arch,
        problem=p,
        **extra,
    )
    if reduction == "workspace":
        rspec = WgradReduceSpec(
            problem=p,
            dtype_d=spec.data.dtype_d,
            groups=p.groups,
            ws_replicas=spec.ws_replicas,
        )
        rart = compile_kernel(
            build_conv_wgrad_workspace_reduce(rspec, arch=arch), arch=arch
        )
        reducer = KernelLauncher(
            hsaco=rart.hsaco,
            kernel_name=rart.kernel_name,
            signature=wgrad_reduce_signature(rspec),
        )
        rvalues = {
            "ws_ptr": ws,
            "dw_ptr": bufs[2],
            "wg_M": p.K // p.groups,
            "wg_N": p.Y * p.X * (p.C // p.groups),
            "ws_bytes": ws_bytes,
            "dw_bytes": dW.nbytes,
            "groups": p.groups,
        }

    outs = []
    try:
        for _ in range(launches):
            # The documented zeroing contract, per reduction: atomic accumulates
            # into dW, workspace into the scratch, linear/tree only reset flags.
            # Everything the contract leaves alone is poisoned (0xFF bytes are
            # NaN in every float dtype): a stale value from the previous
            # launch would otherwise be bit-identical to the right one and
            # hide an early read of a partial or an unwritten dW element.
            if reduction == "atomic":
                rt.memset(bufs[2], 0, dW.nbytes)
            else:
                rt.memset(bufs[2], 0xFF, dW.nbytes)
            if reduction == "workspace":
                rt.memset(ws, 0, ws_bytes)
            elif reduction in ("linear", "tree"):
                if partials_bytes:
                    rt.memset(ws + flags_bytes, 0xFF, partials_bytes)
                if flags_bytes:
                    rt.memset(ws, 0, flags_bytes)
            launcher(
                values,
                config=LaunchConfig(
                    grid=wgrad_streamk_grid(spec, arch=arch, problem=p),
                    block=(spec.block_size, 1, 1),
                    fence=True,
                ),
            )
            if reduction == "workspace":
                reducer(
                    rvalues,
                    config=LaunchConfig(
                        grid=wgrad_reduce_grid(rspec),
                        block=(rspec.block_size, 1, 1),
                        fence=True,
                    ),
                )
            out = torch.empty_like(dW)
            rt.memcpy_d2h(_u8(out), bufs[2], dW.nbytes)
            outs.append(out)
    finally:
        for buf in bufs:
            rt.free(buf)
    return outs, ref


def _check(test, spec, *, launches: int = 1, problem=None, artifact=None):
    outs, ref = _run(spec, launches=launches, problem=problem, artifact=artifact)
    p = problem if problem is not None else spec.problem
    out = outs[0].float()
    written = out.reshape(p.K, -1).abs().sum(dim=1) > 0
    expected = ref.reshape(p.K, -1).abs().sum(dim=1) > 0
    missing = int((expected & ~written).sum())
    test.assertEqual(
        missing, 0, f"{missing}/{p.K} dW output-channel rows were never written"
    )
    rel = float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    test.assertLess(rel, _TOL[spec.data.dtype_d], f"rel_err={rel:.3e}")
    return outs


def _spec(problem, *, mode, reduction, dtype_d, ctas):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    ab = "bf16" if dtype_d == "bf16" else "fp16"
    edge = 32 if GPU_ARCH == "gfx950" else 16
    return WgradConvSpec(
        problem=problem,
        data=ConvDataSpec(dtype_a=ab, dtype_b=ab, dtype_d=dtype_d),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=16,
        streamk=mode,
        streamk_reduction=reduction,
        streamk_ctas=ctas,
    )


def _problems():
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return {
        # 6 tiles, 32 iterations each, over 16 CTAs: up to 3 partners per tile,
        # two tree rounds.
        "multi_partner": (
            ConvProblem(N=8, Hi=16, Wi=16, C=16, K=96, Y=3, X=3, pH=1, pW=1),
            16,
        ),
        # 25 tiles over 7 CTAs: 14 data-parallel tiles plus 11 stream-K tiles.
        "dp_plus_sk": (
            ConvProblem(N=2, Hi=8, Wi=8, C=32, K=320, Y=3, X=3, pH=1, pW=1),
            7,
        ),
        "strided": (
            ConvProblem(
                N=3, Hi=13, Wi=11, C=24, K=72, Y=3, X=3, sH=2, sW=2, pH=1, pW=1
            ),
            5,
        ),
    }


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKNumerics(unittest.TestCase):
    def test_modes_and_reductions(self):
        for name, (problem, ctas) in _problems().items():
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in _REDUCTIONS:
                    spec = _spec(
                        problem,
                        mode=mode,
                        reduction=reduction,
                        dtype_d=dtype_d,
                        ctas=ctas,
                    )
                    with self.subTest(shape=name, mode=mode, reduction=reduction):
                        _check(self, spec)

    def test_default_pool(self):
        # streamk_ctas=-1: one CTA per CU for linear/tree, the occupancy-sized
        # pool (several CTAs per CU) for the reductions that never wait.
        # dp_plus_sk / multi_partner have far fewer tiles than that pool, so
        # they run all-DP; "splits" (17 x 27 = 459 tiles) is the one that
        # actually takes a linear/tree/workspace/atomic fixup at the default
        # pool (on gfx950: 256 linear/tree CTAs, 1024 atomic/workspace CTAs).
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_wgrad import wgrad_streamk_partition

        shapes = {
            name: _problems()[name][0] for name in ("dp_plus_sk", "multi_partner")
        }
        shapes["splits"] = ConvProblem(
            N=2, Hi=12, Wi=12, C=192, K=1056, Y=3, X=3, pH=1, pW=1
        )
        for name, problem in shapes.items():
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in _REDUCTIONS:
                    spec = _spec(
                        problem,
                        mode=mode,
                        reduction=reduction,
                        dtype_d=dtype_d,
                        ctas=-1,
                    )
                    with self.subTest(shape=name, mode=mode, reduction=reduction):
                        if name == "splits":
                            part = wgrad_streamk_partition(spec, arch=GPU_ARCH)
                            self.assertGreater(part.sk_ctas, 0, part)
                            self.assertGreater(part.sk_tiles, 0, part)
                        _check(self, spec)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKGrouped(unittest.TestCase):
    """Groups folded into GEMM-M, including a partial last M tile per group."""

    def test_grouped(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        problems = {
            "g3_partial_m": (
                ConvProblem(
                    N=4, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3
                ),
                6,
            ),
            "g8": (
                ConvProblem(
                    N=2, Hi=12, Wi=12, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=8
                ),
                5,
            ),
        }
        for name, (problem, ctas) in problems.items():
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in _REDUCTIONS:
                    spec = _spec(
                        problem,
                        mode=mode,
                        reduction=reduction,
                        dtype_d=dtype_d,
                        ctas=ctas,
                    )
                    with self.subTest(shape=name, mode=mode, reduction=reduction):
                        _check(self, spec)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKAot(unittest.TestCase):
    """One binary per spec serves every launch shape.

    The partition is a kernarg block, so a kernel built for one problem must
    be correct on others -- with a different tile count, iteration count,
    data-parallel/stream-K split and group count.
    """

    def test_one_binary_many_shapes(self):
        from rocke import compile_kernel
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
        )

        build = ConvProblem(N=2, Hi=8, Wi=8, C=48, K=96, Y=3, X=3, pH=1, pW=1, groups=3)
        launches = [
            ConvProblem(N=4, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3),
            ConvProblem(N=8, Hi=16, Wi=16, C=24, K=96, Y=3, X=3, pH=1, pW=1, groups=3),
            ConvProblem(
                N=3,
                Hi=13,
                Wi=11,
                C=72,
                K=72,
                Y=3,
                X=3,
                sH=2,
                sW=2,
                pH=1,
                pW=1,
                groups=3,
            ),
        ]
        for mode in ("dp_sk", "persistent"):
            for reduction, dtype_d in _REDUCTIONS:
                spec = _spec(
                    build, mode=mode, reduction=reduction, dtype_d=dtype_d, ctas=6
                )
                art = compile_kernel(
                    build_implicit_gemm_conv_wgrad(spec, arch=GPU_ARCH), arch=GPU_ARCH
                )
                for i, problem in enumerate(launches):
                    with self.subTest(mode=mode, reduction=reduction, shape=i):
                        _check(self, spec, problem=problem, artifact=art)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKDeterminism(unittest.TestCase):
    """Linear and tree fold partials in a fixed order: dW is bit-exact.

    With the partial slots NaN-poisoned this also catches a missing release on
    the partial hand-off, but not a missing acquire -- see the module
    docstring and ``TestConvWgradStreamKFlagOrderingIR``/``...Isa``.
    """

    def test_repeat_launches_are_bit_identical(self):
        import torch

        problem, ctas = _problems()["multi_partner"]
        for mode in ("dp_sk", "persistent"):
            for reduction in ("linear", "tree"):
                spec = _spec(
                    problem, mode=mode, reduction=reduction, dtype_d="fp16", ctas=ctas
                )
                with self.subTest(mode=mode, reduction=reduction):
                    outs = _check(self, spec, launches=20)
                    for i, out in enumerate(outs[1:], start=1):
                        self.assertTrue(
                            torch.equal(out, outs[0]),
                            f"launch {i} differs from launch 0",
                        )


# --------------------------------------------------------------------------
# Flag hand-off ordering, pinned in codegen.
#
# The repeat-launch test above catches a missing release but cannot catch a
# missing acquire (see the module docstring), so the acquire -- and, for
# symmetry, the release -- is asserted on the code a real linear and tree
# kernel lowers to. Host-only: no GPU is needed for either level.

_FLAG_ARCHES = ("gfx950", "gfx942")
_FLAG_KERNELS = (
    ("dp_sk", "linear"),
    ("dp_sk", "tree"),
    ("persistent", "linear"),
    ("persistent", "tree"),
)


def _flag_comgr_input(arch: str, mode: str, reduction: str):
    """The lowered ``.ll`` + COMGR target of a real stream-K fixup kernel.

    Always the Python engine (the C++ engine is byte-identical by contract and
    its extension may not be importable), and lowered with ``arch`` as the
    target rather than the detected GPU, so the gfx942 code is checked on a
    gfx950 host and vice versa.
    """
    from rocke.helpers.compile import lower_kernel_for_comgr

    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        build_implicit_gemm_conv_wgrad,
    )

    edge = 32 if arch == "gfx950" else 16
    spec = WgradConvSpec(
        problem=ConvProblem(N=8, Hi=16, Wi=16, C=16, K=96, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=16,
        streamk=mode,
        streamk_reduction=reduction,
        streamk_ctas=16,
    )
    kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
    return lower_kernel_for_comgr(kernel, arch=arch, backend="python")


def _ll_blocks(ll: str) -> dict:
    """``{label: [instruction lines]}`` of every basic block in ``ll``."""
    blocks, cur = {}, None
    for line in ll.splitlines():
        m = re.match(r"^([A-Za-z0-9_.$-]+):", line)
        if m:
            cur = blocks.setdefault(m.group(1), [])
        elif cur is not None and line.startswith("  "):
            cur.append(line.strip())
    return blocks


def _gep_base(ll: str, ptr: str) -> str:
    """Base pointer of the ``getelementptr`` that defines ``ptr``."""
    m = re.search(
        rf"^\s*{re.escape(ptr)} = getelementptr\b[^,]*, ptr addrspace\(1\) (%[\w.]+),",
        ll,
        re.MULTILINE,
    )
    return m.group(1) if m else ""


def _assert_flag_ll(test, ll: str) -> None:
    """Acquire on every flag spin, fence + release on every flag publish."""
    blocks = _ll_blocks(ll)
    load_re = re.compile(
        r"^(%[\w.]+) = load atomic i32, ptr addrspace\(1\) (%[\w.]+) (.*), align 4$"
    )
    store_re = re.compile(
        r"^store atomic i32 1, ptr addrspace\(1\) (%[\w.]+) (.*), align 4$"
    )
    spins = publishes = 0
    for label, insts in blocks.items():
        for i, inst in enumerate(insts):
            m = load_re.match(inst)
            if m:
                spins += 1
                test.assertEqual(_gep_base(ll, m.group(2)), "%sk_flags", inst)
                test.assertEqual(m.group(3), 'syncscope("agent") acquire', inst)
                # A spin: the block re-branches to itself until the flag hits.
                test.assertRegex(
                    insts[-1], rf"^br i1 .*label %{re.escape(label)}\b", label
                )
                continue
            m = store_re.match(inst)
            if m:
                publishes += 1
                test.assertEqual(_gep_base(ll, m.group(1)), "%sk_flags", inst)
                test.assertEqual(m.group(2), 'syncscope("agent") release', inst)
                # The nearest preceding memory op is the agent release fence,
                # and every partial store of the block is ordered before it.
                mem = [
                    j
                    for j in range(i)
                    if re.match(
                        r"^(fence|store|%[\w.]+ = (load|atomicrmw))\b", insts[j]
                    )
                ]
                test.assertTrue(mem, f"{label}: flag store with no fence before it")
                test.assertEqual(
                    insts[mem[-1]], 'fence syncscope("agent") release', label
                )
                test.assertTrue(
                    any(re.match(r"^store float ", insts[j]) for j in mem[:-1]),
                    f"{label}: no partial store before the release fence",
                )
    test.assertGreater(spins, 0, "no flag-wait spin loop in the kernel")
    test.assertGreater(publishes, 0, "no flag publish in the kernel")


def _objdump() -> str | None:
    for candidate in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _isa_lines(objdump: str, comgr_input, arch: str) -> list:
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    hsaco, _ = build_hsaco_from_llvm_ir(
        comgr_input.llvm_text,
        isa=comgr_input.isa,
        options=list(comgr_input.options),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "streamk.hsaco"
        path.write_bytes(hsaco)
        text = subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    lines = (ln.split("//")[0].strip() for ln in text.splitlines())
    return [ln for ln in lines if ln and not ln.endswith(">:")]


_SC1_LOAD = re.compile(r"^global_load_dword\b.*\bsc1\b")
_SC1_STORE = re.compile(r"^global_store_dword\b.*\bsc1\b")


def _assert_flag_isa(test, isa: list) -> None:
    """gfx94x agent-scope acquire/release: ``buffer_inv sc1`` after each spin
    load, ``buffer_wbl2 sc1`` + ``s_waitcnt vmcnt(0)`` before each flag store.
    """
    spins = [i for i, ln in enumerate(isa) if _SC1_LOAD.match(ln)]
    stores = [i for i, ln in enumerate(isa) if _SC1_STORE.match(ln)]
    test.assertTrue(spins, "no sc1 flag load in the ISA")
    test.assertTrue(stores, "no sc1 flag store in the ISA")
    for i in spins:
        # Up to the spin's back-branch: wait for the load, then invalidate so
        # the partial loads after the exit cannot hit a stale line.
        end = next(
            (j for j in range(i + 1, len(isa)) if isa[j].startswith("s_cbranch")),
            len(isa),
        )
        body = isa[i + 1 : end]
        test.assertIn("buffer_inv sc1", body, f"spin at {i}: no acquire invalidate")
        test.assertLess(
            body.index("s_waitcnt vmcnt(0)"),
            body.index("buffer_inv sc1"),
            f"spin at {i}: buffer_inv before the flag load completed",
        )
    for i in stores:
        # Back to the last partial store: write back L2 and drain every
        # outstanding store before the flag goes out.
        start = max(
            (
                j
                for j in range(i)
                if isa[j].startswith(("global_store", "buffer_store", "global_atomic"))
            ),
            default=-1,
        )
        window = isa[start + 1 : i]
        test.assertIn("buffer_wbl2 sc1", window, f"flag store at {i}: no writeback")
        test.assertIn(
            "s_waitcnt vmcnt(0)", window, f"flag store at {i}: stores not drained"
        )


class TestConvWgradStreamKFlagOrderingIR(unittest.TestCase):
    """The lowered ``.ll`` of real linear/tree kernels orders the hand-off."""

    def test_acquire_spin_and_release_publish(self):
        for arch in _FLAG_ARCHES:
            for mode, reduction in _FLAG_KERNELS:
                with self.subTest(arch=arch, mode=mode, reduction=reduction):
                    ll = _flag_comgr_input(arch, mode, reduction).llvm_text
                    _assert_flag_ll(self, ll)


class TestConvWgradStreamKFlagOrderingIsa(unittest.TestCase):
    """The same kernels' gfx942/gfx950 ISA keeps the cache maintenance.

    Offline compile (COMGR) + ``llvm-objdump``; no GPU. Skips when either
    tool is unavailable.
    """

    def test_acquire_invalidate_and_release_writeback(self):
        from rocke.runtime import comgr

        objdump = _objdump()
        if objdump is None:
            self.skipTest("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
        try:
            comgr._resolve_lib()
        except (comgr.ComgrError, OSError) as e:
            self.skipTest(f"COMGR not loadable: {str(e)[:200]}")
        for arch in _FLAG_ARCHES:
            for mode, reduction in _FLAG_KERNELS:
                with self.subTest(arch=arch, mode=mode, reduction=reduction):
                    c = _flag_comgr_input(arch, mode, reduction)
                    _assert_flag_isa(self, _isa_lines(objdump, c, arch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
