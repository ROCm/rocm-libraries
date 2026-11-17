# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The binding seam: turning a selection into something runnable.

Selection says *which* kernel; :class:`ProblemBinding` says *how to call it*.
These tests pin the seam itself and the GEMM implementation of it, using a
fake runtime so they run on a machine with no GPU -- which is the property the
binding's ``rt``-as-parameter shape exists to preserve.
"""

from __future__ import annotations

import struct
import unittest

from rocke.core.arch import known_arches
from rocke.dispatch.core import Capability, KernelCandidate, ProblemBinding
from rocke.dispatch.gemm.bf16_rcr import dispatch_gemm_bf16, gemm_bf16_candidates
from rocke.dispatch.gemm.common import GemmRequest
from rocke.dispatch.gemm.fp16_rcr import dispatch_gemm_fp16, gemm_fp16_candidates


class FakeRuntime:
    """Host-memory stand-in for the HIP runtime's allocate/copy surface."""

    def __init__(self) -> None:
        self.mem: dict[int, bytearray] = {}
        self._next = 0x1000

    def alloc(self, n: int) -> int:
        ptr = self._next
        self._next += max(n, 1) + 4096
        self.mem[ptr] = bytearray(n)
        return ptr

    def memcpy_h2d(self, ptr: int, buf, n: int) -> None:
        self.mem[ptr][:n] = bytes(memoryview(buf))[:n]

    def memcpy_d2h(self, buf, ptr: int, n: int) -> None:
        memoryview(buf).cast("B")[:n] = self.mem[ptr][:n]

    def memset(self, ptr: int, value: int, n: int) -> None:
        self.mem[ptr][:n] = bytes([value]) * n


def _write_reference_output(rt, binding, ptrs, req, dtype):
    """Fill the C buffer with the exact expected result, as a perfect kernel would."""
    import numpy as np

    from rocke.dispatch.gemm.binding import _SEED, _bf16_from_f32

    rng = np.random.default_rng(_SEED)
    a = rng.integers(-5, 6, size=(req.M, req.K), dtype=np.int16)
    b = rng.integers(-5, 6, size=(req.N, req.K), dtype=np.int16)
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    encoded = ref.astype(np.float16) if dtype == "fp16" else _bf16_from_f32(np, ref)
    rt.mem[ptrs[2]][: encoded.nbytes] = encoded.tobytes()


class TestBindSeam(unittest.TestCase):
    def test_candidate_without_bind_explains_itself(self):
        candidate = KernelCandidate(
            name="selection_only",
            family="demo",
            algorithm="demo",
            spec_id="demo",
            abi_version="demo/v1",
            priority=10,
            capability=Capability(arches=known_arches()),
            _supports=lambda req: (True, ""),
            select_spec=lambda req: None,
            signature=lambda spec: (),
            grid=lambda spec, req: (1, 1, 1),
            block=lambda spec: (1, 1, 1),
            sweep_space=lambda req: (),
        )
        with self.assertRaises(NotImplementedError) as ctx:
            candidate.bound(object())
        # The message has to name the candidate and the remedy: a selectable
        # but unlaunchable candidate is a gap, not an error in the caller.
        self.assertIn("selection_only", str(ctx.exception))
        self.assertIn("bind", str(ctx.exception))

    def test_binding_adapts_to_the_manifest_problem_builder_tuple(self):
        binding = ProblemBinding(
            grid=(2, 3, 4),
            block=(64, 1, 1),
            make_args=lambda rt: (b"", ()),
            check=lambda rt, ptrs: (0.0, 0, 0),
            flop=1.0,
            bytes_moved=2.0,
        )
        make_args, grid, block, flop, bytes_xfer, check = binding.as_problem_builder()
        self.assertEqual(
            (grid, block, flop, bytes_xfer), ((2, 3, 4), (64, 1, 1), 1.0, 2.0)
        )
        self.assertIs(make_args, binding.make_args)
        self.assertIs(check, binding.check)


class TestGemmBinding(unittest.TestCase):
    CASES = (
        ("fp16", dispatch_gemm_fp16),
        ("bf16", dispatch_gemm_bf16),
    )

    def _request(self, dtype):
        return GemmRequest(M=64, N=128, K=256, arch="gfx950", dtype=dtype)

    def test_geometry_comes_from_the_dispatcher_not_recomputed(self):
        # The whole point of binding on the candidate: one definition of the
        # launch geometry. If a binding ever recomputed it from tile fields,
        # this is the test that catches the drift.
        for dtype, dispatch in self.CASES:
            with self.subTest(dtype=dtype):
                result = dispatch(self._request(dtype))
                binding = result.bind()
                self.assertEqual(binding.grid, result.grid)
                self.assertEqual(binding.block, result.block)

    def test_roofline_denominators(self):
        for dtype, dispatch in self.CASES:
            with self.subTest(dtype=dtype):
                req = self._request(dtype)
                binding = dispatch(req).bind()
                self.assertEqual(binding.flop, 2.0 * req.M * req.N * req.K)
                self.assertEqual(
                    binding.bytes_moved,
                    2.0 * (req.M * req.K + req.N * req.K + req.M * req.N),
                )

    def test_packed_args_match_the_declared_signature(self):
        for dtype, dispatch in self.CASES:
            with self.subTest(dtype=dtype):
                req = self._request(dtype)
                result = dispatch(req)
                args, ptrs = result.bind().make_args(FakeRuntime())
                self.assertEqual(len(ptrs), 3)
                a, b, c, m, n, k = struct.unpack("<QQQiii", args)
                self.assertEqual((m, n, k), (req.M, req.N, req.K))
                self.assertEqual((a, b, c), ptrs)

    def test_verify_accepts_an_exact_kernel_and_rejects_a_wrong_one(self):
        for dtype, dispatch in self.CASES:
            with self.subTest(dtype=dtype):
                req = self._request(dtype)
                binding = dispatch(req).bind(verify=True)
                rt = FakeRuntime()
                _args, ptrs = binding.make_args(rt)

                _write_reference_output(rt, binding, ptrs, req, dtype)
                _diff, bad, total = binding.check(rt, ptrs)
                self.assertEqual(bad, 0, "an exact kernel must verify clean")
                self.assertEqual(total, req.M * req.N)

                # make_args memset C to zero, so re-binding gives a "kernel
                # that never wrote anything" -- the check must notice.
                rt_zero = FakeRuntime()
                _args, zero_ptrs = binding.make_args(rt_zero)
                _diff, bad_zero, _total = binding.check(rt_zero, zero_ptrs)
                self.assertGreater(bad_zero, 0, "a silent kernel must fail verify")

    def test_binding_without_verify_does_no_reference_work(self):
        req = self._request("fp16")
        binding = dispatch_gemm_fp16(req).bind()
        rt = FakeRuntime()
        _args, ptrs = binding.make_args(rt)
        diff, bad, total = binding.check(rt, ptrs)
        self.assertEqual((diff, bad), (0.0, 0))
        self.assertEqual(total, req.M * req.N)

    def test_every_registered_gemm_candidate_can_bind(self):
        # Registration and launchability should not drift apart within a
        # family that has opted into binding.
        for candidate in gemm_fp16_candidates() + gemm_bf16_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.bind)


class TestBindingRatchet(unittest.TestCase):
    """`bind` is mandatory per family, and the set of exempt families shrinks.

    Not a global mandate like `capability`, because `capability` is a
    declaration every candidate can make and `bind` is behavior that has to be
    written. A family opts in once backfilled; these tests make sure the opt-in
    is real and that the families still exempt are exempt on purpose.
    """

    # Every family whose candidates cannot yet be launched, with the reason.
    # Deleting an entry here is how a family graduates; adding one should
    # require the same argument the originals did.
    _EXEMPT = {
        # No args signature declared yet (`signature=lambda _spec: ()`), so
        # there is nothing to pack. Backfilling the signature comes first.
        "norm2d": "no args signature",
        # Also has no grid: it is computed at runtime from num_m_blocks.
        "moe_fused_mega": "no args signature, runtime grid",
    }

    def _registries(self):
        from rocke.dispatch.families.moe import MOE_REGISTRY
        from rocke.dispatch.families.norm import NORM_REGISTRY
        from rocke.dispatch.gemm.bf16_rcr import GEMM_BF16_REGISTRY
        from rocke.dispatch.gemm.fp16_rcr import GEMM_FP16_REGISTRY

        return (
            GEMM_FP16_REGISTRY,
            GEMM_BF16_REGISTRY,
            MOE_REGISTRY,
            NORM_REGISTRY,
        )

    def test_a_requiring_family_has_no_unbound_candidates(self):
        for registry in self._registries():
            if not registry.require_binding:
                continue
            for candidate in registry.candidates():
                with self.subTest(family=registry.family, candidate=candidate.name):
                    self.assertIsNotNone(candidate.bind)

    def test_exempt_families_are_exactly_the_documented_ones(self):
        exempt = {r.family for r in self._registries() if not r.require_binding}
        self.assertEqual(
            exempt,
            set(self._EXEMPT),
            "a family changed its binding stance; update _EXEMPT with the reason",
        )

    def test_registration_fails_when_a_required_binding_is_missing(self):
        from rocke.dispatch.core import CandidateRegistry

        registry = CandidateRegistry("demo", require_binding=True)
        candidate = KernelCandidate(
            name="unbound",
            family="demo",
            algorithm="demo",
            spec_id="demo",
            abi_version="demo/v1",
            priority=10,
            capability=Capability(arches=known_arches()),
            _supports=lambda req: (True, ""),
            select_spec=lambda req: None,
            signature=lambda spec: (),
            grid=lambda spec, req: (1, 1, 1),
            block=lambda spec: (1, 1, 1),
            sweep_space=lambda req: (),
        )
        with self.assertRaises(ValueError) as ctx:
            registry.register(candidate)
        message = str(ctx.exception)
        self.assertIn("unbound", message)
        self.assertIn("bind", message)

    def test_a_non_requiring_family_still_accepts_an_unbound_candidate(self):
        from rocke.dispatch.core import CandidateRegistry

        registry = CandidateRegistry("demo")
        registry.register(
            KernelCandidate(
                name="unbound",
                family="demo",
                algorithm="demo",
                spec_id="demo",
                abi_version="demo/v1",
                priority=10,
                capability=Capability(arches=known_arches()),
                _supports=lambda req: (True, ""),
                select_spec=lambda req: None,
                signature=lambda spec: (),
                grid=lambda spec, req: (1, 1, 1),
                block=lambda spec: (1, 1, 1),
                sweep_space=lambda req: (),
            )
        )
        self.assertEqual(len(registry.candidates()), 1)

    def test_coverage_reports_bindability(self):
        from rocke.dispatch.gemm.fp16_rcr import GEMM_FP16_REGISTRY

        coverage = GEMM_FP16_REGISTRY.coverage()
        self.assertTrue(coverage["requires_binding"])
        self.assertTrue(all(c["bindable"] for c in coverage["candidates"]))


class TestBf16Encoding(unittest.TestCase):
    def test_round_trip_is_exact_for_bf16_representable_values(self):
        import numpy as np

        from rocke.dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16

        values = np.array([0.0, 1.0, -1.0, 5.0, -256.0, 1024.0], dtype=np.float32)
        back = _f32_from_bf16(np, _bf16_from_f32(np, values))
        np.testing.assert_array_equal(back, values)

    def test_rounding_is_nearest_even_not_truncation(self):
        import numpy as np

        from rocke.dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16

        # A value just above a bf16 tick must round up, not truncate down.
        one_ulp = np.float32(1.0 + 2.0**-8)
        nudged = np.array([one_ulp * (1.0 + 2.0**-12)], dtype=np.float32)
        back = _f32_from_bf16(np, _bf16_from_f32(np, nudged))
        self.assertGreaterEqual(float(back[0]), float(one_ulp))


if __name__ == "__main__":
    unittest.main()
