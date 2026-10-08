# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text validation for the named gfx1250 prefetch user (Phase 6).

``BlockScaledGemmSpec(prefetch=True)`` wires the phase 1 ops into the
block-scaled GEMM: one scalar prefetch of the workgroup's A_scale rows and a
per-lane ``global_prefetch`` of the next K step's A/B rows. The default
(``prefetch=False``) IR is pinned by the representative-IR golden hashes; the
ISA half of the proof, flag on against flag off, lives in
``rocke/examples/gfx1250/isa_features/prefetch_users_verify.py``.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python, lower_kernel_to_llvm
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

_PREFETCH_CALLS = (
    "call void @llvm.amdgcn.s.setreg(",
    "call void @llvm.amdgcn.s.prefetch.data.",
    "call void @llvm.amdgcn.global.prefetch(",
)


def _spec(matrix_path="wmma_scale", *, K=256, prefetch=True, **overrides):
    block_k, scale_dtype = {
        "wmma": (128, "fp32"),
        "wmma_scale": (32, "e8m0"),
        "wmma_scale16": (16, "e8m0"),
    }[matrix_path]
    args = dict(
        name="pf_user",
        M=32,
        N=48,
        K=K,
        block_k=block_k,
        scale_dtype=scale_dtype,
        matrix_path=matrix_path,
        prefetch=prefetch,
    )
    args.update(overrides)
    return BlockScaledGemmSpec(**args)


def _lower(spec, *, flavor="llvm23"):
    return _lower_kernel_to_llvm_python(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor=flavor
    )


def _traffic(llvm):
    """Loads, stores, and MMA calls in order, with SSA names erased."""
    lines = [
        line.strip()
        for line in llvm.splitlines()
        if re.search(r"= load |^\s*store |@llvm\.amdgcn\.wmma\.", line)
        and not line.startswith("declare")
    ]
    return [re.sub(r"%[\w.]+", "%v", line) for line in lines]


class TestBlockScaledGemmPrefetch(unittest.TestCase):
    def test_default_is_off(self):
        spec = BlockScaledGemmSpec(name="pf_user", M=32, N=48, K=256)
        self.assertFalse(spec.prefetch)
        self.assertNotIn("_pf", spec.kernel_name())
        llvm = _lower(spec)
        for call in _PREFETCH_CALLS:
            with self.subTest(call=call):
                self.assertNotIn(call, llvm)

    def test_kernel_name_suffix(self):
        off = _spec(prefetch=False)
        on = _spec()
        self.assertEqual(on.kernel_name(), off.kernel_name() + "_pf")
        self.assertIn(f"@{on.kernel_name()}(", _lower(on))

    def test_scalar_prefetch_of_scale_rows(self):
        # (path, K, scale bytes per row, cache lines of 128 bytes, at least 1).
        cases = (
            ("wmma_scale", 256, 8, 1),
            ("wmma_scale16", 256, 16, 2),
            ("wmma", 512, 16, 2),  # four fp32 groups
            ("wmma", 256, 8, 1),  # two fp32 groups
        )
        for path, k, row_bytes, lines in cases:
            with self.subTest(path=path, K=k):
                llvm = _lower(_spec(path, K=k))
                self.assertEqual(llvm.count("call void @llvm.amdgcn.s.setreg("), 1)
                self.assertIn("call void @llvm.amdgcn.s.setreg(i32 1537, i32 1)", llvm)
                m0 = re.search(
                    r"%(bid\d+) = call i32 @llvm\.amdgcn\.workgroup\.id\.y\(\)\n"
                    r"\s*%(mul\d+) = mul nsw i32 %\1, 16",
                    llvm,
                )
                self.assertIsNotNone(m0)
                self.assertRegex(
                    llvm,
                    rf"%(mul\d+) = mul nsw i32 %{m0.group(2)}, {row_bytes}\n"
                    r"\s*%(goff64\.\d+) = zext i32 %\1 to i64\n"
                    r"\s*%(gptr\d+) = getelementptr inbounds i8, "
                    r"ptr addrspace\(1\) %A_scale, i64 %\2\n"
                    r"\s*call void @llvm\.amdgcn\.s\.prefetch\.data\.p1\("
                    rf"ptr addrspace\(1\) %\3, i32 {lines}\)",
                )
                self.assertEqual(
                    llvm.count("call void @llvm.amdgcn.s.prefetch.data.p1("), 1
                )
                self.assertLess(
                    llvm.index("@llvm.amdgcn.s.setreg(i32"),
                    llvm.index("call void @llvm.amdgcn.s.prefetch.data.p1("),
                )

    def test_global_prefetch_of_next_step(self):
        # (path, K, K per step): one A/B pair for every step but the last.
        for path, k, step in (
            ("wmma_scale", 128, 128),
            ("wmma_scale", 512, 128),
            ("wmma_scale16", 256, 128),
            ("wmma", 128, 64),
            ("wmma", 256, 64),
        ):
            with self.subTest(path=path, K=k):
                llvm = _lower(_spec(path, K=k))
                pairs = re.findall(
                    r"%(add\d+) = add nsw i32 %mul\d+, (\d+)\n"
                    r"\s*%goff64\.\d+ = zext i32 %\1 to i64\n"
                    r"\s*%(gptr\d+) = getelementptr inbounds i8, "
                    r"ptr addrspace\(1\) %([AB]), i64 %goff64\.\d+\n"
                    r"\s*call void @llvm\.amdgcn\.global\.prefetch\("
                    r"ptr addrspace\(1\) %\3, i32 0\)",
                    llvm,
                )
                expected = [
                    (str(k_next), operand)
                    for k_next in range(step, k, step)
                    for operand in ("A", "B")
                ]
                self.assertEqual([(off, op) for _, off, _, op in pairs], expected)
                self.assertEqual(
                    llvm.count("call void @llvm.amdgcn.global.prefetch("),
                    len(expected),
                )

    def test_prefetch_precedes_the_step_it_serves(self):
        llvm = _lower(_spec("wmma_scale", K=256))
        first_prefetch = llvm.index("call void @llvm.amdgcn.global.prefetch(")
        first_mma = llvm.index("call <8 x float> @llvm.amdgcn.wmma.")
        self.assertLess(first_prefetch, first_mma)

    def test_memory_traffic_unchanged(self):
        # The hints add no loads or stores and leave the MMA chain intact.
        for path, k in (("wmma", 256), ("wmma_scale", 256), ("wmma_scale16", 256)):
            with self.subTest(path=path):
                on = _traffic(_lower(_spec(path, K=k)))
                off = _traffic(_lower(_spec(path, K=k, prefetch=False)))
                self.assertTrue(off)
                self.assertEqual(on, off)

    def test_requires_llvm23(self):
        # The legacy wmma path is the one that also lowers under llvm22.
        with self.assertRaisesRegex(ValueError, "requires LLVM flavor llvm23"):
            _lower(_spec("wmma"), flavor="llvm22")
        # Flag off keeps lowering on every flavor the kernel supported before.
        self.assertIn(
            "define amdgpu_kernel",
            _lower(_spec("wmma", prefetch=False), flavor="llvm22"),
        )

    def test_serialization_roundtrip(self):
        kernel = build_block_scaled_gemm(_spec("wmma", K=256))
        text = serialize(kernel)
        parsed = parse(text)
        self.assertEqual(text, serialize(parsed))
        self.assertEqual(
            _lower_kernel_to_llvm_python(kernel, arch="gfx1250", llvm_flavor="llvm23"),
            _lower_kernel_to_llvm_python(parsed, arch="gfx1250", llvm_flavor="llvm23"),
        )

    def test_backend_dispatch_matches_python(self):
        # Under ROCKE_BACKEND=cpp or both this is the C++ engine's byte check.
        for path in ("wmma", "wmma_scale"):
            with self.subTest(path=path):
                kernel = build_block_scaled_gemm(_spec(path, K=256))
                self.assertEqual(
                    lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23"),
                    _lower_kernel_to_llvm_python(
                        kernel, arch="gfx1250", llvm_flavor="llvm23"
                    ),
                )

    def test_spec_is_otherwise_unchanged(self):
        spec = _spec(prefetch=False)
        self.assertEqual(replace(spec, prefetch=True).prefetch, True)
        self.assertEqual(
            serialize(build_block_scaled_gemm(spec)),
            serialize(
                build_block_scaled_gemm(
                    BlockScaledGemmSpec(
                        name=spec.name,
                        M=spec.M,
                        N=spec.N,
                        K=spec.K,
                        block_k=spec.block_k,
                        scale_dtype=spec.scale_dtype,
                        matrix_path=spec.matrix_path,
                    )
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
