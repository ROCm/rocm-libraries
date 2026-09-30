# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text validation for the gfx1250 workgroup-cluster ops (Phase 2).

The ISA half of the proof lives in
``rocke/examples/gfx1250/isa_features/cluster_ids_verify.py``, which needs an
llvm23 toolchain; these tests pin the LLVM the lowering hands it. Nothing here
launches with a cluster shape -- that is Phase 3.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import I32, IRBuilder, PtrType, is_pure_op_name
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_AXES = ("x", "y", "z")
_AXIS_STEMS = ("cluster.id", "cluster.workgroup.id", "cluster.workgroup.max.id")
_FLAT = ("cluster.workgroup.flat.id", "cluster.workgroup.max.flat.id")

_BARRIER = (
    '  fence syncscope("cluster") release\n'
    "  call void @llvm.amdgcn.s.cluster.barrier()\n"
    '  fence syncscope("cluster") acquire\n'
)


def _build_all():
    """Every read on every axis, stored so the pure reads stay live, then one
    cluster barrier. Mirrors build_gfx1250_cluster in
    future_intrinsic_lowering.cpp."""
    b = IRBuilder("gfx1250_cluster")
    out = b.param("out", PtrType(I32, "global"))
    slot = 0

    def store(v):
        nonlocal slot
        b.global_store(out, b.const_i32(slot), v, align=4)
        slot += 1

    for axis in _AXES:
        store(b.cluster_id(axis))
        store(b.cluster_workgroup_id(axis))
        store(b.cluster_workgroup_max_id(axis))
    store(b.cluster_workgroup_flat_id())
    store(b.cluster_workgroup_max_flat_id())
    b.cluster_barrier()
    b.ret()
    return b.kernel


def _one(emit):
    b = IRBuilder("one")
    out = b.param("out", PtrType(I32, "global"))
    v = emit(b)
    if v is not None:
        b.global_store(out, b.const_i32(0), v)
    b.ret()
    return b.kernel


def _lower(kernel, *, arch="gfx1250", flavor="llvm23"):
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


_GATED = {
    "cluster_id": lambda b: b.cluster_id("x"),
    "cluster_workgroup_id": lambda b: b.cluster_workgroup_id("y"),
    "cluster_workgroup_max_id": lambda b: b.cluster_workgroup_max_id("z"),
    "cluster_workgroup_flat_id": lambda b: b.cluster_workgroup_flat_id(),
    "cluster_workgroup_max_flat_id": lambda b: b.cluster_workgroup_max_flat_id(),
    "cluster_barrier": lambda b: b.cluster_barrier(),
}


class TestGfx1250Cluster(unittest.TestCase):
    def test_exact_calls(self):
        llvm = _lower(_build_all())
        for stem in _AXIS_STEMS:
            for axis in _AXES:
                text = f" = call i32 @llvm.amdgcn.{stem}.{axis}()"
                with self.subTest(text=text):
                    self.assertEqual(llvm.count(text), 1)
        for name in _FLAT:
            text = f" = call i32 @llvm.amdgcn.{name}()"
            with self.subTest(text=text):
                self.assertEqual(llvm.count(text), 1)

    def test_barrier_is_fenced_at_cluster_scope(self):
        llvm = _lower(_build_all())
        # Release before, acquire after, nothing in between.
        self.assertEqual(llvm.count(_BARRIER), 1)
        self.assertNotIn('syncscope("workgroup")', llvm)

    def test_exact_declares(self):
        llvm = _lower(_build_all())
        expected = [f"declare i32 @llvm.amdgcn.{s}.{a}()" for s in _AXIS_STEMS for a in _AXES]
        expected += [f"declare i32 @llvm.amdgcn.{n}()" for n in _FLAT]
        expected.append("declare void @llvm.amdgcn.s.cluster.barrier()")
        for text in expected:
            with self.subTest(text=text):
                self.assertEqual(llvm.count(text), 1)

    def test_unused_ops_declare_nothing(self):
        llvm = _lower(_one(lambda b: b.cluster_id("z")))
        self.assertIn("call i32 @llvm.amdgcn.cluster.id.z()", llvm)
        for name in ("cluster.id.x", "cluster.id.y", "cluster.workgroup", "s.cluster.barrier"):
            with self.subTest(name=name):
                self.assertNotIn(f"@llvm.amdgcn.{name}", llvm)

    def test_cluster_size_is_max_id_plus_one(self):
        llvm = _lower(_one(lambda b: b.cluster_size("y")))
        self.assertRegex(
            llvm,
            r"(%cwmax\d+) = call i32 @llvm\.amdgcn\.cluster\.workgroup\.max\.id\.y\(\)\n"
            r"  %\S+ = add nsw i32 \1, 1\n",
        )
        self.assertNotIn("@llvm.amdgcn.cluster.id.", llvm)

    def test_distinct_from_workgroup_split_barrier(self):
        llvm = _lower(_build_all())
        self.assertNotIn("@llvm.amdgcn.s.barrier.signal", llvm)
        self.assertNotIn("@llvm.amdgcn.s.barrier.wait", llvm)

    def test_serialization_roundtrip_preserves_all_ops(self):
        kernel = _build_all()
        text = serialize(kernel)
        parsed = parse(text)
        self.assertEqual(text, serialize(parsed))
        self.assertEqual(_lower(kernel), _lower(parsed))

    def test_each_op_is_gated(self):
        bad = (
            ("gfx950", "llvm22", "requires gfx1250"),
            ("gfx1201", "llvm23", "requires gfx1250"),
            ("gfx1250", "llvm22", "requires LLVM flavor llvm23"),
        )
        for name, emit in _GATED.items():
            for arch, flavor, why in bad:
                with self.subTest(op=name, arch=arch, flavor=flavor):
                    with self.assertRaisesRegex(ValueError, f"{name} {why}"):
                        _lower(_one(emit), arch=arch, flavor=flavor)

    def test_builder_rejects_bad_axis(self):
        b = IRBuilder("bad")
        cases = (
            (b.cluster_id, "w", "cluster_id axis must be x, y, or z, got 'w'"),
            (b.cluster_workgroup_id, "X", "cluster_workgroup_id axis must be x, y, or z, got 'X'"),
            (b.cluster_workgroup_max_id, "", "cluster_workgroup_max_id axis must be x, y, or z"),
            (b.cluster_size, "xy", "cluster_workgroup_max_id axis must be x, y, or z, got 'xy'"),
            (b.cluster_id, None, "cluster_id axis must be x, y, or z, got None"),
        )
        for fn, axis, msg in cases:
            with self.subTest(fn=fn.__name__, axis=axis):
                with self.assertRaisesRegex(ValueError, msg):
                    fn(axis)

    def test_lowerer_rechecks_axis(self):
        # Serialized IR reaches the lowerer without passing through the builder.
        kernel = _one(lambda b: b.cluster_workgroup_id("x"))
        (op,) = [op for op in kernel.body.ops if op.name.startswith("gpu.cluster")]
        op.attrs["axis"] = "w"
        with self.assertRaisesRegex(
            ValueError, "cluster_workgroup_id axis must be x, y, or z, got 'w'"
        ):
            _lower(kernel)

    def test_purity(self):
        for name in (
            "gpu.cluster_id",
            "gpu.cluster_workgroup_id",
            "gpu.cluster_workgroup_max_id",
            "gpu.cluster_workgroup_flat_id",
            "gpu.cluster_workgroup_max_flat_id",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_pure_op_name(name))
        self.assertFalse(is_pure_op_name("tile.cluster_barrier"))


if __name__ == "__main__":
    unittest.main()
