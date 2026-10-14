# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The kernarg ABI derived from each op contract equals every committed object's, and
an object whose tuple, segment size or hidden arguments differ is rejected with
TRITONGEN-ABI, at generate time and when descriptors are derived from it."""

import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

import gen_descriptors  # noqa: E402
from tests.test_generator_contract import committed_compiled  # noqa: E402
from tritongen import _family as F  # noqa: E402
from tritongen import gen  # noqa: E402
from tritongen._codeobject import (  # noqa: E402
    describe,
    expected_abi,
    hkp_pack_module,
    verify_abi,
)
from tritongen._family import GeneratorError, derive_abi  # noqa: E402
from tritongen.ops import OPS  # noqa: E402

# The metadata a compile without LLVM optimization reports: the explicit arguments
# followed by the hidden ones the AMDGPU backend appends, in a 304-byte segment.
HIDDEN_ARGS = [
    {".offset": 48, ".size": 4, ".value_kind": "hidden_block_count_x"},
    {".offset": 52, ".size": 4, ".value_kind": "hidden_block_count_y"},
    {".offset": 64, ".size": 2, ".value_kind": "hidden_group_size_x"},
    {".offset": 136, ".size": 8, ".value_kind": "hidden_hostcall_buffer"},
]
UNOPTIMIZED_SEGMENT = 304


def unoptimized(document: dict) -> dict:
    """``document`` as an unoptimized compile reports it (hidden args, 304 B)."""
    changed = copy.deepcopy(document)
    for kernel in changed["amdhsa.kernels"]:
        kernel[".args"] = list(kernel[".args"]) + HIDDEN_ARGS
        kernel[".kernarg_segment_size"] = UNOPTIMIZED_SEGMENT
    return changed


def patched_metadata(change):
    """Patch the public metadata reader so every object reports ``change(document)``."""
    hkp = hkp_pack_module()
    real = hkp.metadata_document
    return mock.patch.object(
        hkp,
        "metadata_document",
        side_effect=lambda elf, where: change(real(elf, where)),
    )


def first(op):
    return op.ARCHES[0], op.instances()[0]


class DerivedAbiTest(unittest.TestCase):
    def test_every_committed_object_has_the_derived_abi(self):
        for op in OPS.values():
            for arch in op.ARCHES:
                for instance in op.instances():
                    with self.subTest(op=op.OP, arch=arch, instance=instance.name):
                        abi = expected_abi(op, instance)
                        blob = committed_compiled(op, arch, instance).hsaco
                        described = describe(blob, "t")
                        self.assertEqual(described["signature"], abi.signature)
                        self.assertEqual(
                            described["kernarg_segment_size"], abi.segment_size
                        )
                        self.assertEqual(described["hidden_kinds"], ())

    def test_scalars_take_natural_alignment_and_scratch_follows(self):
        abi = derive_abi(
            "k",
            {"a": "*fp16", "n": "i32", "b": "*fp16", "C": "constexpr"},
            {"a": "a", "n": "n", "b": "b"},
        )
        self.assertEqual(
            [(a.name, a.kind, a.size, a.offset) for a in abi.args],
            [
                ("a", "global_buffer", 8, 0),
                ("n", "by_value", 4, 8),
                ("b", "global_buffer", 8, 16),
                ("global_scratch", "global_buffer", 8, 24),
                ("profile_scratch", "global_buffer", 8, 32),
            ],
        )
        self.assertEqual(abi.segment_size, 40)

    def test_type_without_a_rule_is_abi(self):
        with self.assertRaises(GeneratorError) as raised:
            derive_abi("k", {"x": "i64"}, {"x": "x"})
        self.assertTrue(str(raised.exception).startswith(F.ERR_ABI))


class AbiRejectionTest(unittest.TestCase):
    def described(self, op):
        arch, instance = first(op)
        return describe(committed_compiled(op, arch, instance).hsaco, "t"), instance

    def assert_abi_rejected(self, op, described, instance, *fragments):
        with self.assertRaises(GeneratorError) as raised:
            verify_abi(described, expected_abi(op, instance), instance.name, "obj")
        message = str(raised.exception)
        self.assertTrue(message.startswith(f"{F.ERR_ABI}: obj: "), message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def mutated_signature(self, op, change):
        described, instance = self.described(op)
        signature = list(described["signature"])
        change(signature)
        self.assert_abi_rejected(
            op, {**described, "signature": tuple(signature)}, instance
        )

    def test_changed_offset_rejected(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):

                def shift(sig):
                    kind, size, offset = sig[-1]
                    sig[-1] = (kind, size, offset + 8)

                self.mutated_signature(op, shift)

    def test_changed_kind_rejected(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):

                def rekind(sig):
                    _, size, offset = sig[0]
                    sig[0] = ("by_value", size, offset)

                self.mutated_signature(op, rekind)

    def test_missing_scratch_slot_rejected(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                self.mutated_signature(op, lambda sig: sig.pop())

    def test_extra_non_hidden_argument_rejected(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                described, _ = self.described(op)
                end = described["kernarg_segment_size"]
                self.mutated_signature(
                    op, lambda sig: sig.append(("global_buffer", 8, end))
                )

    def test_wrong_kernel_symbol_rejected(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                described, instance = self.described(op)
                self.assert_abi_rejected(
                    op,
                    {**described, "symbol": "other_kernel"},
                    instance,
                    "other_kernel",
                )

    def test_larger_segment_rejected_naming_both_sizes(self):
        for op in OPS.values():
            with self.subTest(op=op.OP):
                described, instance = self.described(op)
                expected = expected_abi(op, instance).segment_size
                self.assert_abi_rejected(
                    op,
                    {**described, "kernarg_segment_size": expected + 8},
                    instance,
                    f"kernarg_segment_size {expected + 8}",
                    f"expected {expected}",
                )

    def test_unoptimized_metadata_rejected_naming_hidden_kinds(self):
        for op in OPS.values():
            with self.subTest(op=op.OP), patched_metadata(unoptimized):
                described, instance = self.described(op)
                self.assertEqual(
                    described["signature"], expected_abi(op, instance).signature
                )
                self.assert_abi_rejected(
                    op,
                    described,
                    instance,
                    f"kernarg_segment_size {UNOPTIMIZED_SEGMENT}",
                    "hidden_block_count_x",
                    "hidden_hostcall_buffer",
                )

    def test_unoptimized_metadata_rejected_at_generate_time(self):
        for op in OPS.values():
            with self.subTest(op=op.OP), patched_metadata(unoptimized):
                arch, instance = first(op)
                with self.assertRaises(GeneratorError) as raised:
                    gen.verify_compiled(
                        op, committed_compiled(op, arch, instance), instance, arch
                    )
                self.assertTrue(str(raised.exception).startswith(F.ERR_ABI))
                self.assertIn(str(UNOPTIMIZED_SEGMENT), str(raised.exception))

    def test_unoptimized_metadata_rejected_by_collect(self):
        for op in OPS.values():
            with self.subTest(op=op.OP), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "root"
                shutil.copytree(F.CONTENT_DIR, root)
                arch, _ = first(op)
                with patched_metadata(unoptimized), self.assertRaises(
                    GeneratorError
                ) as raised:
                    gen_descriptors.collect(op, root, arch)
                message = str(raised.exception)
                self.assertTrue(message.startswith(F.ERR_ABI), message)
                self.assertIn("hidden_", message)


if __name__ == "__main__":
    unittest.main()
