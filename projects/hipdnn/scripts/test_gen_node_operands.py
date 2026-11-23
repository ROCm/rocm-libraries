#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for gen_node_operands.py's operand policy, on hand-built schemas."""

import os
import re
import shutil
import unittest

from gen_cache_key import _HIPDNN_DIR, SDK_DIR, _resolve_flatc
from gen_node_operands import OperandsEmitter, generated_headers, sdk_emitter
from test_gen_cache_key import field, schema, table

NS = "test_ns"


def work(entry):
    entry["attributes"].append({"key": "work_data_dependent"})
    return entry


def node_schema(members, tensor_fields=None):
    """A root with a `Tensor` domain, a `Node` holding an `Attrs` union of @p members.

    @p members is a list of (name, fields); member tables are laid out after the fixed
    Tensor/Root/Node tables, in order.
    """
    tensor = table(
        f"{NS}.Tensor",
        [field("uid", "Long", 0, uid_key=True)] + (tensor_fields or []),
    )
    root = table(
        f"{NS}.Root",
        [
            field("tensors", "Vector", 0, index=0, element="Obj", uid_domain=True),
            field("nodes", "Vector", 1, index=2, element="Obj"),
        ],
    )
    node = table(
        f"{NS}.Node",
        [
            field("attributes_type", "UType", 0, index=0),
            field("attributes", "Union", 1, index=0),
        ],
    )
    objects = [tensor, root, node]
    values = [{"name": "NONE", "value": 0}]
    for position, (name, fields) in enumerate(members):
        objects.append(table(f"{NS}.{name}", fields))
        values.append(
            {
                "name": name,
                "value": position + 1,
                "union_type": {"base_type": "Obj", "index": len(objects) - 1},
            }
        )
    return schema(objects, [{"name": f"{NS}.Attrs", "values": values}])


def emit(members, tensor_fields=None):
    return OperandsEmitter(
        node_schema(members, tensor_fields), f"{NS}.Root", f"{NS}.Node"
    ).emit()


def body(header, member):
    """The emitted visitor body for @p member."""
    match = re.search(
        rf"void visit\(const {member}&[^)]*\)\n\{{\n(.*?)\n\}}\n", header, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"no visitor emitted for {member}")
    return match.group(1)


class TestEveryMemberIsVisited(unittest.TestCase):
    def test_each_union_member_gets_a_visitor_and_a_dispatch_case(self):
        # A member without a case would publish nothing and report "unknown type".
        members = [
            ("Alpha", [field("x_tensor_uid", "Long", 0, uid=True)]),
            ("Beta", [field("y_tensor_uid", "Long", 0, uid=True)]),
            ("Gamma", [field("mode", "Byte", 0)]),
        ]
        header = emit(members)
        for name, _ in members:
            body(header, name)
            self.assertIn(f"case Attrs::{name}:", header)
            self.assertIn(f"node.attributes_as_{name}()", header)
        self.assertNotIn("case Attrs::NONE:", header)

    def test_a_member_with_no_operands_still_compiles(self):
        # -Wunused-parameter is an error in this project.
        header = emit([("Empty", [field("label", "String", 0)])])
        self.assertIn("void visit(const Empty&, V&)", header)

    def test_dispatch_pins_the_last_union_member_at_compile_time(self):
        # A header generated before a member was added must fail to compile against the
        # new schema, whose MAX moves, instead of reaching `default: return false`.
        header = emit(
            [
                ("Alpha", [field("x_tensor_uid", "Long", 0, uid=True)]),
                ("Beta", [field("mode", "Byte", 0)]),
            ]
        )
        dispatch = header[header.index("bool visit(const Node& node") :]
        self.assertRegex(
            dispatch[: dispatch.index("switch(")],
            r"static_assert\(Attrs::MAX == Attrs::Beta,",
        )


class TestTensorOperands(unittest.TestCase):
    def test_a_uid_reports_its_role_and_annotation(self):
        header = emit(
            [
                (
                    "Attn",
                    [
                        field("q_tensor_uid", "Long", 0, uid=True),
                        work(
                            field(
                                "seq_len_q_tensor_uid",
                                "Long",
                                1,
                                uid=True,
                                optional=True,
                            )
                        ),
                    ],
                )
            ]
        )
        visitor = body(header, "Attn")
        self.assertIn('visitor.tensor(std::string_view("q"), ', visitor)
        self.assertRegex(visitor, r'"q"\), [^;]*, false\);')
        self.assertIn("uid.has_value()", visitor)
        self.assertRegex(visitor, r'"seq_len_q"\), [^;]*, true\);')

    def test_a_uid_vector_reports_one_indexed_role_per_element(self):
        header = emit(
            [
                (
                    "Custom",
                    [field("input_tensor_uids", "Vector", 0, element="Long", uid=True)],
                )
            ]
        )
        visitor = body(header, "Custom")
        self.assertIn('detail::IndexedRole role("input", index);', visitor)
        self.assertIn("visitor.tensor(role.view(), ", visitor)
        # The stack buffer must fit the longest vector role plus "[4294967295]".
        self.assertIn(f"char _text[{len('input')} + 12]", header)

    def test_a_tensor_named_field_without_cache_uid_is_a_scalar(self):
        # PointwiseAttributes.axis_tensor_uid is an axis index despite its name.
        header = emit(
            [
                (
                    "Pointwise",
                    [
                        field("axis_tensor_uid", "Long", 0, optional=True),
                        field("in_0_tensor_uid", "Long", 1, uid=True),
                    ],
                )
            ]
        )
        visitor = body(header, "Pointwise")
        self.assertIn(
            'visitor.scalar(std::string_view("axis_tensor_uid"), '
            "static_cast<int64_t>(*value));",
            visitor,
        )
        self.assertNotIn('"axis"', visitor)

    def test_the_tensor_table_annotation_becomes_a_predicate(self):
        header = emit(
            [("Op", [field("x_tensor_uid", "Long", 0, uid=True)])],
            tensor_fields=[
                work(
                    field(
                        "ragged_offset_tensor_uid",
                        "Long",
                        1,
                        uid=True,
                        optional=True,
                    )
                )
            ],
        )
        self.assertIn(
            "return tensor.ragged_offset_tensor_uid().has_value();",
            header,
        )

    def test_an_unannotated_tensor_table_is_never_data_dependent(self):
        header = emit([("Op", [field("x_tensor_uid", "Long", 0, uid=True)])])
        self.assertIn("inline bool workDataDependent(const Tensor&)", header)


class TestAttributeOperands(unittest.TestCase):
    def setUp(self):
        self.visitor = body(
            emit(
                [
                    (
                        "Conv",
                        [
                            field("stride", "Vector", 0, element="Long"),
                            field("scales", "Vector", 1, element="Float"),
                            field("mode", "Byte", 2, index=0),
                            field("alpha", "Float", 3, optional=True),
                            field("causal", "Bool", 4),
                            field("groups", "Int", 5),
                            field("name", "String", 6),
                            field("payload", "Vector", 7, element="UByte"),
                            field("labels", "Vector", 8, element="String"),
                            field("old", "Int", 9, deprecated=True),
                        ],
                    )
                ]
            ),
            "Conv",
        )

    def test_numeric_vectors_report_each_element(self):
        self.assertIn(
            'visitor.element(std::string_view("stride"), static_cast<size_t>(index), '
            "static_cast<int64_t>(values->Get(index)));",
            self.visitor,
        )
        self.assertIn(
            'visitor.element(std::string_view("scales"), static_cast<size_t>(index), '
            "static_cast<double>(values->Get(index)));",
            self.visitor,
        )

    def test_scalars_are_widened_by_kind(self):
        self.assertIn(
            'visitor.scalar(std::string_view("mode"), static_cast<int64_t>(op.mode()));',
            self.visitor,
        )
        self.assertIn(
            'visitor.scalar(std::string_view("causal"), static_cast<bool>(op.causal()));',
            self.visitor,
        )
        self.assertIn(
            'visitor.scalar(std::string_view("groups"), static_cast<int64_t>(op.groups()));',
            self.visitor,
        )

    def test_an_absent_optional_is_skipped(self):
        self.assertIn(
            "if(const auto value = op.alpha(); value.has_value())", self.visitor
        )

    def test_strings_payloads_and_deprecated_fields_are_not_visited(self):
        for name in ("name", "payload", "labels", "old"):
            self.assertNotIn(f'"{name}"', self.visitor)


class TestOperandTable(unittest.TestCase):
    """OPERANDS marks an operand optional when a valid node may leave it out; the UHD tools
    read each such operand through a default, so a wrong flag either hides an unguarded read
    or demands a default for an operand that is always there."""

    @staticmethod
    def rows(fields, tensor_fields=None):
        emitter = OperandsEmitter(
            node_schema([("Op", fields)], tensor_fields), f"{NS}.Root", f"{NS}.Node"
        )
        return {
            name: (kind, optional) for _, name, kind, optional in emitter.operand_rows()
        }

    def test_a_required_tensor_reference_stays_required(self):
        rows = self.rows([field("x_tensor_uid", "Long", 0, uid=True)])
        self.assertEqual(rows, {"x": ("TENSOR", False)})

    def test_an_optional_tensor_reference_is_optional(self):
        rows = self.rows([field("bias_tensor_uid", "Long", 0, uid=True, optional=True)])
        self.assertEqual(rows, {"bias": ("TENSOR", True)})

    def test_a_vector_of_tensor_references_is_optional_though_the_field_is_not(self):
        # The schema fixes no length, so any element index may be missing.
        rows = self.rows(
            [field("input_tensor_uids", "Vector", 0, element="Long", uid=True)]
        )
        self.assertEqual(rows, {"input": ("TENSORS", True)})

    def test_an_optional_scalar_is_optional_and_a_required_one_is_not(self):
        rows = self.rows(
            [
                field("alpha", "Float", 0, optional=True),
                field("groups", "Int", 1),
                field("causal", "Bool", 2),
            ]
        )
        self.assertEqual(
            rows,
            {
                "alpha": ("SCALAR", True),
                "groups": ("SCALAR", False),
                "causal": ("SCALAR", False),
            },
        )

    def test_a_vector_attribute_is_optional_though_the_field_is_not(self):
        rows = self.rows([field("stride", "Vector", 0, element="Long")])
        self.assertEqual(rows, {"stride": ("ELEMENTS", True)})

    def test_operands_without_a_published_name_have_no_row(self):
        rows = self.rows(
            [
                field("label", "String", 0),
                field("payload", "Vector", 1, element="UByte"),
                field("old", "Int", 2, deprecated=True),
            ]
        )
        self.assertEqual(rows, {})

    def test_the_emitted_table_carries_each_flag(self):
        emitter = OperandsEmitter(
            node_schema(
                [
                    (
                        "Op",
                        [
                            field("x_tensor_uid", "Long", 0, uid=True),
                            field(
                                "input_tensor_uids",
                                "Vector",
                                1,
                                element="Long",
                                uid=True,
                            ),
                        ],
                    )
                ]
            ),
            f"{NS}.Root",
            f"{NS}.Node",
        )
        table_text = emitter.emit_table()
        self.assertIn('{"Op", "x", OperandKind::TENSOR, false},', table_text)
        self.assertIn('{"Op", "input", OperandKind::TENSORS, true},', table_text)
        self.assertIn("std::array<OperandEntry, 2> OPERANDS", table_text)


class TestGeneratedHeaders(unittest.TestCase):
    """One run writes two committed headers; flatbuffers_sdk/CMakeLists.txt declares both as
    the rule's outputs, so a path that drifts leaves a declared output unwritten."""

    @staticmethod
    def declared_outputs():
        with open(
            os.path.join(_HIPDNN_DIR, SDK_DIR, "CMakeLists.txt"), encoding="utf-8"
        ) as f:
            text = f.read()
        paths = {}
        for variable in ("_node_operands_header", "_node_operand_table_header"):
            match = re.search(
                rf'set\({variable}\s+"\$\{{CMAKE_CURRENT_SOURCE_DIR\}}/([^"]+)"', text
            )
            if match is None:
                raise AssertionError(
                    f"flatbuffers_sdk/CMakeLists.txt does not set {variable}"
                )
            paths[variable] = os.path.normpath(
                os.path.join(_HIPDNN_DIR, SDK_DIR, match.group(1))
            )
        return paths

    def test_both_headers_land_where_the_cmake_rule_declares_them(self):
        emitter = OperandsEmitter(
            node_schema([("Op", [field("x_tensor_uid", "Long", 0, uid=True)])]),
            f"{NS}.Root",
            f"{NS}.Node",
        )
        written = [os.path.normpath(path) for path, _ in generated_headers(emitter)]
        declared = self.declared_outputs()
        self.assertEqual(
            written,
            [declared["_node_operands_header"], declared["_node_operand_table_header"]],
        )
        for path in written:
            self.assertTrue(os.path.isfile(path), path)

    def test_the_committed_headers_equal_a_fresh_generation(self):
        # The pre-commit hook proves this too; here the table is held to the schema in a
        # plain test run, where flatc is on PATH at the pinned version.
        if shutil.which("flatc") is None:
            self.skipTest("flatc is not on PATH")
        try:
            flatc_path = _resolve_flatc()
        except SystemExit:
            self.skipTest("flatc is not the pinned version")
        for path, text in generated_headers(sdk_emitter(flatc_path)):
            with open(path, "rb") as f:
                committed = f.read().replace(b"\r\n", b"\n")
            self.assertEqual(committed, text.encode("utf-8"), path)


class TestRejectedSchemas(unittest.TestCase):
    def test_work_data_dependent_requires_a_tensor_reference(self):
        with self.assertRaises(SystemExit):
            emit([("Op", [work(field("size", "Long", 0))])])

    def test_work_data_dependent_outside_the_visited_tables_is_rejected(self):
        members = [("Op", [field("x_tensor_uid", "Long", 0, uid=True)])]
        document = node_schema(members)
        document["objects"].append(
            table(f"{NS}.Other", [work(field("y_tensor_uid", "Long", 0, uid=True))])
        )
        with self.assertRaises(SystemExit):
            OperandsEmitter(document, f"{NS}.Root", f"{NS}.Node")

    def test_a_role_colliding_with_an_attribute_is_rejected(self):
        # `x` and `x.dims[0]` would publish side by side under one name.
        with self.assertRaises(SystemExit):
            emit(
                [
                    (
                        "Op",
                        [
                            field("x_tensor_uid", "Long", 0, uid=True),
                            field("x", "Long", 1),
                        ],
                    )
                ]
            )

    def test_a_vector_without_an_element_rule_is_rejected(self):
        with self.assertRaises(SystemExit):
            emit([("Op", [field("flags", "Vector", 0, element="Bool")])])


if __name__ == "__main__":
    unittest.main()
