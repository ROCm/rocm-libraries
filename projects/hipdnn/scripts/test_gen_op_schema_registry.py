#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for gen_op_schema_registry.py's classification and emitters.

Uses hand-built reflection schemas, so neither flatc nor a compiler is needed. The tests
cover the generator's policy (RFC 0020 Appendix B.3): which fields become edges, which
become attributes and with what value kind, and that any schema the registry cannot
describe faithfully fails the run.
"""

import json
import unittest

from gen_op_schema_registry import Registry, emit_json

NS = "test_ns"
NODE = f"{NS}.Node"

# Enum indices in every schema built here.
DATA_TYPE_ENUM = 0
NODE_ATTRIBUTES_UNION = 1
MODE_ENUM = 2


def field(
    name, base_type, field_id, *, index=-1, element=None, optional=False, **attrs
):
    """A reflection Field. @p attrs maps annotation keys to values (True for a flag)."""
    entry = {
        "name": name,
        "id": field_id,
        "type": {"base_type": base_type, "index": index},
        "attributes": [],
    }
    if element is not None:
        entry["type"]["element"] = element
    if optional:
        entry["optional"] = True
    for key, value in attrs.items():
        entry["attributes"].append(
            {"key": key, "value": "0" if value is True else value}
        )
    return entry


def edge(name, field_id, direction, matcher_name, *, optional=False):
    return field(
        name,
        "Long",
        field_id,
        optional=optional,
        cache_uid=True,
        **{direction: True, "matcher_name": matcher_name},
    )


def edge_list(name, field_id, direction):
    return field(
        name, "Vector", field_id, element="Long", cache_uid=True, **{direction: True}
    )


def table(name, fields, opcode=None):
    entry = {"name": f"{NS}.{name}", "fields": fields}
    if opcode is not None:
        entry["attributes"] = [{"key": "matcher_opcode", "value": opcode}]
    return entry


def schema(tables, *, node_fields=None, extra_objects=()):
    """A schema whose Node.attributes union holds @p tables, in order."""
    if node_fields is None:
        node_fields = [
            field("name", "String", 0, cache_ignore=True),
            field("compute_data_type", "Byte", 1, index=DATA_TYPE_ENUM),
        ]
    node = table(
        "Node",
        node_fields
        + [
            field("attributes_type", "UType", 2, index=NODE_ATTRIBUTES_UNION),
            field("attributes", "Union", 3, index=NODE_ATTRIBUTES_UNION),
        ],
    )
    objects = [node] + list(tables) + list(extra_objects)
    union_values = [{"name": "NONE", "value": 0, "union_type": {}}]
    for position, member in enumerate(tables, start=1):
        union_values.append(
            {
                "name": member["name"].rsplit(".", 1)[-1],
                "value": position,
                "union_type": {"base_type": "Obj", "index": position},
            }
        )
    enums = [
        {"name": f"{NS}.DataType", "values": []},
        {"name": f"{NS}.NodeAttributes", "values": union_values, "is_union": True},
        {"name": f"{NS}.Mode", "values": []},
    ]
    return {"objects": objects, "enums": enums}


def build(tables, **kwargs):
    return Registry(schema(tables, **kwargs), NODE)


def pointwise_like():
    """Edges in_0 (required), in_1 (optional), out_0; scalars of every kind."""
    return table(
        "PointwiseAttributes",
        [
            field("operation", "Byte", 0, index=MODE_ENUM),
            field("relu_lower_clip", "Float", 1, optional=True),
            # Named like a uid, but an axis index with no cache_uid and no flag.
            field("axis_tensor_uid", "Long", 2, optional=True),
            edge("in_0_tensor_uid", 3, "matcher_input_tensor", "in_0"),
            edge("in_1_tensor_uid", 4, "matcher_input_tensor", "in_1", optional=True),
            edge("out_0_tensor_uid", 5, "matcher_output_tensor", "out_0"),
            field("block_size", "Vector", 6, element="Int"),
            field("in_place", "Bool", 7),
        ],
        opcode="pointwise",
    )


class TestRegistryEntry(unittest.TestCase):
    """What a well-annotated table publishes."""

    def setUp(self):
        self.registry = build([pointwise_like()])
        self.ops = json.loads(emit_json(self.registry))["ops"]

    def test_a_pointwise_like_table_publishes_its_edges_and_attributes(self):
        self.assertEqual(
            self.ops["pointwise"],
            {
                "attributes_table": "PointwiseAttributes",
                "attributes_type": 1,
                "operands": [
                    {"name": "in_0", "optional": False},
                    {"name": "in_1", "optional": True},
                ],
                "results": [{"name": "out_0", "optional": False}],
                "operand_lists": [],
                "result_lists": [],
                # Declaration order. The vector is skipped and the Node scalar comes last.
                "attributes": [
                    {"name": "operation", "kind": "enum_name", "optional": False},
                    {"name": "relu_lower_clip", "kind": "float", "optional": True},
                    {"name": "axis_tensor_uid", "kind": "int", "optional": True},
                    {"name": "in_place", "kind": "bool", "optional": False},
                    {
                        "name": "compute_data_type",
                        "kind": "enum_name",
                        "optional": False,
                    },
                ],
            },
        )

    def test_directed_uid_vectors_publish_as_edge_lists_not_edges(self):
        registry = build(
            [
                table(
                    "CustomOpAttributes",
                    [
                        edge_list("input_tensor_uids", 0, "matcher_input_tensor"),
                        edge("x_tensor_uid", 1, "matcher_input_tensor", "x"),
                        edge_list("output_tensor_uids", 2, "matcher_output_tensor"),
                        edge_list("aux_tensor_uids", 3, "matcher_input_tensor"),
                    ],
                    opcode="custom_op",
                )
            ]
        )
        op = json.loads(emit_json(registry))["ops"]["custom_op"]
        # Lists keep declaration order and never appear among the bindable edges.
        self.assertEqual(
            op["operand_lists"],
            [{"name": "input_tensor_uids"}, {"name": "aux_tensor_uids"}],
        )
        self.assertEqual(op["result_lists"], [{"name": "output_tensor_uids"}])
        self.assertEqual(op["operands"], [{"name": "x", "optional": False}])
        self.assertEqual(op["results"], [])

    def test_ops_are_sorted_by_opcode(self):
        registry = build(
            [
                table("ZTable", [], opcode="b_op"),
                table("ATable", [], opcode="c_op"),
                table("MTable", [], opcode="a_op"),
            ]
        )
        self.assertEqual([op.opcode for op in registry.ops], ["a_op", "b_op", "c_op"])
        self.assertEqual([op.attributes_type for op in registry.ops], [3, 1, 2])


class TestBuildErrors(unittest.TestCase):
    """Each annotation the registry cannot honour fails the run, naming the field."""

    def assertFails(self, tables, pattern, **kwargs):
        with self.assertRaisesRegex(SystemExit, "^ERROR: .*" + pattern):
            build(tables, **kwargs)

    def test_both_direction_flags_fail(self):
        bad = field(
            "x_tensor_uid",
            "Long",
            0,
            cache_uid=True,
            matcher_input_tensor=True,
            matcher_output_tensor=True,
            matcher_name="x",
        )
        self.assertFails([table("Op", [bad], "op")], r"Op\.x_tensor_uid")

    def test_matcher_name_without_a_flag_fails(self):
        bad = field("x_tensor_uid", "Long", 0, matcher_name="x")
        self.assertFails([table("Op", [bad], "op")], r"Op\.x_tensor_uid.*matcher_name")

    def test_a_flag_without_matcher_name_fails(self):
        bad = field(
            "x_tensor_uid", "Long", 0, cache_uid=True, matcher_input_tensor=True
        )
        self.assertFails([table("Op", [bad], "op")], r"Op\.x_tensor_uid.*matcher_name")

    def test_a_flag_with_an_empty_matcher_name_fails(self):
        bad = field(
            "x_tensor_uid",
            "Long",
            0,
            cache_uid=True,
            matcher_input_tensor=True,
            matcher_name="",
        )
        self.assertFails([table("Op", [bad], "op")], r"Op\.x_tensor_uid.*matcher_name")

    def test_a_flag_on_a_float_field_fails(self):
        bad = field("x", "Float", 0, matcher_input_tensor=True, matcher_name="x")
        self.assertFails([table("Op", [bad], "op")], r"Op\.x'.*integer")

    def test_a_cache_uid_vector_without_a_direction_flag_fails(self):
        # Skipping it like other vectors would hide the tensors it references.
        bad = field("xs", "Vector", 0, element="Long", cache_uid=True)
        self.assertFails([table("Op", [bad], "op")], r"Op\.xs'.*cache_uid.*edge list")

    def test_a_cache_uid_vector_with_both_direction_flags_fails(self):
        bad = field(
            "xs",
            "Vector",
            0,
            element="Long",
            cache_uid=True,
            matcher_input_tensor=True,
            matcher_output_tensor=True,
        )
        self.assertFails([table("Op", [bad], "op")], r"Op\.xs'.*both")

    def test_matcher_name_on_an_edge_list_fails(self):
        bad = field(
            "xs",
            "Vector",
            0,
            element="Long",
            cache_uid=True,
            matcher_input_tensor=True,
            matcher_name="xs",
        )
        self.assertFails([table("Op", [bad], "op")], r"Op\.xs'.*matcher_name")

    def test_a_flag_on_a_vector_of_32_bit_integers_fails(self):
        bad = field(
            "xs", "Vector", 0, element="Int", cache_uid=True, matcher_input_tensor=True
        )
        self.assertFails([table("Op", [bad], "op")], r"Op\.xs'.*\[long\]")

    def test_a_flag_on_a_32_bit_integer_fails(self):
        bad = field("x", "Int", 0, matcher_input_tensor=True, matcher_name="x")
        self.assertFails([table("Op", [bad], "op")], r"Op\.x'.*64-bit")

    def test_a_duplicate_matcher_name_across_operands_and_results_fails(self):
        fields = [
            edge("x_tensor_uid", 0, "matcher_input_tensor", "x"),
            edge("y_tensor_uid", 1, "matcher_output_tensor", "x"),
        ]
        self.assertFails([table("Op", fields, "op")], r"'x'.*Op\.y_tensor_uid")

    def test_a_duplicate_attribute_name_with_the_node_table_fails(self):
        fields = [field("compute_data_type", "Byte", 0, index=DATA_TYPE_ENUM)]
        self.assertFails(
            [table("Op", fields, "op")], r"'compute_data_type'.*Node\.compute_data_type"
        )

    def test_an_edge_named_for_a_reserved_root_fails(self):
        for root in ("graph", "kernel", "device"):
            with self.subTest(root=root):
                fields = [edge("g_tensor_uid", 0, "matcher_input_tensor", root)]
                self.assertFails(
                    [table("Op", fields, "op")], r"Op\.g_tensor_uid.*reserved"
                )

    def test_a_duplicate_opcode_across_tables_fails(self):
        self.assertFails(
            [table("First", [], "op"), table("Second", [], "op")], r"'First'.*'Second'"
        )

    def test_a_table_without_matcher_opcode_fails(self):
        self.assertFails(
            [table("First", [], "first"), table("MatmulAttributes", [])],
            r"'MatmulAttributes'.*matcher_opcode",
        )

    def test_a_non_identifier_matcher_opcode_fails(self):
        self.assertFails(
            [table("Op", [], "op-name")], r"matcher_opcode.*'Op'.*identifier"
        )

    def test_a_non_identifier_matcher_name_fails(self):
        fields = [edge("x_tensor_uid", 0, "matcher_input_tensor", "x.y")]
        self.assertFails(
            [table("Op", fields, "op")], r"matcher_name.*Op\.x_tensor_uid.*identifier"
        )

    def test_a_matcher_annotation_on_a_node_field_fails(self):
        node_fields = [
            field(
                "compute_data_type",
                "Byte",
                0,
                index=DATA_TYPE_ENUM,
                matcher_name="dtype",
            ),
            field("name", "String", 1, cache_ignore=True, matcher_input_tensor=True),
        ]
        for bad in node_fields:
            with self.subTest(field=bad["name"]):
                self.assertFails(
                    [table("Op", [], "op")],
                    rf"matcher_.*Node\.{bad['name']}",
                    node_fields=[bad],
                )

    def test_an_unknown_matcher_field_annotation_fails(self):
        bad = field("x", "Long", 0, matcher_inptu_tensor=True)
        self.assertFails([table("Op", [bad], "op")], r"matcher_inptu_tensor.*Op\.x")

    def test_an_unknown_matcher_table_annotation_fails(self):
        bad = table("Op", [], "op")
        bad["attributes"].append({"key": "matcher_opcodes", "value": "x"})
        self.assertFails([bad], r"matcher_opcodes.*'Op'")

    def test_a_cache_uid_integer_without_a_direction_flag_fails(self):
        # A tensor reference left unflagged would otherwise publish as an Int attribute.
        bad = field("x_tensor_uid", "Long", 0, cache_uid=True)
        self.assertFails([table("Op", [bad], "op")], r"Op\.x_tensor_uid.*cache_uid")

    def test_matcher_opcode_on_a_table_outside_the_union_fails(self):
        stray = table("Stray", [], "stray")
        self.assertFails([table("Op", [], "op")], r"'Stray'", extra_objects=[stray])

    def test_an_unsigned_64_bit_attribute_fails(self):
        bad = field("count", "ULong", 0)
        self.assertFails([table("Op", [bad], "op")], r"Op\.count.*unsigned 64-bit")


if __name__ == "__main__":
    unittest.main()
