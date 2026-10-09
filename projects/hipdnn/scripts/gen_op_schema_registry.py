# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generate the op-schema registry from the FlatBuffers schemas.

For every table in the ``NodeAttributes`` union, the registry records the opcode a UED
pattern names it by, its operand and result tensor edges, its operand and result edge
lists (tensor-uid vectors), and its scalar attributes (RFC 0020 Appendix B). The
``matcher_opcode``, ``matcher_input_tensor``, ``matcher_output_tensor``, and
``matcher_name`` schema annotations define this. flatc's C++ output does not expose
annotations, so they are read from the binary schema.

Writes two files from one schema walk, so they always agree:

- ``op_schema_registry_generated.h``: the header-only registry the kernel ingestor
  compiles, with a typed accessor per edge and attribute.
- ``op_schema_registry.json``: the same registry for build-time tools (hkp_pack).

If the registry cannot describe a schema faithfully, the run fails with an ``ERROR:``
that names the table and field. No registry is written.

Each run regenerates both files. Run it manually, from the build through the custom
target in flatbuffers_sdk/CMakeLists.txt, or through the ``op-schema-registry-hipdnn``
pre-commit hook.
"""

import argparse
import json
import os
import re
import tempfile

from gen_cache_key import (
    NAMESPACE,
    SCALAR_BASE_TYPES,
    SDK_DIR,
    _HIPDNN_DIR,
    _resolve_flatc,
    accessor,
    load_schema,
    short_name,
)

# Table annotation: the opcode a pattern names the op by. Required on every union member.
OPCODE_ATTRIBUTE = "matcher_opcode"

# Field flags: the field is an input (operand) or output (result) tensor uid, or a
# vector of them (an edge list).
INPUT_ATTRIBUTE = "matcher_input_tensor"
OUTPUT_ATTRIBUTE = "matcher_output_tensor"

# Field string: the edge name a pattern binds a flagged scalar field by. Edge lists
# are never bound, so they take no name.
NAME_ATTRIBUTE = "matcher_name"

FIELD_ATTRIBUTES = frozenset([INPUT_ATTRIBUTE, OUTPUT_ATTRIBUTE, NAME_ATTRIBUTE])
ANNOTATION_PREFIX = "matcher_"

# Defined by gen_cache_key.py. An integer or integer vector carrying cache_uid is a
# tensor reference. The registry must classify it as an edge or edge list, never
# publish it as an attribute or skip it.
UID_ATTRIBUTE = "cache_uid"
IGNORE_ATTRIBUTE = "cache_ignore"

# Symbol roots a pattern variable may not take (RFC 0020 section 6.1).
RESERVED_ROOTS = frozenset(["device", "graph", "kernel"])

# Opcodes, edge names, and attribute names are pattern identifiers.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCHEMA_FILE = "graph.fbs"
SCHEMA_NAMESPACE = f"{NAMESPACE}.data_objects"
NODE_TABLE = f"{SCHEMA_NAMESPACE}.Node"
NODE_UNION_FIELD = "attributes"

HEADER_NAME = "op_schema_registry_generated.h"
JSON_NAME = "op_schema_registry.json"
GENERATOR = "scripts/gen_op_schema_registry.py"

# Value kinds, as (JSON name, C++ ValueKind enumerator, ScalarValue alternative).
INT = ("int", "INT", "int64_t")
FLOAT = ("float", "FLOAT", "double")
BOOL = ("bool", "BOOL", "bool")
ENUM_NAME = ("enum_name", "ENUM_NAME", "std::string_view")


def _fail(message):
    raise SystemExit(f"ERROR: {message}")


def _camel(name):
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


class Edge:
    def __init__(self, name, field, optional):
        self.name = name
        self.field = field
        self.optional = optional


class EdgeList:
    def __init__(self, field):
        self.field = field


class Attribute:
    def __init__(self, name, kind, optional, enum, on_node):
        self.name = name
        self.kind = kind
        self.optional = optional
        # Short enum type name for ENUM_NAME, else None.
        self.enum = enum
        # Read from the Node table rather than the attribute table.
        self.on_node = on_node


class OpSchema:
    def __init__(self, opcode, table, attributes_type):
        self.opcode = opcode
        self.table = table
        self.attributes_type = attributes_type
        self.operands = []
        self.results = []
        self.operand_lists = []
        self.result_lists = []
        self.attributes = []


class Registry:
    """Classifies the NodeAttributes tables per RFC 0020 Appendix B.3."""

    def __init__(self, schema, node_name=NODE_TABLE):
        self.schema = schema
        self.objects = {o["name"]: o for o in schema["objects"]}
        node = self.objects.get(node_name)
        if node is None:
            _fail(f"the schema has no '{short_name(node_name)}' table")
        self.node = node
        self.node_attributes = self._node_attributes()
        self.ops = self._build()

    @staticmethod
    def _annotations(entry):
        return {a["key"]: a.get("value", "") for a in entry.get("attributes") or []}

    @staticmethod
    def _fields(obj):
        # The binary schema sorts fields alphabetically; restore declaration order.
        return sorted(
            (f for f in obj["fields"] if not f.get("deprecated")),
            key=lambda f: f.get("id", 0),
        )

    def _union_members(self):
        """(union value, table object) per NodeAttributes member, NONE excluded."""
        unions = [
            f
            for f in self.node["fields"]
            if f["name"] == NODE_UNION_FIELD and f["type"]["base_type"] == "Union"
        ]
        if len(unions) != 1:
            _fail(
                f"'{short_name(self.node['name'])}.{NODE_UNION_FIELD}' must be the "
                "attributes union"
            )
        union = self.schema["enums"][unions[0]["type"]["index"]]
        members = []
        for value in union["values"]:
            member = value.get("union_type") or {}
            if member.get("base_type") != "Obj" or member.get("index", -1) < 0:
                continue
            members.append((value["value"], self.schema["objects"][member["index"]]))
        return members

    def _opcode(self, table):
        where = short_name(table["name"])
        annotations = self._annotations(table)
        for key in annotations:
            if key.startswith(ANNOTATION_PREFIX) and key != OPCODE_ATTRIBUTE:
                _fail(f"unknown annotation '{key}' on table '{where}'")
        if OPCODE_ATTRIBUTE not in annotations:
            _fail(
                f"table '{where}' is a NodeAttributes member without {OPCODE_ATTRIBUTE}; "
                "every op must declare the opcode a pattern names it by"
            )
        opcode = annotations[OPCODE_ATTRIBUTE]
        if not IDENTIFIER.match(opcode):
            _fail(
                f"{OPCODE_ATTRIBUTE} on table '{where}' must be an identifier, "
                f"got '{opcode}'"
            )
        return opcode

    def _attribute(self, field, where, on_node):
        """The Attribute an unflagged field binds as, or None for a non-scalar."""
        ftype = field["type"]
        base = ftype["base_type"]
        if base == "Vector" and self.has_uid(field):
            _fail(
                f"'{where}' is a {UID_ATTRIBUTE} vector without {INPUT_ATTRIBUTE} or "
                f"{OUTPUT_ATTRIBUTE}; a tensor-uid vector must be declared an edge list"
            )
        if base not in SCALAR_BASE_TYPES:
            return None
        optional = bool(field.get("optional"))
        index = ftype.get("index", -1)
        if index >= 0:
            enum = short_name(self.schema["enums"][index]["name"])
            return Attribute(field["name"], ENUM_NAME, optional, enum, on_node)
        if base == "Bool":
            return Attribute(field["name"], BOOL, optional, None, on_node)
        if base in ("Float", "Double"):
            return Attribute(field["name"], FLOAT, optional, None, on_node)
        if base == "ULong":
            _fail(
                f"'{where}' is an unsigned 64-bit attribute; its values are not "
                "representable as a signed 64-bit Int"
            )
        if self.has_uid(field):
            _fail(
                f"'{where}' carries {UID_ATTRIBUTE} but neither {INPUT_ATTRIBUTE} nor "
                f"{OUTPUT_ATTRIBUTE}; a tensor reference must be declared an edge"
            )
        return Attribute(field["name"], INT, optional, None, on_node)

    def has_uid(self, field):
        return UID_ATTRIBUTE in self._annotations(field)

    def _classify_field(self, op, field, table_name):
        where = f"{table_name}.{field['name']}"
        annotations = self._annotations(field)
        for key in annotations:
            if key.startswith(ANNOTATION_PREFIX) and key not in FIELD_ATTRIBUTES:
                _fail(f"unknown annotation '{key}' on '{where}'")
        is_input = INPUT_ATTRIBUTE in annotations
        is_output = OUTPUT_ATTRIBUTE in annotations
        if is_input and is_output:
            _fail(f"'{where}' carries both {INPUT_ATTRIBUTE} and {OUTPUT_ATTRIBUTE}")
        if not is_input and not is_output:
            if NAME_ATTRIBUTE in annotations:
                _fail(
                    f"'{where}' carries {NAME_ATTRIBUTE} without {INPUT_ATTRIBUTE} or "
                    f"{OUTPUT_ATTRIBUTE}"
                )
            attribute = self._attribute(field, where, on_node=False)
            if attribute is not None:
                op.attributes.append(attribute)
            return
        flag = INPUT_ATTRIBUTE if is_input else OUTPUT_ATTRIBUTE
        ftype = field["type"]
        if ftype["base_type"] == "Vector":
            self._classify_edge_list(op, field, where, flag, annotations)
            return
        name = annotations.get(NAME_ATTRIBUTE)
        if not name:
            _fail(f"'{where}' carries {flag} without a non-empty {NAME_ATTRIBUTE}")
        if not IDENTIFIER.match(name):
            _fail(f"{NAME_ATTRIBUTE} on '{where}' must be an identifier, got '{name}'")
        if ftype["base_type"] != "Long" or ftype.get("index", -1) >= 0:
            _fail(
                f"{flag} on '{where}' requires a scalar signed 64-bit integer (long) "
                "tensor uid"
            )
        if name in RESERVED_ROOTS:
            _fail(f"{NAME_ATTRIBUTE} '{name}' on '{where}' is a reserved symbol root")
        edges = op.operands if is_input else op.results
        edges.append(Edge(name, field["name"], bool(field.get("optional"))))

    @staticmethod
    def _classify_edge_list(op, field, where, flag, annotations):
        """A flagged vector: an edge list of the node's operand or result uids."""
        if NAME_ATTRIBUTE in annotations:
            _fail(
                f"'{where}' is a tensor-uid vector carrying {NAME_ATTRIBUTE}; an edge "
                "list is never bound by a pattern, so it takes no name"
            )
        ftype = field["type"]
        if ftype.get("element") != "Long" or ftype.get("index", -1) >= 0:
            _fail(
                f"{flag} on '{where}' requires a vector of signed 64-bit integer "
                "([long]) tensor uids"
            )
        edge_lists = op.operand_lists if flag == INPUT_ATTRIBUTE else op.result_lists
        edge_lists.append(EdgeList(field["name"]))

    def _node_attributes(self):
        """The Node table's scalars, merged into every op (RFC 0020 Appendix B.3)."""
        table_name = short_name(self.node["name"])
        attributes = []
        for field in self._fields(self.node):
            where = f"{table_name}.{field['name']}"
            annotations = self._annotations(field)
            for key in annotations:
                if key.startswith(ANNOTATION_PREFIX):
                    _fail(
                        f"annotation '{key}' on '{where}': Node fields bind as "
                        "attributes only"
                    )
            if IGNORE_ATTRIBUTE in annotations:
                continue
            attribute = self._attribute(field, where, on_node=True)
            if attribute is not None:
                attributes.append(attribute)
        return attributes

    def _build(self):
        members = self._union_members()
        member_names = {table["name"] for _, table in members}
        for obj in self.schema["objects"]:
            if (
                OPCODE_ATTRIBUTE in self._annotations(obj)
                and obj["name"] not in member_names
            ):
                _fail(
                    f"{OPCODE_ATTRIBUTE} on table '{short_name(obj['name'])}', which is "
                    "not a NodeAttributes member"
                )

        node_attributes = self.node_attributes
        ops = {}
        for value, table in members:
            table_name = short_name(table["name"])
            opcode = self._opcode(table)
            if opcode in ops:
                _fail(
                    f"opcode '{opcode}' is declared by both "
                    f"'{ops[opcode].table}' and '{table_name}'"
                )
            op = OpSchema(opcode, table_name, value)
            for field in self._fields(table):
                self._classify_field(op, field, table_name)
            op.attributes.extend(node_attributes)

            edge_names = set()
            for edge in op.operands + op.results:
                if edge.name in edge_names:
                    _fail(
                        f"{NAME_ATTRIBUTE} '{edge.name}' is declared twice in "
                        f"'{table_name}' (second on '{table_name}.{edge.field}')"
                    )
                edge_names.add(edge.name)
            attribute_names = set()
            for attribute in op.attributes:
                if attribute.name in attribute_names:
                    owner = "Node" if attribute.on_node else table_name
                    _fail(
                        f"attribute '{attribute.name}' is declared twice for "
                        f"'{table_name}' (second on '{owner}.{attribute.name}')"
                    )
                attribute_names.add(attribute.name)
            ops[opcode] = op
        return [ops[opcode] for opcode in sorted(ops)]


class HeaderEmitter:
    """Emits op_schema_registry_generated.h for a classified Registry."""

    def __init__(self, registry):
        self.registry = registry
        self.node_table = short_name(registry.node["name"])
        self.lines = []

    def w(self, line=""):
        self.lines.append(line)

    def edge_reader(self, op, edge):
        return f"read{op.table}{_camel(edge.field)}"

    def attribute_reader(self, op, attribute):
        if attribute.on_node:
            return f"read{self.node_table}{_camel(attribute.name)}"
        return f"read{op.table}{_camel(attribute.name)}"

    @staticmethod
    def array_name(op, suffix):
        return f"{op.opcode.upper()}_{suffix}"

    def emit(self):
        self.emit_prologue()
        self.w("namespace detail")
        self.w("{")
        self.w()
        for attribute in self.registry.node_attributes:
            self.emit_attribute_reader(None, attribute)
        for op in self.registry.ops:
            for edge in op.operands + op.results:
                self.emit_edge_reader(op, edge)
            for edge_list in op.operand_lists + op.result_lists:
                self.emit_edge_list_reader(op, edge_list)
            for attribute in op.attributes:
                if not attribute.on_node:
                    self.emit_attribute_reader(op, attribute)
            self.emit_arrays(op)
        self.emit_op_table()
        self.w("} // namespace detail")
        self.w()
        self.emit_lookups()
        self.w(f"}} // namespace {NAMESPACE}::data_objects::op_schema")
        return "\n".join(self.lines) + "\n"

    def emit_prologue(self):
        self.w(f"// Automatically generated by {GENERATOR}, do not modify.")
        self.w("//")
        self.w(
            "// The op-schema registry: per NodeAttributes table, the opcode a UED pattern"
        )
        self.w(
            "// names it by, its operand and result tensor edges and edge lists, and its"
        )
        self.w(
            f"// scalar attributes, taken from the '{OPCODE_ATTRIBUTE}', '{INPUT_ATTRIBUTE}',"
        )
        self.w(
            f"// '{OUTPUT_ATTRIBUTE}', and '{NAME_ATTRIBUTE}' schema annotations. Static tables"
        )
        self.w("// read the caller's buffer in place: no allocation, no reflection.")
        self.w()
        self.w("#pragma once")
        self.w()
        self.w("#include <array>")
        self.w("#include <cstddef>")
        self.w("#include <cstdint>")
        self.w("#include <optional>")
        self.w("#include <string_view>")
        self.w("#include <utility>")
        self.w("#include <variant>")
        self.w()
        self.w(f"#include <{NAMESPACE}/data_objects/graph_generated.h>")
        self.w()
        self.w(f"namespace {NAMESPACE}::data_objects::op_schema")
        self.w("{")
        self.w()
        self.w("/// How a scalar attribute's value is represented in a ScalarValue.")
        self.w("enum class ValueKind : uint8_t")
        self.w("{")
        self.w("    INT, ///< int64_t, widened from any integer field.")
        self.w("    FLOAT, ///< double, widened exactly from a float or double field.")
        self.w("    BOOL, ///< bool.")
        self.w(
            "    ENUM_NAME, ///< std::string_view naming the enum value; empty when out of range."
        )
        self.w("};")
        self.w()
        self.w(
            "using ScalarValue = std::variant<bool, int64_t, double, std::string_view>;"
        )
        self.w()
        self.w("/// An operand or result tensor edge of an op.")
        self.w("struct EdgeSchema")
        self.w("{")
        self.w(
            "    std::string_view name; ///< The edge name a pattern binds (matcher_name)."
        )
        self.w("    bool optional; ///< The uid field is `= null`.")
        self.w(
            "    /// The edge's tensor uid; nullopt when an optional uid is absent or the node"
        )
        self.w("    /// does not carry this op's attribute table. Never throws.")
        self.w("    std::optional<int64_t> (*read)(const Node& node);")
        self.w("};")
        self.w()
        self.w("/// A scalar attribute of an op, published as `<node id>.<name>`.")
        self.w("struct AttributeSchema")
        self.w("{")
        self.w("    std::string_view name; ///< The field name.")
        self.w("    ValueKind kind;")
        self.w("    bool optional; ///< The field is `= null`.")
        self.w(
            "    /// The attribute's value; nullopt when an optional field is absent or the node"
        )
        self.w(
            "    /// does not carry this op's attribute table. Node-table attributes read the"
        )
        self.w("    /// Node itself. Never throws.")
        self.w("    std::optional<ScalarValue> (*read)(const Node& node);")
        self.w("};")
        self.w()
        self.w(
            "/// A tensor-uid vector field: its uids are the node's operands or results, but a"
        )
        self.w("/// pattern never binds it by name.")
        self.w("struct EdgeListSchema")
        self.w("{")
        self.w("    std::string_view name; ///< The field name.")
        self.w(
            "    /// The uid vector; nullptr when absent or the node does not carry this op's"
        )
        self.w("    /// attribute table. Never throws.")
        self.w("    const ::flatbuffers::Vector<int64_t>* (*read)(const Node& node);")
        self.w("};")
        self.w()
        self.w(
            "/// A read-only view over a static array (C++17 stand-in for std::span)."
        )
        self.w("template <typename T>")
        self.w("struct SchemaSpan")
        self.w("{")
        self.w("    const T* data;")
        self.w("    size_t size;")
        self.w()
        self.w("    constexpr const T* begin() const")
        self.w("    {")
        self.w("        return data;")
        self.w("    }")
        self.w()
        self.w("    constexpr const T* end() const")
        self.w("    {")
        self.w("        return data + size;")
        self.w("    }")
        self.w()
        self.w("    constexpr bool empty() const")
        self.w("    {")
        self.w("        return size == 0;")
        self.w("    }")
        self.w("};")
        self.w()
        self.w("/// One op: a NodeAttributes table and what a pattern can bind on it.")
        self.w("struct OpSchema")
        self.w("{")
        self.w(
            "    std::string_view opcode; ///< The table's matcher_opcode annotation."
        )
        self.w('    std::string_view attributesTable; ///< e.g. "PointwiseAttributes".')
        self.w("    NodeAttributes attributesType; ///< The union discriminant.")
        self.w("    SchemaSpan<EdgeSchema> operands; ///< Declaration order.")
        self.w("    SchemaSpan<EdgeSchema> results; ///< Declaration order.")
        self.w(
            "    /// Attribute-table scalars in declaration order, then the Node table's scalars."
        )
        self.w("    SchemaSpan<AttributeSchema> attributes;")
        self.w("    SchemaSpan<EdgeListSchema> operandLists; ///< Declaration order.")
        self.w("    SchemaSpan<EdgeListSchema> resultLists; ///< Declaration order.")
        self.w("};")
        self.w()

    def emit_edge_reader(self, op, edge):
        self.w(
            f"inline std::optional<int64_t> {self.edge_reader(op, edge)}(const Node& node) noexcept"
        )
        self.w("{")
        self.w(f"    const auto* table = node.attributes_as<{op.table}>();")
        self.w("    if(table == nullptr)")
        self.w("    {")
        self.w("        return std::nullopt;")
        self.w("    }")
        field = accessor(edge.field)
        if edge.optional:
            self.w(f"    const auto value = table->{field}();")
            self.w("    if(!value.has_value())")
            self.w("    {")
            self.w("        return std::nullopt;")
            self.w("    }")
            self.w("    return value.value();")
        else:
            self.w(f"    return table->{field}();")
        self.w("}")
        self.w()

    def emit_edge_list_reader(self, op, edge_list):
        self.w(
            "inline const ::flatbuffers::Vector<int64_t>* "
            f"{self.edge_reader(op, edge_list)}(const Node& node) noexcept"
        )
        self.w("{")
        self.w(f"    const auto* table = node.attributes_as<{op.table}>();")
        self.w("    if(table == nullptr)")
        self.w("    {")
        self.w("        return nullptr;")
        self.w("    }")
        self.w(f"    return table->{accessor(edge_list.field)}();")
        self.w("}")
        self.w()

    def emit_attribute_reader(self, op, attribute):
        name = self.attribute_reader(op, attribute)
        self.w(f"inline std::optional<ScalarValue> {name}(const Node& node) noexcept")
        self.w("{")
        if attribute.on_node:
            source = "node."
        else:
            self.w(f"    const auto* table = node.attributes_as<{op.table}>();")
            self.w("    if(table == nullptr)")
            self.w("    {")
            self.w("        return std::nullopt;")
            self.w("    }")
            source = "table->"
        read = f"{source}{accessor(attribute.name)}()"
        if attribute.optional:
            self.w(f"    const auto value = {read};")
            self.w("    if(!value.has_value())")
            self.w("    {")
            self.w("        return std::nullopt;")
            self.w("    }")
            read = "value.value()"
        if attribute.kind is ENUM_NAME:
            read = f"EnumName{attribute.enum}({read})"
        self.w(
            f"    return ScalarValue{{std::in_place_type<{attribute.kind[2]}>, {read}}};"
        )
        self.w("}")
        self.w()

    def emit_arrays(self, op):
        for suffix, edges in (("OPERANDS", op.operands), ("RESULTS", op.results)):
            if not edges:
                continue
            self.w(
                f"inline constexpr std::array<EdgeSchema, {len(edges)}> "
                f"{self.array_name(op, suffix)} = {{{{"
            )
            for edge in edges:
                self.w(
                    f'    {{"{edge.name}", {self.bool(edge.optional)}, '
                    f"&{self.edge_reader(op, edge)}}},"
                )
            self.w("}};")
            self.w()
        for suffix, edge_lists in (
            ("OPERAND_LISTS", op.operand_lists),
            ("RESULT_LISTS", op.result_lists),
        ):
            if not edge_lists:
                continue
            self.w(
                f"inline constexpr std::array<EdgeListSchema, {len(edge_lists)}> "
                f"{self.array_name(op, suffix)} = {{{{"
            )
            for edge_list in edge_lists:
                self.w(
                    f'    {{"{edge_list.field}", &{self.edge_reader(op, edge_list)}}},'
                )
            self.w("}};")
            self.w()
        if op.attributes:
            self.w(
                f"inline constexpr std::array<AttributeSchema, {len(op.attributes)}> "
                f"{self.array_name(op, 'ATTRIBUTES')} = {{{{"
            )
            for attribute in op.attributes:
                self.w(
                    f'    {{"{attribute.name}", ValueKind::{attribute.kind[1]}, '
                    f"{self.bool(attribute.optional)}, "
                    f"&{self.attribute_reader(op, attribute)}}},"
                )
            self.w("}};")
            self.w()

    @staticmethod
    def bool(value):
        return "true" if value else "false"

    def span(self, op, suffix, element, entries):
        if not entries:
            return f"SchemaSpan<{element}>{{nullptr, 0}}"
        name = self.array_name(op, suffix)
        return f"SchemaSpan<{element}>{{{name}.data(), {name}.size()}}"

    def emit_op_table(self):
        ops = self.registry.ops
        self.w("/// Sorted by opcode.")
        self.w(f"inline constexpr std::array<OpSchema, {len(ops)}> OP_SCHEMAS = {{{{")
        for op in ops:
            self.w(f'    {{"{op.opcode}",')
            self.w(f'     "{op.table}",')
            self.w(f"     NodeAttributes::{op.table},")
            self.w(f"     {self.span(op, 'OPERANDS', 'EdgeSchema', op.operands)},")
            self.w(f"     {self.span(op, 'RESULTS', 'EdgeSchema', op.results)},")
            self.w(
                f"     {self.span(op, 'ATTRIBUTES', 'AttributeSchema', op.attributes)},"
            )
            self.w(
                f"     {self.span(op, 'OPERAND_LISTS', 'EdgeListSchema', op.operand_lists)},"
            )
            self.w(
                f"     {self.span(op, 'RESULT_LISTS', 'EdgeListSchema', op.result_lists)}}},"
            )
        self.w("}};")
        self.w()

    def emit_lookups(self):
        self.w("/// Every op, sorted by opcode.")
        self.w("constexpr SchemaSpan<OpSchema> allOpSchemas() noexcept")
        self.w("{")
        self.w(
            "    return SchemaSpan<OpSchema>{detail::OP_SCHEMAS.data(), detail::OP_SCHEMAS.size()};"
        )
        self.w("}")
        self.w()
        self.w("/// The op named @p opcode, or nullptr when no op declares it.")
        self.w(
            "constexpr const OpSchema* findOpSchema(std::string_view opcode) noexcept"
        )
        self.w("{")
        self.w("    size_t low  = 0;")
        self.w("    size_t high = detail::OP_SCHEMAS.size();")
        self.w("    while(low < high)")
        self.w("    {")
        self.w("        const size_t middle = low + ((high - low) / 2);")
        self.w("        if(detail::OP_SCHEMAS[middle].opcode < opcode)")
        self.w("        {")
        self.w("            low = middle + 1;")
        self.w("        }")
        self.w("        else")
        self.w("        {")
        self.w("            high = middle;")
        self.w("        }")
        self.w("    }")
        self.w(
            "    if(low < detail::OP_SCHEMAS.size() && detail::OP_SCHEMAS[low].opcode == opcode)"
        )
        self.w("    {")
        self.w("        return &detail::OP_SCHEMAS[low];")
        self.w("    }")
        self.w("    return nullptr;")
        self.w("}")
        self.w()
        self.w(
            "/// The op whose attribute table is @p attributesType, or nullptr for NONE or a"
        )
        self.w("/// value no op declares.")
        self.w(
            "constexpr const OpSchema* findOpSchema(NodeAttributes attributesType) noexcept"
        )
        self.w("{")
        self.w("    for(const auto& op : detail::OP_SCHEMAS)")
        self.w("    {")
        self.w("        if(op.attributesType == attributesType)")
        self.w("        {")
        self.w("            return &op;")
        self.w("        }")
        self.w("    }")
        self.w("    return nullptr;")
        self.w("}")
        self.w()


def emit_json(registry):
    def edges(entries):
        return [{"name": e.name, "optional": e.optional} for e in entries]

    def edge_lists(entries):
        return [{"name": e.field} for e in entries]

    ops = {}
    for op in registry.ops:
        ops[op.opcode] = {
            "attributes_table": op.table,
            "attributes_type": op.attributes_type,
            "operands": edges(op.operands),
            "results": edges(op.results),
            "operand_lists": edge_lists(op.operand_lists),
            "result_lists": edge_lists(op.result_lists),
            "attributes": [
                {"name": a.name, "kind": a.kind[0], "optional": a.optional}
                for a in op.attributes
            ],
        }
    document = {"generator": GENERATOR, "ops": ops}
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flatc",
        default=None,
        help="flatc binary to use; defaults to the one on PATH. The build passes the "
        "binary CMake already resolved, which need not be on PATH.",
    )
    flatc_path = _resolve_flatc(parser.parse_args().flatc)
    sdk_dir = os.path.join(_HIPDNN_DIR, SDK_DIR)
    schemas_dir = os.path.join(sdk_dir, "schemas")
    header_dir = os.path.join(sdk_dir, "include", NAMESPACE, "data_objects")

    with tempfile.TemporaryDirectory() as work_dir:
        schema = load_schema(flatc_path, schemas_dir, SCHEMA_FILE, work_dir)
    registry = Registry(schema)
    # Both outputs are rendered before either is written, so a failure leaves the
    # checked-in pair consistent.
    outputs = [
        (os.path.join(header_dir, HEADER_NAME), HeaderEmitter(registry).emit()),
        (os.path.join(sdk_dir, JSON_NAME), emit_json(registry)),
    ]
    os.makedirs(header_dir, exist_ok=True)
    for destination, content in outputs:
        with open(destination, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)


if __name__ == "__main__":
    main()
