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
  compiles. Each edge and attribute reads its field through a reader template
  instantiated on the FlatBuffers accessor.
- ``op_schema_registry.json``: each op's operand and result edges, for build-time
  tools (hkp_pack).

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
from dataclasses import dataclass, field as dataclass_field
from string import Template
from typing import Optional

from gen_cache_key import (
    HIPDNN_DIR,
    IGNORE_ATTRIBUTE,
    NAMESPACE,
    SCALAR_BASE_TYPES,
    SDK_DIR,
    UID_ATTRIBUTE,
    accessor,
    load_schema,
    resolve_flatc,
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

# Value kinds, named by their C++ ValueKind enumerator.
INT = "INT"
FLOAT = "FLOAT"
BOOL = "BOOL"
ENUM_NAME = "ENUM_NAME"

# The ScalarValue alternative each non-enum kind is held as.
SCALAR_ALTERNATIVES = {INT: "int64_t", FLOAT: "double", BOOL: "bool"}


def _fail(message):
    raise SystemExit(f"ERROR: {message}")


@dataclass
class Edge:
    name: str
    field: str
    optional: bool


@dataclass
class EdgeList:
    field: str


@dataclass
class Attribute:
    name: str
    kind: str
    optional: bool
    # Short enum type name for ENUM_NAME, else None.
    enum: Optional[str]
    # Read from the Node table rather than the attribute table.
    on_node: bool


@dataclass
class OpSchema:
    opcode: str
    table: str
    operands: list = dataclass_field(default_factory=list)
    results: list = dataclass_field(default_factory=list)
    operand_lists: list = dataclass_field(default_factory=list)
    result_lists: list = dataclass_field(default_factory=list)
    attributes: list = dataclass_field(default_factory=list)


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
        """The table object of each NodeAttributes member, NONE excluded."""
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
            members.append(self.schema["objects"][member["index"]])
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
        member_names = {table["name"] for table in members}
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
        for table in members:
            table_name = short_name(table["name"])
            opcode = self._opcode(table)
            if opcode in ops:
                _fail(
                    f"opcode '{opcode}' is declared by both "
                    f"'{ops[opcode].table}' and '{table_name}'"
                )
            op = OpSchema(opcode, table_name)
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


# Everything before the per-op tables: the schema types and the reader templates every
# table row instantiates.
HEADER_PROLOGUE = Template(
    """\
// Automatically generated by ${generator}, do not modify.
//
// The op-schema registry: per NodeAttributes table, the opcode a UED pattern
// names it by, its operand and result tensor edges and edge lists, and its
// scalar attributes, taken from the '${opcode}', '${input}',
// '${output}', and '${name}' schema annotations. Static tables
// read the caller's buffer in place: no allocation, no reflection.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <${namespace}/data_objects/graph_generated.h>

namespace ${namespace}::data_objects::op_schema
{

/// How a scalar attribute's value is represented in a ScalarValue.
enum class ValueKind : uint8_t
{
    INT, ///< int64_t, widened from any integer field.
    FLOAT, ///< double, widened exactly from a float or double field.
    BOOL, ///< bool.
    ENUM_NAME, ///< std::string_view naming the enum value; empty when out of range.
};

using ScalarValue = std::variant<bool, int64_t, double, std::string_view>;

/// An operand or result tensor edge of an op.
struct EdgeSchema
{
    std::string_view name; ///< The edge name a pattern binds (matcher_name).
    bool optional; ///< The uid field is `= null`.
    /// The edge's tensor uid; nullopt when an optional uid is absent or the node
    /// does not carry this op's attribute table. Never throws.
    std::optional<int64_t> (*read)(const Node& node);
};

/// A scalar attribute of an op, published as `<node id>.<name>`.
struct AttributeSchema
{
    std::string_view name; ///< The field name.
    ValueKind kind;
    bool optional; ///< The field is `= null`.
    /// The attribute's value; nullopt when an optional field is absent or the node
    /// does not carry this op's attribute table. Node-table attributes read the
    /// Node itself. Never throws.
    std::optional<ScalarValue> (*read)(const Node& node);
};

/// A tensor-uid vector field: its uids are the node's operands or results, but a
/// pattern never binds it by name.
struct EdgeListSchema
{
    std::string_view name; ///< The field name.
    /// The uid vector; nullptr when absent or the node does not carry this op's
    /// attribute table. Never throws.
    const ::flatbuffers::Vector<int64_t>* (*read)(const Node& node);
};

/// A read-only view over a static array (C++17 stand-in for std::span).
template <typename T>
struct SchemaSpan
{
    const T* data;
    size_t size;

    constexpr const T* begin() const
    {
        return data;
    }

    constexpr const T* end() const
    {
        return data + size;
    }

    constexpr bool empty() const
    {
        return size == 0;
    }
};

/// One op: a NodeAttributes table and what a pattern can bind on it.
struct OpSchema
{
    std::string_view opcode; ///< The table's matcher_opcode annotation.
    NodeAttributes attributesType; ///< The union discriminant.
    SchemaSpan<EdgeSchema> operands; ///< Declaration order.
    SchemaSpan<EdgeSchema> results; ///< Declaration order.
    /// Attribute-table scalars in declaration order, then the Node table's scalars.
    SchemaSpan<AttributeSchema> attributes;
    SchemaSpan<EdgeListSchema> operandLists; ///< Declaration order.
    SchemaSpan<EdgeListSchema> resultLists; ///< Declaration order.
};

namespace detail
{

template <typename Accessor>
struct AccessorTable;

template <typename Table, typename Value>
struct AccessorTable<Value (Table::*)() const>
{
    using type = Table;
};

/// The table @p Field reads: the Node itself for a Node field, else the node's
/// attribute table when it is @p Field's table, else nullptr.
template <auto Field>
const auto* fieldTable(const Node& node) noexcept
{
    using Table = typename AccessorTable<decltype(Field)>::type;
    if constexpr(std::is_same_v<Table, Node>)
    {
        return &node;
    }
    else
    {
        return node.attributes_as<Table>();
    }
}

template <typename T>
std::optional<T> present(T value) noexcept
{
    return value;
}

template <typename T>
std::optional<T> present(const ::flatbuffers::Optional<T>& value) noexcept
{
    if(!value.has_value())
    {
        return std::nullopt;
    }
    return *value;
}

/// @p Field's value; nullopt when the node does not carry its table or an optional
/// field is absent.
template <auto Field>
auto readField(const Node& node) noexcept
{
    const auto* table = fieldTable<Field>(node);
    using Value = decltype(present((table->*Field)()));
    if(table == nullptr)
    {
        return Value{};
    }
    return present((table->*Field)());
}

template <auto Field>
std::optional<int64_t> readEdge(const Node& node) noexcept
{
    return readField<Field>(node);
}

template <auto Field>
const ::flatbuffers::Vector<int64_t>* readEdgeList(const Node& node) noexcept
{
    const auto* table = fieldTable<Field>(node);
    return table == nullptr ? nullptr : (table->*Field)();
}

/// A scalar attribute held as @p Alternative.
template <typename Alternative, auto Field>
std::optional<ScalarValue> readScalar(const Node& node) noexcept
{
    const auto value = readField<Field>(node);
    if(!value.has_value())
    {
        return std::nullopt;
    }
    return ScalarValue{std::in_place_type<Alternative>, *value};
}

/// An enum attribute held as the name @p EnumName gives its value.
template <auto Field, auto EnumName>
std::optional<ScalarValue> readEnumName(const Node& node) noexcept
{
    const auto value = readField<Field>(node);
    if(!value.has_value())
    {
        return std::nullopt;
    }
    return ScalarValue{std::in_place_type<std::string_view>, EnumName(*value)};
}

template <typename T, size_t N>
constexpr SchemaSpan<T> spanOf(const std::array<T, N>& entries) noexcept
{
    return SchemaSpan<T>{entries.data(), N};
}

"""
)

# Everything after the op table: the lookups.
HEADER_EPILOGUE = Template(
    """\
} // namespace detail

/// Every op, sorted by opcode.
constexpr SchemaSpan<OpSchema> allOpSchemas() noexcept
{
    return detail::spanOf(detail::OP_SCHEMAS);
}

/// The op named @p opcode, or nullptr when no op declares it.
constexpr const OpSchema* findOpSchema(std::string_view opcode) noexcept
{
    size_t low  = 0;
    size_t high = detail::OP_SCHEMAS.size();
    while(low < high)
    {
        const size_t middle = low + ((high - low) / 2);
        if(detail::OP_SCHEMAS[middle].opcode < opcode)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    if(low < detail::OP_SCHEMAS.size() && detail::OP_SCHEMAS[low].opcode == opcode)
    {
        return &detail::OP_SCHEMAS[low];
    }
    return nullptr;
}

/// The op whose attribute table is @p attributesType, or nullptr for NONE or a
/// value no op declares.
constexpr const OpSchema* findOpSchema(NodeAttributes attributesType) noexcept
{
    for(const auto& op : detail::OP_SCHEMAS)
    {
        if(op.attributesType == attributesType)
        {
            return &op;
        }
    }
    return nullptr;
}

} // namespace ${namespace}::data_objects::op_schema
"""
)


def _bool(value):
    return "true" if value else "false"


def _member(table, name):
    return f"&{table}::{accessor(name)}"


def _attribute_reader(op, attribute, node_table):
    member = _member(node_table if attribute.on_node else op.table, attribute.name)
    if attribute.kind == ENUM_NAME:
        return f"&readEnumName<{member}, &EnumName{attribute.enum}>"
    return f"&readScalar<{SCALAR_ALTERNATIVES[attribute.kind]}, {member}>"


def emit_header(registry):
    node_table = short_name(registry.node["name"])
    out = [
        HEADER_PROLOGUE.substitute(
            generator=GENERATOR,
            opcode=OPCODE_ATTRIBUTE,
            input=INPUT_ATTRIBUTE,
            output=OUTPUT_ATTRIBUTE,
            name=NAME_ATTRIBUTE,
            namespace=NAMESPACE,
        )
    ]

    def array(op, suffix, element, rows):
        """Emits one op's array; returns the OP_SCHEMAS span expression for it."""
        if not rows:
            return "{}"
        name = f"{op.opcode.upper()}_{suffix}"
        out.append(
            f"inline constexpr std::array<{element}, {len(rows)}> {name} = {{{{\n"
        )
        out.extend(f"    {{{row}}},\n" for row in rows)
        out.append("}};\n\n")
        return f"spanOf({name})"

    op_rows = []
    for op in registry.ops:
        spans = [
            array(
                op,
                suffix,
                "EdgeSchema",
                [
                    f'"{e.name}", {_bool(e.optional)}, &readEdge<{_member(op.table, e.field)}>'
                    for e in edges
                ],
            )
            for suffix, edges in (("OPERANDS", op.operands), ("RESULTS", op.results))
        ]
        spans.append(
            array(
                op,
                "ATTRIBUTES",
                "AttributeSchema",
                [
                    f'"{a.name}", ValueKind::{a.kind}, {_bool(a.optional)}, '
                    f"{_attribute_reader(op, a, node_table)}"
                    for a in op.attributes
                ],
            )
        )
        spans.extend(
            array(
                op,
                suffix,
                "EdgeListSchema",
                [
                    f'"{e.field}", &readEdgeList<{_member(op.table, e.field)}>'
                    for e in edge_lists
                ],
            )
            for suffix, edge_lists in (
                ("OPERAND_LISTS", op.operand_lists),
                ("RESULT_LISTS", op.result_lists),
            )
        )
        op_rows.append(
            f'    {{"{op.opcode}", NodeAttributes::{op.table}, {", ".join(spans)}}},\n'
        )

    out.append("/// Sorted by opcode.\n")
    out.append(
        f"inline constexpr std::array<OpSchema, {len(op_rows)}> OP_SCHEMAS = {{{{\n"
    )
    out.extend(op_rows)
    out.append("}};\n\n")
    out.append(HEADER_EPILOGUE.substitute(namespace=NAMESPACE))
    return "".join(out)


def emit_json(registry):
    """One edge object per line, so a schema change diffs as the edges it touches."""

    def edges(key, entries, last):
        rows = [json.dumps({"name": e.name, "optional": e.optional}) for e in entries]
        end = "" if last else ","
        if not rows:
            return [f'      "{key}": []{end}']
        return (
            [f'      "{key}": [']
            + [f"        {row}," for row in rows[:-1]]
            + [f"        {rows[-1]}", f"      ]{end}"]
        )

    lines = ["{", f'  "generator": {json.dumps(GENERATOR)},', '  "ops": {']
    for position, op in enumerate(registry.ops):
        lines.append(f"    {json.dumps(op.opcode)}: {{")
        lines += edges("operands", op.operands, last=False)
        lines += edges("results", op.results, last=True)
        lines.append("    }" if position == len(registry.ops) - 1 else "    },")
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flatc",
        default=None,
        help="flatc binary to use; defaults to the one on PATH. The build passes the "
        "binary CMake already resolved, which need not be on PATH.",
    )
    flatc_path = resolve_flatc(parser.parse_args().flatc)
    sdk_dir = os.path.join(HIPDNN_DIR, SDK_DIR)
    schemas_dir = os.path.join(sdk_dir, "schemas")
    header_dir = os.path.join(sdk_dir, "include", NAMESPACE, "data_objects")

    with tempfile.TemporaryDirectory() as work_dir:
        schema = load_schema(flatc_path, schemas_dir, SCHEMA_FILE, work_dir)
    registry = Registry(schema)
    # Both outputs are rendered before either is written, so a failure leaves the
    # checked-in pair consistent.
    outputs = [
        (os.path.join(header_dir, HEADER_NAME), emit_header(registry)),
        (os.path.join(sdk_dir, JSON_NAME), emit_json(registry)),
    ]
    os.makedirs(header_dir, exist_ok=True)
    for destination, content in outputs:
        with open(destination, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)


if __name__ == "__main__":
    main()
