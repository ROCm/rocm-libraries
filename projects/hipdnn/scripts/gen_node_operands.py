# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generate the node-operand visitor header (node_operands_generated.h) from the schemas.

Emits, per ``NodeAttributes`` union member, a zero-allocation visitor over the node's
operands: every ``cache_uid`` field as a tensor role, every scalar/enum/bool attribute,
and every element of a numeric-vector attribute. Heuristics publish these as features
(plugin_sdk EngineFeatures.hpp), so a node type added to the schema publishes its
operands without hand-written per-op code.

Shares gen_cache_key.py's binary-schema reader: flatc parses and resolves the schema,
and the annotations deciding what a field is -- ``cache_uid`` for a tensor reference,
``work_data_dependent`` for one whose contents decide the node's work -- are read from
the compiled schema, never inferred from a field's name.

Takes no arguments: it re-derives the whole header. Run manually, from the build via
the custom target in flatbuffers_sdk/CMakeLists.txt, or through the
``node-operands-hipdnn`` pre-commit hook.
"""

import argparse
import os
import tempfile

from gen_cache_key import (
    INTEGER_BASE_TYPES,
    NAMESPACE,
    SDK_DIR,
    UID_ATTRIBUTE,
    UID_DOMAIN_ATTRIBUTE,
    _HIPDNN_DIR,
    _resolve_flatc,
    accessor,
    load_schema,
    short_name,
)

# On a `cache_uid` field: the referenced tensor's contents decide how much work the
# node does. Meaningless on anything that is not a tensor reference.
WORK_ATTRIBUTE = "work_data_dependent"

SCHEMA_FILE = "graph.fbs"
ROOT_TABLE = "hipdnn_flatbuffers_sdk.data_objects.Graph"
NODE_TABLE = "hipdnn_flatbuffers_sdk.data_objects.Node"
HEADER_NAME = "node_operands_generated.h"

# Stripped from a tensor-reference field's name to form its role (`x_tensor_uid` ->
# `x`, `input_tensor_uids` -> `input`). Longest first so `_uids` is not left behind.
ROLE_SUFFIXES = ("_tensor_uids", "_tensor_uid")

FLOAT_BASE_TYPES = frozenset(["Float", "Double"])

# A byte vector is an opaque payload (`CustomOpAttributes.data`), not per-element
# attributes: publishing it element by element would describe bytes, not the problem.
OPAQUE_ELEMENT_TYPES = frozenset(["Byte", "UByte"])

# Base types that carry no scalar operand: text, nested tables/structs and unions.
UNVISITED_BASE_TYPES = frozenset(["String", "Obj", "Union"])


class Operand:
    """One visitor call site: what a field contributes, derived from its schema type."""

    TENSOR = "tensor"
    SCALAR = "scalar"
    ELEMENTS = "elements"

    def __init__(self, kind, field, name, cpp_type=None, work=False):
        self.kind = kind
        self.field = field
        self.name = name
        self.cpp_type = cpp_type
        self.work = work

    @property
    def accessor(self):
        return accessor(self.field["name"])

    @property
    def optional(self):
        return bool(self.field.get("optional"))

    @property
    def vector(self):
        return self.field["type"]["base_type"] == "Vector"


class OperandsEmitter:
    """Walks the node union's member tables and emits the visitor header."""

    def __init__(self, schema, root_name=ROOT_TABLE, node_name=NODE_TABLE):
        self.schema = schema
        self.objects = {o["name"]: o for o in schema["objects"]}
        self.node_name = node_name
        self.union_field, self.union = self._resolve_union()
        self.tensor_table = self._resolve_tensor_table(root_name)
        self.members = [
            (value["name"], self.schema["objects"][value["union_type"]["index"]])
            for value in self.union["values"]
            if value.get("union_type") and value["union_type"].get("index", -1) >= 0
        ]
        self.operands = {
            member["name"]: self.operands_of(member) for _, member in self.members
        }
        self._validate_work_annotations()

    @staticmethod
    def has_attribute(field, key):
        return any(a["key"] == key for a in field.get("attributes", []))

    @staticmethod
    def fields_of(obj):
        # The binary schema sorts fields alphabetically; restore declaration order so
        # the visitor reports operands in the order the .fbs reads.
        return sorted(
            (f for f in obj["fields"] if not f.get("deprecated")),
            key=lambda f: f.get("id", 0),
        )

    def _resolve_union(self):
        node = self.objects.get(self.node_name)
        if node is None:
            raise SystemExit(f"ERROR: the schema has no '{self.node_name}' table")
        unions = [f for f in self.fields_of(node) if f["type"]["base_type"] == "Union"]
        if len(unions) != 1:
            raise SystemExit(
                f"ERROR: '{short_name(self.node_name)}' must hold exactly one union of "
                f"node attributes, found {len(unions)}"
            )
        field = unions[0]
        return accessor(field["name"]), self.schema["enums"][field["type"]["index"]]

    def _resolve_tensor_table(self, root_name):
        """The table `cache_uid` references resolve to, from the root's domain vector.

        None when the schema declares no domain: there is then no tensor table whose
        own fields could make work data-dependent.
        """
        root = self.objects.get(root_name)
        if root is None:
            return None
        for field in self.fields_of(root):
            if self.has_attribute(field, UID_DOMAIN_ATTRIBUTE):
                return self.schema["objects"][field["type"]["index"]]
        return None

    def operands_of(self, obj):
        owner = short_name(obj["name"])
        operands = []
        for field in self.fields_of(obj):
            operand = self.classify(owner, field)
            if operand is not None:
                operands.append(operand)
        seen = {}
        for operand in operands:
            # A role and an attribute sharing a name would publish `<name>` beside
            # `<name>.dims[i]`, and two roles sharing one would overwrite each other.
            if operand.name in seen:
                raise SystemExit(
                    f"ERROR: '{owner}.{operand.field['name']}' and "
                    f"'{owner}.{seen[operand.name]}' both publish as '{operand.name}'"
                )
            seen[operand.name] = operand.field["name"]
        return operands

    @staticmethod
    def role(name):
        for suffix in ROLE_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                return name[: -len(suffix)]
        return name

    def classify(self, owner, field):
        """The operand @p field contributes, or None for a field the visitor skips."""
        ftype = field["type"]
        base = ftype["base_type"]
        name = field["name"]
        where = f"'{owner}.{name}'"
        work = self.has_attribute(field, WORK_ATTRIBUTE)

        if self.has_attribute(field, UID_ATTRIBUTE):
            integer = (
                ftype.get("element") if base == "Vector" else base
            ) in INTEGER_BASE_TYPES and ftype.get("index", -1) < 0
            if not integer:
                raise SystemExit(
                    f"ERROR: {UID_ATTRIBUTE} on {where} requires an integer scalar or a "
                    "vector of integers"
                )
            return Operand(Operand.TENSOR, field, self.role(name), work=work)
        if work:
            raise SystemExit(
                f"ERROR: {WORK_ATTRIBUTE} on {where} requires a {UID_ATTRIBUTE} tensor "
                "reference; only a tensor's contents can make work data-dependent"
            )
        if base == "UType" or base in UNVISITED_BASE_TYPES:
            return None
        if base == "Vector":
            element = ftype.get("element")
            if element in UNVISITED_BASE_TYPES:
                return None
            is_enum = ftype.get("index", -1) >= 0
            if element in OPAQUE_ELEMENT_TYPES and not is_enum:
                return None
            if element in INTEGER_BASE_TYPES:
                return Operand(Operand.ELEMENTS, field, name, "int64_t")
            if element in FLOAT_BASE_TYPES:
                return Operand(Operand.ELEMENTS, field, name, "double")
            raise SystemExit(
                f"ERROR: {where} is a vector of '{element}'; the generator has no rule "
                "to visit its elements"
            )
        if base == "Bool":
            return Operand(Operand.SCALAR, field, name, "bool")
        if base in INTEGER_BASE_TYPES:
            # Enums included: their values are their identity, widened like integers.
            return Operand(Operand.SCALAR, field, name, "int64_t")
        if base in FLOAT_BASE_TYPES:
            return Operand(Operand.SCALAR, field, name, "double")
        raise SystemExit(
            f"ERROR: {where} has unhandled base type '{base}'; the generator has no "
            "rule to visit it"
        )

    def _validate_work_annotations(self):
        """`work_data_dependent` must sit where the visitor reports it.

        Anywhere else it would compile and silently mark nothing.
        """
        reported = {member["name"] for _, member in self.members}
        if self.tensor_table is not None:
            reported.add(self.tensor_table["name"])
        for obj in self.schema["objects"]:
            for field in self.fields_of(obj):
                if not self.has_attribute(field, WORK_ATTRIBUTE):
                    continue
                where = f"'{short_name(obj['name'])}.{field['name']}'"
                if obj["name"] not in reported:
                    raise SystemExit(
                        f"ERROR: {WORK_ATTRIBUTE} on {where} is never reported: only "
                        f"'{short_name(self.union['name'])}' members and the tensor table "
                        "are visited"
                    )
                if not self.has_attribute(field, UID_ATTRIBUTE):
                    raise SystemExit(
                        f"ERROR: {WORK_ATTRIBUTE} on {where} requires a {UID_ATTRIBUTE} "
                        "tensor reference; only a tensor's contents can make work "
                        "data-dependent"
                    )

    def emit(self):
        lines = []
        w = lines.append
        namespace = NODE_TABLE.rsplit(".", 1)[0].replace(".", "::") + "::node_operands"
        self.emit_prologue(w, namespace)
        self.emit_indexed_role(w)
        self.emit_tensor_predicate(w)
        for _, member in self.members:
            self.emit_member(w, member)
        self.emit_entry(w)
        w(f"}} // namespace {namespace}")
        return "\n".join(lines) + "\n"

    def emit_prologue(self, w, namespace):
        w("// Automatically generated by scripts/gen_node_operands.py, do not modify.")
        w("//")
        w("// Per node type, a visitor over the node's operands, from the FlatBuffers schemas:")
        w("//")
        w("//   visitor.tensor(std::string_view role, int64_t uid, bool workDataDependent)")
        w(f"//     for every '{UID_ATTRIBUTE}' field present. The role is the field name")
        w("//     without '_tensor_uid' ('x_tensor_uid' -> \"x\"); a vector of uids reports")
        w("//     one call per element as \"role[i]\". An absent optional uid is skipped.")
        w(f"//     workDataDependent is the field's '{WORK_ATTRIBUTE}' annotation: the")
        w("//     tensor's contents, not its shape, decide how much work the node does.")
        w("//   visitor.scalar(std::string_view name, int64_t | double | bool)")
        w("//     for every present scalar attribute that is not a tensor reference: integers")
        w("//     and enums as int64_t, floats as double, bools as bool. An absent optional")
        w("//     is skipped; a non-optional field reports its schema default when unset.")
        w("//   visitor.element(std::string_view name, size_t index, int64_t | double)")
        w("//     for every element of a numeric vector attribute (e.g. conv 'stride').")
        w("//")
        w("// Strings, nested tables and structs, unions and byte vectors (opaque payloads)")
        w("// are not visited. Operands are reported in schema declaration order. Reads the")
        w("// caller's buffer in place: no UnPack, no allocation, no reflection. Every")
        w("// std::string_view is valid only for the duration of the call.")
        w("")
        w("#pragma once")
        w("")
        w("#include <charconv>")
        w("#include <cstddef>")
        w("#include <cstdint>")
        w("#include <string_view>")
        w("")
        w("#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>")
        w("")
        w(f"namespace {namespace}")
        w("{")
        w("")

    def emit_indexed_role(self, w):
        longest = max(
            (
                len(operand.name)
                for operands in self.operands.values()
                for operand in operands
                if operand.kind == Operand.TENSOR and operand.vector
            ),
            default=0,
        )
        w("namespace detail")
        w("{")
        w("/// \"role[index]\" formatted on the stack, so each element of a uid vector reaches")
        w("/// the visitor as its own role without allocating.")
        w("class IndexedRole")
        w("{")
        w("public:")
        w("    IndexedRole(std::string_view role, uint32_t index)")
        w("    {")
        w("        _size = role.copy(_text, role.size());")
        w("        _text[_size++] = '[';")
        w("        _size = static_cast<size_t>(")
        w("            std::to_chars(_text + _size, _text + sizeof(_text) - 1, index).ptr - _text);")
        w("        _text[_size++] = ']';")
        w("    }")
        w("")
        w("    std::string_view view() const")
        w("    {")
        w("        return {_text, _size};")
        w("    }")
        w("")
        w("private:")
        w("    // The longest vector role in the schema, '[', a uint32_t's ten digits, ']'.")
        w(f"    char _text[{longest} + 12] = {{}};")
        w("    size_t _size = 0;")
        w("};")
        w("} // namespace detail")
        w("")

    def emit_tensor_predicate(self, w):
        if self.tensor_table is None:
            return
        short = short_name(self.tensor_table["name"])
        terms = []
        for field in self.fields_of(self.tensor_table):
            if not self.has_attribute(field, WORK_ATTRIBUTE):
                continue
            get = f"tensor.{accessor(field['name'])}()"
            if field["type"]["base_type"] == "Vector":
                terms.append(f"({get} != nullptr && {get}->size() != 0)")
            elif field.get("optional"):
                terms.append(f"{get}.has_value()")
            else:
                terms.append("true")
        w("/// Whether @p tensor's own fields make the work of every node reading or writing")
        w(f"/// it depend on tensor contents ('{WORK_ATTRIBUTE}' on a {short} field).")
        if not terms:
            w(f"inline bool workDataDependent(const {short}&)")
            w("{")
            w("    return false;")
        else:
            w(f"inline bool workDataDependent(const {short}& tensor)")
            w("{")
            w(f"    return {' || '.join(terms)};")
        w("}")
        w("")

    def emit_member(self, w, member):
        short = short_name(member["name"])
        operands = self.operands[member["name"]]
        if operands:
            w("template <typename V>")
            w(f"void visit(const {short}& op, V& visitor)")
        else:
            w("template <typename V>")
            w(f"void visit(const {short}&, V&)")
        w("{")
        for operand in operands:
            self.emit_operand(w, operand)
        w("}")
        w("")

    @staticmethod
    def emit_operand(w, operand):
        get = f"op.{operand.accessor}()"
        name = f'std::string_view("{operand.name}")'
        if operand.kind == Operand.TENSOR:
            work = "true" if operand.work else "false"
            if operand.vector:
                w(f"    if(const auto* uids = {get})")
                w("    {")
                w("        for(uint32_t index = 0; index < uids->size(); ++index)")
                w("        {")
                w(f'            const detail::IndexedRole role("{operand.name}", index);')
                w(
                    "            visitor.tensor(role.view(), "
                    f"static_cast<int64_t>(uids->Get(index)), {work});"
                )
                w("        }")
                w("    }")
            elif operand.optional:
                w(f"    if(const auto uid = {get}; uid.has_value())")
                w("    {")
                w(f"        visitor.tensor({name}, static_cast<int64_t>(*uid), {work});")
                w("    }")
            else:
                w(f"    visitor.tensor({name}, static_cast<int64_t>({get}), {work});")
        elif operand.kind == Operand.SCALAR:
            if operand.optional:
                w(f"    if(const auto value = {get}; value.has_value())")
                w("    {")
                w(
                    f"        visitor.scalar({name}, "
                    f"static_cast<{operand.cpp_type}>(*value));"
                )
                w("    }")
            else:
                w(f"    visitor.scalar({name}, static_cast<{operand.cpp_type}>({get}));")
        else:
            w(f"    if(const auto* values = {get})")
            w("    {")
            w("        for(uint32_t index = 0; index < values->size(); ++index)")
            w("        {")
            w(
                f"            visitor.element({name}, static_cast<size_t>(index), "
                f"static_cast<{operand.cpp_type}>(values->Get(index)));"
            )
            w("        }")
            w("    }")

    def emit_entry(self, w):
        union = short_name(self.union["name"])
        node = short_name(self.node_name)
        w("/// Visits @p node's operands. Returns false, visiting nothing, when the node has")
        w("/// no attributes or a type this header predates.")
        w("template <typename V>")
        w(f"bool visit(const {node}& node, V&& visitor)")
        w("{")
        w(f"    switch(node.{self.union_field}_type())")
        w("    {")
        for tag, member in self.members:
            w(f"    case {union}::{tag}:")
            w(
                f"        if(const auto* op = node.{self.union_field}_as_{tag}(); "
                "op != nullptr)"
            )
            w("        {")
            w("            visit(*op, visitor);")
            w("            return true;")
            w("        }")
            w("        return false;")
        w("    default:")
        w("        return false;")
        w("    }")
        w("}")
        w("")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flatc",
        default=None,
        help="flatc binary to use; defaults to the one on PATH. The build passes the "
        "binary CMake already resolved, which need not be on PATH.",
    )
    flatc_path = _resolve_flatc(parser.parse_args().flatc)
    schemas_dir = os.path.join(_HIPDNN_DIR, SDK_DIR, "schemas")
    output_dir = os.path.join(
        _HIPDNN_DIR, SDK_DIR, "include", NAMESPACE, "data_objects"
    )
    with tempfile.TemporaryDirectory() as work_dir:
        schema = load_schema(flatc_path, schemas_dir, SCHEMA_FILE, work_dir)
    header = OperandsEmitter(schema).emit()
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, HEADER_NAME), "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write(header)


if __name__ == "__main__":
    main()
