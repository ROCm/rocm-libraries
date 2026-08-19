# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generate the graph cache-key header from the FlatBuffers schemas.

Emits ``hashAppend()`` and ``logicallyEqual()`` over the zero-copy FlatBuffers
accessors for every type reachable from a root table, honouring the
``cache_ignore`` schema annotation. The generated code has no runtime
dependency on FlatBuffers reflection and performs no allocation.

The annotation is invisible to flatc's C++ backend -- it survives only into the
binary schema -- so the field policy is read back from a ``.bfbs`` produced by
flatc itself rather than by parsing ``.fbs`` text. Parsing, resolving and
validating the schema stays flatc's job; this script only walks the result.

Run manually, from the build via cmake/FlatBuffersGenerate.cmake, or through the
``flatc-hipdnn`` pre-commit hook. Mirrors run_flatc.py's conventions: same
flatc-version gate, same shared flag file, same failure reporting.
"""

import os
import shutil
import subprocess
import sys
import tempfile

_HIPDNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fields carrying this annotation are excluded from the hash and the comparison.
# The policy is opt-out: an unannotated field participates, so adding a field to
# a schema cannot silently drop it from the key.
IGNORE_ATTRIBUTE = "cache_ignore"

# Roots to generate for, as (schema file, fully-qualified root table, output header).
TARGETS = [
    (
        "graph.fbs",
        "hipdnn_flatbuffers_sdk.data_objects.Graph",
        "cachekey_generated.h",
    ),
]

SDK_DIR = "flatbuffers_sdk"
NAMESPACE = "hipdnn_flatbuffers_sdk"

# flatc escapes C++ keywords in accessor names by appending an underscore
# (idl_gen_cpp.cpp). Mirror that or the emitted code will not compile: the
# `virtual` field on TensorAttributes is a live example.
CPP_KEYWORDS = frozenset(
    """alignas alignof and and_eq asm auto bitand bitor bool break case catch char
    char16_t char32_t class compl concept const constexpr const_cast continue decltype
    default delete do double dynamic_cast else enum explicit export extern false float
    for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr
    operator or or_eq private protected public register reinterpret_cast requires return
    short signed sizeof static static_assert static_cast struct switch template this
    thread_local throw true try typedef typeid typename union unsigned using virtual void
    volatile wchar_t while xor xor_eq""".split()
)


def _read_required_version():
    with open(
        os.path.join(_HIPDNN_DIR, "cmake", "default_flatc_version.txt"),
        encoding="utf-8",
    ) as f:
        return f.read().strip()


def _resolve_flatc():
    """Locate flatc and gate on the pinned version, as run_flatc.py does."""
    required = _read_required_version()
    flatc_path = shutil.which("flatc")
    current = ""
    if flatc_path:
        try:
            current = subprocess.check_output(
                [flatc_path, "--version"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            pass
    if required not in current:
        print(
            f'ERROR: flatc version {required} required. Found: {current or "None"}',
            file=sys.stderr,
        )
        print(
            "Download the following and include the executable in PATH:",
            file=sys.stderr,
        )
        print(
            f"  Windows: Download https://github.com/google/flatbuffers/releases/download/v{required}/Windows.flatc.binary.zip",
            file=sys.stderr,
        )
        print(
            f"  Linux:   wget https://github.com/google/flatbuffers/releases/download/v{required}/Linux.flatc.binary.g++-13.zip",
            file=sys.stderr,
        )
        sys.exit(1)
    return flatc_path


def _run(argv):
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: command failed: {' '.join(argv)}", file=sys.stderr)
        print("STDOUT:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("STDERR:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def _resolve_reflection_schema(flatc_path, work_dir):
    """Path to the reflection schema flatc needs to decode a .bfbs.

    Prefers a copy installed next to flatc -- TheRock ships one at
    share/flatbuffers/reflection/reflection.fbs -- because that copy is upstream's
    own, tracked at the same version as the flatc producing the .bfbs. Falls back to
    the embedded copy below when building against a stock FlatBuffers install, which
    does not ship the schema at all.
    """
    if flatc_path:
        prefix = os.path.dirname(os.path.dirname(os.path.abspath(flatc_path)))
        installed = os.path.join(
            prefix, "share", "flatbuffers", "reflection", "reflection.fbs"
        )
        if os.path.isfile(installed):
            return installed

    fallback = os.path.join(work_dir, "reflection.fbs")
    with open(fallback, "w", encoding="utf-8") as f:
        f.write(_REFLECTION_SCHEMA)
    return fallback


def load_schema(flatc_path, schemas_dir, schema_file, work_dir):
    """Return the parsed reflection.Schema for `schema_file` as plain Python.

    flatc emits the binary schema, then decodes it back to JSON. Using flatc for
    both steps keeps hipDNN free of any .fbs parser: parsing, resolving and
    validating the schema stays flatc's job.
    """
    import json

    stem = os.path.splitext(schema_file)[0]
    _run(
        [
            flatc_path,
            "-b",
            "--schema",
            "--bfbs-builtins",
            "-o",
            work_dir,
            "-I",
            schemas_dir,
            os.path.join(schemas_dir, schema_file),
        ]
    )
    bfbs = os.path.join(work_dir, f"{stem}.bfbs")
    reflection_fbs = _resolve_reflection_schema(flatc_path, work_dir)
    _run(
        [
            flatc_path,
            "-t",
            "--strict-json",
            "--raw-binary",
            "-o",
            work_dir,
            reflection_fbs,
            "--",
            bfbs,
        ]
    )
    with open(os.path.join(work_dir, f"{stem}.json"), encoding="utf-8") as f:
        return json.load(f)


def accessor(name):
    return name + "_" if name in CPP_KEYWORDS else name


def short_name(fully_qualified):
    return fully_qualified.rsplit(".", 1)[-1]


class Emitter:
    """Walks the reflection schema and emits the header."""

    def __init__(self, schema, root_name):
        self.schema = schema
        self.root_name = root_name
        self.objects = {o["name"]: o for o in schema["objects"]}
        self.enums = {e["name"]: e for e in schema["enums"]}
        self.lines = []
        self.unions = set()

    def keep(self, field):
        if field.get("deprecated"):
            return False
        return not any(
            a["key"] == IGNORE_ATTRIBUTE for a in field.get("attributes", [])
        )

    def fields_of(self, obj):
        # The binary schema sorts fields alphabetically; restore declaration
        # order so the emitted stream is stable and readable against the .fbs.
        return sorted(
            (f for f in obj["fields"] if self.keep(f)), key=lambda f: f.get("id", 0)
        )

    def reachable(self):
        """Types reachable from the root through non-ignored fields only."""
        seen, stack = set(), [self.root_name]
        while stack:
            name = stack.pop()
            if name in seen or name not in self.objects:
                continue
            seen.add(name)
            for field in self.fields_of(self.objects[name]):
                ftype = field["type"]
                base, index = ftype["base_type"], ftype.get("index", -1)
                if base == "Obj" and index >= 0:
                    stack.append(self.schema["objects"][index]["name"])
                elif base == "Vector" and ftype.get("element") == "Obj" and index >= 0:
                    stack.append(self.schema["objects"][index]["name"])
                elif base == "Union" and index >= 0:
                    union = self.schema["enums"][index]
                    self.unions.add(union["name"])
                    for value in union["values"]:
                        member = value.get("union_type")
                        if member and member.get("index", -1) >= 0:
                            stack.append(
                                self.schema["objects"][member["index"]]["name"]
                            )
        return [o["name"] for o in self.schema["objects"] if o["name"] in seen]

    def w(self, line=""):
        self.lines.append(line)

    def emit(self):
        order = self.reachable()
        namespace = self.root_name.rsplit(".", 1)[0].replace(".", "::")
        self.emit_prologue(namespace)
        for name in order:
            short = short_name(name)
            self.w(f"inline void hashAppend(Hasher& hasher, const {short}* value);")
            self.w(f"inline bool logicallyEqual(const {short}* a, const {short}* b);")
        for union in sorted(self.unions):
            short = short_name(union)
            self.w(
                f"inline void hashAppend(Hasher& hasher, {short} type, const void* value);"
            )
            self.w(
                f"inline bool logicallyEqual({short} aType, const void* a, {short} bType, const void* b);"
            )
        self.w()
        for union in sorted(self.unions):
            self.emit_union(union)
        for name in order:
            self.emit_table(name)
        self.w(f"}} // namespace {namespace}::cachekey")
        return "\n".join(self.lines) + "\n"

    def emit_prologue(self, namespace):
        self.w("// Automatically generated by scripts/gen_cache_key.py, do not modify.")
        self.w("//")
        self.w(
            "// Hash and logical-equality over the FlatBuffers accessors, with the field"
        )
        self.w(
            f"// policy taken from the '{IGNORE_ATTRIBUTE}' schema annotation. Reads the"
        )
        self.w("// caller's buffer in place: no UnPack, no allocation, no reflection.")
        self.w("")
        self.w("#pragma once")
        self.w("")
        self.w("#include <cstdint>")
        self.w("#include <string_view>")
        self.w("#include <type_traits>")
        self.w("")
        self.w("#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>")
        self.w("")
        self.w(f"namespace {namespace}::cachekey")
        self.w("{")
        self.w("")
        self.w("/// Streaming FNV-1a. Folds a value at a time so nothing is buffered.")
        self.w("class Hasher")
        self.w("{")
        self.w("public:")
        self.w(
            "    /// Presence and union discriminators are folded through this, so an absent"
        )
        self.w("    /// field and a field holding the tag's value cannot collide.")
        self.w("    void tag(uint8_t value)")
        self.w("    {")
        self.w("        _state ^= value;")
        self.w("        _state *= PRIME;")
        self.w("    }")
        self.w("")
        self.w("    template <typename TValue>")
        self.w("    void raw(TValue value)")
        self.w("    {")
        self.w("        static_assert(std::is_trivially_copyable_v<TValue>);")
        self.w("        uint8_t bytes[sizeof(TValue)];")
        self.w("        __builtin_memcpy(bytes, &value, sizeof(TValue));")
        self.w("        for(uint8_t byte : bytes)")
        self.w("        {")
        self.w("            tag(byte);")
        self.w("        }")
        self.w("    }")
        self.w("")
        self.w("    uint64_t value() const")
        self.w("    {")
        self.w("        return _state;")
        self.w("    }")
        self.w("")
        self.w("private:")
        self.w("    static constexpr uint64_t OFFSET_BASIS = 1469598103934665603ULL;")
        self.w("    static constexpr uint64_t PRIME        = 1099511628211ULL;")
        self.w("")
        self.w("    uint64_t _state = OFFSET_BASIS;")
        self.w("};")
        self.w("")

    def emit_union(self, union_name):
        enum = self.enums[union_name]
        short = short_name(union_name)
        members = [
            (
                v["name"],
                short_name(self.schema["objects"][v["union_type"]["index"]]["name"]),
            )
            for v in enum["values"]
            if v.get("union_type") and v["union_type"].get("index", -1) >= 0
        ]
        self.w(
            f"inline void hashAppend(Hasher& hasher, {short} type, const void* value)"
        )
        self.w("{")
        self.w("    hasher.raw(static_cast<uint8_t>(type));")
        self.w("    switch(type)")
        self.w("    {")
        for tag, member in members:
            self.w(f"    case {short}::{tag}:")
            self.w(f"        hashAppend(hasher, static_cast<const {member}*>(value));")
            self.w("        break;")
        self.w("    default:")
        self.w("        break;")
        self.w("    }")
        self.w("}")
        self.w("")
        self.w(
            f"inline bool logicallyEqual({short} aType, const void* a, {short} bType, const void* b)"
        )
        self.w("{")
        self.w("    if(aType != bType)")
        self.w("    {")
        self.w("        return false;")
        self.w("    }")
        self.w("    switch(aType)")
        self.w("    {")
        for tag, member in members:
            self.w(f"    case {short}::{tag}:")
            self.w(
                f"        return logicallyEqual(static_cast<const {member}*>(a), static_cast<const {member}*>(b));"
            )
        self.w("    default:")
        self.w("        return true;")
        self.w("    }")
        self.w("}")
        self.w("")

    def emit_table(self, name):
        obj = self.objects[name]
        short = short_name(name)
        fields = self.fields_of(obj)

        self.w(f"inline void hashAppend(Hasher& hasher, const {short}* value)")
        self.w("{")
        self.w("    if(value == nullptr)")
        self.w("    {")
        self.w("        hasher.tag(0);")
        self.w("        return;")
        self.w("    }")
        self.w("    hasher.tag(1);")
        for field in fields:
            self.emit_hash_field(field)
        self.w("}")
        self.w("")

        self.w(f"inline bool logicallyEqual(const {short}* a, const {short}* b)")
        self.w("{")
        self.w("    if(a == b)")
        self.w("    {")
        self.w("        return true;")
        self.w("    }")
        self.w("    if(a == nullptr || b == nullptr)")
        self.w("    {")
        self.w("        return false;")
        self.w("    }")
        for field in fields:
            self.emit_equal_field(field)
        self.w("    return true;")
        self.w("}")
        self.w("")

    def emit_hash_field(self, field):
        ftype = field["type"]
        base = ftype["base_type"]
        name = accessor(field["name"])
        get = f"value->{name}()"

        if base == "UType":
            # Emitted alongside its union payload.
            return
        if base == "Vector":
            element = ftype.get("element")
            self.w(f"    {{")
            self.w(f"        const auto* items = {get};")
            self.w(
                "        const uint32_t count = items == nullptr ? 0u : items->size();"
            )
            self.w("        // Length first: {1, 2} and {1, 2, 0} must not fold alike.")
            self.w("        hasher.raw(count);")
            self.w("        for(uint32_t index = 0; index < count; ++index)")
            self.w("        {")
            if element == "Obj":
                self.w("            hashAppend(hasher, items->Get(index));")
            elif element == "String":
                self.w("            const auto* item = items->Get(index);")
                self.w(
                    "            hasher.raw(static_cast<uint32_t>(item == nullptr ? 0 : item->size()));"
                )
                self.w("            if(item != nullptr)")
                self.w("            {")
                self.w("                for(char character : *item)")
                self.w("                {")
                self.w(
                    "                    hasher.tag(static_cast<uint8_t>(character));"
                )
                self.w("                }")
                self.w("            }")
            else:
                self.w("            hasher.raw(items->Get(index));")
            self.w("        }")
            self.w("    }")
        elif base == "String":
            self.w(f"    {{")
            self.w(f"        const auto* text = {get};")
            self.w(
                "        hasher.raw(static_cast<uint32_t>(text == nullptr ? 0 : text->size()));"
            )
            self.w("        if(text != nullptr)")
            self.w("        {")
            self.w("            for(char character : *text)")
            self.w("            {")
            self.w("                hasher.tag(static_cast<uint8_t>(character));")
            self.w("            }")
            self.w("        }")
            self.w("    }")
        elif base == "Union":
            self.w(f"    hashAppend(hasher, value->{name}_type(), {get});")
        elif base == "Obj":
            self.w(f"    hashAppend(hasher, {get});")
        elif field.get("optional"):
            # An absent optional and a present one holding the same value are
            # different content, so the presence tag is folded either way.
            self.w(f"    {{")
            self.w(f"        const auto optional = {get};")
            self.w("        hasher.tag(optional ? 1 : 0);")
            self.w("        if(optional)")
            self.w("        {")
            self.w("            hasher.raw(*optional);")
            self.w("        }")
            self.w("    }")
        elif ftype.get("index", -1) >= 0:
            # Enum-typed scalar: widen so the underlying type's width cannot
            # change the fold when a schema switches byte -> short.
            self.w(f"    hasher.raw(static_cast<int64_t>({get}));")
        else:
            self.w(f"    hasher.raw({get});")

    def emit_equal_field(self, field):
        ftype = field["type"]
        base = ftype["base_type"]
        name = accessor(field["name"])
        left, right = f"a->{name}()", f"b->{name}()"

        if base == "UType":
            return
        if base == "Vector":
            element = ftype.get("element")
            self.w("    {")
            self.w(f"        const auto* aItems = {left};")
            self.w(f"        const auto* bItems = {right};")
            self.w(
                "        const uint32_t aCount = aItems == nullptr ? 0u : aItems->size();"
            )
            self.w(
                "        const uint32_t bCount = bItems == nullptr ? 0u : bItems->size();"
            )
            self.w("        if(aCount != bCount)")
            self.w("        {")
            self.w("            return false;")
            self.w("        }")
            self.w("        for(uint32_t index = 0; index < aCount; ++index)")
            self.w("        {")
            if element == "Obj":
                self.w(
                    "            if(!logicallyEqual(aItems->Get(index), bItems->Get(index)))"
                )
                self.w("            {")
                self.w("                return false;")
                self.w("            }")
            elif element == "String":
                self.w("            const auto* aItem = aItems->Get(index);")
                self.w("            const auto* bItem = bItems->Get(index);")
                self.w("            if((aItem == nullptr) != (bItem == nullptr))")
                self.w("            {")
                self.w("                return false;")
                self.w("            }")
                self.w(
                    "            if(aItem != nullptr && aItem->string_view() != bItem->string_view())"
                )
                self.w("            {")
                self.w("                return false;")
                self.w("            }")
            else:
                self.w("            if(aItems->Get(index) != bItems->Get(index))")
                self.w("            {")
                self.w("                return false;")
                self.w("            }")
            self.w("        }")
            self.w("    }")
        elif base == "String":
            self.w("    {")
            self.w(f"        const auto* aText = {left};")
            self.w(f"        const auto* bText = {right};")
            self.w("        if((aText == nullptr) != (bText == nullptr))")
            self.w("        {")
            self.w("            return false;")
            self.w("        }")
            self.w(
                "        if(aText != nullptr && aText->string_view() != bText->string_view())"
            )
            self.w("        {")
            self.w("            return false;")
            self.w("        }")
            self.w("    }")
        elif base == "Union":
            self.w(
                f"    if(!logicallyEqual(a->{name}_type(), {left}, b->{name}_type(), {right}))"
            )
            self.w("    {")
            self.w("        return false;")
            self.w("    }")
        elif base == "Obj":
            self.w(f"    if(!logicallyEqual({left}, {right}))")
            self.w("    {")
            self.w("        return false;")
            self.w("    }")
        else:
            self.w(f"    if({left} != {right})")
            self.w("    {")
            self.w("        return false;")
            self.w("    }")


# Fallback only, used when flatc's install tree has no reflection.fbs -- upstream
# FlatBuffers installs the C++ reflection headers but not the schema itself.
# TheRock ships it (third-party/flatbuffers/post_hook_therock-flatbuffers.cmake),
# and _resolve_reflection_schema() prefers that copy.
#
# This is the upstream reflection.fbs trimmed to the fields this generator reads.
# FlatBuffers field additions are backward compatible, so a newer flatc's .bfbs
# still decodes against it and simply carries fields we ignore.
_REFLECTION_SCHEMA = """
namespace reflection;

enum BaseType : byte {
    None, UType, Bool,
    Byte, UByte, Short, UShort, Int, UInt, Long, ULong,
    Float, Double,
    String, Vector, Obj, Union, Array, Vector64
}

table Type {
    base_type: BaseType;
    element: BaseType = None;
    index: int = -1;
    fixed_length: uint16 = 0;
    base_size: uint = 4;
    element_size: uint = 0;
}

table KeyValue {
    key: string (required, key);
    value: string;
}

table EnumVal {
    name: string (required);
    value: long (key);
    unused_documentation: [string];
    union_type: Type;
    documentation: [string];
    attributes: [KeyValue];
}

table Enum {
    name: string (required, key);
    values: [EnumVal] (required);
    is_union: bool = false;
    underlying_type: Type (required);
    attributes: [KeyValue];
    documentation: [string];
    declaration_file: string;
}

table Field {
    name: string (required, key);
    type: Type (required);
    id: ushort;
    offset: ushort;
    default_integer: long = 0;
    default_real: double = 0.0;
    deprecated: bool = false;
    required: bool = false;
    key: bool = false;
    attributes: [KeyValue];
    documentation: [string];
    optional: bool = false;
    padding: uint16 = 0;
    offset64: bool = false;
}

table Object {
    name: string (required, key);
    fields: [Field] (required);
    is_struct: bool = false;
    minalign: int;
    bytesize: int;
    attributes: [KeyValue];
    documentation: [string];
    declaration_file: string;
}

table RPCCall {
    name: string (required, key);
    request: Object (required);
    response: Object (required);
    attributes: [KeyValue];
    documentation: [string];
}

table Service {
    name: string (required, key);
    calls: [RPCCall];
    attributes: [KeyValue];
    documentation: [string];
    declaration_file: string;
}

table SchemaFile {
    filename: string (required, key);
    included_filenames: [string];
}

table Schema {
    objects: [Object] (required);
    enums: [Enum] (required);
    file_ident: string;
    file_ext: string;
    root_table: Object;
    services: [Service];
    advanced_features: ulong;
    fbs_files: [SchemaFile];
}

root_type Schema;
file_identifier "BFBS";
file_extension "bfbs";
"""


def main():
    flatc_path = _resolve_flatc()
    schemas_dir = os.path.join(_HIPDNN_DIR, SDK_DIR, "schemas")
    output_dir = os.path.join(
        _HIPDNN_DIR, SDK_DIR, "include", NAMESPACE, "data_objects"
    )

    for schema_file, root_name, header_name in TARGETS:
        with tempfile.TemporaryDirectory() as work_dir:
            schema = load_schema(flatc_path, schemas_dir, schema_file, work_dir)
        # TemporaryDirectory is released before emitting: the schema is already
        # in memory and the generator touches no file until it writes the header.
        header = Emitter(schema, root_name).emit()
        destination = os.path.join(output_dir, header_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(destination, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)


if __name__ == "__main__":
    main()
