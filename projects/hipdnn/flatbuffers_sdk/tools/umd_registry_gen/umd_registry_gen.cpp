// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// umd_registry_gen -- build-time generator for the RFC 0018 op-schema registry.
//
// Reads the FlatBuffers binary reflection schema (graph.bfbs), enumerates the
// NodeAttributes union members (opcode -> attribute table), classifies each
// field per RFC 0018 Appendix B.3, and emits a header-only, strongly-typed
// registry (op_schema_registry_generated.hpp) whose readers call the generated
// FlatBuffers accessors directly -- no runtime reflection (RFC 0018 B.4).
//
// Field classification (B.3):
//   * umd_operand + umd_name  -> operand edge  (field MUST be a `long` UID)
//   * umd_result  + umd_name  -> result  edge  (field MUST be a `long` UID)
//   * neither flag, scalar    -> scalar attribute, bind-named by field name
//   * neither flag, non-scalar-> skipped (a vector/table/union is not a UMD
//                                scalar; the RFC's attribute tables are UID +
//                                scalar, so this only affects ops outside the
//                                UMD scalar namespace)
// Optionality is derived from the `= null` default (reflection `optional()`).
//
// Build errors (fail closed, non-zero exit): both flags on one field; umd_name
// without a flag; a flag on a non-integer field; a duplicate umd_name within an
// op; a role name colliding with a reserved root (graph/kernel/device).

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "flatbuffers/reflection.h"
#include "flatbuffers/reflection_generated.h"

namespace
{

[[noreturn]] void fail(const std::string& msg)
{
    std::fprintf(stderr, "umd_registry_gen: error: %s\n", msg.c_str());
    std::exit(1);
}

std::string readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
        fail("cannot open input schema: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Short (unqualified) name: text after the last '.'.
std::string shortName(const std::string& qualified)
{
    auto pos = qualified.rfind('.');
    return pos == std::string::npos ? qualified : qualified.substr(pos + 1);
}

const reflection::KeyValue* attr(const reflection::Field* f, const char* key)
{
    return f->attributes() ? f->attributes()->LookupByKey(key) : nullptr;
}

// Table-level attribute value (e.g. `umd_opcode`), or empty when absent.
std::string objAttr(const reflection::Object* o, const char* key)
{
    if(!o->attributes())
        return {};
    const auto* kv = o->attributes()->LookupByKey(key);
    if(!kv)
        return {};
    return kv->value() ? kv->value()->str() : std::string();
}

bool isIntegerBase(reflection::BaseType t)
{
    return t == reflection::Byte || t == reflection::UByte || t == reflection::Short
           || t == reflection::UShort || t == reflection::Int || t == reflection::UInt
           || t == reflection::Long || t == reflection::ULong;
}

bool isScalarBase(reflection::BaseType t)
{
    return t == reflection::Bool || t == reflection::Float || t == reflection::Double
           || isIntegerBase(t);
}

// C++ keywords flatc suffixes with '_' when emitting an accessor name.
const std::set<std::string>& cppKeywords()
{
    static const std::set<std::string> k
        = {"alignas", "alignof",   "and",      "asm",      "auto",      "bool",     "break",
           "case",    "catch",     "char",     "class",    "const",     "continue", "default",
           "delete",  "do",        "double",   "else",     "enum",      "explicit", "export",
           "extern",  "false",     "float",    "for",      "friend",    "goto",     "if",
           "inline",  "int",       "long",     "mutable",  "namespace", "new",      "operator",
           "private", "protected", "public",   "register", "return",    "short",    "signed",
           "sizeof",  "static",    "struct",   "switch",   "template",  "this",     "throw",
           "true",    "try",       "typedef",  "typename", "union",     "unsigned", "using",
           "virtual", "void",      "volatile", "while",    "and_eq",    "bitand",   "bitor",
           "compl",   "not",       "not_eq",   "or",       "or_eq",     "xor",      "xor_eq"};
    return k;
}

std::string accessorName(const std::string& fieldName)
{
    return cppKeywords().count(fieldName) ? fieldName + "_" : fieldName;
}

bool isReservedRoot(const std::string& n)
{
    return n == "graph" || n == "kernel" || n == "device";
}

constexpr const char* kNs = "::hipdnn_flatbuffers_sdk::data_objects";

struct Emitted
{
    std::string operands; // OperandBinding initializers
    std::string results; // ResultBinding initializers
    std::string attributes; // AttrBinding initializers
    std::size_t operandCount = 0;
    std::size_t resultCount = 0;
    std::size_t attributeCount = 0;
};

// Emit the reader lambda + binding line for one operand/result UID field.
std::string emitUidBinding(const std::string& table,
                           const std::string& role,
                           const std::string& acc,
                           bool optional)
{
    std::ostringstream o;
    o << "    {\"" << role << "\", " << (optional ? "true" : "false") << ", "
      << "+[](const void* a, std::int64_t& out) -> bool { ";
    if(optional)
        o << "auto v = static_cast<const " << kNs << "::" << table << "*>(a)->" << acc
          << "(); if(!v) return false; out = *v; return true;";
    else
        o << "out = static_cast<const " << kNs << "::" << table << "*>(a)->" << acc
          << "(); return true;";
    o << " }},\n";
    return o.str();
}

// Emit the reader lambda + binding line for one scalar attribute field.
std::string emitAttrBinding(const reflection::Schema* schema,
                            const std::string& table,
                            const reflection::Field* fld)
{
    const std::string name = fld->name()->str();
    const std::string acc = accessorName(name);
    const auto base = fld->type()->base_type();
    const bool optional = fld->optional();
    const int enumIdx = fld->type()->index();
    const bool isEnum = enumIdx >= 0 && isIntegerBase(base);

    std::string attrType;
    std::string body;
    if(isEnum)
    {
        const std::string enumShort = shortName(schema->enums()->Get(enumIdx)->name()->str());
        attrType = "AttrType::Dtype";
        std::ostringstream b;
        b << "ScalarValue s; s.type = AttrType::Dtype; ";
        if(optional)
            b << "auto v = static_cast<const " << kNs << "::" << table << "*>(a)->" << acc
              << "(); s.present = v.has_value(); if(v) s.dtype = " << kNs << "::EnumName"
              << enumShort << "(*v);";
        else
            b << "s.present = true; s.dtype = " << kNs << "::EnumName" << enumShort
              << "(static_cast<const " << kNs << "::" << table << "*>(a)->" << acc << "());";
        b << " return s;";
        body = b.str();
    }
    else
    {
        std::string member;
        if(base == reflection::Bool)
        {
            attrType = "AttrType::Bool";
            member = "b";
        }
        else if(base == reflection::Float || base == reflection::Double)
        {
            attrType = "AttrType::Float";
            member = "f";
        }
        else
        {
            attrType = "AttrType::Int";
            member = "i";
        }
        std::string cast;
        if(member == "i")
            cast = "std::int64_t";
        else if(member == "f")
            cast = "double";
        const std::string readExpr
            = "static_cast<const " + std::string(kNs) + "::" + table + "*>(a)->" + acc + "()";
        std::ostringstream b;
        b << "ScalarValue s; s.type = " << attrType << "; ";
        if(optional)
            b << "auto v = " << readExpr << "; s.present = v.has_value(); if(v) s." << member
              << " = " << (cast.empty() ? "*v" : "static_cast<" + cast + ">(*v)") << ";";
        else
            b << "s.present = true; s." << member << " = "
              << (cast.empty() ? readExpr : "static_cast<" + cast + ">(" + readExpr + ")") << ";";
        b << " return s;";
        body = b.str();
    }

    std::ostringstream o;
    o << "    {\"" << name << "\", " << (optional ? "true" : "false") << ", " << attrType
      << ", +[](const void* a) -> ScalarValue { " << body << " }},\n";
    return o.str();
}

Emitted classifyTable(const reflection::Schema* schema, const reflection::Object* obj)
{
    const std::string table = shortName(obj->name()->str());
    Emitted out;
    std::set<std::string> seenRoles;

    for(const auto* fld : *obj->fields())
    {
        const std::string name = fld->name()->str();
        const auto* opFlag = attr(fld, "umd_operand");
        const auto* resFlag = attr(fld, "umd_result");
        const auto* nameAttr = attr(fld, "umd_name");
        const auto base = fld->type()->base_type();

        if(opFlag && resFlag)
            fail(table + "." + name + ": has both umd_operand and umd_result");

        const bool isEdge = opFlag || resFlag;
        if(isEdge)
        {
            if(!nameAttr)
                fail(table + "." + name + ": umd_operand/umd_result requires umd_name");
            const std::string role = nameAttr->value() ? nameAttr->value()->str() : "";
            if(role.empty())
                fail(table + "." + name + ": umd_name must be non-empty");
            if(base != reflection::Long)
                fail(table + "." + name + ": umd_operand/umd_result field must be `long` (a UID)");
            if(isReservedRoot(role))
                fail(table + "." + name + ": role name '" + role
                     + "' collides with a reserved root (graph/kernel/device)");
            if(!seenRoles.insert(role).second)
                fail(table + ": duplicate umd_name '" + role + "'");

            const std::string acc = accessorName(name);
            const bool optl = fld->optional();
            if(opFlag)
            {
                out.operands += emitUidBinding(table, role, acc, optl);
                ++out.operandCount;
            }
            else
            {
                out.results += emitUidBinding(table, role, acc, optl);
                ++out.resultCount;
            }
            continue;
        }

        if(nameAttr)
            fail(table + "." + name + ": umd_name without umd_operand/umd_result");

        // Unannotated: a scalar attribute (if a scalar base type). Non-scalar
        // fields (vector/table/union/string) are not UMD scalars -- skip.
        if(isScalarBase(base))
        {
            out.attributes += emitAttrBinding(schema, table, fld);
            ++out.attributeCount;
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc != 3)
        fail("usage: umd_registry_gen <graph.bfbs> <output.hpp>");
    const std::string bfbsPath = argv[1];
    const std::string outPath = argv[2];

    const std::string buf = readFile(bfbsPath);
    const auto* schema = reflection::GetSchema(buf.data());
    if(!schema)
        fail("input is not a valid reflection schema: " + bfbsPath);

    const reflection::Enum* nodeAttrs = nullptr;
    for(const auto* e : *schema->enums())
    {
        if(e->is_union() && shortName(e->name()->str()) == "NodeAttributes")
            nodeAttrs = e;
    }
    if(!nodeAttrs)
        fail("NodeAttributes union not found in schema");

    std::ostringstream body;
    std::ostringstream entries;
    std::size_t opCount = 0;
    std::set<std::string> seenOpcodes;

    for(const auto* ev : *nodeAttrs->values())
    {
        if(!ev->union_type())
            continue;
        const int objIdx = ev->union_type()->index();
        if(objIdx < 0)
            continue; // NONE
        const reflection::Object* obj = schema->objects()->Get(objIdx);
        const std::string table = shortName(obj->name()->str());
        const std::string member = ev->name()->str(); // union value name == table type name

        // Opcode key: the table's `umd_opcode` shorthand (RFC 0018), else the
        // table type name. Must be unique across ops.
        std::string opcode = objAttr(obj, "umd_opcode");
        if(opcode.empty())
            opcode = member;
        if(!seenOpcodes.insert(opcode).second)
            fail("duplicate umd_opcode '" + opcode + "'");

        const Emitted em = classifyTable(schema, obj);

        const std::string tag = "op" + std::to_string(opCount);
        std::string operandsRef = "nullptr";
        std::string resultsRef = "nullptr";
        std::string attrsRef = "nullptr";
        const bool anyTable = em.operandCount || em.resultCount || em.attributeCount;
        if(anyTable)
            body << "namespace " << tag << " {\n";
        if(em.operandCount)
        {
            body << "inline const OperandBinding operands[] = {\n" << em.operands << "};\n";
            operandsRef = tag + "::operands";
        }
        if(em.resultCount)
        {
            body << "inline const ResultBinding results[] = {\n" << em.results << "};\n";
            resultsRef = tag + "::results";
        }
        if(em.attributeCount)
        {
            body << "inline const AttrBinding attributes[] = {\n" << em.attributes << "};\n";
            attrsRef = tag + "::attributes";
        }
        if(anyTable)
            body << "} // namespace " << tag << "\n\n";

        entries << "    {\"" << opcode << "\", \"" << member << "\", static_cast<int>(" << kNs
                << "::NodeAttributes::" << member << "), " << operandsRef << ", " << em.operandCount
                << "u, " << resultsRef << ", " << em.resultCount << "u, " << attrsRef << ", "
                << em.attributeCount << "u},\n";
        ++opCount;
    }

    std::ostringstream hdr;
    hdr << "// AUTOMATICALLY GENERATED by umd_registry_gen -- do not modify.\n"
        << "// Source: FlatBuffers reflection schema (graph.bfbs). See RFC 0018 Appendix B.\n"
        << "#pragma once\n\n"
        << "#include <cstddef>\n#include <cstdint>\n#include <string_view>\n\n"
        << "#include <hipdnn_flatbuffers_sdk/umd/OpSchemaRegistry.hpp>\n"
        << "#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>\n\n"
        << "namespace hipdnn_flatbuffers_sdk::umd\n{\nnamespace generated\n{\n\n"
        << body.str() << "inline const OpSchemaEntry entries[] = {\n"
        << entries.str() << "};\n\n} // namespace generated\n\n"
        << "inline const OpSchemaEntry* opSchemaEntries(std::size_t& count)\n{\n"
        << "    count = sizeof(generated::entries) / sizeof(generated::entries[0]);\n"
        << "    return generated::entries;\n}\n\n"
        << "inline const OpSchemaEntry* lookupOpByName(std::string_view opcode)\n{\n"
        << "    for(const auto& e : generated::entries)\n        if(e.opcode == opcode)\n"
        << "            return &e;\n    return nullptr;\n}\n\n"
        << "inline const OpSchemaEntry* lookupOpByType(int attributesType)\n{\n"
        << "    for(const auto& e : generated::entries)\n        if(e.attributesType == "
           "attributesType)\n"
        << "            return &e;\n    return nullptr;\n}\n\n"
        << "} // namespace hipdnn_flatbuffers_sdk::umd\n";

    std::ofstream ofs(outPath, std::ios::binary);
    if(!ofs)
        fail("cannot open output for writing: " + outPath);
    ofs << hdr.str();
    if(!ofs)
        fail("failed writing output: " + outPath);

    std::fprintf(
        stderr, "umd_registry_gen: emitted %zu op entries to %s\n", opCount, outPath.c_str());
    return 0;
}
