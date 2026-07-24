// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Neutral (JsonLogic-agnostic) op-schema registry types for RFC 0018 (UMD).
//
// This header defines the data shapes the generated registry populates. It is
// hand-written and stable; the per-op tables and lookup functions live in the
// build-time-generated op_schema_registry_generated.hpp, emitted by the
// umd_registry_gen tool from the FlatBuffers `.bfbs` reflection schema
// (RFC 0018 Appendix B). Keeping these types free of any provider / JsonLogic
// dependency lets flatbuffers_sdk own the registry without depending on a
// consumer.
//
// The registry reconstructs a UID-centric graph's edges and auto-binds symbols:
// each op entry lists its operand and result roles (each with a typed UID
// reader) and its scalar attributes (each with a typed value reader). Readers
// use the generated FlatBuffers accessors directly -- no runtime reflection.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hipdnn_flatbuffers_sdk::umd
{

// The value kind of a scalar attribute. Mirrors the JsonLogic value domain a
// criteria expression type-checks against (RFC 0018 Appendix A.7): integer,
// float, boolean, or an enum name (`Dtype`). Enum-typed attributes surface as
// `Dtype`, carrying the enum-value name string.
enum class AttrType : std::uint8_t
{
    Int,
    Float,
    Bool,
    Dtype,
};

// A resolved scalar attribute value. `present` is false only for an optional
// attribute the graph omitted; a present value populates exactly the member
// selected by `type`. `dtype` points at a static enum-name string (no
// ownership) when `type == Dtype`.
struct ScalarValue
{
    AttrType type = AttrType::Int;
    bool present = false;
    std::int64_t i = 0;
    double f = 0.0;
    bool b = false;
    const char* dtype = nullptr;
};

// Reads a role's tensor UID out of a concrete attribute table (passed as a
// `const void*` to keep this header type-erased). Returns false when the role
// is optional and absent from the graph; returns true and writes `out`
// otherwise. A required role always returns true.
using UidReader = bool (*)(const void* attributes, std::int64_t& out);

// Reads a scalar attribute value out of a concrete attribute table.
using ScalarReader = ScalarValue (*)(const void* attributes);

struct OperandBinding
{
    std::string_view role;
    bool optional = false;
    UidReader read = nullptr;
};

struct ResultBinding
{
    std::string_view role;
    bool optional = false;
    UidReader read = nullptr;
};

struct AttrBinding
{
    std::string_view name;
    bool optional = false;
    AttrType type = AttrType::Int;
    ScalarReader read = nullptr;
};

// One op's schema: its `opcode` (the UMD-facing shorthand from the table's
// `umd_opcode` attribute, e.g. "sdpa_fwd", falling back to the table type name
// when the attribute is absent), the `tableName` (the NodeAttributes union member,
// e.g. "SdpaAttributes", for diagnostics), and the integer `attributesType` (the
// value of the NodeAttributes enum, for O(1) lookup against
// Node::attributes_type()), plus its operand, result, and scalar-attribute
// bindings.
struct OpSchemaEntry
{
    std::string_view opcode;
    std::string_view tableName;
    int attributesType = 0;
    const OperandBinding* operands = nullptr;
    std::size_t operandCount = 0;
    const ResultBinding* results = nullptr;
    std::size_t resultCount = 0;
    const AttrBinding* attributes = nullptr;
    std::size_t attributeCount = 0;
};

// Defined (inline) in the generated header.
//   const OpSchemaEntry* opSchemaEntries(std::size_t& count);
//   const OpSchemaEntry* lookupOpByName(std::string_view opcode);
//   const OpSchemaEntry* lookupOpByType(int attributesType);

} // namespace hipdnn_flatbuffers_sdk::umd
