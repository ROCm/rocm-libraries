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
// each op entry lists its input-tensor and output-tensor names (each with a typed
// reader) and its scalar attributes (each with a typed value reader). Readers
// use the generated FlatBuffers accessors directly -- no runtime reflection.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// Opaque forward declaration of the FlatBuffers NodeAttributes union discriminant
// (fully defined in the generated graph_generated.h). Declaring it here with its
// fixed underlying type makes it a complete type, so OpSchemaEntry can carry the
// real enum value -- no int laundering -- while this neutral header stays free of
// the generated schema include. The generated registry TU sees both this
// declaration and the real definition, so any underlying-type mismatch is a
// compile error there.
namespace hipdnn_flatbuffers_sdk::data_objects
{
enum class NodeAttributes : std::uint8_t;
} // namespace hipdnn_flatbuffers_sdk::data_objects

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

// Reads a bound tensor's UID out of a concrete attribute table (passed as a
// `const void*` to keep this header type-erased). Returns false when the tensor
// is optional and absent from the graph; returns true and writes `out`
// otherwise. A required tensor always returns true.
using UidReader = bool (*)(const void* attributes, std::int64_t& out);

// Reads a scalar attribute value out of a concrete attribute table.
using ScalarReader = ScalarValue (*)(const void* attributes);

struct InputTensorBinding
{
    std::string_view name;
    bool optional = false;
    UidReader read = nullptr;
};

struct OutputTensorBinding
{
    std::string_view name;
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
// e.g. "SdpaAttributes", for diagnostics), and the `attributesType` (the
// NodeAttributes enum value, for O(1) lookup against Node::attributes_type()),
// plus its input-tensor, output-tensor, and scalar bindings.
struct OpSchemaEntry
{
    std::string_view opcode;
    std::string_view tableName;
    data_objects::NodeAttributes attributesType{};
    const InputTensorBinding* inputTensors = nullptr;
    std::size_t inputTensorCount = 0;
    const OutputTensorBinding* outputTensors = nullptr;
    std::size_t outputTensorCount = 0;
    const AttrBinding* attributes = nullptr;
    std::size_t attributeCount = 0;
};

// Defined (inline) in the generated header.
//   const OpSchemaEntry* opSchemaEntries(std::size_t& count);
//   const OpSchemaEntry* lookupOpByName(std::string_view opcode);
//   const OpSchemaEntry* lookupOpByType(data_objects::NodeAttributes attributesType);

} // namespace hipdnn_flatbuffers_sdk::umd
