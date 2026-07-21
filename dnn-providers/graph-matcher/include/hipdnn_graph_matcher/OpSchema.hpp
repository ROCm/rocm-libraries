// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// OpSchema: the generic operand/result edge model over hipDNN's FlatBuffer op
// graph. hipDNN stores a node's edges as named `*_tensor_uid` fields *inside*
#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <array>
#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace hipdnn::graph_matcher {

using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

// How many tensor UIDs a role contributes.
//   Required  - exactly one, always present.
//   Optional  - zero or one (a FlatBuffers Optional<int64> field).
//   Variadic  - zero or more (a FlatBuffers [long] vector field).
enum class Arity : uint8_t { Required, Optional, Variadic };

// Reads the UID(s) a role contributes from a node's attribute table, appending
// them in slot order. `attrs` is the raw NodeAttributes union pointer
// (INodeWrapper::attributes()); the reader casts it to the concrete type.
using EdgeReader = void (*)(const void* attrs, std::vector<int64_t>& out);

struct EdgeRole {
    std::string_view name;
    Arity arity;
    EdgeReader read;
};

// Reads a scalar op attribute (enum/bool/int) as int64, or nullopt if absent.
// Attributes live inside the op's attribute table, not as tensor edges.
using AttrReader = std::optional<int64_t> (*)(const void* attrs);

struct AttrAccessor {
    std::string_view name;
    AttrReader read;
};

// One op's edge schema. `operands` are consumed tensors (inputs), `results` are
// produced tensors (outputs). `opcode` is the canonical name patterns match on.
struct OpSchema {
    NodeAttributes type;
    std::string_view opcode;
    std::vector<EdgeRole> operands;
    std::vector<EdgeRole> results;
    std::vector<AttrAccessor> attributes{};  // scalar attrs constraints can read

    // Attribute accessor by name, or nullptr if this op has none so named.
    const AttrAccessor* findAttr(std::string_view attrName) const noexcept {
        for (const auto& attr : attributes) {
            if (attr.name == attrName) {
                return &attr;
            }
        }
        return nullptr;
    }
};

// --- Reader templates: bind a concrete FlatBuffers getter to an EdgeReader. ---

template <class T, int64_t (T::*Getter)() const>
void readRequired(const void* attrs, std::vector<int64_t>& out) {
    out.push_back((static_cast<const T*>(attrs)->*Getter)());
}

template <class T, ::flatbuffers::Optional<int64_t> (T::*Getter)() const>
void readOptional(const void* attrs, std::vector<int64_t>& out) {
    const auto value = (static_cast<const T*>(attrs)->*Getter)();
    if (value) {
        out.push_back(*value);
    }
}

template <class T, const ::flatbuffers::Vector<int64_t>* (T::*Getter)() const>
void readVariadic(const void* attrs, std::vector<int64_t>& out) {
    const auto* vec = (static_cast<const T*>(attrs)->*Getter)();
    if (vec != nullptr) {
        for (const int64_t uid : *vec) {
            out.push_back(uid);
        }
    }
}

// Reads a scalar member (enum/bool/int) coerced to int64. Non-optional fields.
template <class T, class R, R (T::*Getter)() const>
std::optional<int64_t> readAttrScalar(const void* attrs) {
    return static_cast<int64_t>((static_cast<const T*>(attrs)->*Getter)());
}

// Reads an Optional<int64> member: nullopt when the field is absent.
template <class T, ::flatbuffers::Optional<int64_t> (T::*Getter)() const>
std::optional<int64_t> readAttrOptInt(const void* attrs) {
    const auto value = (static_cast<const T*>(attrs)->*Getter)();
    return value ? std::optional<int64_t>{static_cast<int64_t>(*value)} : std::nullopt;
}

// Registry of every built-in op schema, indexed by NodeAttributes enum value.
class OpSchemaRegistry {
   public:
    // The process-wide registry covering all ops in graph.fbs today.
    static const OpSchemaRegistry& builtin();

    // Schema for a union member, or nullptr for NONE / an unregistered op.
    const OpSchema* find(NodeAttributes type) const noexcept;

    // Schema for a live node, resolved from its attributes_type().
    const OpSchema* forNode(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node) const noexcept;

    // Schema whose canonical opcode string equals `opcode`, or nullptr if none.
    const OpSchema* findByOpcode(std::string_view opcode) const noexcept;

    // Number of registered schemas (excludes NONE).
    size_t size() const noexcept {
        return _schemas.size();
    }

   private:
    OpSchemaRegistry();

    static constexpr size_t kTypeSlots =
        static_cast<size_t>(NodeAttributes::MAX) + 1;  // enum values 0..MAX

    std::vector<OpSchema> _schemas;
    std::array<const OpSchema*, kTypeSlots> _byType{};
};

}  // namespace hipdnn::graph_matcher
