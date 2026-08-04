// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// BindingContext.hpp - the queryable "bindings object" for RFC 0018 (UMD).
//
// All names below live in namespace hip_kernel_provider_common::umd; these
// examples assume `namespace umd = hip_kernel_provider_common::umd;`.
//
// After a UniversalGraphMatcher structurally matches a graph, it builds one
// BindingContext per match. The context is both:
//
//   1. the JsonLogic data source the criteria expression evaluates against
//      (it satisfies the `Value getData(const std::string&) const` contract), and
//   2. the queryable bindings object the caller inspects post-match
//      (`ctx.get("$q.head_size")`).
//
// It resolves the RFC's five `$`-namespaces over a live flatbuffer graph
// (RFC 0018 §4):
//   - Tensor    `$q`, `$q.uid`, `$q.rank`, `$q.dtype`, `$q.dims[i]`,
//               `$q.strides[i]`, `$q.<named-dim>`, `$q.stride_order`,
//               `$q.packed`, `$q.virtual`, `$q.present`,
//               `$q.is_runtime_pass_by_value`, `$q.value_f32`
//   - Graph     `$graph.node_count`, `$graph.is_override_shape_enabled`
//   - Attributes `$<node_id>.<attr>`, `$<node_id>.<attr>.present`
//   - Kernel    `$kernel.<field>` (from caller-supplied, fully-resolved KMD
//               metadata; an unbound document is a defensive fallback that
//               reads every field as null)
//   - Device    `$device.<field>`
//
// The two resolvers (RFC 0018 §4, "binding architecture"): edge (name->UID->
// tensor) and scalar-attribute reads use the Phase 0 *generated* typed
// accessors carried on the op-schema registry entry; the Tensor-namespace path
// resolver (dims/strides/rank/stride_order/packed/virtual/uid) is hand-written
// once over the single TensorAttributes shape. jlogic::Value is the sole
// type-erasure boundary.
//
// Fail-closed (RFC 0018 §4/§14): every unresolved reference -- an unknown root,
// an out-of-range `dims[i]`, an unknown dim-name, or a read of a field on an
// absent optional operand -- resolves to null, which makes the enclosing
// criterion false, declining the match rather than matching on a wrong value.

#include "hip_kernel_provider_common/JsonDataSource.hpp"
#include "hip_kernel_provider_common/JsonLogic.hpp"
#include "hip_kernel_provider_common/umd/UmdPathParse.hpp"

#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E8M0.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/umd/op_schema_registry_generated.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider_common::umd
{

namespace jlogic = hip_kernel_provider_common::jsonlogic;

// A tensor pattern-variable bound to a concrete graph tensor. `tensor` is null
// only for an optional operand the graph omitted; `dimNames` points at the
// UMD-static name->index map the compiler's `shape` lowering produced (owned by
// the CompiledUmd, which outlives every BindingContext it spawns).
struct BoundTensor
{
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor = nullptr;
    bool optional = false;
    const std::unordered_map<std::string, std::size_t>* dimNames = nullptr;
};

// A pattern node bound to a concrete graph node's attributes, resolving the
// Attributes namespace `$<node_id>.<attr>` for that node.
struct BoundNode
{
    const hipdnn_flatbuffers_sdk::umd::OpSchemaEntry* schema = nullptr;
    const void* attributes = nullptr;
};

class BindingContext
{
public:
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using OpSchemaEntry = hipdnn_flatbuffers_sdk::umd::OpSchemaEntry;

    BindingContext() = default;

    BindingContext(const IGraph* graph, std::unordered_map<std::string, jlogic::Value> device)
        : _graph(graph)
        , _device(std::move(device))
    {
    }

    // Bind a pattern node id (`$sdpa_fwd`) to a graph node's attributes. A UMD
    // may declare several nodes, each resolving its own Attributes namespace.
    void bindNode(const std::string& nodeId, const OpSchemaEntry* schema, const void* attributes)
    {
        _nodes[nodeId] = BoundNode{schema, attributes};
    }

    // Bind a tensor pattern variable (`$q`) to a graph tensor. A null `tensor`
    // records an absent optional operand (`$q.present` reads false; any other
    // field read declines).
    void bindTensor(const std::string& tvar,
                    const TensorAttributes* tensor,
                    bool optional,
                    const std::unordered_map<std::string, std::size_t>* dimNames)
    {
        _tensors[tvar] = BoundTensor{tensor, optional, dimNames};
    }

    // Bind the kernel metadata the criteria's `$kernel.<field>` references
    // resolve against (RFC 0018 §4). The document is a UKD's fully-resolved KMD
    // values (optional fields already filled with schema defaults, so every
    // field a referenced criterion needs is present). An unset or empty
    // document is a defensive fallback in which every kernel field reads null.
    void bindKernelMetadata(nlohmann::json metadata)
    {
        _kernel = jlogic::JsonDataSource(std::move(metadata));
    }

    // JsonLogic data-source contract: resolve a sigil-stripped variable path
    // (`q.head_size`, `graph.node_count`, `sdpa_fwd.dropout_probability.present`)
    // to a Value. Unresolved -> null (fail closed).
    jlogic::Value getData(const std::string& varPath) const
    {
        std::string root;
        std::string rest;
        path::splitRoot(varPath, root, rest);

        if(root == "graph")
        {
            return resolveGraph(rest);
        }
        if(root == "device")
        {
            return resolveDevice(rest);
        }
        if(root == "kernel")
        {
            return _kernel.getData(rest);
        }
        const auto nit = _nodes.find(root);
        if(nit != _nodes.end())
        {
            return resolveAttr(nit->second, rest);
        }
        const auto it = _tensors.find(root);
        if(it != _tensors.end())
        {
            return resolveTensor(it->second, rest);
        }
        return {}; // unknown root -> decline
    }

    // Query helper for post-match inspection: accepts a leading `$` sigil.
    jlogic::Value get(const std::string& ref) const
    {
        if(!ref.empty() && ref.front() == '$')
        {
            return getData(ref.substr(1));
        }
        return getData(ref);
    }

    // The tensor pattern variables bound in this match, for symbol-table
    // enumeration (RFC 0018 §15). `tensor` is null for an absent optional.
    const std::unordered_map<std::string, BoundTensor>& boundTensors() const
    {
        return _tensors;
    }

    // The pattern nodes bound in this match, keyed by node id.
    const std::unordered_map<std::string, BoundNode>& boundNodes() const
    {
        return _nodes;
    }

private:
    static std::vector<std::int64_t> toVector(const ::flatbuffers::Vector<std::int64_t>* v)
    {
        if(v == nullptr)
        {
            return {};
        }
        // NOLINTNEXTLINE(modernize-return-braced-init-list) - iterator-range ctor, not initializer_list
        return std::vector<std::int64_t>(v->begin(), v->end());
    }

    jlogic::Value resolveGraph(const std::string& rest) const
    {
        if(_graph == nullptr)
        {
            return {};
        }
        if(rest == "node_count")
        {
            return {static_cast<std::int64_t>(_graph->nodeCount())};
        }
        if(rest == "is_override_shape_enabled")
        {
            // The graph's own state, read from the same accessor the matcher's
            // override-shape gate uses. Distinct from the descriptor's
            // `allow_override_shape` key, which is the matcher's opt-in.
            return {_graph->getGraph().is_override_shape_enabled()};
        }
        return {};
    }

    jlogic::Value resolveDevice(const std::string& rest) const
    {
        const auto it = _device.find(rest);
        return it != _device.end() ? it->second : jlogic::Value();
    }

    static jlogic::Value resolveAttr(const BoundNode& node, const std::string& rest)
    {
        if(node.schema == nullptr || node.attributes == nullptr)
        {
            return {};
        }
        if(rest.empty())
        {
            return {}; // a bare node id is not a value reference (compiler-rejected)
        }

        std::string attr = rest;
        const bool wantPresent = path::stripPresentSuffix(attr);

        const hipdnn_flatbuffers_sdk::umd::AttrBinding* binding = nullptr;
        for(std::size_t i = 0; i < node.schema->attributeCount; ++i)
        {
            if(node.schema->attributes[i].name == attr)
            {
                binding = &node.schema->attributes[i];
                break;
            }
        }
        if(binding == nullptr || binding->read == nullptr)
        {
            return {};
        }

        const hipdnn_flatbuffers_sdk::umd::ScalarValue sv = binding->read(node.attributes);
        if(wantPresent)
        {
            return {sv.present};
        }
        if(!sv.present)
        {
            return {}; // absent optional attribute read -> decline
        }
        switch(sv.type)
        {
        case hipdnn_flatbuffers_sdk::umd::AttrType::INT:
            return {sv.i};
        case hipdnn_flatbuffers_sdk::umd::AttrType::FLOAT:
            return {sv.f};
        case hipdnn_flatbuffers_sdk::umd::AttrType::BOOL:
            return {sv.b};
        case hipdnn_flatbuffers_sdk::umd::AttrType::DTYPE:
            return {std::string(sv.dtype != nullptr ? sv.dtype : "")};
        default:
            break;
        }
        return {};
    }

    static jlogic::Value resolveTensor(const BoundTensor& bt, const std::string& rest)
    {
        if(rest == "present")
        {
            if(!bt.optional)
            {
                return {}; // present on a required operand is refused (compiler-gated)
            }
            return {bt.tensor != nullptr};
        }
        if(bt.tensor == nullptr)
        {
            return {}; // field read on an absent optional operand -> decline
        }
        const TensorAttributes* t = bt.tensor;

        if(rest.empty() || rest == "uid")
        {
            return {static_cast<std::int64_t>(t->uid())};
        }
        if(rest == "rank")
        {
            return {static_cast<std::int64_t>(t->dims() != nullptr ? t->dims()->size() : 0)};
        }
        if(rest == "dtype")
        {
            return {std::string(
                hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(t->data_type()))};
        }
        if(rest == "virtual")
        {
            return {t->virtual_()};
        }
        if(rest == "packed")
        {
            return {hipdnn_data_sdk::utilities::isTensorPacked(toVector(t->dims()),
                                                               toVector(t->strides()))};
        }
        if(rest == "is_runtime_pass_by_value")
        {
            return {t->is_runtime_pass_by_value()};
        }
        if(rest == "value_f32")
        {
            return resolveValueF32(*t);
        }
        if(rest == "stride_order")
        {
            // Published in the RFC 0017 §5 form: logical dimension indices
            // ordered outermost (largest-stride) first, so `[0,2,3,1]` over an
            // (n,c,h,w) dim order spells N,H,W,C. extractStrideOrder returns
            // the inverse -- entry `d` is dimension `d`'s stride rank, higher
            // meaning slower-varying -- so invert it here, once, at the one
            // place a descriptor-visible value is minted.
            const std::vector<std::int64_t> ranks
                = hipdnn_data_sdk::utilities::extractStrideOrder(toVector(t->strides()));
            jlogic::Value::Array arr(ranks.size(), jlogic::Value(std::int64_t{0}));
            const auto rank = static_cast<std::int64_t>(ranks.size());
            for(std::size_t dim = 0; dim < ranks.size(); ++dim)
            {
                // A dim of stride rank r sits at physical position rank-1-r.
                const std::int64_t position = rank - 1 - ranks[dim];
                if(position < 0 || position >= rank)
                {
                    return {}; // malformed rank vector -> decline (fail closed)
                }
                arr[static_cast<std::size_t>(position)]
                    = jlogic::Value(static_cast<std::int64_t>(dim));
            }
            return {std::move(arr)};
        }

        std::size_t idx = 0;
        if(path::parseSubscript(rest, "dims", idx))
        {
            const auto* dims = t->dims();
            if(dims == nullptr || idx >= dims->size())
            {
                return {};
            }
            return {
                static_cast<std::int64_t>(dims->Get(static_cast<::flatbuffers::uoffset_t>(idx)))};
        }
        if(path::parseSubscript(rest, "strides", idx))
        {
            const auto* strides = t->strides();
            if(strides == nullptr || idx >= strides->size())
            {
                return {};
            }
            return {static_cast<std::int64_t>(
                strides->Get(static_cast<::flatbuffers::uoffset_t>(idx)))};
        }

        // Named dim introduced by a `shape` short-hand.
        if(bt.dimNames != nullptr)
        {
            const auto dit = bt.dimNames->find(rest);
            if(dit != bt.dimNames->end())
            {
                const auto* dims = t->dims();
                if(dims == nullptr || dit->second >= dims->size())
                {
                    return {};
                }
                return {static_cast<std::int64_t>(
                    dims->Get(static_cast<::flatbuffers::uoffset_t>(dit->second)))};
            }
        }
        return {}; // unknown tensor field -> decline
    }

    // Coerce whichever arm of the tensor's `value` union is set to f32 and
    // publish it as the single `$q.value_f32` token (RFC 0017 §5). A tensor
    // carrying no compile-time value resolves to null, so a criterion over it
    // declines rather than reading a zero that would satisfy a comparison.
    static jlogic::Value resolveValueF32(const TensorAttributes& t)
    {
        namespace data = hipdnn_flatbuffers_sdk::data_objects;
        switch(t.value_type())
        {
        case data::TensorValue::Float32Value:
            return {static_cast<double>(t.value_as_Float32Value()->value())};
        case data::TensorValue::Float16Value:
            return {static_cast<double>(t.value_as_Float16Value()->value())};
        case data::TensorValue::BFloat16Value:
            return {static_cast<double>(t.value_as_BFloat16Value()->value())};
        case data::TensorValue::Float64Value:
            return {t.value_as_Float64Value()->value()};
        case data::TensorValue::Int32Value:
            return {static_cast<double>(t.value_as_Int32Value()->value())};
        case data::TensorValue::Int64Value:
            return {static_cast<double>(t.value_as_Int64Value()->value())};
        case data::TensorValue::BoolValue:
            return {t.value_as_BoolValue()->value() ? 1.0 : 0.0};
        case data::TensorValue::Float8Value:
            return fp8ValueToF32(t.value_as_Float8Value()->value(), t.data_type());
        default:
            break;
        }
        return {}; // TensorValue::NONE -> no compile-time value -> decline
    }

    // Float8Value stores raw bits; the tensor's data_type says which 8-bit
    // format they encode. An arm/data_type pairing that names no 8-bit format
    // resolves to null (fail closed) rather than reinterpreting the bits.
    static jlogic::Value fp8ValueToF32(std::uint8_t bits,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType dtype)
    {
        namespace data = hipdnn_flatbuffers_sdk::data_objects;
        namespace types = hipdnn_data_sdk::types;
        switch(dtype)
        {
        case data::DataType::FP8_E4M3:
            return {static_cast<double>(static_cast<float>(types::fp8_e4m3::from_bits(bits)))};
        case data::DataType::FP8_E5M2:
            return {static_cast<double>(static_cast<float>(types::fp8_e5m2::from_bits(bits)))};
        case data::DataType::FP8_E4M3_FNUZ:
            return {static_cast<double>(static_cast<float>(types::fp8_e4m3_fnuz::from_bits(bits)))};
        case data::DataType::FP8_E5M2_FNUZ:
            return {static_cast<double>(static_cast<float>(types::fp8_e5m2_fnuz::from_bits(bits)))};
        case data::DataType::FP8_E8M0:
            return {static_cast<double>(static_cast<float>(types::fp8_e8m0::from_bits(bits)))};
        case data::DataType::UINT8:
            return {static_cast<double>(bits)};
        case data::DataType::INT8:
            // Reinterpret the raw byte as signed: an INT8 -1 is stored as 0xFF
            // and must read as -1.0, not 255.0.
            return {static_cast<double>(static_cast<std::int8_t>(bits))};
        default:
            break;
        }
        return {};
    }

    const IGraph* _graph = nullptr;
    std::unordered_map<std::string, jlogic::Value> _device;
    std::unordered_map<std::string, BoundNode> _nodes;
    std::unordered_map<std::string, BoundTensor> _tensors;
    jlogic::JsonDataSource _kernel; // empty until bindKernelMetadata()
};

} // namespace hip_kernel_provider_common::umd
