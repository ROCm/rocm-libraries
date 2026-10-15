// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Gated like the Triton native files that include it: outside the gate it declares nothing.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"

/**
 * @file TritonPackCommon.hpp
 * @brief What every Triton AOT pack's native file shares: the graph helpers its matchers
 *        build on, the dtype spelling its kernels' metadata uses, scoring, the prepared
 *        dispatch, and the process-lifetime module cache and dispatch handler.
 *
 * Adapted from dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp at commit 03911535bd0 (FlyDSL provider)
 *
 * Each pack keeps its own graph matcher, kernel admission rule, binding, kernarg signature
 * and launch argument list, and the symbol strings its descriptors name.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::triton
{

namespace ingestor = hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Triton's global and profile scratch kernarg slots. Always null: the generator rejects
/// any object that needs scratch.
inline constexpr void* NO_SCRATCH = nullptr;

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

inline const data_objects::TensorAttributes* findTensor(const ingestor::MatchContext& context,
                                                        int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// A regular device operand: real storage, readable by the kernel through a pointer.
inline bool isDeviceOperand(const data_objects::TensorAttributes& tensor)
{
    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/// Runs on an unvalidated graph, so it must be total: dims/strides may be null or
/// mismatched, or an extent non-positive. Later helpers may assume positive extents.
inline bool hasUsableShape(const data_objects::TensorAttributes& tensor, uint32_t minimumRank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() < minimumRank)
    {
        return false;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) <= 0)
        {
            return false;
        }
    }
    return true;
}

/// Dense row-major: innermost stride 1 and every outer stride the product of the extents
/// inside it. This makes collapsing rank > 2 to (rows, N) sound, since a kernel addresses
/// row `r` at `r * N`. Also false when the element count overflows int64, so products of
/// extents after a true result (see rowCount) cannot overflow. Requires positive extents.
inline bool isPackedRowMajor(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    int64_t expected = 1;
    for(auto axis = static_cast<int64_t>(dims->size()) - 1; axis >= 0; --axis)
    {
        const auto index = static_cast<flatbuffers::uoffset_t>(axis);
        if(strides->Get(index) != expected)
        {
            return false;
        }
        const auto extent = dims->Get(index);
        if(expected > std::numeric_limits<int64_t>::max() / extent)
        {
            return false;
        }
        expected *= extent;
    }
    return true;
}

/// Product of every extent except the innermost: the row count a grid is sized from.
/// Requires a tensor that passed isPackedRowMajor, which bounds the product.
inline int64_t rowCount(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    int64_t rows = 1;
    for(flatbuffers::uoffset_t axis = 0; axis + 1 < dims->size(); ++axis)
    {
        rows *= dims->Get(axis);
    }
    return rows;
}

inline int64_t innermostExtent(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    return dims->Get(dims->size() - 1);
}

inline bool sameShape(const data_objects::TensorAttributes& lhs,
                      const data_objects::TensorAttributes& rhs)
{
    const auto* lhsDims = lhs.dims();
    const auto* rhsDims = rhs.dims();
    if(lhsDims->size() != rhsDims->size())
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < lhsDims->size(); ++axis)
    {
        if(lhsDims->Get(axis) != rhsDims->Get(axis))
        {
            return false;
        }
    }
    return true;
}

/// The kernel metadata spelling of a graph dtype (`bf16`/`f16`, not the flatbuffer
/// `BFLOAT16`/`HALF`); nullopt for a dtype no Triton kernel is built for.
inline std::optional<std::string> kernelDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    default:
        return std::nullopt;
    }
}

/// True when the tensor @p uid can be classified as a scalar operand (an unknown dtype
/// cannot). The value is read at execute, never compared at match time.
inline bool
    scalarIsClassifiable(const ingestor::MatchContext& context, int64_t uid, const char* name)
{
    try
    {
        (void)hipdnn_plugin_sdk::makeScalarOperand(context.graph.getTensorMap(), uid, name);
        return true;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Scoring and dispatch
// ---------------------------------------------------------------------------

/// Ranks the kernels a pack's matchers admitted by descriptor `priority`.
inline double priorityScore(const ingestor::MatchContext& /*context*/,
                            const ingestor::BoundTokens& /*bound*/,
                            const ingestor::KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

/// The loaded kernel plus everything a launch reads from the graph, resolved once at
/// prepare(). @p TLaunchState is the pack's own: its bindings and kernarg values.
template <typename TLaunchState>
class PreparedTritonDispatch : public ingestor::PreparedDispatch
{
public:
    PreparedTritonDispatch(IngestorKernelCode code, TLaunchState state)
        : _code(std::move(code))
        , _state(std::move(state))
    {
    }

    /// The kernel for the stream's device; resolved per launch because a plan outlives the
    /// handle it was built from.
    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const TLaunchState& state() const
    {
        return _state;
    }

private:
    // Owns each device's program alongside the kernel viewing into it, so the module
    // outlives every function resolved from it.
    IngestorKernelCode _code;
    TLaunchState _state;
};

/// The kpack module cache @p THandler loads through. Process-lifetime and per pack: the
/// key is (archive, toc_key, arch), so a shared cache would answer other packs' lookups.
template <typename THandler>
compilation::KpackModuleCache& kpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// The one @p THandler instance, loading through kpackModuleCache<THandler>().
///
/// Process-lifetime: the registry holds a non-owning pointer, and Containers are created
/// and destroyed per handle, so the handler and its loader must outlive every Container.
template <typename THandler>
const THandler& processDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(kpackModuleCache<THandler>());
    static const THandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::triton

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
