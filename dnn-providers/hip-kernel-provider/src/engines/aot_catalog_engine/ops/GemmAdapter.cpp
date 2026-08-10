// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ops/GemmAdapter.hpp"

#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "launch/PluginError.hpp"

namespace aot_catalog_engine::ops
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;

namespace
{

// hipDNN element dtype -> the provider dtype token kernel authors use in
// family.json constraints. Unsupported dtypes yield nullopt (decode declines).
std::optional<std::string> providerDtype(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::HALF:
        return std::string("f16");
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::FLOAT:
        return std::string("f32");
    default:
        return std::nullopt;
    }
}

int64_t problemInt(const catalog::ProblemShape& problem, const std::string& key)
{
    auto it = problem.find(key);
    if(it == problem.end() || !std::holds_alternative<int64_t>(it->second))
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog(gemm): problem missing integer key '" + key + "'");
    }
    return std::get<int64_t>(it->second);
}

} // namespace

std::optional<catalog::ProblemShape> GemmAdapter::decode(const IGraph& graph) const
{
    // Single MatmulAttributes node only (Tier-D allowlist for the POC).
    if(!graph.isValid() || graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::MatmulAttributes)
    {
        return std::nullopt;
    }
    const auto& attrs = node.attributesAs<data_objects::MatmulAttributes>();

    const auto& tensorMap = graph.getTensorMap();
    auto findTensor = [&](int64_t uid) -> const data_objects::TensorAttributes* {
        auto it = tensorMap.find(uid);
        return it == tensorMap.end() ? nullptr : it->second;
    };

    const auto* aTensor = findTensor(attrs.a_tensor_uid());
    const auto* bTensor = findTensor(attrs.b_tensor_uid());
    const auto* cTensor = findTensor(attrs.c_tensor_uid());
    if(aTensor == nullptr || bTensor == nullptr || cTensor == nullptr)
    {
        return std::nullopt;
    }

    // TensorAttributesWrapper::dims()/strides()/dataType() throw on null
    // dims/strides; a malformed graph should make the engine decline, not throw
    // out of isApplicable, so treat any decode failure as "not applicable".
    try
    {
        const TensorAttributesWrapper a(aTensor);
        const TensorAttributesWrapper b(bTensor);
        const TensorAttributesWrapper c(cTensor);

        // hipDNN matmul is C = A @ B on LOGICAL dims A[...,M,K], B[...,K,N],
        // C[...,M,N] -- there is no transpose flag; the physical memory layout
        // is carried entirely by strides (frontend docs: "memory layout is
        // controlled by strides, not by dimension order"). So we read M/N/K
        // from the logical dims and then require the strides to match exactly
        // the packed RCR layout the shipped wmma_gemm implements:
        //   A row-major [M,K]      -> strides {K, 1}
        //   B "transposed" [K,N]   -> strides {1, K}   (physical [N,K] weight)
        //   C row-major [M,N]      -> strides {N, 1}
        // That RCR triple is exactly nn.Linear (y = x @ W^T, W = [N,K]). Any
        // other stride layout (e.g. standard row-major B {N,1}) is a matmul we
        // do NOT have a kernel for, so we fail closed and decline rather than
        // launch the RCR kernel on it and return wrong results.
        const auto aDims = a.dims();
        const auto bDims = b.dims();
        const auto cDims = c.dims();
        if(aDims.size() != 2 || bDims.size() != 2 || cDims.size() != 2)
        {
            return std::nullopt; // 2D only for the POC (no batched matmul yet)
        }

        const int64_t m = aDims[0];
        const int64_t k = aDims[1];
        const int64_t n = bDims[1];
        if(bDims[0] != k || cDims[0] != m || cDims[1] != n)
        {
            return std::nullopt; // inconsistent matmul shapes
        }

        const auto aStrides = a.strides();
        const auto bStrides = b.strides();
        const auto cStrides = c.strides();
        const bool aRowMajor = aStrides.size() == 2 && aStrides[0] == k && aStrides[1] == 1;
        const bool bTransposed = bStrides.size() == 2 && bStrides[0] == 1 && bStrides[1] == k;
        const bool cRowMajor = cStrides.size() == 2 && cStrides[0] == n && cStrides[1] == 1;
        if(!aRowMajor || !bTransposed || !cRowMajor)
        {
            return std::nullopt; // not the packed RCR layout our kernel serves
        }

        // Require a single element dtype across A/B/C for the POC.
        const auto dtype = providerDtype(a.dataType());
        if(!dtype.has_value() || b.dataType() != a.dataType() || c.dataType() != a.dataType())
        {
            return std::nullopt;
        }

        catalog::ProblemShape shape;
        shape.emplace("dtype", catalog::ShapeValue{*dtype});
        shape.emplace("M", catalog::ShapeValue{m});
        shape.emplace("N", catalog::ShapeValue{n});
        shape.emplace("K", catalog::ShapeValue{k});
        return shape;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO("aot-catalog(gemm): declining graph, decode failed: " << e.what());
        return std::nullopt;
    }
}

catalog::LaunchBindings GemmAdapter::buildBindings(const IGraph& graph,
                                                   const catalog::ProblemShape& problem,
                                                   const catalog::KernelEntry& kernel) const
{
    (void)kernel; // GEMM binds a fixed (A,B,C,M,N,K) ABI regardless of kernel

    const auto& node = graph.getNodeWrapper(0);
    const auto& attrs = node.attributesAs<data_objects::MatmulAttributes>();

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", attrs.a_tensor_uid());
    bindings.pointerUids.emplace("B", attrs.b_tensor_uid());
    bindings.pointerUids.emplace("C", attrs.c_tensor_uid());
    bindings.scalars.emplace("M", catalog::ScalarValue{problemInt(problem, "M")});
    bindings.scalars.emplace("N", catalog::ScalarValue{problemInt(problem, "N")});
    bindings.scalars.emplace("K", catalog::ScalarValue{problemInt(problem, "K")});
    return bindings;
}

launch::SymbolTable GemmAdapter::gridSymbols(const catalog::ProblemShape& problem,
                                             const catalog::KernelEntry& kernel) const
{
    (void)kernel;

    launch::SymbolTable symbols;
    symbols.emplace("M", problemInt(problem, "M"));
    symbols.emplace("N", problemInt(problem, "N"));
    symbols.emplace("K", problemInt(problem, "K"));
    return symbols;
}

} // namespace aot_catalog_engine::ops
