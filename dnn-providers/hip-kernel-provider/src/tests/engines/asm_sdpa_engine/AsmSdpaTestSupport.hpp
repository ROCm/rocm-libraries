// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Test helpers that do not include a generated kernel registry. The registry headers
// (asm_fmha_v3_fwd_configs.hpp and asm_fmha_v3_bwd_configs.hpp) each define the macro
// ADD_CFG and the alias CFG at global scope, so one translation unit can include only one
// of them. This header is safe to include next to either.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace asm_sdpa_engine
{

/// A packed SDPA backward graph: Q, K, dQ and dK take @p qkDims; V, O, dO and dV take
/// @p voDims.
inline flatbuffers::FlatBufferBuilder
    createPackedSdpaBwdGraph(const std::vector<int64_t>& qkDims,
                             const std::vector<int64_t>& voDims,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16)
{
    const auto qkStrides = hipdnn_data_sdk::utilities::generateStrides(qkDims);
    const auto voStrides = hipdnn_data_sdk::utilities::generateStrides(voDims);
    return hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        qkDims, qkStrides, qkDims, qkStrides, voDims, voStrides, voDims, voStrides, dataType);
}

/// A packed SDPA forward graph with FLOAT compute: Q and K take @p qkDims; V and O take
/// @p voDims.
inline flatbuffers::FlatBufferBuilder
    createPackedSdpaFwdGraph(const std::vector<int64_t>& qkDims,
                             const std::vector<int64_t>& voDims,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16)
{
    const auto qkStrides = hipdnn_data_sdk::utilities::generateStrides(qkDims);
    const auto voStrides = hipdnn_data_sdk::utilities::generateStrides(voDims);
    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qkDims,
        qkStrides,
        qkDims,
        qkStrides,
        voDims,
        voStrides,
        voDims,
        voStrides,
        dataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
}

/// One code object a kernel registry row names: the key it has in its architecture's
/// kpack TOC, and the kernel symbol it must define.
struct AsmKernelTocEntry
{
    std::string tocKey;
    std::string knlName;
};

/// Every cfg_fmha_fwd row for @p arch, keyed as SdpaFwdPlanBuilder looks it up: a gfx942
/// row once under MI300/ and once under MI308/, a gfx950 row once without a variant.
/// Defined in AsmSdpaConfigHelpers.cpp, which includes the forward registry.
std::vector<AsmKernelTocEntry> getFwdKernelTocEntries(const std::string& arch);

} // namespace asm_sdpa_engine
