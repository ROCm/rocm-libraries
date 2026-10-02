// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <string>

#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/CustomLibraryAdapter.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/IUhdAdapter.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/NativeAdapter.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TableAdapter.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TreeDataAdapter.hpp>

/// @file AdapterFactory.hpp
/// @brief Builds the scorer a UhdConfig names.
namespace hipdnn_plugin_sdk::uhd
{

/// @brief Construct the adapter @p cfg names, or nullptr if it cannot be built.
///
/// nullptr never changes applicability. `static_order` always yields nullptr (it is ranked
/// by declared order), as does `onnx`, whose runtime not every provider links.
inline std::shared_ptr<IUhdAdapter> makeUhdAdapter(const UhdConfig& cfg)
{
    if(cfg.adapterType == "tree_data")
    {
        if(!cfg.modelArtifactPath.empty())
        {
            // A grouped model chooses its group, so it applies RFC 0019 §8.3's score rule
            // itself and needs the transform and metric.
            return TreeDataAdapter::load(cfg.modelArtifactPath,
                                         cfg.featuresHash,
                                         cfg.modelHash,
                                         cfg.objective,
                                         cfg.scoreTransform,
                                         cfg.scoreMetric);
        }
    }
    else if(cfg.adapterType == "table")
    {
        if(!cfg.modelArtifactPath.empty())
        {
            // modelHash is the content identity caches key on; the adapter verifies it.
            return TableAdapter::load(cfg.modelArtifactPath, cfg.featuresHash, cfg.modelHash);
        }
    }
    else if(cfg.adapterType == "native")
    {
        // In-process scorer; nothing is loaded from disk (RFC 0019 §7.1).
        if(!cfg.nativeSymbol.empty())
        {
            return NativeAdapter::resolve(
                cfg.nativeSymbol, cfg.featuresSignature.size(), cfg.featuresHash);
        }
    }

    else if(cfg.adapterType == "custom_library")
    {
        if(!cfg.modelArtifactPath.empty() && !cfg.customLibrarySymbol.empty())
        {
            return CustomLibraryAdapter::load(cfg.modelArtifactPath,
                                              cfg.customLibrarySymbol,
                                              cfg.featuresSignature.size(),
                                              cfg.featuresHash,
                                              cfg.modelHash);
        }
        HIPDNN_SDK_LOG_ERROR("uhd: custom_library needs both a model artifact path and a "
                             "symbol name; scorer unavailable");
    }

    return nullptr;
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
