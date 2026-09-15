// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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
/// An unavailable scorer leaves the caller's applicability decision intact: kernel
/// selection falls back to declared order; engine prediction returns no estimate.
///
/// `static_order` is not a scorer and yields nullptr by design -- selection ranks it with
/// the declared-order comparator instead of building an adapter.
///
/// `onnx` is not supported here: its runtime is not a dependency of every provider.
inline std::shared_ptr<IUhdAdapter> makeUhdAdapter(const UhdConfig& cfg)
{
    if(cfg.adapterType == "tree_data")
    {
        if(!cfg.modelArtifactPath.empty())
        {
            return TreeDataAdapter::load(cfg.modelArtifactPath, cfg.featuresHash, cfg.modelHash);
        }
    }
    else if(cfg.adapterType == "table")
    {
        if(!cfg.modelArtifactPath.empty())
        {
            return TableAdapter::load(cfg.modelArtifactPath, cfg.featuresHash);
        }
    }
    else if(cfg.adapterType == "native")
    {
        // Resolves a scorer the engine registered in-process; nothing is loaded from disk
        // (RFC 0019 §7.1).
        if(!cfg.nativeSymbol.empty())
        {
            return NativeAdapter::resolve(
                cfg.nativeSymbol, cfg.featuresSignature.size(), cfg.featuresHash);
        }
    }

    else if(cfg.adapterType == "custom_library")
    {
        // The platform loader supports the compiled-scorer escape hatch on every host.
        if(!cfg.modelArtifactPath.empty() && !cfg.customLibrarySymbol.empty())
        {
            return CustomLibraryAdapter::load(cfg.modelArtifactPath,
                                              cfg.customLibrarySymbol,
                                              cfg.featuresSignature.size(),
                                              cfg.featuresHash);
        }
        HIPDNN_SDK_LOG_ERROR("uhd: custom_library needs both a model artifact path and a "
                             "symbol name; scorer unavailable");
    }

    return nullptr;
}

} // namespace hipdnn_plugin_sdk::uhd
