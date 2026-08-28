// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/IUhdAdapter.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/NativeAdapter.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/TableAdapter.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/TreeDataAdapter.hpp>

/// @file AdapterFactory.hpp
/// @brief Builds the scorer a UhdConfig names.
namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief Construct the adapter @p cfg names, or nullptr if it cannot be built.
///
/// Returning nullptr rather than throwing is the contract: RFC 0019 §5 requires a UHD
/// that will not come up to degrade to declared order, not to fail the request. A caller
/// that gets nullptr ranks by `priority` then `id`.
///
/// `static_order` is not a scorer and yields nullptr by design -- selection ranks it with
/// the declared-order comparator instead of building an adapter.
///
/// Two adapter kinds are absent here and handled by the caller: `onnx` needs a runtime
/// this header will not pull onto every provider's include path, and `custom_library`
/// needs dlopen, which is not portable enough for a header. Both live in the backend
/// (see EngineRegistry), which checks for them before delegating here.
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
            return NativeAdapter::resolve(cfg.nativeSymbol,
                                          cfg.featuresSignature.size(),
                                          cfg.featuresHash);
        }
    }

    return nullptr;
}

/// @brief Whether @p adapterType is one this factory can build.
///
/// Lets a caller that handles extra kinds decide whether to delegate, without duplicating
/// the list of names.
inline bool isFactoryAdapterType(const std::string& adapterType)
{
    return adapterType == "tree_data" || adapterType == "table" || adapterType == "native";
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
