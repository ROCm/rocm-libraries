// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/IUhdAdapter.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/CustomLibraryAdapter.hpp>
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

    else if(cfg.adapterType == "custom_library")
    {
        // RFC 0019 §7.2's escape hatch. It lived beside the backend's kernel-ranking path and
        // so was unreachable from here, which meant the adapter kind was implemented and not
        // delivered. It needs only dlopen, unlike `onnx`, whose runtime deliberately does not
        // reach every provider's include path.
        if(!cfg.modelArtifactPath.empty() && !cfg.customLibrarySymbol.empty())
        {
            return CustomLibraryAdapter::load(cfg.modelArtifactPath,
                                              cfg.customLibrarySymbol,
                                              cfg.featuresSignature.size(),
                                              cfg.featuresHash);
        }
        HIPDNN_SDK_LOG_ERROR("uhd: custom_library needs both a model artifact path and a "
                             "symbol name; ranking falls back to declared order");
    }

    return nullptr;
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
