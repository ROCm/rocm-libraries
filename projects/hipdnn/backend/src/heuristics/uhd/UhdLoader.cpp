// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "UhdLoader.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>

#include <flatbuffers/flatbuffers.h>

#include <fstream>
#include <sstream>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

std::optional<UhdConfig> UhdLoader::load(const std::filesystem::path& uhdPath)
{
    if(!std::filesystem::exists(uhdPath))
    {
        HIPDNN_SDK_LOG_ERROR("UHD file not found: " << uhdPath);
        return std::nullopt;
    }

    // Read entire file into memory
    std::ifstream file(uhdPath, std::ios::binary | std::ios::ate);
    if(!file)
    {
        HIPDNN_SDK_LOG_ERROR("Failed to open UHD file: " << uhdPath);
        return std::nullopt;
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        HIPDNN_SDK_LOG_ERROR("Failed to read UHD file: " << uhdPath);
        return std::nullopt;
    }

    // Resolve base path for model artifact path
    const auto basePath = uhdPath.parent_path();

    return loadFromBuffer(buffer.data(), buffer.size(), basePath);
}

std::optional<UhdConfig>
    UhdLoader::loadFromBuffer(const void* buffer, size_t bufferSize, const std::filesystem::path& basePath)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(buffer == nullptr || bufferSize == 0)
    {
        HIPDNN_SDK_LOG_ERROR("UHD buffer is null or empty");
        return std::nullopt;
    }

    // Verify FlatBuffer integrity
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), bufferSize);
    if(!verifier.VerifyBuffer<UHD>(nullptr))
    {
        HIPDNN_SDK_LOG_ERROR("UHD FlatBuffer verification failed (corrupt or malformed buffer)");
        return std::nullopt;
    }

    // Get root UHD table
    const auto* uhd = flatbuffers::GetRoot<UHD>(buffer);
    if(uhd == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("Failed to get UHD root table");
        return std::nullopt;
    }

    // Populate UhdConfig
    UhdConfig config;

    // Required fields
    if(uhd->id() == nullptr || uhd->features_hash() == nullptr || uhd->objective() == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("UHD missing required fields (id, features_hash, or objective)");
        return std::nullopt;
    }

    config.uhdId = uhd->id()->str();
    config.featuresHash = uhd->features_hash()->str();
    config.objective = uhd->objective()->str();

    // Validate objective value
    if(config.objective != "max" && config.objective != "min")
    {
        HIPDNN_SDK_LOG_ERROR("UHD objective must be 'max' or 'min', got: " << config.objective);
        return std::nullopt;
    }

    // Optional fields
    if(uhd->name() != nullptr)
    {
        config.name = uhd->name()->str();
    }

    // Derived values (RFC 0019 §6.4)
    if(uhd->derived() != nullptr)
    {
        config.derived.reserve(uhd->derived()->size());
        for(const auto* entry : *uhd->derived())
        {
            if(entry != nullptr && entry->name() != nullptr && entry->expression() != nullptr)
            {
                config.derived.emplace_back(entry->name()->str(), entry->expression()->str());
            }
        }
    }

    // Features signature
    if(uhd->features_signature() != nullptr)
    {
        config.featuresSignature.reserve(uhd->features_signature()->size());
        for(const auto* feature : *uhd->features_signature())
        {
            if(feature != nullptr)
            {
                config.featuresSignature.push_back(feature->str());
            }
        }
    }

    // Score metadata
    if(uhd->score() != nullptr)
    {
        const auto* scoreMetadata = uhd->score();
        if(scoreMetadata->units() != nullptr)
        {
            config.scoreUnits = scoreMetadata->units()->str();
        }
        config.scoreCalibrated = scoreMetadata->calibrated();
        if(scoreMetadata->transform() != nullptr)
        {
            config.scoreTransform = scoreMetadata->transform()->str();
        }
    }

    // Adapter type
    config.adapterType = [](UhdAdapter adapter) -> std::string {
        switch(adapter)
        {
        case UhdAdapter::STATIC_ORDER: return "static_order";
        case UhdAdapter::TREE_DATA: return "tree_data";
        case UhdAdapter::TABLE: return "table";
        case UhdAdapter::ONNX: return "onnx";
        case UhdAdapter::CUSTOM_LIBRARY: return "custom_library";
        default: return "unknown";
        }
    }(uhd->adapter());

    // Model artifact path (resolve relative to base path if provided)
    if(uhd->model_artifact_path() != nullptr)
    {
        std::filesystem::path artifactPath(uhd->model_artifact_path()->str());
        if(!basePath.empty() && artifactPath.is_relative())
        {
            artifactPath = basePath / artifactPath;
        }
        config.modelArtifactPath = artifactPath.string();
    }

    // Model hash for integrity validation
    if(uhd->model_hash() != nullptr)
    {
        config.modelHash = uhd->model_hash()->str();
    }

    // Static order fields
    if(uhd->static_order_fields() != nullptr)
    {
        config.staticOrderFields.clear();
        config.staticOrderFields.reserve(uhd->static_order_fields()->size());
        for(const auto* field : *uhd->static_order_fields())
        {
            if(field != nullptr)
            {
                config.staticOrderFields.push_back(field->str());
            }
        }
    }

    // Custom library symbol
    if(uhd->custom_library_symbol() != nullptr)
    {
        config.customLibrarySymbol = uhd->custom_library_symbol()->str();
    }

    // Validate adapter-specific requirements
    if(config.adapterType == "tree_data" || config.adapterType == "onnx" ||
       config.adapterType == "custom_library")
    {
        if(config.modelArtifactPath.empty())
        {
            HIPDNN_SDK_LOG_ERROR("UHD adapter '" << config.adapterType
                                                  << "' requires model_artifact_path");
            return std::nullopt;
        }
    }

    if(config.adapterType == "static_order" && config.staticOrderFields.empty())
    {
        HIPDNN_SDK_LOG_WARN("UHD static_order adapter has no static_order_fields, "
                            << "defaulting to ['priority', 'id']");
        config.staticOrderFields = {"priority", "id"};
    }

    if(config.adapterType == "custom_library" && config.customLibrarySymbol.empty())
    {
        HIPDNN_SDK_LOG_ERROR("UHD custom_library adapter requires custom_library_symbol");
        return std::nullopt;
    }

    return config;
}

} // namespace hipdnn_backend::heuristics::uhd
