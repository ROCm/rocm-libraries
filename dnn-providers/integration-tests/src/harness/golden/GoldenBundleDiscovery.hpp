// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>


namespace hipdnn_integration_tests::golden
{

struct DiscoveredBundle
{
    std::filesystem::path jsonPath;
    std::string suiteName;
    std::string testName;
};

inline constexpr std::array<const char*, 4> K_TIER_NAMES = {
    "quick", "standard", "comprehensive", "full"};

// RFC 0011 §4.3 test-naming scheme: the tier becomes a GTest suite prefix.
// `quick` is the default smoke tier and carries no prefix; the others are
// Capitalized (e.g. `Standard/`). The prefix includes its trailing '/'.
inline std::string tierPrefix(const std::string& tierName)
{
    if(tierName == "quick")
    {
        return "";
    }
    if(tierName == "standard")
    {
        return "Standard/";
    }
    if(tierName == "comprehensive")
    {
        return "Comprehensive/";
    }
    if(tierName == "full")
    {
        return "Full/";
    }
    return "";
}

inline std::string sanitizeForGtest(const std::string& input)
{
    std::string result;
    result.reserve(input.size());
    for(const char c : input)
    {
        result += (std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_') ? c : '_';
    }
    return result;
}

inline std::string dataTypeToShortString(
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DT::FLOAT:
        return "fp32";
    case DT::HALF:
        return "fp16";
    case DT::BFLOAT16:
        return "bfp16";
    case DT::INT8:
        return "int8";
    case DT::FP8_E4M3:
        return "fp8e4m3";
    case DT::FP8_E5M2:
        return "fp8e5m2";
    case DT::INT32:
        return "int32";
    case DT::INT64:
        return "int64";
    case DT::DOUBLE:
        return "fp64";
    case DT::BOOLEAN:
        return "bool";
    default:
        return "unknown";
    }
}

inline std::string deriveLayoutFromStrides(
    const flatbuffers::Vector<int64_t>* dims,
    const flatbuffers::Vector<int64_t>* strides)
{
    if(dims == nullptr || strides == nullptr || dims->size() < 4)
    {
        return "unknown";
    }

    auto ndim = dims->size();

    // Build index-by-stride (descending stride → dimension order)
    std::vector<size_t> indices(ndim);
    for(size_t i = 0; i < ndim; ++i)
    {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return strides->Get(static_cast<flatbuffers::uoffset_t>(a))
               > strides->Get(static_cast<flatbuffers::uoffset_t>(b));
    });

    if(ndim == 4)
    {
        // NCHW: N(0) C(1) H(2) W(3) — strides descending in that order
        if(indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[3] == 3)
        {
            return "nchw";
        }
        // NHWC: N(0) H(2) W(3) C(1)
        if(indices[0] == 0 && indices[1] == 2 && indices[2] == 3 && indices[3] == 1)
        {
            return "nhwc";
        }
    }
    else if(ndim == 5)
    {
        // NCDHW: N(0) C(1) D(2) H(3) W(4)
        if(indices[0] == 0 && indices[1] == 1 && indices[2] == 2 && indices[3] == 3
           && indices[4] == 4)
        {
            return "ncdhw";
        }
        // NDHWC: N(0) D(2) H(3) W(4) C(1)
        if(indices[0] == 0 && indices[1] == 2 && indices[2] == 3 && indices[3] == 4
           && indices[4] == 1)
        {
            return "ndhwc";
        }
    }

    return "unknown";
}

inline std::string nodeAttributesToOperationName(
    hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    switch(attrType)
    {
    case NA::ConvolutionFwdAttributes:
        return "ConvFprop";
    case NA::ConvolutionBwdAttributes:
        return "ConvDgrad";
    case NA::ConvolutionWrwAttributes:
        return "ConvWgrad";
    case NA::BatchnormInferenceAttributes:
        return "BatchnormInference";
    case NA::BatchnormInferenceAttributesVarianceExt:
        return "BatchnormInferenceVarianceExt";
    case NA::BatchnormAttributes:
        return "Batchnorm";
    case NA::BatchnormBackwardAttributes:
        return "BatchnormBackward";
    case NA::PointwiseAttributes:
        return "Pointwise";
    case NA::MatmulAttributes:
        return "Matmul";
    case NA::RMSNormAttributes:
        return "RmsNorm";
    case NA::LayernormAttributes:
        return "LayerNorm";
    case NA::SdpaAttributes:
        return "SdpaFwd";
    case NA::SdpaBackwardAttributes:
        return "SdpaBwd";
    case NA::BlockScaleQuantizeAttributes:
        return "BlockScaleQuantize";
    case NA::BlockScaleDequantizeAttributes:
        return "BlockScaleDequantize";
    case NA::ReductionAttributes:
        return "Reduction";
    case NA::CustomOpAttributes:
        return "CustomOp";
    default:
        return "Unknown";
    }
}

inline std::string deriveOperationName(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper)
{
    std::string opName;
    auto nodeCount = wrapper.nodeCount();
    for(uint32_t i = 0; i < nodeCount; ++i)
    {
        auto& node = wrapper.getNode(i);
        auto attrType = node.attributes_type();
        auto name = nodeAttributesToOperationName(attrType);
        if(!opName.empty())
        {
            opName += "_";
        }
        opName += name;
    }
    return opName.empty() ? "UnknownOp" : opName;
}

inline std::string deriveDataTypeFromGraph(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper)
{
    auto tensorMap = wrapper.getTensorMap();
    for(auto& [uid, attrs] : tensorMap)
    {
        if(attrs == nullptr)
        {
            continue;
        }
        return dataTypeToShortString(attrs->data_type());
    }
    return "unknown";
}

inline std::string deriveLayoutFromGraph(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper)
{
    auto tensorMap = wrapper.getTensorMap();
    for(auto& [uid, attrs] : tensorMap)
    {
        if(attrs == nullptr || attrs->dims() == nullptr || attrs->strides() == nullptr)
        {
            continue;
        }
        if(attrs->dims()->size() >= 4)
        {
            return deriveLayoutFromStrides(attrs->dims(), attrs->strides());
        }
    }
    return "unknown";
}

struct DerivedTestName
{
    std::string suiteName;
    std::string testName;
};

inline DerivedTestName deriveTestName(const std::filesystem::path& jsonPath,
                                      const std::string& tierName)
{
    std::ifstream file(jsonPath);
    if(!file)
    {
        throw std::runtime_error("Cannot open bundle JSON: " + jsonPath.string());
    }

    auto graphJson = nlohmann::json::parse(file);
    flatbuffers::FlatBufferBuilder builder;
    auto offset
        = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            builder, graphJson);
    builder.Finish(offset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const auto opName = deriveOperationName(wrapper);
    const auto layout = deriveLayoutFromGraph(wrapper);
    const auto dtype = deriveDataTypeFromGraph(wrapper);

    const std::string suite = tierPrefix(tierName) + sanitizeForGtest(opName) + "_"
                              + sanitizeForGtest(layout) + "_" + sanitizeForGtest(dtype);

    // Test name = bundle directory name (the immediate parent of the .json)
    const auto bundleDirName = jsonPath.parent_path().filename().string();
    const std::string test = sanitizeForGtest(bundleDirName);

    return {suite, test};
}

// Recursively discovers golden bundles under each tier directory.
//
// Per ALMIOPEN-1968, structural problems are hard errors (throw), not warnings:
//   - a stray top-level directory that is not one of the four tiers
//   - a tier directory that is missing or empty
//   - an unparseable bundle .json
//   - a generated test-name collision (names both producing paths)
// The caller registers tests only on success, so any throw aborts startup and
// surfaces the authoring mistake loudly rather than silently dropping coverage.
inline std::vector<DiscoveredBundle> discoverGoldenBundles(
    const std::filesystem::path& goldenDataDir)
{
    std::vector<DiscoveredBundle> bundles;
    std::unordered_map<std::string, std::filesystem::path> nameToPath;

    // Reject stray top-level directories that are not recognized tiers.
    for(const auto& entry : std::filesystem::directory_iterator(goldenDataDir))
    {
        if(!entry.is_directory())
        {
            continue;
        }
        auto dirName = entry.path().filename().string();
        const bool isTier = std::any_of(
            K_TIER_NAMES.begin(), K_TIER_NAMES.end(), [&](const char* tier) {
                return dirName == tier;
            });
        if(!isTier)
        {
            throw std::runtime_error(
                "Unexpected top-level directory '" + dirName
                + "' in golden reference data at " + goldenDataDir.string()
                + "; expected one of: quick, standard, comprehensive, full");
        }
    }

    for(const auto& tierName : K_TIER_NAMES)
    {
        auto tierDir = goldenDataDir / tierName;
        if(!std::filesystem::exists(tierDir) || !std::filesystem::is_directory(tierDir))
        {
            throw std::runtime_error(
                "Golden reference tier directory missing: " + tierDir.string()
                + "; every tier (quick, standard, comprehensive, full) must exist");
        }

        auto jsonPaths = hipdnn_test_sdk::utilities::scanBundleJsonFiles(tierDir);
        if(jsonPaths.empty())
        {
            throw std::runtime_error(
                "Golden reference tier directory is empty: " + tierDir.string()
                + "; every tier must contain at least one bundle");
        }

        for(const auto& jsonPath : jsonPaths)
        {
            // deriveTestName throws on an unparseable .json; let it propagate.
            const DerivedTestName derived = deriveTestName(jsonPath, tierName);

            auto fullName = derived.suiteName + "." + derived.testName;
            auto it = nameToPath.find(fullName);
            if(it != nameToPath.end())
            {
                throw std::runtime_error(
                    "Golden bundle name collision: '" + fullName
                    + "' produced by both:\n  " + it->second.string()
                    + "\n  " + jsonPath.string());
            }
            nameToPath[fullName] = jsonPath;

            bundles.push_back({jsonPath, derived.suiteName, derived.testName});
        }
    }

    return bundles;
}

} // namespace hipdnn_integration_tests::golden
