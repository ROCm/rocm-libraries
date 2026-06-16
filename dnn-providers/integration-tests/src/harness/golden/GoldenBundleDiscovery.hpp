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

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

namespace hipdnn_integration_tests::golden
{

struct DiscoveredBundle
{
    std::filesystem::path jsonPath;
    std::string suiteName;
    std::string testName;
};

// Generic recursive file scanner: returns every file under `directory` whose
// extension matches `extension` (e.g. ".json"), sorted for deterministic test
// ordering. It carries NO golden-ref knowledge — meta-file exclusion is layered
// on top by the caller (see isGoldenMetaFile / discoverGoldenBundles). This is
// the clean split called for in ALMIOPEN-1968: a generic scan, with golden-ref
// filtering applied separately rather than baked into the directory walk.
inline std::vector<std::filesystem::path>
    scanFilesByExtension(const std::filesystem::path& directory, const std::string& extension)
{
    std::vector<std::filesystem::path> paths;
    for(const auto& entry : std::filesystem::recursive_directory_iterator(directory))
    {
        if(entry.is_regular_file() && entry.path().extension() == extension)
        {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Golden-ref filter: true for companion metadata files, i.e. either a bare
// `meta.json` or any `{Name}.meta.json`. These are not bundle graphs and must
// be excluded from discovery.
inline bool isGoldenMetaFile(const std::filesystem::path& jsonPath)
{
    if(jsonPath.filename() == "meta.json")
    {
        return true;
    }
    const auto stem = jsonPath.stem().string();
    return stem.size() >= 5 && stem.substr(stem.size() - 5) == ".meta";
}

inline constexpr std::array<const char*, 4> K_TIER_NAMES
    = {"quick", "standard", "comprehensive", "full"};

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

inline std::string dataTypeToShortString(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
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

inline std::string
    nodeAttributesToOperationName(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType)
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
    case NA::RMSNormBackwardAttributes:
        return "RmsNormBwd";
    case NA::ResampleFwdAttributes:
        return "ResampleFwd";
    case NA::LayernormAttributes:
        return "LayerNorm";
    case NA::LayernormBackwardAttributes:
        return "LayerNormBwd";
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

inline std::string
    deriveOperationName(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper)
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

inline bool isPointwiseOp(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType)
{
    return attrType == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes;
}

inline bool isSdpaOp(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes attrType)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    return attrType == NA::SdpaAttributes || attrType == NA::SdpaBackwardAttributes;
}

inline int64_t primaryInputUid(const hipdnn_flatbuffers_sdk::data_objects::Node* node)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    switch(node->attributes_type())
    {
    case NA::ConvolutionFwdAttributes:
        return node->attributes_as_ConvolutionFwdAttributes()->x_tensor_uid();
    case NA::ConvolutionBwdAttributes:
        return node->attributes_as_ConvolutionBwdAttributes()->dy_tensor_uid();
    case NA::ConvolutionWrwAttributes:
        return node->attributes_as_ConvolutionWrwAttributes()->x_tensor_uid();
    case NA::BatchnormInferenceAttributes:
        return node->attributes_as_BatchnormInferenceAttributes()->x_tensor_uid();
    case NA::BatchnormInferenceAttributesVarianceExt:
        return node->attributes_as_BatchnormInferenceAttributesVarianceExt()->x_tensor_uid();
    case NA::BatchnormAttributes:
        return node->attributes_as_BatchnormAttributes()->x_tensor_uid();
    case NA::BatchnormBackwardAttributes:
        return node->attributes_as_BatchnormBackwardAttributes()->x_tensor_uid();
    case NA::SdpaAttributes:
        return node->attributes_as_SdpaAttributes()->q_tensor_uid();
    case NA::SdpaBackwardAttributes:
        return node->attributes_as_SdpaBackwardAttributes()->q_tensor_uid();
    case NA::MatmulAttributes:
        return node->attributes_as_MatmulAttributes()->a_tensor_uid();
    case NA::LayernormAttributes:
        return node->attributes_as_LayernormAttributes()->x_tensor_uid();
    case NA::LayernormBackwardAttributes:
        return node->attributes_as_LayernormBackwardAttributes()->x_tensor_uid();
    case NA::RMSNormAttributes:
        return node->attributes_as_RMSNormAttributes()->x_tensor_uid();
    case NA::RMSNormBackwardAttributes:
        return node->attributes_as_RMSNormBackwardAttributes()->x_tensor_uid();
    case NA::ReductionAttributes:
        return node->attributes_as_ReductionAttributes()->in_tensor_uid();
    case NA::BlockScaleQuantizeAttributes:
        return node->attributes_as_BlockScaleQuantizeAttributes()->x_tensor_uid();
    case NA::BlockScaleDequantizeAttributes:
        return node->attributes_as_BlockScaleDequantizeAttributes()->x_tensor_uid();
    case NA::ResampleFwdAttributes:
        return node->attributes_as_ResampleFwdAttributes()->x_tensor_uid();
    case NA::CustomOpAttributes:
    {
        const auto* uids = node->attributes_as_CustomOpAttributes()->input_tensor_uids();
        if(uids != nullptr && !uids->empty())
        {
            return uids->Get(0);
        }
        return -1;
    }
    default:
        return -1;
    }
}

inline const std::vector<const hipdnn_data_sdk::utilities::TensorLayout*>*
    layoutCandidates(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes opType, size_t ndim)
{
    using TL = hipdnn_data_sdk::utilities::TensorLayout;
    static const std::vector<const TL*> s_sdpaLayouts = {&TL::BHSD, &TL::BSHD};
    static const std::vector<const TL*> s_convLayouts5 = {&TL::NCDHW, &TL::NDHWC};
    static const std::vector<const TL*> s_convLayouts4 = {&TL::NCHW, &TL::NHWC};
    static const std::vector<const TL*> s_convLayouts3 = {&TL::NCL, &TL::NLC};
    if(isSdpaOp(opType))
    {
        return &s_sdpaLayouts;
    }
    if(ndim == 5)
    {
        return &s_convLayouts5;
    }
    if(ndim == 4)
    {
        return &s_convLayouts4;
    }
    if(ndim == 3)
    {
        return &s_convLayouts3;
    }
    return nullptr;
}

inline std::string layoutNameFromGraph(const hipdnn_flatbuffers_sdk::data_objects::Graph* graph)
{
    if(graph == nullptr)
    {
        return "unknown";
    }

    const auto* nodes = graph->nodes();
    if(nodes == nullptr || nodes->empty())
    {
        return "unknown";
    }

    const hipdnn_flatbuffers_sdk::data_objects::Node* primaryNode = nullptr;
    for(flatbuffers::uoffset_t i = 0; i < nodes->size(); ++i)
    {
        const auto* node = nodes->Get(i);
        if(node != nullptr && !isPointwiseOp(node->attributes_type()))
        {
            primaryNode = node;
            break;
        }
    }
    if(primaryNode == nullptr)
    {
        return "unknown";
    }

    const auto uid = primaryInputUid(primaryNode);
    if(uid == -1)
    {
        return "unknown";
    }

    const auto* tensors = graph->tensors();
    if(tensors == nullptr)
    {
        return "unknown";
    }

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* primaryTensor = nullptr;
    for(flatbuffers::uoffset_t i = 0; i < tensors->size(); ++i)
    {
        const auto* t = tensors->Get(i);
        if(t != nullptr && t->uid() == uid)
        {
            primaryTensor = t;
            break;
        }
    }
    if(primaryTensor == nullptr || primaryTensor->dims() == nullptr
       || primaryTensor->strides() == nullptr)
    {
        return "unknown";
    }

    const auto ndim = primaryTensor->dims()->size();
    if(ndim < 3)
    {
        return "unknown";
    }

    std::vector<int64_t> strides(ndim);
    for(flatbuffers::uoffset_t i = 0; i < ndim; ++i)
    {
        strides[i] = primaryTensor->strides()->Get(i);
    }

    const auto order = hipdnn_data_sdk::utilities::extractStrideOrder(strides);

    const auto* candidates = layoutCandidates(primaryNode->attributes_type(), ndim);
    if(candidates == nullptr)
    {
        return "unknown";
    }

    for(const auto* layout : *candidates)
    {
        if(order == layout->strideOrder)
        {
            std::string name = layout->name;
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return name;
        }
    }
    return "unknown";
}

inline std::string
    deriveLayoutFromGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper)
{
    return layoutNameFromGraph(&wrapper.getGraph());
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

    nlohmann::json graphJson;
    try
    {
        graphJson = nlohmann::json::parse(file);
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Failed to parse bundle JSON " + jsonPath.string() + ": "
                                 + e.what());
    }
    flatbuffers::FlatBufferBuilder builder;
    auto offset = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
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
// Scans a single tier directory for bundle .json files: a recursive scan with
// the golden-ref meta-file filter layered on top. This is the "recursive .json
// scan per tier" the ticket (ALMIOPEN-1968) describes. It deliberately does NOT
// own the root-level rules (stray-dir rejection, all-tiers-exist, cross-tier
// collision) — those need visibility across all tiers and live in
// discoverGoldenBundles, which is why that entry point takes the data root
// rather than a single tierDir.
inline std::vector<std::filesystem::path> scanTier(const std::filesystem::path& tierDir)
{
    std::vector<std::filesystem::path> jsonPaths;
    for(auto& p : scanFilesByExtension(tierDir, ".json"))
    {
        if(!isGoldenMetaFile(p))
        {
            jsonPaths.push_back(std::move(p));
        }
    }
    return jsonPaths;
}

//   - a generated test-name collision (names both producing paths)
// The caller registers tests only on success, so any throw aborts startup and
// surfaces the authoring mistake loudly rather than silently dropping coverage.
inline std::vector<DiscoveredBundle>
    discoverGoldenBundles(const std::filesystem::path& goldenDataDir)
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
        const bool isTier = std::any_of(K_TIER_NAMES.begin(),
                                        K_TIER_NAMES.end(),
                                        [&](const char* tier) { return dirName == tier; });
        if(!isTier)
        {
            throw std::runtime_error("Unexpected top-level directory '" + dirName
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

        const auto jsonPaths = scanTier(tierDir);
        if(jsonPaths.empty())
        {
            throw std::runtime_error("Golden reference tier directory is empty: " + tierDir.string()
                                     + "; every tier must contain at least one bundle");
        }

        for(const auto& jsonPath : jsonPaths)
        {
            const DerivedTestName derived = deriveTestName(jsonPath, tierName);

            auto fullName = derived.suiteName + "." + derived.testName;
            auto it = nameToPath.find(fullName);
            if(it != nameToPath.end())
            {
                throw std::runtime_error("Golden bundle name collision: '" + fullName
                                         + "' produced by both:\n  " + it->second.string() + "\n  "
                                         + jsonPath.string());
            }
            nameToPath[fullName] = jsonPath;

            bundles.push_back({jsonPath, derived.suiteName, derived.testName});
        }
    }

    return bundles;
}

} // namespace hipdnn_integration_tests::golden
