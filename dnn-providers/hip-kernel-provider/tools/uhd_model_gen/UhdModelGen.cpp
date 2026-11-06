// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file UhdModelGen.cpp
/// @brief Emits the pointwise_model pack's tree_data UHD and its hand-written model artifact.
///
/// Both are generated so `features_hash` comes from the runtime's own function and cannot
/// drift from the signature. Output must be byte-identical run to run, or the build
/// re-triggers everything downstream.

#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

namespace fbs = hipdnn_flatbuffers_sdk::data_objects;

/// Only `$kernel.*` is usable: the pointwise matcher binds tensor uids, not sizes.
const std::vector<nlohmann::json> SIGNATURE = {"$kernel.block_size"};

/// The fixture UED's catalog-ranking model, resolved before generation.
constexpr const char* UHD_ID = "5a1c0000-0000-4000-8000-000000000002";
constexpr const char* MODEL_FILE = "pointwise_model.bin";
constexpr const char* UHD_FILE = "pointwise_model.uhd.json";

/// Prefers the small block, the opposite of the native scorer and the declared order, so a
/// test can tell this model's ranking apart from both.
///   block_size <= 96  -> 9.0   (the 64 kernel)
///   block_size >  96  -> 1.0   (the 256 kernel)
void writeModel(const std::filesystem::path& path, const std::string& featuresHash)
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int32_t> featureIndices = {0, 0, 0};
    const std::vector<double> thresholds = {96.0, 0.0, 0.0};
    const std::vector<int32_t> leftChildren = {1, -1, -1};
    const std::vector<int32_t> rightChildren = {2, -1, -1};
    const std::vector<double> leafValues = {0.0, 9.0, 1.0};
    const std::vector<uint8_t> defaultLeft = {1, 1, 1};
    const std::vector<uint8_t> decisionLte = {1, 1, 1};

    const auto tree = fbs::CreateGbdtTree(builder,
                                          builder.CreateVector(featureIndices),
                                          builder.CreateVector(thresholds),
                                          builder.CreateVector(leftChildren),
                                          builder.CreateVector(rightChildren),
                                          builder.CreateVector(leafValues),
                                          builder.CreateVector(defaultLeft),
                                          builder.CreateVector(decisionLte));

    const std::vector<flatbuffers::Offset<fbs::GbdtTree>> trees = {tree};
    const std::vector<flatbuffers::Offset<flatbuffers::String>> arches;

    const auto model = fbs::CreateGbdtModel(
        builder,
        builder.CreateVector(trees),
        static_cast<int32_t>(SIGNATURE.size()),
        builder.CreateString(featuresHash),
        0.0, // base_score
        1.0, // learning_rate: leaf values are already final, as LightGBM emits them
        builder.CreateString("hand-authored"),
        // Fixed, not the build time, so the artifact is reproducible.
        builder.CreateString("1970-01-01T00:00:00Z"),
        0, // num_training_samples: nothing was measured
        builder.CreateString("regression"),
        // Empty: nothing reads as out-of-distribution (RFC 0019 §9.3) on any device.
        builder.CreateVector(arches),
        builder.CreateString("0.0.0"));

    builder.Finish(model, fbs::GbdtModelIdentifier());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
              static_cast<std::streamsize>(builder.GetSize()));
}

/// The `trained_against` record for the descriptor set whose `default` catalog ranker
/// names this UHD.
nlohmann::json snapshotProvenance(const std::vector<std::filesystem::path>& roots)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    const auto dependency = [](const auto& descriptor) {
        return nlohmann::json{{"id", toString(descriptor.id)},
                              {"revision",
                               std::to_string(descriptor.revision.major) + "."
                                   + std::to_string(descriptor.revision.minor)}};
    };
    for(const auto& set : resolveDescriptorSets(loadDescriptorCatalog(roots)))
    {
        // Each arch maps to a list of UHDs, one per metric.
        const auto models = set.engine.sortKernelCatalog.find("default");
        if(models == set.engine.sortKernelCatalog.end()
           || std::none_of(models->second.begin(), models->second.end(), [](const auto& id) {
                  return toString(id) == UHD_ID;
              }))
        {
            continue;
        }
        auto matchers = nlohmann::json::array();
        for(const auto& matcher : set.matchers)
        {
            matchers.push_back(dependency(matcher));
        }
        return {{"ued", dependency(set.engine)},
                {"kmd", dependency(set.schema)},
                {"umd", std::move(matchers)}};
    }
    throw std::runtime_error(
        "Model fixture UED did not resolve from the supplied descriptor roots");
}

/// Writes the UHD text with fields in a fixed order, for reproducible output.
void writeUhd(const std::filesystem::path& path,
              const std::string& featuresHash,
              const nlohmann::json& provenance)
{
    std::ostringstream json;
    json << "{\n";
    json << "  \"version\": \"1.0\",\n";
    json << R"(  "id": ")" << UHD_ID << "\",\n";
    json << "  \"name\": \"pointwise model selector\",\n";
    json << "  \"adapter\": \"tree_data\",\n";

    json << "  \"features_signature\": " << nlohmann::json(SIGNATURE).dump() << ",\n";
    json << "  \"trained_against\": " << provenance.dump() << ",\n";

    json << R"(  "features_hash": ")" << featuresHash << "\",\n";
    json << "  \"objective\": \"max\",\n";

    // Uncalibrated and metric-less: the leaf values only order kernels, so this is the
    // engine's metric-less `sort_kernel_catalog` ranker (RFC 0019 §4.4, §12.3).
    json << "  \"score\": { \"calibrated\": false, \"transform\": \"identity\" },\n";

    // Relative to this file's directory, wherever the pack is staged.
    json << R"(  "tree_data": { "artifact": ")" << MODEL_FILE << "\" }\n";
    json << "}\n";

    const auto text = json.str();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "usage: uhd_model_gen <output-directory> <descriptor-root>...\n";
        return 1;
    }

    try
    {
        const std::filesystem::path outputDir(argv[1]);
        std::vector<std::filesystem::path> roots;
        for(int i = 2; i < argc; ++i)
        {
            roots.emplace_back(argv[i]);
        }
        const auto provenance = snapshotProvenance(roots);
        std::filesystem::create_directories(outputDir);

        // The runtime rejects a UHD whose hash disagrees with its signature.
        const std::string featuresHash
            = hipdnn_plugin_sdk::uhd::FeatureExtractor::computeHash(SIGNATURE);

        writeModel(outputDir / MODEL_FILE, featuresHash);
        writeUhd(outputDir / UHD_FILE, featuresHash, provenance);
    }
    catch(const std::exception& error)
    {
        std::cerr << "uhd_model_gen: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
