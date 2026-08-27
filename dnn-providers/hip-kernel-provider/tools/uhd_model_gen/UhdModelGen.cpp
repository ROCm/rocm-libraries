// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file UhdModelGen.cpp
/// @brief Emits the trained-model artifacts the pointwise_model descriptor pack names.
///
/// A MODEL heuristic (RFC 0019 §7) points at a `.uhd.fb` that in turn points at a model
/// artifact, and neither is text. `HkpPackaging.cmake` carries `.json` and `.cpp` only, so
/// something has to produce them; committing them instead would put two opaque binaries
/// under review and let them drift silently the moment the feature signature changes.
///
/// The "training" here is a two-leaf tree written by hand. That is the point: the pack
/// exists to prove a model-backed UHD is loaded and ranks the catalog, not to be a good
/// model. A real one comes from `tools/uhd_gen` and a benchmark corpus.
///
/// Output is byte-identical run to run -- no timestamps, no ordering from a hash map --
/// because a build step whose output churns re-triggers everything downstream of it.
///
/// Links the flatbuffers SDK and the plugin SDK's feature extractor. Deliberately not the
/// test SDK, whose `GbdtModelTestBuilder` does much the same thing: a shipped tool must not
/// depend on test scaffolding. The feature extractor is a different matter and is used on
/// purpose -- computing the signature hash with the same code the runtime validates against
/// is what makes a mismatch impossible by construction rather than by inspection.

#include <hipdnn_plugin_sdk/ingestor/uhd/FeatureExtractor.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

namespace fbs = hipdnn_flatbuffers_sdk::data_objects;

/// The one feature. `$kernel.*` is all this model can see: the pointwise graph matcher
/// binds tensor uids rather than sizes, so there is no useful `$q.*` to split on. A real
/// pack would want problem features too -- read this as a demonstration of the mechanism,
/// not of a well-chosen feature set.
const std::vector<std::string> SIGNATURE = {"$kernel.block_size"};

constexpr const char* UHD_ID = "5e8d21a7-9c34-4b16-8f70-2a63d8e14c90";
constexpr const char* MODEL_FILE = "pointwise_model.bin";
constexpr const char* UHD_FILE = "pointwise_model.uhd.fb";

/// Prefers the small block where the native scorer (`hipkernel.pointwise.score`, which
/// returns `block_size`) prefers the large one. The disagreement is the whole point: it is
/// what lets a test tell a ranking model apart from the declared-order fallback, which the
/// pack's kernel ids are arranged to send the other way.
///
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
        // Fixed, not the build time: a changing timestamp would rewrite the artifact on
        // every configure and re-trigger everything staged downstream of it.
        builder.CreateString("1970-01-01T00:00:00Z"),
        0, // num_training_samples: nothing was measured
        builder.CreateString("regression"),
        // Empty: no architecture was trained against, so nothing should read as
        // out-of-distribution (RFC 0019 §9.3) on any device.
        builder.CreateVector(arches),
        builder.CreateString("0.0.0"));

    builder.Finish(model, fbs::GbdtModelIdentifier());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
              static_cast<std::streamsize>(builder.GetSize()));
}

void writeUhd(const std::filesystem::path& path, const std::string& featuresHash)
{
    flatbuffers::FlatBufferBuilder builder;

    const auto id = builder.CreateString(UHD_ID);
    const auto name = builder.CreateString("pointwise model selector");
    const auto hash = builder.CreateString(featuresHash);
    const auto objective = builder.CreateString("max");
    const auto units = builder.CreateString("score");
    // The leaf values are the score itself, so nothing has to be undone to read them.
    const auto transform = builder.CreateString("identity");
    // Relative: resolved against this file's own directory, wherever the pack is staged.
    const auto artifact = builder.CreateString(MODEL_FILE);

    std::vector<flatbuffers::Offset<flatbuffers::String>> signature;
    signature.reserve(SIGNATURE.size());
    for(const auto& entry : SIGNATURE)
    {
        signature.push_back(builder.CreateString(entry));
    }

    // Not calibrated: 9.0 and 1.0 are ordering, not throughput, so this score means nothing
    // against another engine's (RFC 0019 §12.3).
    const auto score = fbs::CreateUhdScoreMetadata(builder, units, false, transform);

    const auto uhd = fbs::CreateUHD(builder,
                                    id,
                                    name,
                                    fbs::UhdAdapter::TREE_DATA,
                                    0, // derived: the one feature is a bare field reference
                                    builder.CreateVector(signature),
                                    hash,
                                    objective,
                                    score,
                                    artifact);

    builder.Finish(uhd, fbs::UHDIdentifier());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
              static_cast<std::streamsize>(builder.GetSize()));
}

} // namespace

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "usage: uhd_model_gen <output-directory>\n";
        return 1;
    }

    try
    {
        const std::filesystem::path outputDir(argv[1]);
        std::filesystem::create_directories(outputDir);

        // The runtime rejects a UHD whose declared hash disagrees with the signature it
        // carries, so computing it here with the runtime's own function is what keeps the
        // pair consistent by construction.
        const std::string featuresHash
            = hipdnn_plugin_sdk::ingestor::uhd::FeatureExtractor::computeHash(SIGNATURE);

        writeModel(outputDir / MODEL_FILE, featuresHash);
        writeUhd(outputDir / UHD_FILE, featuresHash);
    }
    catch(const std::exception& error)
    {
        std::cerr << "uhd_model_gen: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
