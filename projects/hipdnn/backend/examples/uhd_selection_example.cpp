// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file uhd_selection_example.cpp
 * @brief End-to-end example of UHD (Universal Heuristic Descriptor) selection flow.
 *
 * This example demonstrates the complete RFC 0019 UHD selection flow:
 * 1. Building a GBDT model artifact with FlatBuffers
 * 2. Registering an engine with UHD configuration and kernel candidates
 * 3. Extracting features from device/query/kernel context
 * 4. Scoring candidates with derived values
 * 5. Selecting the best kernel based on objective and score transforms
 *
 * Usage:
 *   cd <workspace>/projects/hipdnn/build
 *   ./examples/uhd_selection_example
 *
 * Expected output:
 *   - Model loaded successfully with hash validation
 *   - Features extracted with derived values
 *   - Candidates scored and ranked
 *   - Best kernel selected based on UHD objective
 */

#include <heuristics/uhd/EngineRegistry.hpp>
#include <heuristics/uhd/SelectionEngine.hpp>
#include <heuristics/uhd/adapters/TreeDataAdapter.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace hipdnn_backend::heuristics::uhd;
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

// ========== Helper: Build a GBDT Model Artifact ==========

/**
 * @brief Build a simple GBDT model for demonstration.
 *
 * Creates a 2-tree GBDT model that scores based on:
 * - Feature 0: tile_m (kernel tiling parameter)
 * - Feature 1: batch (query parameter)
 *
 * Tree 1: if (tile_m <= 96) then 2.0 else 1.0
 * Tree 2: if (batch <= 16) then 0.5 else 1.5
 *
 * Base score: 0.0, Learning rate: 1.0
 */
std::vector<uint8_t> buildExampleGbdtModel()
{
    flatbuffers::FlatBufferBuilder builder(1024);

    // Tree 1: split on feature 0 (tile_m)
    //   [0] tile_m <= 96.0
    //   / \
    // [1]  [2]
    // 2.0  1.0
    std::vector<int32_t> tree1_features = {0, 0, 0};
    std::vector<double> tree1_thresholds = {96.0, 0.0, 0.0};
    std::vector<int32_t> tree1_left = {1, -1, -1};
    std::vector<int32_t> tree1_right = {2, -1, -1};
    std::vector<double> tree1_values = {0.0, 2.0, 1.0};
    std::vector<uint8_t> tree1_default = {1, 1, 1};

    auto tree1 = fb::CreateGbdtTree(builder,
                                    builder.CreateVector(tree1_features),
                                    builder.CreateVector(tree1_thresholds),
                                    builder.CreateVector(tree1_left),
                                    builder.CreateVector(tree1_right),
                                    builder.CreateVector(tree1_values),
                                    builder.CreateVector(tree1_default));

    // Tree 2: split on feature 1 (batch)
    //   [0] batch <= 16.0
    //   / \
    // [1]  [2]
    // 0.5  1.5
    std::vector<int32_t> tree2_features = {1, 0, 0};
    std::vector<double> tree2_thresholds = {16.0, 0.0, 0.0};
    std::vector<int32_t> tree2_left = {1, -1, -1};
    std::vector<int32_t> tree2_right = {2, -1, -1};
    std::vector<double> tree2_values = {0.0, 0.5, 1.5};
    std::vector<uint8_t> tree2_default = {1, 1, 1};

    auto tree2 = fb::CreateGbdtTree(builder,
                                    builder.CreateVector(tree2_features),
                                    builder.CreateVector(tree2_thresholds),
                                    builder.CreateVector(tree2_left),
                                    builder.CreateVector(tree2_right),
                                    builder.CreateVector(tree2_values),
                                    builder.CreateVector(tree2_default));

    std::vector<flatbuffers::Offset<fb::GbdtTree>> trees = {tree1, tree2};

    // Compute hash for features signature to match UHD config
    std::vector<std::string> signature = {"\"$kernel.tile_m\"", "\"$q.batch\""};
    std::string hashStr = FeatureExtractor::computeHash(signature);

    auto featuresHash = builder.CreateString(hashStr);
    auto modelVersion = builder.CreateString("1.0.0");

    auto model = fb::CreateGbdtModel(builder,
                                     builder.CreateVector(trees),          // trees
                                     2,                                    // num_features
                                     featuresHash,                         // features_hash
                                     0.0,                                  // base_score
                                     1.0,                                  // learning_rate
                                     0,                                    // framework
                                     0,                                    // training_date
                                     0,                                    // num_training_samples
                                     0,                                    // training_objective
                                     0,                                    // training_arches
                                     modelVersion);                        // model_version

    builder.Finish(model, fb::GbdtModelIdentifier());

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    return std::vector<uint8_t>(buf, buf + size);
}

// ========== Main Example ==========

int main()
{
    std::cout << "=== UHD Selection Flow Example ===" << std::endl;
    std::cout << std::endl;

    // Step 1: Build a GBDT model artifact
    std::cout << "[Step 1] Building GBDT model artifact..." << std::endl;
    auto modelBuffer = buildExampleGbdtModel();
    std::cout << "  Model size: " << modelBuffer.size() << " bytes" << std::endl;
    std::cout << "  Trees: 2, Features: 2 (tile_m, batch)" << std::endl;
    std::cout << std::endl;

    // Step 2: Register an engine with UHD configuration
    std::cout << "[Step 2] Registering engine with multi-arch UHD configs..." << std::endl;

    EngineEntry entry;
    entry.engineId = 1000;
    entry.engineName = "ExampleConvEngine";

    // RFC 0019 §3.1: Demonstrate multi-arch UHD registration
    // Each architecture can have its own tuned heuristic

    // GFX942 (MI300X): Tuned model for MI300 series
    UhdConfig gfx942Config;
    gfx942Config.uhdId = "example_uhd_gfx942_v1";
    gfx942Config.name = "GFX942 Tuned GBDT";
    gfx942Config.adapterType = "tree_data";
    gfx942Config.objective = "max";
    gfx942Config.featuresSignature = {"\"$kernel.tile_m\"", "\"$q.batch\""};
    gfx942Config.featuresHash = FeatureExtractor::computeHash(gfx942Config.featuresSignature);
    gfx942Config.derived = {
        {"num_tiles", "{\"ceil_div\": [\"$q.seqlen_q\", \"$kernel.tile_m\"]}"}
    };
    gfx942Config.scoreUnits = "tflops";
    gfx942Config.scoreCalibrated = true;
    gfx942Config.scoreTransform = "log1p";
    entry.sortKernelCatalog["gfx942"] = gfx942Config;

    // GFX950 (MI355X): Different tuning for newer arch
    UhdConfig gfx950Config;
    gfx950Config.uhdId = "example_uhd_gfx950_v1";
    gfx950Config.name = "GFX950 Tuned GBDT";
    gfx950Config.adapterType = "tree_data";
    gfx950Config.objective = "max";
    gfx950Config.featuresSignature = {"\"$kernel.tile_m\"", "\"$q.batch\""};
    gfx950Config.featuresHash = FeatureExtractor::computeHash(gfx950Config.featuresSignature);
    gfx950Config.derived = {
        {"num_tiles", "{\"ceil_div\": [\"$q.seqlen_q\", \"$kernel.tile_m\"]}"}
    };
    gfx950Config.scoreUnits = "tflops";
    gfx950Config.scoreCalibrated = true;
    gfx950Config.scoreTransform = "sqrt"; // Different transform for MI355X
    entry.sortKernelCatalog["gfx950"] = gfx950Config;

    // Default fallback: Conservative heuristic for unknown archs
    UhdConfig defaultConfig;
    defaultConfig.uhdId = "example_uhd_default_v1";
    defaultConfig.name = "Default Conservative GBDT";
    defaultConfig.adapterType = "tree_data";
    defaultConfig.objective = "max";
    defaultConfig.featuresSignature = {"\"$kernel.tile_m\"", "\"$q.batch\""};
    defaultConfig.featuresHash = FeatureExtractor::computeHash(defaultConfig.featuresSignature);
    defaultConfig.derived = {
        {"num_tiles", "{\"ceil_div\": [\"$q.seqlen_q\", \"$kernel.tile_m\"]}"}
    };
    defaultConfig.scoreUnits = "tflops";
    defaultConfig.scoreCalibrated = false; // Not calibrated across archs
    defaultConfig.scoreTransform = "identity";
    entry.sortKernelCatalog["default"] = defaultConfig;

    // Backward compatibility: populate legacy field with default
    entry.uhdConfig = defaultConfig;

    // Register 3 kernel candidates with different tile_m values
    KernelCandidate k1;
    k1.kernelId = 101;
    k1.priority = 0;
    k1.metadata = {{"tile_m", 64.0}};

    KernelCandidate k2;
    k2.kernelId = 102;
    k2.priority = 0;
    k2.metadata = {{"tile_m", 128.0}};

    KernelCandidate k3;
    k3.kernelId = 103;
    k3.priority = 0;
    k3.metadata = {{"tile_m", 256.0}};

    entry.candidates = {k1, k2, k3};

    // Write model to a temporary file so the registry can load it via modelArtifactPath
    // In production, UHDs reference persistent model files; this example uses a temp file
    const char* tmpPath = "/tmp/example_uhd_model.gbdt";
    {
        std::ofstream out(tmpPath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(modelBuffer.data()),
                  static_cast<std::streamsize>(modelBuffer.size()));
    }

    // Set artifact paths for all arch UHDs to point at the same model
    // (In production, each arch could have its own separately-trained model)
    gfx942Config.modelArtifactPath = tmpPath;
    gfx950Config.modelArtifactPath = tmpPath;
    defaultConfig.modelArtifactPath = tmpPath;

    // Re-assign configs with artifact paths
    entry.sortKernelCatalog["gfx942"] = gfx942Config;
    entry.sortKernelCatalog["gfx950"] = gfx950Config;
    entry.sortKernelCatalog["default"] = defaultConfig;
    entry.uhdConfig = defaultConfig; // Legacy path

    EngineRegistry::instance().registerEngine(std::move(entry));

    std::cout << "  Engine ID: 1000" << std::endl;
    std::cout << "  Registered UHDs:" << std::endl;
    std::cout << "    - gfx942: " << gfx942Config.uhdId << " (transform=log1p)" << std::endl;
    std::cout << "    - gfx950: " << gfx950Config.uhdId << " (transform=sqrt)" << std::endl;
    std::cout << "    - default: " << defaultConfig.uhdId << " (transform=identity)" << std::endl;
    std::cout << "  Candidates: 3 (tile_m = 64, 128, 256)" << std::endl;
    std::cout << "  Objective: max (higher score = better)" << std::endl;
    std::cout << std::endl;

    // Step 3: Set up query context (device + query parameters)
    std::cout << "[Step 3] Setting up query context..." << std::endl;

    // Demonstrate arch resolution: add DEVICE_ARCH_KEY to deviceVars
    // This triggers RFC 0019 §8.3 resolution: exact → "default" → nullopt

    // Test case 1: gfx942 (exact match exists)
    const std::string testArch = "gfx942";

    FeatureExtractionContext::ValueMap deviceVars = {
        {"cu_count", int64_t{120}},
        {"multi_processor_count", int64_t{120}},
        {"total_global_mem", int64_t{32LL * 1024 * 1024 * 1024}},
        {"device_id", int64_t{0}},
        {"arch", testArch}, // DEVICE_ARCH_KEY = "arch" triggers UHD arch resolution
    };

    FeatureExtractionContext::ValueMap queryVars = {
        {"batch", 32.0},      // Feature 1 in model
        {"seqlen_q", 2048.0}, // Used in derived value
        {"seqlen_k", 2048.0},
        {"num_heads", 32.0},
        {"head_dim", 128.0},
    };

    std::cout << "  Device: arch=" << testArch << ", cu_count=120, total_mem=32GB" << std::endl;
    std::cout << "  Query: batch=32, seqlen_q=2048, heads=32, hdim=128" << std::endl;
    std::cout << std::endl;

    // Step 4: Run selection with arch resolution
    std::cout << "[Step 4] Running UHD selection (arch=" << testArch << ")..." << std::endl;

    auto result = SelectionEngine::select(1000, deviceVars, queryVars);

    if(!result.applied)
    {
        std::cerr << "ERROR: Selection failed - " << result.fallbackReason << std::endl;
        return 1;
    }

    std::cout << "  Selection applied: YES" << std::endl;
    std::cout << "  UHD resolved: " << (result.trace.uhdId.empty() ? "(none)" : result.trace.uhdId) << std::endl;
    std::cout << "  Candidates scored: " << result.sortedKernelIds.size() << std::endl;
    std::cout << std::endl;

    // Step 5: Display results
    std::cout << "[Step 5] Selection Results:" << std::endl;
    if(result.bestKernelId.has_value())
    {
        std::cout << "  Best kernel: " << result.bestKernelId.value() << std::endl;
    }
    else
    {
        std::cout << "  Best kernel: (none)" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "  Ranked candidates (best to worst):" << std::endl;
    for(size_t i = 0; i < result.sortedKernelIds.size(); ++i)
    {
        int64_t kernelId = result.sortedKernelIds[i];

        // Find the candidate to show its tile_m
        const auto& candidates = EngineRegistry::instance().getEngine(1000)->candidates;
        auto it = std::find_if(candidates.begin(), candidates.end(),
                               [kernelId](const KernelCandidate& c) { return c.kernelId == kernelId; });

        if(it != candidates.end())
        {
            double tileM = it->metadata.at("tile_m");
            std::cout << "    " << (i + 1) << ". Kernel " << kernelId
                     << " (tile_m=" << tileM << ")" << std::endl;
        }
    }

    std::cout << std::endl;

    // Explain the scoring and arch resolution
    std::cout << "[Explanation]" << std::endl;
    std::cout << "  Arch Resolution (RFC 0019 §8.3):" << std::endl;
    std::cout << "    - Requested arch: " << testArch << std::endl;
    std::cout << "    - Resolution path: exact match → gfx942Config" << std::endl;
    std::cout << "    - UHD used: " << gfx942Config.uhdId << std::endl;
    std::cout << "    - Score transform: " << gfx942Config.scoreTransform << std::endl;
    std::cout << std::endl;
    std::cout << "  Model scoring logic:" << std::endl;
    std::cout << "    Tree 1: if (tile_m <= 96) then +2.0 else +1.0" << std::endl;
    std::cout << "    Tree 2: if (batch <= 16) then +0.5 else +1.5" << std::endl;
    std::cout << "    Score = base_score(0.0) + tree1_output + tree2_output" << std::endl;
    std::cout << std::endl;
    std::cout << "  For batch=32 (> 16), tree2 always outputs 1.5" << std::endl;
    std::cout << "  So the ranking depends on tree1 (tile_m):" << std::endl;
    std::cout << "    - tile_m=64:  score = 0.0 + 2.0 + 1.5 = 3.5 (best)" << std::endl;
    std::cout << "    - tile_m=128: score = 0.0 + 1.0 + 1.5 = 2.5" << std::endl;
    std::cout << "    - tile_m=256: score = 0.0 + 1.0 + 1.5 = 2.5" << std::endl;
    std::cout << std::endl;
    std::cout << "  After " << gfx942Config.scoreTransform << " transform:" << std::endl;
    std::cout << "    - log1p(3.5) ≈ 1.504" << std::endl;
    std::cout << "    - log1p(2.5) ≈ 1.253" << std::endl;
    std::cout << std::endl;
    std::cout << "  Objective is 'max', so highest score wins → Kernel 101 (tile_m=64)" << std::endl;
    std::cout << std::endl;
    std::cout << "  Try changing testArch to \"gfx950\", \"gfx1100\", or \"default\"" << std::endl;
    std::cout << "  to see different UHD resolution paths!" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Example Complete ===" << std::endl;

    return 0;
}
