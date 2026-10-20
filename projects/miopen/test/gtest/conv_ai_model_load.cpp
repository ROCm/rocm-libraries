// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <fdeep/fdeep.hpp>
#include <gtest/gtest.h>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>

#include <string>

namespace {

TEST(CPU_ConvAIModelLoad_NONE, BundledFdeepModelsLoad)
{
    const auto db = miopen::GetSystemDbPath();
    ASSERT_TRUE(miopen::fs::exists(db)) << "system db path does not exist: " << db;

    std::size_t immediate_models = 0;
    std::size_t tuning_models    = 0;
    for(const auto& entry : miopen::fs::directory_iterator(db))
    {
        const auto path = entry.path();
        const auto name = path.filename().string();
        if(name.find("metadata") != std::string::npos)
            continue;
        const bool immediate = name.ends_with(".tn.model");
        const bool tuning    = name.ends_with(".ktn.model");
        if(!immediate && !tuning)
            continue;

        SCOPED_TRACE(name);
        // Verification also executes the embedded TensorFlow reference predictions.
        // Include KTN decoders: their LSTMs require explicit hidden and cell inputs.
        EXPECT_NO_THROW(fdeep::load_model(path.string(), true, fdeep::dev_null_logger));
        immediate_models += immediate;
        tuning_models += tuning;
    }
    EXPECT_GT(immediate_models, 0u) << "no immediate-mode models found in " << db;
    EXPECT_GT(tuning_models, 0u) << "no kernel-tuning models found in " << db;
}

} // namespace
#endif
