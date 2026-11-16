// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "compilation/HiprtcDiskCache.hpp"

#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using namespace hip_kernel_provider::compilation;

namespace
{

/// The inputs of one key, each editable by a case.
struct KeyInputs
{
    std::string identity = "hiprtc=7.13;hip-runtime=71326176;comgr=3.0";
    std::string program = "ConvIgemm.cpp";
    std::string source = "extern \"C\" __global__ void ConvIgemm() {}";
    std::vector<std::string> includeNames = {"ConvCommon.hpp", "Tile.hpp"};
    std::vector<std::string> includeTexts
        = {"#pragma once\nconstexpr int A = 1;", "#pragma once\nconstexpr int B = 2;"};
    std::vector<std::string> options = {"-std=c++17", "--offload-arch=gfx942", "-DTILE=64"};

    std::string key() const
    {
        std::vector<std::string_view> texts(includeTexts.begin(), includeTexts.end());
        std::vector<const char*> names;
        names.reserve(includeNames.size());
        for(const auto& name : includeNames)
        {
            names.push_back(name.c_str());
        }
        return hiprtc_cache::codeObjectKey(identity, program, source, texts, names, options);
    }
};

} // namespace

TEST(TestHiprtcCacheKey, IsAStableSha256OfItsInputs)
{
    const KeyInputs inputs;
    const auto key = inputs.key();
    EXPECT_EQ(key.size(), 64u);
    EXPECT_EQ(key.find_first_not_of("0123456789abcdef"), std::string::npos);
    EXPECT_EQ(inputs.key(), key);
}

TEST(TestHiprtcCacheKey, ChangesWithTheCompilerIdentity)
{
    KeyInputs inputs;
    const auto before = inputs.key();
    // A nightly with the same HIPRTC major.minor but another build of the library.
    inputs.identity += ";hiprtc-library=file:libhiprtc.so.7.13.26177-0123456789:985760";
    EXPECT_NE(inputs.key(), before);
}

TEST(TestHiprtcCacheKey, ChangesWithASourceEdit)
{
    KeyInputs inputs;
    const auto before = inputs.key();
    inputs.source += "\n// edited";
    EXPECT_NE(inputs.key(), before);
}

TEST(TestHiprtcCacheKey, ChangesWithAHeaderEdit)
{
    KeyInputs inputs;
    const auto before = inputs.key();
    inputs.includeTexts[1] = "#pragma once\nconstexpr int B = 3;";
    EXPECT_NE(inputs.key(), before);

    KeyInputs renamed;
    renamed.includeNames[0] = "ConvCommon2.hpp";
    EXPECT_NE(renamed.key(), before);
}

TEST(TestHiprtcCacheKey, ChangesWithAnOptionChange)
{
    const auto before = KeyInputs().key();

    KeyInputs value;
    value.options[2] = "-DTILE=128";
    EXPECT_NE(value.key(), before);

    KeyInputs arch;
    arch.options[1] = "--offload-arch=gfx942:sramecc+:xnack-";
    EXPECT_NE(arch.key(), before);

    KeyInputs added;
    added.options.emplace_back("-DEPILOGUE=1");
    EXPECT_NE(added.key(), before);
}

TEST(TestHiprtcCacheKey, KeepsFieldBoundaries)
{
    // The same characters split differently between two fields must not collide.
    KeyInputs first;
    first.options = {"-DA", "=1"};
    KeyInputs second;
    second.options = {"-DA=", "1"};
    EXPECT_NE(first.key(), second.key());
}

TEST(TestHiprtcCacheKey, CompilerIdentityNamesTheCompilerItRuns)
{
    const auto& identity = hiprtc_cache::compilerIdentity();
    EXPECT_NE(identity.find("hiprtc="), std::string::npos) << identity;
    EXPECT_NE(identity.find(";hip-runtime="), std::string::npos) << identity;
    EXPECT_NE(identity.find(";hiprtc-library="), std::string::npos) << identity;
    EXPECT_EQ(identity.find(";hiprtc-library=unknown"), std::string::npos) << identity;
    EXPECT_NE(identity.find(";comgr="), std::string::npos) << identity;
#if !defined(_WIN32)
    // HIPRTC links comgr, so it is loaded whenever the provider is.
    EXPECT_NE(identity.find(";comgr-library="), std::string::npos) << identity;
#endif
    // Computed once per process.
    EXPECT_EQ(&hiprtc_cache::compilerIdentity(), &identity);
}

TEST(TestHiprtcCachePath, LivesUnderTheCacheRootAndIsOffWhenTheCacheIsDisabled)
{
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hiprtc-path", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const std::string key(64, 'a');
    const auto path = hiprtc_cache::codeObjectPath(key);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.filename().string(), key + ".co");
    EXPECT_EQ(path.parent_path().filename().string(), "hiprtc");
    EXPECT_EQ(path.parent_path().parent_path().filename().string(), "hip-kernel-provider");

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disabled(
        "HIPDNN_DISABLE_CACHE", "1");
    EXPECT_TRUE(hiprtc_cache::codeObjectPath(key).empty());
}
