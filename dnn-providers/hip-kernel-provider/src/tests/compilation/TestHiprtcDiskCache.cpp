// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "compilation/HiprtcDiskCache.hpp"

#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
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
        const std::vector<std::string_view> texts(includeTexts.begin(), includeTexts.end());
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

namespace
{

std::vector<char> sampleCodeObject(std::size_t size, unsigned seed)
{
    std::vector<char> bytes(size);
    for(std::size_t i = 0; i < size; ++i)
    {
        bytes[i] = static_cast<char>((i * 131 + std::size_t{seed} * 7919) & 0xFF);
    }
    return bytes;
}

void writeRaw(const std::filesystem::path& path, const std::vector<char>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::filesystem::path> temporaryFiles(const std::filesystem::path& directory)
{
    std::vector<std::filesystem::path> found;
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(directory, error))
    {
        if(entry.path().extension() == ".tmp")
        {
            found.push_back(entry.path());
        }
    }
    return found;
}

} // namespace

TEST(TestHiprtcCacheEntry, RoundTripsThroughItsHeader)
{
    const auto codeObject = sampleCodeObject(4099, 1);
    const auto entry = hiprtc_cache::encodeEntry(codeObject);
    EXPECT_GT(entry.size(), codeObject.size());
    std::string problem;
    EXPECT_EQ(hiprtc_cache::decodeEntry(entry, problem), codeObject);
    EXPECT_TRUE(problem.empty()) << problem;
}

TEST(TestHiprtcCacheEntry, RejectsATruncatedEntry)
{
    auto entry = hiprtc_cache::encodeEntry(sampleCodeObject(4099, 2));
    entry.pop_back();
    std::string problem;
    EXPECT_TRUE(hiprtc_cache::decodeEntry(entry, problem).empty());
    EXPECT_NE(problem.find("payload bytes"), std::string::npos) << problem;

    // Cut inside the header too.
    entry.resize(20);
    EXPECT_TRUE(hiprtc_cache::decodeEntry(entry, problem).empty());
    EXPECT_FALSE(problem.empty());
}

TEST(TestHiprtcCacheEntry, RejectsAFlippedPayloadBit)
{
    auto entry = hiprtc_cache::encodeEntry(sampleCodeObject(4099, 3));
    entry[entry.size() - 100] = static_cast<char>(entry[entry.size() - 100] ^ 0x10);
    std::string problem;
    EXPECT_TRUE(hiprtc_cache::decodeEntry(entry, problem).empty());
    EXPECT_NE(problem.find("checksum"), std::string::npos) << problem;
}

TEST(TestHiprtcCacheEntry, RejectsAFileWithoutAHeader)
{
    // A bare code object, as an older format stored it.
    const auto bare = sampleCodeObject(4099, 4);
    std::string problem;
    EXPECT_TRUE(hiprtc_cache::decodeEntry(bare, problem).empty());
    EXPECT_NE(problem.find("header"), std::string::npos) << problem;
    EXPECT_TRUE(hiprtc_cache::decodeEntry({}, problem).empty());
    EXPECT_FALSE(problem.empty());
}

TEST(TestHiprtcCacheFile, ReadsWhatItWroteAndLeavesNoTemporaryFile)
{
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hiprtc-file", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const auto path = hiprtc_cache::codeObjectPath(std::string(64, 'b'));
    ASSERT_FALSE(path.empty());

    EXPECT_EQ(hiprtc_cache::readCodeObject(path).outcome, hiprtc_cache::ReadOutcome::ABSENT);

    const auto codeObject = sampleCodeObject(70000, 5);
    ASSERT_TRUE(hiprtc_cache::writeCodeObject(path, codeObject));
    const auto read = hiprtc_cache::readCodeObject(path);
    EXPECT_EQ(read.outcome, hiprtc_cache::ReadOutcome::VALID) << read.problem;
    EXPECT_EQ(read.codeObject, codeObject);
    EXPECT_TRUE(temporaryFiles(path.parent_path()).empty());

    // A rewrite replaces the entry.
    const auto other = sampleCodeObject(1000, 6);
    ASSERT_TRUE(hiprtc_cache::writeCodeObject(path, other));
    EXPECT_EQ(hiprtc_cache::readCodeObject(path).codeObject, other);
}

TEST(TestHiprtcCacheFile, ReadsATruncatedOrCorruptedFileAsCorrupt)
{
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "hiprtc-corrupt", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const auto path = hiprtc_cache::codeObjectPath(std::string(64, 'c'));
    ASSERT_FALSE(path.empty());
    auto entry = hiprtc_cache::encodeEntry(sampleCodeObject(9000, 7));

    // Truncated, as a crash or a full disk leaves it.
    writeRaw(path, std::vector<char>(entry.begin(), entry.begin() + 5000));
    auto read = hiprtc_cache::readCodeObject(path);
    EXPECT_EQ(read.outcome, hiprtc_cache::ReadOutcome::CORRUPT);
    EXPECT_FALSE(read.problem.empty());
    EXPECT_TRUE(read.codeObject.empty());

    // Empty.
    writeRaw(path, {});
    EXPECT_EQ(hiprtc_cache::readCodeObject(path).outcome, hiprtc_cache::ReadOutcome::CORRUPT);

    // Full length, one byte changed.
    entry[entry.size() / 2] = static_cast<char>(entry[entry.size() / 2] + 1);
    writeRaw(path, entry);
    read = hiprtc_cache::readCodeObject(path);
    EXPECT_EQ(read.outcome, hiprtc_cache::ReadOutcome::CORRUPT);
    EXPECT_NE(read.problem.find("checksum"), std::string::npos) << read.problem;
}
