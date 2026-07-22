/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Unit tests for AIHPBLAS-3751 Phase 1 safe tuning-file replay. New rows persist
// solution_name alongside solution_index so runtime lookup can verify that an
// index still identifies the tuned kernel. A mismatch is rejected and falls
// back to default selection; the library does not relocate kernels by name.
//
// This file is split into two parts by build requirement:
//
// 1. Identity-validation and persisted-name normalization coverage (always
//    built/run). Their lightweight headers are available to the normal client
//    target, so these tests run in the default "smoke" CI lane.
//
// 2. Parser and OverrideMap coverage, gated behind
//    CODE_COVERAGE like auxiliary_gtest.cpp's other rocBLASLt-internal-header tests
//    (e.g. testing_aux_tensile_host_func in testing_auxiliary.hpp). UserDrivenTuningParser.hpp
//    is a private rocBLASLt header whose transitive includes (tensile_host.hpp ->
//    handle.h -> rocblaslt.h -> ...) reach into several more private directories
//    (library/src/amd_detail/rocblaslt/{src/include,include}, rocroller/include,
//    tensilelite/rocisa, shared/origami/include) that clients/CMakeLists.txt only
//    adds to this target's include path under HIPBLASLT_ENABLE_COVERAGE. That is not
//    just a missing "-I": library/src/amd_detail/rocblaslt/src/include/utility.hpp
//    and clients/common/include/utility.hpp are two DIFFERENT files with the same
//    name, so unconditionally adding the private directory to this target's shared
//    include path (to reach it from this one file) would risk silently resolving some
//    other translation unit's `#include "utility.hpp"` to the wrong one. That risk -
//    not merely convenience - is why this part stays behind the existing convention
//    rather than forcing it into the default build.
//
// Parser file tests load into a local OverrideMap through
// loadContractionProblemsFromFile(), avoiding process-global singleton state.
// End-to-end replay still requires a live Tensile solution library and belongs
// in hardware/integration coverage.

#include <gtest/gtest.h>

#include "UserDrivenTuningTypes.hpp"
#include "argument_model.hpp"

// getSolutionNameFromData() (tensile_host.cpp), used by the ext API's
// GemmInstance::getSolutionName(), can append a " (Custom tuning: GSU: x, WGM: y)"
// display suffix. If that decorated string were persisted as solution_name,
// validation against the plain index->name path could never match it, so
// hipblaslt_strip_custom_tuning_suffix() must remove it before the value reaches
// HIPBLASLT_TUNING_FILE. Always built/run - see file header comment part 1.
TEST(UserDrivenTuningParser, smoke_StripsCustomTuningSuffixFromPersistedName)
{
    EXPECT_EQ(hipblaslt_strip_custom_tuning_suffix(
                  "Cijk_Alik_Bljk_HHS_BH_MT128x128x16 (Custom tuning: GSU: 4)"),
              "Cijk_Alik_Bljk_HHS_BH_MT128x128x16");
    EXPECT_EQ(hipblaslt_strip_custom_tuning_suffix(
                  "Cijk_Alik_Bljk_HHS_BH_MT128x128x16 (Custom tuning: GSU: 4, WGM: 2)"),
              "Cijk_Alik_Bljk_HHS_BH_MT128x128x16");
    // Names without the suffix must pass through unchanged.
    EXPECT_EQ(hipblaslt_strip_custom_tuning_suffix("Cijk_Alik_Bljk_HHS_BH_MT128x128x16"),
              "Cijk_Alik_Bljk_HHS_BH_MT128x128x16");
    EXPECT_EQ(hipblaslt_strip_custom_tuning_suffix(""), "");
}

// Named entries are accepted only when the stored index resolves and its current
// name exactly matches the recorded name. Build identity is irrelevant when a
// name is available.
TEST(UserDrivenTuningParser, smoke_ValidatesNamedSolutionIdentity)
{
    const TensileLite::TunedSolution tuned{333, "ExpectedKernel"};

    EXPECT_TRUE(TensileLite::isTunedSolutionIdentityValid(tuned, true, "ExpectedKernel", false));
    EXPECT_TRUE(TensileLite::isTunedSolutionIdentityValid(tuned, true, "ExpectedKernel", true));
    EXPECT_FALSE(TensileLite::isTunedSolutionIdentityValid(tuned, true, "DifferentKernel", true));
    EXPECT_FALSE(TensileLite::isTunedSolutionIdentityValid(tuned, false, "ExpectedKernel", true));
}

// Legacy rows have no name to authorize an index, so both a resolvable index and
// a matching build identity are required.
TEST(UserDrivenTuningParser, smoke_ValidatesLegacySolutionIdentity)
{
    const TensileLite::TunedSolution legacy{333, ""};

    EXPECT_TRUE(TensileLite::isTunedSolutionIdentityValid(legacy, true, "IgnoredKernel", true));
    EXPECT_FALSE(TensileLite::isTunedSolutionIdentityValid(legacy, true, "IgnoredKernel", false));
    EXPECT_FALSE(TensileLite::isTunedSolutionIdentityValid(legacy, false, "IgnoredKernel", true));
}

TEST(UserDrivenTuningParser, smoke_ValidatesTuningFileVersionHeader)
{
    EXPECT_TRUE(TensileLite::isTuningFileVersionCurrent("Git Version: abc123", "abc123"));
    EXPECT_FALSE(TensileLite::isTuningFileVersionCurrent("Git Version: old", "abc123"));
    EXPECT_FALSE(TensileLite::isTuningFileVersionCurrent("Git Version: abc123-extra", "abc123"));
    EXPECT_FALSE(TensileLite::isTuningFileVersionCurrent("Version: abc123", "abc123"));
    EXPECT_FALSE(TensileLite::isTuningFileVersionCurrent("", "abc123"));
}

// See file header comment part 2 for why this part is CODE_COVERAGE-only.
#ifdef CODE_COVERAGE
#include "UserDrivenTuningParser.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace TensileLite;

namespace
{
    // Matches the column order ArgumentModel::log_args() (argument_model.hpp) writes to
    // HIPBLASLT_TUNING_FILE: transA,transB,batch_count,m,n,k,a_type,b_type,c_type,
    // compute_type,solution_index[,solution_name].
    std::vector<std::string> legacyEntries(const std::string& solutionIndex,
                                           const std::string& m = "1024")
    {
        return {"N", "N", "1", m, "512", "1024", "f16_r", "f16_r", "f16_r", "f32_r", solutionIndex};
    }

    std::vector<std::string> namedEntries(const std::string& solutionIndex,
                                          const std::string& solutionName,
                                          const std::string& m = "1024")
    {
        auto entries = legacyEntries(solutionIndex, m);
        entries.push_back(solutionName);
        return entries;
    }
} // namespace

// Legacy rows keep their index and an empty name. Runtime replay applies the
// build-version check because these rows cannot be validated by name.
TEST(UserDrivenTuningParser, smoke_ParsesLegacyRowWithoutSolutionName)
{
    auto result = problemFromEntries(legacyEntries("56537"));

    EXPECT_EQ(result.second.index, 56537);
    EXPECT_TRUE(result.second.name.empty());
}

// New-format rows must capture the trailing solution_name alongside the index.
TEST(UserDrivenTuningParser, smoke_ParsesRowWithSolutionName)
{
    auto result = problemFromEntries(namedEntries("56537", "Cijk_Alik_Bljk_HHS_BH_MT128x128x16"));

    EXPECT_EQ(result.second.index, 56537);
    EXPECT_EQ(result.second.name, "Cijk_Alik_Bljk_HHS_BH_MT128x128x16");
}

// An explicit empty solution_name receives the same legacy safety treatment.
TEST(UserDrivenTuningParser, smoke_ParsesRowWithEmptySolutionNameAsLegacy)
{
    auto result = problemFromEntries(namedEntries("56537", ""));

    EXPECT_EQ(result.second.index, 56537);
    EXPECT_TRUE(result.second.name.empty());
}

// Too few columns (neither the legacy nor the new field count) must be rejected
// outright (sentinel index <= 0), not partially parsed.
TEST(UserDrivenTuningParser, smoke_RejectsTooFewFields)
{
    std::vector<std::string> tooFew{"N", "N", "1", "1024", "512", "1024"};

    auto result = problemFromEntries(tooFew);

    EXPECT_LE(result.second.index, 0);
    EXPECT_TRUE(result.second.name.empty());
}

// Too many columns (e.g. accidentally including bench's trailing gcnArchName/CUs
// columns in the entries vector) must also be rejected, not silently misread.
TEST(UserDrivenTuningParser, smoke_RejectsTooManyFields)
{
    auto entries = namedEntries("56537", "SomeKernel");
    entries.push_back("gfx942");

    auto result = problemFromEntries(entries);

    EXPECT_LE(result.second.index, 0);
}

// A non-numeric solution_index must be rejected rather than throwing past
// problemFromEntries (it wraps std::stoi in try/catch).
TEST(UserDrivenTuningParser, smoke_RejectsNonNumericSolutionIndex)
{
    auto result = problemFromEntries(legacyEntries("not_a_number"));

    EXPECT_LE(result.second.index, 0);
}

TEST(UserDrivenTuningParser, smoke_RejectsOutOfRangeSolutionIndex)
{
    auto result = problemFromEntries(legacyEntries("999999999999999999999999999"));

    EXPECT_LE(result.second.index, 0);
}

TEST(UserDrivenTuningParser, smoke_RejectsNonNumericProblemDimension)
{
    auto entries                                  = legacyEntries("56537");
    entries[static_cast<size_t>(HeaderFields::m)] = "not_a_dimension";

    auto result = problemFromEntries(entries);

    EXPECT_LE(result.second.index, 0);
}

TEST(UserDrivenTuningParser, smoke_ParsesTransposeAndShapeFields)
{
    auto entries                                            = namedEntries("56537", "SomeKernel");
    entries[static_cast<size_t>(HeaderFields::transA)]      = "T";
    entries[static_cast<size_t>(HeaderFields::transB)]      = "C";
    entries[static_cast<size_t>(HeaderFields::batch_count)] = "3";
    entries[static_cast<size_t>(HeaderFields::m)]           = "2048";
    entries[static_cast<size_t>(HeaderFields::n)]           = "256";
    entries[static_cast<size_t>(HeaderFields::k)]           = "4096";

    auto result = problemFromEntries(entries);

    EXPECT_TRUE(result.first.transA());
    EXPECT_TRUE(result.first.transB());
    EXPECT_EQ(result.first.batchSize(), 3);
    EXPECT_EQ(result.first.m(), 2048);
    EXPECT_EQ(result.first.n(), 256);
    EXPECT_EQ(result.first.k(), 4096);
}

// Note: an "unrecognized datatype string" case is deliberately not covered here.
// hipDataType_to_tensile_type() (tensile_host.hpp), which problemFromEntries() calls
// for a_type/b_type/c_type/compute_type, reaches an assert(!"...") on an unmapped
// hipDataType - a hard abort() when assertions are enabled, not a catchable/testable
// failure - for any row whose type string round-trips through string_to_hip_datatype()
// to a value it doesn't recognize. That is pre-existing behavior this feature doesn't
// touch, so it is out of scope to test (or fix) here.

// Two problems that differ only by M must produce distinct ProblemOverride keys -
// sanity check that problemFromEntries actually threads shape fields through.
TEST(UserDrivenTuningParser, smoke_DistinctShapesProduceDistinctKeys)
{
    auto problemA = problemFromEntries(legacyEntries("1", "1024")).first;
    auto problemB = problemFromEntries(legacyEntries("2", "2048")).first;

    EXPECT_NE(problemA, problemB);
}

TEST(UserDrivenTuningParser, smoke_OverrideMapDeduplicatesIndexesAndReturnsLockedRange)
{
    OverrideMap     map;
    ProblemOverride key = problemFromEntries(legacyEntries("0", "4096")).first;

    EXPECT_TRUE(map.add({key, TunedSolution{333, "KernelA"}}));
    EXPECT_FALSE(map.add({key, TunedSolution{333, "ConflictingName"}}));
    EXPECT_TRUE(map.add({key, TunedSolution{444, "KernelB"}}));

    auto matches = map.find(key);
    ASSERT_FALSE(matches.empty());

    int count = 0;
    for(const auto& match : matches)
    {
        if(count == 0)
        {
            EXPECT_EQ(match.second.index, 333);
            EXPECT_EQ(match.second.name, "KernelA");
        }
        else if(count == 1)
        {
            EXPECT_EQ(match.second.index, 444);
            EXPECT_EQ(match.second.name, "KernelB");
        }
        ++count;
    }
    EXPECT_EQ(count, 2);
}

TEST(UserDrivenTuningParser, smoke_OverrideMapSupportsConcurrentReaders)
{
    OverrideMap     map;
    ProblemOverride key = problemFromEntries(legacyEntries("0", "8192")).first;
    ASSERT_TRUE(map.add({key, TunedSolution{777, "ConcurrentKernel"}}));

    std::atomic<bool>        valid{true};
    std::vector<std::thread> readers;
    for(int thread = 0; thread < 8; ++thread)
    {
        readers.emplace_back([&]() {
            for(int iteration = 0; iteration < 100; ++iteration)
            {
                auto matches = map.find(key);
                if(matches.empty() || matches.begin()->second.index != 777
                   || matches.begin()->second.name != "ConcurrentKernel")
                {
                    valid = false;
                    return;
                }
            }
        });
    }

    for(auto& reader : readers)
        reader.join();

    EXPECT_TRUE(valid);
}

// End-to-end parser population into an isolated OverrideMap. Exercises
// header/value matching, mixed legacy/new rows, and per-shape lookup.
TEST(UserDrivenTuningParser, smoke_LoadsMixedFormatFileIntoOverrideMap)
{
    auto path
        = std::filesystem::temp_directory_path()
          / ("hipblaslt_tuning_gtest_" + std::to_string(static_cast<long long>(getpid())) + ".csv");

    {
        std::ofstream file(path);
        // New-format row (has solution_name), shape M=1024.
        file << "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,"
                "solution_index,solution_name\n";
        file << "N,N,1,1024,512,1024,f16_r,f16_r,f16_r,f32_r,111,KernelAAA\n";
        // Legacy-format row with bench's trailing gcnArchName/CUs columns instead of
        // solution_name, different shape M=2048.
        file << "    transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,"
                "solution_index,gcnArchName,CUs\n";
        file << "N,N,1,2048,512,1024,f16_r,f16_r,f16_r,f32_r,222,gfx942,304\n";
    }

    OverrideMap map;
    loadContractionProblemsFromFile(path.string(), map);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_GE(map.size(), 2);

    ProblemOverride keyA = problemFromEntries(legacyEntries("0", "1024")).first;
    {
        auto matchesA = map.find(keyA);
        ASSERT_FALSE(matchesA.empty()) << "Shape M=1024 (new-format row) not found";
        EXPECT_EQ(matchesA.begin()->second.index, 111);
        EXPECT_EQ(matchesA.begin()->second.name, "KernelAAA");
    }

    ProblemOverride keyB = problemFromEntries(legacyEntries("0", "2048")).first;
    {
        auto matchesB = map.find(keyB);
        ASSERT_FALSE(matchesB.empty()) << "Shape M=2048 (legacy-format row) not found";
        EXPECT_EQ(matchesB.begin()->second.index, 222);
        EXPECT_TRUE(matchesB.begin()->second.name.empty());
    }
}

#endif // CODE_COVERAGE
