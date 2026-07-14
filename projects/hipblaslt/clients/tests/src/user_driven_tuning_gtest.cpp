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

// Unit tests for the AIHPBLAS-3751 tuning-cache self-healing feature: the
// solution_name column added to HIPBLASLT_TUNING_FILE / HIPBLASLT_TUNING_OVERRIDE_FILE
// entries (UserDrivenTuningParser.hpp/.cpp). solution_name is a stable identifier for
// the tuned kernel that survives a rebuild reordering solution_index, which is what
// lets a tuning-cache entry be healed instead of silently going stale or the whole
// override file being discarded (see rocblaslt_auxiliary.cpp's
// problem_override_from_file[_cpp]() for the runtime healing logic this data feeds).
//
// This file is split into two parts by build requirement:
//
// 1. hipblaslt_strip_custom_tuning_suffix() coverage (always built/run). That helper
//    lives in argument_model.hpp, a normal public client header, so this part needs
//    nothing special and runs in the same "smoke" CI lane as the rest of this binary
//    (test/therock/test_hipblaslt.py, see caching_library_gtest.cpp for the
//    `smoke_` naming convention this file follows).
//
// 2. Everything else (problemFromEntries()/OverrideMap coverage), gated behind
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
//    rather than forcing it into the default build. It was instead verified with a
//    standalone harness compiled directly against UserDrivenTuningParser.cpp and run
//    against real hardware (see the AIHPBLAS-3751 plan's implementation notes).
//
// Why part 2 is unit tests of the parser, not YAML-driven matmul_gtest cases:
// TensileLite::OverrideMap and OverrideSingleton (UserDrivenTuningParser.hpp) are
// process-wide singletons that load HIPBLASLT_TUNING_OVERRIDE_FILE at most once per
// process - getContractionProblemsFromFile() is a no-op once the map is non-empty, and
// OverrideSingleton reads the env var once at first construction. That makes per-case
// override-file scenarios driven through the shared hipblaslt-test binary fundamentally
// order-dependent: only the first case in the whole process to touch the mechanism
// actually exercises it, regardless of whether that's done via YAML or C++. This is a
// pre-existing property of the singleton design, not something introduced here.
// problemFromEntries() has no such issue - it is a pure function - so it carries the
// bulk of this coverage. The OverrideMap-level tests below are deliberately the only
// tests in this binary that call getContractionProblemsFromFile(), so they are not
// racing any other case for the singleton's one-shot load. The runtime healing search
// itself (healTunedSolutionByName[Cpp](), rocblaslt_auxiliary.cpp) requires a live
// Tensile solution library and is intentionally file-local (static); it was verified
// manually against real hardware rather than unit tested here.

#include <gtest/gtest.h>

#include "argument_model.hpp"

// getSolutionNameFromData() (tensile_host.cpp), used by the ext API's
// GemmInstance::getSolutionName(), can append a " (Custom tuning: GSU: x, WGM: y)"
// display suffix. If that decorated string were ever persisted as solution_name,
// healing/validation (which always compares against the plain index->name path -
// see rocblaslt_auxiliary.cpp) could never match it again, so
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

// See file header comment part 2 for why this part is CODE_COVERAGE-only.
#ifdef CODE_COVERAGE
#include "UserDrivenTuningParser.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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

// Legacy (pre solution_name) rows must keep parsing exactly as before: a valid index
// and an empty name (no identifier to heal by, but still usable as an index-only
// fast-path hint).
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

// A row with an explicit but empty solution_name column (trailing comma, no text)
// degrades to the same "index-only, no healing" treatment as a legacy row, rather
// than e.g. being rejected.
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

// Note: an "unrecognized datatype string" case is deliberately not covered here.
// hipDataType_to_tensile_type() (tensile_host.hpp), which problemFromEntries() calls
// for a_type/b_type/c_type/compute_type, reaches an assert(!"...") on an unmapped
// hipDataType - a hard abort() when assertions are enabled, not a catchable/testable
// failure - for any row whose type string round-trips through string_to_hip_datatype()
// to a value it doesn't recognize. That is pre-existing behavior this feature doesn't
// touch, so it is out of scope to test (or fix) here.

// Two problems that differ only by M must produce distinct ProblemOverride keys -
// sanity check that problemFromEntries actually threads shape fields through, since
// OverrideMap's healing/fast-path logic depends on keys being shape-specific.
TEST(UserDrivenTuningParser, smoke_DistinctShapesProduceDistinctKeys)
{
    auto problemA = problemFromEntries(legacyEntries("1", "1024")).first;
    auto problemB = problemFromEntries(legacyEntries("2", "2048")).first;

    EXPECT_NE(problemA, problemB);
}

// End-to-end OverrideMap population via getContractionProblemsFromFile(): the only
// test in this binary allowed to call it (see file header comment - OverrideMap loads
// a given process's override file at most once). Exercises header/value line matching,
// mixed legacy + new-format rows in the same file, and per-shape lookup.
TEST(UserDrivenTuningParser, smoke_LoadsMixedFormatFileIntoOverrideMap)
{
    auto path = std::filesystem::temp_directory_path()
               / ("hipblaslt_tuning_gtest_" + std::to_string(static_cast<long long>(getpid()))
                  + ".csv");

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

    getContractionProblemsFromFile(path.string());
    std::error_code ec;
    std::filesystem::remove(path, ec);

    OverrideMap& map = OverrideMap::getMap();
    ASSERT_GE(map.size(), 2);

    ProblemOverride keyA     = problemFromEntries(legacyEntries("0", "1024")).first;
    auto            matchesA = map.find(keyA);
    ASSERT_FALSE(matchesA.empty()) << "Shape M=1024 (new-format row) not found";
    EXPECT_EQ(matchesA[0].index, 111);
    EXPECT_EQ(matchesA[0].name, "KernelAAA");

    ProblemOverride keyB     = problemFromEntries(legacyEntries("0", "2048")).first;
    auto            matchesB = map.find(keyB);
    ASSERT_FALSE(matchesB.empty()) << "Shape M=2048 (legacy-format row) not found";
    EXPECT_EQ(matchesB[0].index, 222);
    EXPECT_TRUE(matchesB[0].name.empty());
}

// Heals an entry: update() must be locatable by (prob_key, original value) even
// though find() now returns snapshots rather than live iterators (the fix for a data
// race where a caller could read a TunedSolution concurrently with another thread
// healing/updating it via a stale iterator). This is the same call pattern
// rocblaslt_auxiliary.cpp's problem_override_from_file[_cpp]() use after a successful
// heal.
TEST(UserDrivenTuningParser, smoke_UpdateHealsSnapshotEntryInPlace)
{
    auto path = std::filesystem::temp_directory_path()
               / ("hipblaslt_tuning_gtest_update_"
                  + std::to_string(static_cast<long long>(getpid())) + ".csv");

    {
        std::ofstream file(path);
        file << "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,"
                "solution_index,solution_name\n";
        file << "N,N,1,4096,512,1024,f16_r,f16_r,f16_r,f32_r,333,KernelToHeal\n";
    }

    getContractionProblemsFromFile(path.string());
    std::error_code ec;
    std::filesystem::remove(path, ec);

    OverrideMap&    map = OverrideMap::getMap();
    ProblemOverride key = problemFromEntries(legacyEntries("0", "4096")).first;

    auto before = map.find(key);
    ASSERT_FALSE(before.empty());
    EXPECT_EQ(before[0].index, 333);

    map.update(key, before[0], TunedSolution{444, "KernelToHeal"});

    auto after = map.find(key);
    ASSERT_FALSE(after.empty());
    EXPECT_EQ(after[0].index, 444);
    EXPECT_EQ(after[0].name, "KernelToHeal");
}

#endif // CODE_COVERAGE
