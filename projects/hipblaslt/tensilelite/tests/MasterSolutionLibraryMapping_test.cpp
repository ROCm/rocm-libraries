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

// Host-only tests for MasterSolutionLibrary::initLibraryMapping.
//
// These tests exercise the per-arch Mapping-file routing introduced so that
// single-arch shard installs do not collide on a shared
// "TensileLiteLibrary_lazy_Mapping.dat" filename. They run without a GPU
// because initLibraryMapping only inspects filenames and reads a small
// msgpack file from disk; no kernels are launched.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <msgpack.hpp>
#include <zlib.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

namespace fs = std::filesystem;

namespace
{
    using TestLibrary
        = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm,
                                             TensileLite::ContractionSolution>;

    // Write a minimal msgpack mapping file in the format
    // MessagePackLoadLibraryMapping expects: a top-level map of
    // string-encoded-int -> string.
    void writeMappingFile(const fs::path&                          path,
                          const std::map<std::string, std::string>& entries)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, entries);

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good()) << "could not open " << path << " for writing";
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        ASSERT_TRUE(out.good()) << "write failed for " << path;
    }

    struct MasterSolutionLibraryMappingTest : public ::testing::Test
    {
        fs::path tmpDir;

        void SetUp() override
        {
            tmpDir = fs::temp_directory_path()
                     / fs::path("hipblaslt-mapping-test-"
                                + std::to_string(::testing::UnitTest::GetInstance()->random_seed())
                                + "-"
                                + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            fs::create_directories(tmpDir);
        }

        void TearDown() override
        {
            std::error_code ec;
            fs::remove_all(tmpDir, ec);
        }

        // Build a master library file path inside tmpDir for a given arch.
        // initLibraryMapping never reads this file; it only parses the name.
        fs::path masterPath(const std::string& arch) const
        {
            return tmpDir / ("TensileLibrary_lazy_" + arch + ".dat");
        }

        fs::path perArchMappingPath(const std::string& arch) const
        {
            return tmpDir / ("TensileLiteLibrary_lazy_" + arch + "_Mapping.dat");
        }

        fs::path legacyGlobalMappingPath() const
        {
            return tmpDir / "TensileLiteLibrary_lazy_Mapping.dat";
        }
    };
}

// Per-arch file present and well-formed: must succeed and load all entries.
TEST_F(MasterSolutionLibraryMappingTest, LoadsPerArchMapping)
{
    const std::string arch = "gfx9999";
    writeMappingFile(perArchMappingPath(arch),
                     {{"0", "kernel_a_" + arch},
                      {"1", "kernel_b_" + arch},
                      {"42", "kernel_c_" + arch}});

    TestLibrary lib;
    ASSERT_TRUE(lib.initLibraryMapping(masterPath(arch).string()));
    EXPECT_TRUE(lib.uidIndexMapping.empty());
    EXPECT_EQ(lib.libraryMapping.size(), 3u);
    EXPECT_EQ(lib.libraryMapping.at(0), "kernel_a_" + arch);
    EXPECT_EQ(lib.libraryMapping.at(1), "kernel_b_" + arch);
    EXPECT_EQ(lib.libraryMapping.at(42), "kernel_c_" + arch);
}

TEST_F(MasterSolutionLibraryMappingTest, CompressedUidMappingRetainsFullWidthIdentity)
{
    const std::map<uint64_t, int> expected{
        {1, 0}, {uint64_t{1} << 32, 1}, {std::numeric_limits<uint64_t>::max(), 2}};
    msgpack::sbuffer buffer;
    msgpack::pack(buffer, expected);
    uLongf               compressedSize = compressBound(buffer.size());
    std::vector<uint8_t> compressed(compressedSize);
    ASSERT_EQ(compress2(compressed.data(),
                        &compressedSize,
                        reinterpret_cast<const Bytef*>(buffer.data()),
                        buffer.size(),
                        Z_BEST_COMPRESSION),
              Z_OK);
    const fs::path path = tmpDir / "TensileLiteLibrary_lazy_gfx942_UidMapping.dat";
    {
        std::ofstream out(path.string() + ".zlib", std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write(reinterpret_cast<const char*>(compressed.data()), compressedSize);
        ASSERT_TRUE(out.good());
    }

    TestLibrary lib;
    lib.uidIndexMapping = TensileLite::LoadUidMapping(path.string());
    ASSERT_EQ(lib.uidIndexMapping, expected);
    TensileLite::AMDGPU hardware(TensileLite::AMDGPU::Processor::gfx942, 304, "test-device");
    TensileLite::ContractionProblemGemm problem;
    for(const auto& [uid, index] : expected)
    {
        auto solution         = std::make_shared<TensileLite::ContractionSolution>();
        solution->index       = index;
        solution->solutionUid = uid;
        solution->requiredHostWorkspaceSizePerProblem = 0;
        lib.solutions.emplace(index, solution);
        EXPECT_EQ(lib.getSolutionByUid(hardware, uid), solution);
        EXPECT_EQ(lib.getSolutionByUid(problem, hardware, uid), solution);
    }
    EXPECT_EQ(lib.getSolutionByUid(hardware, 0), nullptr);
}

TEST_F(MasterSolutionLibraryMappingTest, UidMappingLoadsOnFirstUidLookup)
{
    const std::string arch = "gfx9999";
    writeMappingFile(perArchMappingPath(arch), {{"10", "kernel_a_" + arch}});
    const uint64_t uid = uint64_t{1} << 40;
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, std::map<uint64_t, int>{{uid, 3}});
        std::ofstream out(tmpDir / ("TensileLiteLibrary_lazy_" + arch + "_UidMapping.dat"),
                          std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        ASSERT_TRUE(out.good());
    }

    TestLibrary lib;
    ASSERT_TRUE(lib.initLibraryMapping(masterPath(arch).string()));
    EXPECT_TRUE(lib.uidIndexMapping.empty());

    // Index 3 is below the first mapped shard, so the lookup resolves it from
    // `solutions` without opening a shard file.
    auto solution         = std::make_shared<TensileLite::ContractionSolution>();
    solution->index       = 3;
    solution->solutionUid = uid;
    solution->requiredHostWorkspaceSizePerProblem = 0;
    lib.solutions.emplace(3, solution);

    TensileLite::AMDGPU hardware(TensileLite::AMDGPU::Processor::gfx942, 304, "test-device");
    EXPECT_EQ(lib.getSolutionByUid(hardware, uid), solution);
    EXPECT_EQ(lib.uidIndexMapping.size(), 1u);
}

// The kpack-collision regression: the legacy global Mapping.dat sitting in
// the same directory must NOT be picked up when the per-arch file is absent.
// If it ever were, two shard installs that overlay onto the same prefix
// would silently load each other's kernels.
TEST_F(MasterSolutionLibraryMappingTest, DoesNotFallBackToLegacyGlobalMapping)
{
    const std::string arch = "gfx9999";
    // Legacy global file exists with plausible content...
    writeMappingFile(legacyGlobalMappingPath(),
                     {{"0", "wrong_kernel_" + arch}, {"1", "wrong_kernel2_" + arch}});
    // ...but the per-arch file does not.

    TestLibrary lib;
    EXPECT_FALSE(lib.initLibraryMapping(masterPath(arch).string()));
    EXPECT_TRUE(lib.libraryMapping.empty());
}

// Arch is derived dynamically from the master filename; a second arch in the
// same directory must not see the first arch's entries, even if both
// per-arch files exist.
TEST_F(MasterSolutionLibraryMappingTest, RoutesByArchDerivedFromMasterFilename)
{
    writeMappingFile(perArchMappingPath("gfx1100"),
                     {{"0", "k1100_a"}, {"1", "k1100_b"}});
    writeMappingFile(perArchMappingPath("gfx1101"),
                     {{"100", "k1101_a"}, {"101", "k1101_b"}, {"102", "k1101_c"}});

    TestLibrary lib1100;
    ASSERT_TRUE(lib1100.initLibraryMapping(masterPath("gfx1100").string()));
    EXPECT_EQ(lib1100.libraryMapping.size(), 2u);
    EXPECT_EQ(lib1100.libraryMapping.at(0), "k1100_a");

    TestLibrary lib1101;
    ASSERT_TRUE(lib1101.initLibraryMapping(masterPath("gfx1101").string()));
    EXPECT_EQ(lib1101.libraryMapping.size(), 3u);
    EXPECT_EQ(lib1101.libraryMapping.at(100), "k1101_a");
    EXPECT_EQ(lib1101.libraryMapping.find(0), lib1101.libraryMapping.end());
}

// Master filename without the expected "TensileLibrary_lazy_" prefix:
// initLibraryMapping must refuse rather than guess at a Mapping path.
//
// Discrimination guard: a legacy implementation that ignores the master
// filename and just opens a hardcoded global "TensileLiteLibrary_lazy_Mapping.dat"
// would happily load the planted global file and return true. The patched
// implementation must reject the malformed name *before* touching the disk.
TEST_F(MasterSolutionLibraryMappingTest, RejectsMalformedMasterFilename)
{
    const fs::path badMaster = tmpDir / "something_unexpected.dat";
    // Plant a well-formed legacy global file. A non-validating implementation
    // would find this and incorrectly succeed.
    writeMappingFile(legacyGlobalMappingPath(),
                     {{"0", "junk_value_a"}, {"1", "junk_value_b"}});
    // Also plant a per-arch file under a plausible arch so an implementation
    // that guesses an arch from anywhere has something to pick up.
    writeMappingFile(perArchMappingPath("gfx1100"), {{"0", "k1100_a"}});

    TestLibrary lib;
    EXPECT_FALSE(lib.initLibraryMapping(badMaster.string()));
    EXPECT_TRUE(lib.libraryMapping.empty());
}
