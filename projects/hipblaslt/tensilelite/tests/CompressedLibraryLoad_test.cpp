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

// Host-only tests for compressed .dat.gz loading via fileToMsgObject.
//
// These tests verify that the runtime can:
// 1. Load zlib-compressed .dat.gz files when no uncompressed .dat exists
// 2. Fall back to uncompressed .dat when no .gz file exists
// 3. Prefer .dat.gz over .dat when both exist
// 4. Handle corrupt .dat.gz gracefully
//
// We test through the public LoadLibraryMapping API which exercises the full
// fileToMsgObject -> msgpack parse path. No GPU required.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include <msgpack.hpp>
#include <zlib.h>

#include <Tensile/Tensile.hpp>

namespace fs = std::filesystem;

namespace
{
    void writeMsgpackMapping(const fs::path&                           path,
                             const std::map<std::string, std::string>& entries)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, entries);

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good()) << "could not open " << path << " for writing";
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        ASSERT_TRUE(out.good()) << "write failed for " << path;
    }

    void writeCompressedMsgpackMapping(const fs::path&                           path,
                                       const std::map<std::string, std::string>& entries)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, entries);

        uLongf compressedSize = compressBound(static_cast<uLong>(buffer.size()));
        std::vector<Bytef> compressed(compressedSize);
        int ret = compress2(compressed.data(),
                            &compressedSize,
                            reinterpret_cast<const Bytef*>(buffer.data()),
                            static_cast<uLong>(buffer.size()),
                            Z_BEST_COMPRESSION);
        ASSERT_EQ(ret, Z_OK) << "zlib compress2 failed";

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good()) << "could not open " << path << " for writing";
        out.write(reinterpret_cast<const char*>(compressed.data()),
                  static_cast<std::streamsize>(compressedSize));
        ASSERT_TRUE(out.good()) << "write failed for " << path;
    }

    struct CompressedLibraryLoadTest : public ::testing::Test
    {
        fs::path tmpDir;

        void SetUp() override
        {
            tmpDir = fs::temp_directory_path()
                     / fs::path("hipblaslt-compressed-test-"
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
    };
}

TEST_F(CompressedLibraryLoadTest, LoadsCompressedDatGz)
{
    fs::path datPath = tmpDir / "test_mapping.dat";
    fs::path gzPath  = tmpDir / "test_mapping.dat.gz";

    writeCompressedMsgpackMapping(gzPath,
                                  {{"0", "kernel_a"}, {"10", "kernel_b"}, {"99", "kernel_c"}});

    auto mapping = TensileLite::LoadLibraryMapping(datPath.string());
    ASSERT_EQ(mapping.size(), 3u);
    EXPECT_EQ(mapping.at(0), "kernel_a");
    EXPECT_EQ(mapping.at(10), "kernel_b");
    EXPECT_EQ(mapping.at(99), "kernel_c");
}

TEST_F(CompressedLibraryLoadTest, FallsBackToUncompressedDat)
{
    fs::path datPath = tmpDir / "test_mapping.dat";

    writeMsgpackMapping(datPath,
                        {{"5", "uncompressed_a"}, {"15", "uncompressed_b"}});

    auto mapping = TensileLite::LoadLibraryMapping(datPath.string());
    ASSERT_EQ(mapping.size(), 2u);
    EXPECT_EQ(mapping.at(5), "uncompressed_a");
    EXPECT_EQ(mapping.at(15), "uncompressed_b");
}

TEST_F(CompressedLibraryLoadTest, PrefersCompressedOverUncompressed)
{
    fs::path datPath = tmpDir / "test_mapping.dat";
    fs::path gzPath  = tmpDir / "test_mapping.dat.gz";

    writeMsgpackMapping(datPath, {{"0", "from_uncompressed"}});
    writeCompressedMsgpackMapping(gzPath, {{"0", "from_compressed"}});

    auto mapping = TensileLite::LoadLibraryMapping(datPath.string());
    ASSERT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping.at(0), "from_compressed");
}

TEST_F(CompressedLibraryLoadTest, HandlesCorruptCompressedFile)
{
    fs::path datPath = tmpDir / "test_mapping.dat";
    fs::path gzPath  = tmpDir / "test_mapping.dat.gz";

    // Write garbage to the .gz file
    {
        std::ofstream out(gzPath, std::ios::binary);
        const char garbage[] = "this is not valid zlib data at all";
        out.write(garbage, sizeof(garbage));
    }

    // Also write a valid uncompressed file to verify fallback
    writeMsgpackMapping(datPath, {{"7", "fallback_kernel"}});

    auto mapping = TensileLite::LoadLibraryMapping(datPath.string());
    ASSERT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping.at(7), "fallback_kernel");
}

TEST_F(CompressedLibraryLoadTest, ReturnsEmptyWhenNeitherExists)
{
    fs::path datPath = tmpDir / "nonexistent.dat";

    auto mapping = TensileLite::LoadLibraryMapping(datPath.string());
    EXPECT_TRUE(mapping.empty());
}
