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

// Host-only tests for compressed .dat.zlib loading via fileToMsgObject.
//
// These tests verify that the runtime can:
// 1. Load zlib-compressed .dat.zlib files when no uncompressed .dat exists
// 2. Fall back to uncompressed .dat when no .zlib file exists
// 3. Prefer .dat.zlib over .dat when both exist
// 4. Handle corrupt .dat.zlib gracefully
//
// We test through the public LoadLibraryMapping API which exercises the full
// fileToMsgObject -> msgpack parse path. No GPU required.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include <msgpack.hpp>
#include <zlib.h>

#include <Tensile/Tensile.hpp>

// Given/When/Then spelled over gtest's native SCOPED_TRACE. Each macro records
// the step in the failure trace and opens a block, so the steps nest and a
// failing assertion reports the Given/When/Then context it sits under. No BDD
// framework is pulled in; this expands to ::testing::ScopedTrace.
#define GIVEN(message) SCOPED_TRACE("Given: " message);
#define WHEN(message) SCOPED_TRACE("When: " message);
#define THEN(message) SCOPED_TRACE("Then: " message);

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

    std::vector<Bytef> packMapping(const std::map<std::string, std::string>& entries)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, entries);
        return std::vector<Bytef>(reinterpret_cast<const Bytef*>(buffer.data()),
                                  reinterpret_cast<const Bytef*>(buffer.data()) + buffer.size());
    }

    // Compress with an explicit zlib window-bits setting so tests can emit
    // non-zlib-wrapped streams (raw deflate at -15, gzip at +31) that the
    // loader's inflateInit (zlib header only) must reject.
    void writeDeflateWithWindowBits(const fs::path&           path,
                                    const std::vector<Bytef>& payload,
                                    int                       windowBits)
    {
        z_stream strm{};
        int      ret = deflateInit2(
            &strm, Z_BEST_COMPRESSION, Z_DEFLATED, windowBits, 8, Z_DEFAULT_STRATEGY);
        ASSERT_EQ(ret, Z_OK) << "deflateInit2 failed";

        std::vector<Bytef> out(payload.size() + 128);
        strm.next_in   = const_cast<Bytef*>(payload.data());
        strm.avail_in  = static_cast<uInt>(payload.size());
        strm.next_out  = out.data();
        strm.avail_out = static_cast<uInt>(out.size());
        ret            = deflate(&strm, Z_FINISH);
        uLong produced = out.size() - strm.avail_out;
        deflateEnd(&strm);
        ASSERT_EQ(ret, Z_STREAM_END) << "deflate did not finish in one pass";

        std::ofstream o(path, std::ios::binary);
        ASSERT_TRUE(o.good()) << "could not open " << path << " for writing";
        o.write(reinterpret_cast<const char*>(out.data()),
                static_cast<std::streamsize>(produced));
        ASSERT_TRUE(o.good()) << "write failed for " << path;
    }

    void writeCompressedBytes(const fs::path& path, const std::vector<Bytef>& bytes)
    {
        uLongf compressedSize = compressBound(static_cast<uLong>(bytes.size()));
        std::vector<Bytef> compressed(compressedSize);
        int ret = compress2(compressed.data(),
                            &compressedSize,
                            bytes.data(),
                            static_cast<uLong>(bytes.size()),
                            Z_BEST_COMPRESSION);
        ASSERT_EQ(ret, Z_OK) << "zlib compress2 failed";

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good()) << "could not open " << path << " for writing";
        out.write(reinterpret_cast<const char*>(compressed.data()),
                  static_cast<std::streamsize>(compressedSize));
        ASSERT_TRUE(out.good()) << "write failed for " << path;
    }

    std::vector<char> readFile(const fs::path& path)
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if(!in.good())
            return {};
        auto size = in.tellg();
        if(size == std::ifstream::pos_type(-1))
            return {};
        std::vector<char> data(static_cast<size_t>(size));
        in.seekg(0);
        in.read(data.data(), static_cast<std::streamsize>(data.size()));
        if(!in.good())
            return {};
        return data;
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
                                + std::to_string(static_cast<long long>(getpid()))
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
    GIVEN("a .dat.zlib catalog and no uncompressed .dat sibling")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        writeCompressedMsgpackMapping(
            gzPath, {{"0", "kernel_a"}, {"10", "kernel_b"}, {"99", "kernel_c"}});

        WHEN("the mapping is loaded by its .dat name")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the decompressed entries are returned")
            {
                ASSERT_EQ(mapping.size(), 3u);
                EXPECT_EQ(mapping.at(0), "kernel_a");
                EXPECT_EQ(mapping.at(10), "kernel_b");
                EXPECT_EQ(mapping.at(99), "kernel_c");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, FallsBackToUncompressedDat)
{
    GIVEN("only an uncompressed .dat exists (no .zlib sibling)")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        writeMsgpackMapping(datPath, {{"5", "uncompressed_a"}, {"15", "uncompressed_b"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the uncompressed entries are returned")
            {
                ASSERT_EQ(mapping.size(), 2u);
                EXPECT_EQ(mapping.at(5), "uncompressed_a");
                EXPECT_EQ(mapping.at(15), "uncompressed_b");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, PrefersCompressedOverUncompressed)
{
    GIVEN("both a .dat and a .dat.zlib exist with different contents")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        writeMsgpackMapping(datPath, {{"0", "from_uncompressed"}});
        writeCompressedMsgpackMapping(gzPath, {{"0", "from_compressed"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the compressed catalog wins")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "from_compressed");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, InconsistentValidInstallPrefersCompressedAndIgnoresDat)
{
    GIVEN("a valid .dat and a valid .dat.zlib whose contents disagree (a stale install)")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // The uncompressed .dat is left over from an older build and disagrees
        // with the freshly written .dat.zlib (different keys and values). The
        // loader has no oracle to reconcile them, so it must use the compressed
        // catalog only and never merge or read the stale .dat.
        writeMsgpackMapping(datPath, {{"0", "stale_a"}, {"1", "stale_b"}});
        writeCompressedMsgpackMapping(gzPath, {{"7", "fresh"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("only the compressed catalog is used and the stale .dat is ignored")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(7), "fresh");
                EXPECT_EQ(mapping.count(0), 0u);
                EXPECT_EQ(mapping.count(1), 0u);
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, HandlesCorruptCompressedFile)
{
    GIVEN("a corrupt .dat.zlib alongside a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        {
            std::ofstream out(gzPath, std::ios::binary);
            const char    garbage[] = "this is not valid zlib data at all";
            out.write(garbage, sizeof(garbage));
        }
        writeMsgpackMapping(datPath, {{"7", "fallback_kernel"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the loader falls back to the uncompressed .dat")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(7), "fallback_kernel");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, FallsBackWhenCompressedPayloadIsInvalidMsgpack)
{
    GIVEN("a valid zlib stream that inflates to non-msgpack bytes, plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // 0xc1 is a reserved msgpack byte. The zlib stream is valid, but the
        // inflated payload is not a valid msgpack object.
        writeCompressedBytes(gzPath, {0xc1});
        writeMsgpackMapping(datPath, {{"11", "fallback_for_bad_msgpack"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the loader falls back to the uncompressed .dat")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(11), "fallback_for_bad_msgpack");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, FallsBackWhenCompressedTrailerIsTruncated)
{
    GIVEN("a .dat.zlib with its Adler-32 trailer truncated, plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        writeCompressedMsgpackMapping(gzPath, {{"0", "from_truncated_compressed"}});
        auto compressed = readFile(gzPath);
        ASSERT_GT(compressed.size(), 4u);
        compressed.resize(compressed.size() - 2);
        {
            std::ofstream out(gzPath, std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(out.good()) << "could not reopen " << gzPath;
            out.write(compressed.data(), static_cast<std::streamsize>(compressed.size()));
            ASSERT_TRUE(out.good()) << "truncate rewrite failed for " << gzPath;
        }
        writeMsgpackMapping(datPath, {{"0", "fallback_for_truncated_checksum"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the integrity failure triggers fallback to the .dat")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "fallback_for_truncated_checksum");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, LoadsWhenFilenameAlreadyEndsWithZlib)
{
    GIVEN("a .dat.zlib catalog")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        writeCompressedMsgpackMapping(gzPath, {{"3", "direct_zlib_path"}});

        WHEN("the mapping is loaded by its literal .zlib path")
        {
            auto mapping = TensileLite::LoadLibraryMapping(gzPath.string());

            THEN("the entries are returned")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(3), "direct_zlib_path");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, ReturnsEmptyWhenNeitherExists)
{
    GIVEN("neither a .dat nor a .dat.zlib exists")
    {
        fs::path datPath = tmpDir / "nonexistent.dat";

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("an empty mapping is returned")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, LoadsCompressedSpanningMultipleInflateChunks)
{
    GIVEN("a compressed catalog larger than one inflate buffer")
    {
        std::map<std::string, std::string> entries;
        std::mt19937                       rng(123);
        for(int i = 0; i < 4000; ++i)
        {
            std::string value(400, 'x');
            for(auto& c : value)
                c = static_cast<char>('a' + (rng() % 26));
            entries[std::to_string(i)] = value;
        }

        fs::path datPath = tmpDir / "big_mapping.dat";
        fs::path gzPath  = tmpDir / "big_mapping.dat.zlib";
        writeCompressedMsgpackMapping(gzPath, entries);

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("every entry is recovered across inflate chunks")
            {
                ASSERT_EQ(mapping.size(), entries.size());
                for(auto const& entry : entries)
                {
                    int  key = std::stoi(entry.first);
                    auto it  = mapping.find(key);
                    ASSERT_NE(it, mapping.end()) << "missing key " << key;
                    EXPECT_EQ(it->second, entry.second) << "mismatch for key " << key;
                }
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, CorruptCompressedFileWithNoFallbackReturnsEmpty)
{
    GIVEN("a corrupt .dat.zlib and no .dat to fall back to")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        {
            std::ofstream out(gzPath, std::ios::binary);
            const char    garbage[] = "definitely not a zlib stream";
            out.write(garbage, sizeof(garbage));
        }

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("an empty mapping is returned without throwing")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, CorruptCompressedWithBrokenFallbackReturnsEmpty)
{
    GIVEN("a corrupt .dat.zlib and a .dat that is itself not valid msgpack")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        {
            std::ofstream out(gzPath, std::ios::binary);
            const char    garbage[] = "definitely not a zlib stream";
            out.write(garbage, sizeof(garbage));
        }
        // The fallback exists but is also broken: 0xc1 is a reserved msgpack
        // byte, so the uncompressed parse fails too.
        {
            std::ofstream out(datPath, std::ios::binary);
            const char    bad = static_cast<char>(0xc1);
            out.write(&bad, 1);
        }

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("an empty mapping is returned without throwing")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, OversizedDecompressionIsRejectedAndFallsBack)
{
    GIVEN("a tiny .dat.zlib that inflates far past the decompression cap, plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // 8 MiB of zeros compresses to a few KB but inflates to 8 MiB; with the
        // cap shrunk below that, the loader must refuse to inflate it (zip-bomb
        // guard) and use the .dat instead of exhausting memory.
        std::vector<Bytef> bomb(8 * 1024 * 1024, 0);
        writeCompressedBytes(gzPath, bomb);
        writeMsgpackMapping(datPath, {{"0", "fallback_after_zip_bomb"}});

        WHEN("the mapping is loaded with the cap set below the inflated size")
        {
            setenv("TENSILE_MAX_DECOMPRESSED_BYTES", "1048576", 1);
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());
            unsetenv("TENSILE_MAX_DECOMPRESSED_BYTES");

            THEN("the oversized stream is rejected and the .dat is used instead")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "fallback_after_zip_bomb");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, EmptyCompressedFileReturnsEmpty)
{
    GIVEN("a zero-byte .dat.zlib and no .dat sibling")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // Zero-byte .zlib: inflate sees no data and never reaches Z_STREAM_END.
        {
            std::ofstream out(gzPath, std::ios::binary);
        }

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("an empty mapping is returned")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, EmptyUncompressedFileReturnsEmpty)
{
    GIVEN("a zero-byte .dat with no .zlib sibling")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        // Zero-byte .dat with no .zlib sibling: parse never completes.
        {
            std::ofstream out(datPath, std::ios::binary);
        }

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("an empty mapping is returned")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, TruncatedCompressedBodyFallsBackToUncompressed)
{
    GIVEN("a .dat.zlib truncated mid-body, plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // Keep only the first half of a valid zlib stream: the header is intact
        // but inflate can never reach Z_STREAM_END (distinct from the
        // trailer-only truncation case). The valid .dat sibling must win.
        writeCompressedMsgpackMapping(gzPath, {{"0", "real"}, {"1", "real_two"}});
        auto compressed = readFile(gzPath);
        ASSERT_GT(compressed.size(), 8u);
        compressed.resize(compressed.size() / 2);
        {
            std::ofstream out(gzPath, std::ios::binary | std::ios::trunc);
            out.write(compressed.data(), static_cast<std::streamsize>(compressed.size()));
            ASSERT_TRUE(out.good()) << "truncate rewrite failed for " << gzPath;
        }
        writeMsgpackMapping(datPath, {{"42", "fallback_after_body_truncation"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the incomplete stream triggers fallback to the .dat")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(42), "fallback_after_body_truncation");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, TruncatedCompressedBodyWithNoFallbackReturnsEmpty)
{
    GIVEN("a .dat.zlib truncated mid-body and no .dat sibling")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        writeCompressedMsgpackMapping(gzPath, {{"0", "real"}, {"1", "real_two"}});
        auto compressed = readFile(gzPath);
        ASSERT_GT(compressed.size(), 8u);
        compressed.resize(compressed.size() / 2);
        {
            std::ofstream out(gzPath, std::ios::binary | std::ios::trunc);
            out.write(compressed.data(), static_cast<std::streamsize>(compressed.size()));
            ASSERT_TRUE(out.good()) << "truncate rewrite failed for " << gzPath;
        }

        WHEN("the mapping is loaded with no .dat sibling")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("an empty mapping is returned without throwing")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, RawDeflatePayloadFallsBackToUncompressed)
{
    GIVEN("a raw-deflate (headerless) .dat.zlib plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // Raw deflate (no zlib header). inflateInit expects a zlib header, so
        // this must be rejected — guards against a future writer switching to
        // raw deflate.
        writeDeflateWithWindowBits(gzPath, packMapping({{"0", "raw_deflate"}}), -15);
        writeMsgpackMapping(datPath, {{"0", "fallback_for_raw_deflate"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the non-zlib header is rejected and the .dat wins")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "fallback_for_raw_deflate");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, GzipWrappedPayloadFallsBackToUncompressed)
{
    GIVEN("a gzip-wrapped .dat.zlib plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // gzip header (wbits +31). inflateInit does NOT auto-detect gzip, so
        // this must be rejected — pins the exact accepted header set to zlib
        // only.
        writeDeflateWithWindowBits(gzPath, packMapping({{"0", "gzip"}}), 31);
        writeMsgpackMapping(datPath, {{"0", "fallback_for_gzip"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the gzip header is rejected and the .dat wins")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "fallback_for_gzip");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, TrailingGarbageAfterValidObjectIsRejected)
{
    GIVEN("a valid zlib stream with a complete msgpack map followed by extra bytes")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // A complete msgpack map followed by extra bytes, all validly
        // zlib-wrapped (Adler-32 passes). Bytes after the first complete
        // object indicate corruption/tampering, so the load is rejected
        // rather than silently returning the leading object.
        auto payload = packMapping({{"0", "leading"}, {"1", "object"}});
        payload.insert(payload.end(), {0xde, 0xad, 0xbe, 0xef, 0x00, 0x11});
        writeCompressedBytes(gzPath, payload);

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the load is rejected and an empty mapping is returned")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, NonNumericMappingKeyIsRejected)
{
    GIVEN("a mapping whose key carries trailing non-numeric characters")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        writeMsgpackMapping(datPath, {{"12bad", "kernel_a"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the load fails gracefully with an empty mapping")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, NonStringMappingValueIsRejected)
{
    GIVEN("a mapping whose value is not a string")
    {
        fs::path                   datPath = tmpDir / "test_mapping.dat";
        std::map<std::string, int> badValues{{"0", 123}};
        msgpack::sbuffer           buffer;
        msgpack::pack(buffer, badValues);
        std::ofstream out(datPath, std::ios::binary);
        ASSERT_TRUE(out.good()) << "could not open " << datPath << " for writing";
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        ASSERT_TRUE(out.good()) << "write failed for " << datPath;
        out.close();

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the load fails gracefully with an empty mapping")
            {
                EXPECT_TRUE(mapping.empty());
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, TrailingBytesAfterZlibStreamFallBackToUncompressed)
{
    GIVEN("a valid .dat.zlib with stray bytes appended after the zlib stream, plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // Distinct from the in-payload trailing-garbage case: here the extra
        // bytes sit *after* a complete zlib stream, so inflate reaches
        // Z_STREAM_END with input left over. That must be rejected, not ignored.
        writeCompressedMsgpackMapping(gzPath, {{"0", "real"}});
        {
            std::ofstream out(gzPath, std::ios::binary | std::ios::app);
            const char    junk[] = {0x01, 0x02, 0x03, 0x04};
            out.write(junk, sizeof(junk));
            ASSERT_TRUE(out.good()) << "append failed for " << gzPath;
        }
        writeMsgpackMapping(datPath, {{"0", "fallback_after_trailing_bytes"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the stream-with-trailing-bytes is rejected and the .dat is used")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "fallback_after_trailing_bytes");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, DirectoryAtCompressedPathFallsBackToUncompressed)
{
    GIVEN("a directory where the .dat.zlib should be, plus a valid .dat")
    {
        fs::path datPath = tmpDir / "test_mapping.dat";
        fs::path gzPath  = tmpDir / "test_mapping.dat.zlib";
        // A directory sitting where the .zlib should be: exists() is true but no
        // readable byte stream can be obtained. Must degrade to the .dat
        // sibling, not crash. (tellg() yields a huge/negative value caught by
        // the size guards, or the read fails — every path leads to fallback.)
        fs::create_directory(gzPath);
        writeMsgpackMapping(datPath, {{"0", "fallback_past_directory"}});

        WHEN("the mapping is loaded")
        {
            auto mapping = TensileLite::LoadLibraryMapping(datPath.string());

            THEN("the unreadable path triggers fallback to the .dat")
            {
                ASSERT_EQ(mapping.size(), 1u);
                EXPECT_EQ(mapping.at(0), "fallback_past_directory");
            }
        }
    }
}

TEST_F(CompressedLibraryLoadTest, DISABLED_LoadTimingComparison)
{
    // Build a realistic payload: 500 entries with long kernel names matching
    // the naming convention used in real hipBLASLt shard files.
    std::map<std::string, std::string> entries;
    for(int i = 0; i < 500; i++)
        entries[std::to_string(i)]
            = "TensileLibrary_gfx942_HPA_BF16_BF16_BF16_BF16_SB_SB_kernels_fallback_gfx942_"
              + std::to_string(i);

    fs::path datPath = tmpDir / "bench_mapping.dat";
    fs::path gzPath  = tmpDir / "bench_mapping.dat.zlib";

    writeMsgpackMapping(datPath, entries);
    writeCompressedMsgpackMapping(gzPath, entries);

    constexpr int N = 20;

    // Time uncompressed loads (only .dat present, no .zlib)
    fs::rename(gzPath, gzPath.string() + ".bak");
    auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < N; i++)
        TensileLite::LoadLibraryMapping(datPath.string());
    auto t1          = std::chrono::steady_clock::now();
    double dat_us    = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    fs::rename(gzPath.string() + ".bak", gzPath);

    // Time compressed loads (.zlib present — .dat also exists but is ignored)
    fs::rename(datPath, datPath.string() + ".bak");
    auto t2          = std::chrono::steady_clock::now();
    for(int i = 0; i < N; i++)
        TensileLite::LoadLibraryMapping(datPath.string());
    auto t3          = std::chrono::steady_clock::now();
    double gz_us     = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;
    fs::rename(datPath.string() + ".bak", datPath);

    std::cout << "\n[LoadTimingComparison] entries=" << entries.size()
              << "  uncompressed .dat: " << dat_us << " µs/call"
              << "  compressed .dat.zlib: " << gz_us << " µs/call"
              << "  overhead: " << (gz_us - dat_us) << " µs\n";

    // Correctness check — no EXPECT on timing values to avoid flakiness.
    auto m = TensileLite::LoadLibraryMapping(datPath.string());
    EXPECT_EQ(m.size(), entries.size());
}

#undef GIVEN
#undef WHEN
#undef THEN
