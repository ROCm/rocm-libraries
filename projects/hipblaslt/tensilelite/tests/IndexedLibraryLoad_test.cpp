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

// Tests for the indexed (format_version 2) library layout, where solutions are
// held as an unparsed blob and deserialized on demand.
//
// Three groups:
//   1. SolutionBlobCache in isolation, driven by a stub deserializer. This is
//      where laziness, dedup, failure handling and thread safety are pinned
//      down without needing a real library file.
//   2. Malformed indexed files, which must be rejected at load rather than
//      surfacing later as a failed query.
//   3. Parity against a real legacy fixture, converted to indexed form in
//      process. Skipped when no fixture is installed, matching
//      RangeLibrary_test.
//
// Laziness is asserted on the cache's materialized count, never on
// MasterSolutionLibrary::solutions: leaf nodes resolve straight through the
// cache and never touch that map, so it legitimately stays empty after a
// query.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <msgpack.hpp>
#include <zlib.h>

#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/SolutionBlobCache.hpp>
#include <Tensile/Tensile.hpp>
// For Serialization::objectToMap, used to walk a legacy document's top level.
#include <Tensile/msgpack/MessagePack.hpp>

#include "TestData.hpp"

namespace fs = std::filesystem;

using namespace TensileLite;

namespace
{
    // Minimal stand-in for ContractionSolution: the cache only ever touches
    // `index` and `codeObjectFilename`.
    struct FakeSolution
    {
        int         index = -1;
        std::string codeObjectFilename;
        std::string tag;
    };

    using FakeCache = SolutionBlobCache<FakeSolution>;

    /// Builds a cache over `count` one-byte slices. The stub deserializer turns
    /// byte value N into a FakeSolution with index N, and counts how many times
    /// it ran so tests can prove work was deferred.
    std::shared_ptr<FakeCache> makeFakeCache(int                   count,
                                             std::shared_ptr<std::atomic<int>> parses,
                                             bool                  failEveryParse = false)
    {
        std::vector<uint8_t>                       blob;
        std::unordered_map<int, FakeCache::Slice>  slices;
        for(int i = 0; i < count; i++)
        {
            slices.emplace(i, FakeCache::Slice(blob.size(), 1));
            blob.push_back(static_cast<uint8_t>(i));
        }

        auto deserialize
            = [parses, failEveryParse](const uint8_t* data,
                                       size_t         size) -> std::shared_ptr<FakeSolution> {
            parses->fetch_add(1);
            if(failEveryParse || size != 1)
                return nullptr;
            auto rv   = std::make_shared<FakeSolution>();
            rv->index = static_cast<int>(*data);
            rv->tag   = "parsed";
            return rv;
        };

        return std::make_shared<FakeCache>(std::move(blob), std::move(slices), deserialize);
    }

    std::vector<uint8_t> deflateBytes(std::vector<uint8_t> const& raw)
    {
        uLongf             bound = compressBound(static_cast<uLong>(raw.size()));
        std::vector<uint8_t> out(bound);
        uLongf             outSize = bound;
        int                ret     = compress2(out.data(),
                             &outSize,
                             raw.data(),
                             static_cast<uLong>(raw.size()),
                             Z_BEST_COMPRESSION);
        EXPECT_EQ(ret, Z_OK);
        out.resize(outSize);
        return out;
    }

    void writeFile(fs::path const& path, std::vector<uint8_t> const& bytes)
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good()) << "cannot open " << path;
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good()) << "write failed for " << path;
    }

    struct TempDirTest : public ::testing::Test
    {
        fs::path tmpDir;

        void SetUp() override
        {
            tmpDir = fs::temp_directory_path()
                     / fs::path("hipblaslt-indexed-"
                                + std::to_string(static_cast<long long>(getpid())) + "-"
                                + ::testing::UnitTest::GetInstance()->current_test_info()->name());
            fs::create_directories(tmpDir);
        }

        void TearDown() override
        {
            std::error_code ec;
            fs::remove_all(tmpDir, ec);
        }
    };
} // namespace

// ---------------------------------------------------------------------------
// 1. SolutionBlobCache in isolation
// ---------------------------------------------------------------------------

TEST(SolutionBlobCacheTest, ParsesNothingUntilAsked)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(100, parses);

    EXPECT_EQ(cache->size(), 100u);
    EXPECT_EQ(cache->materializedCount(), 0u);
    EXPECT_EQ(parses->load(), 0) << "constructing the cache must not parse anything";
}

TEST(SolutionBlobCacheTest, ParsesOnlyTheRequestedSolution)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(100, parses);

    auto solution = cache->get(42);
    ASSERT_NE(solution, nullptr);
    EXPECT_EQ(solution->index, 42);
    EXPECT_EQ(parses->load(), 1);
    EXPECT_EQ(cache->materializedCount(), 1u)
        << "one query must not drag in the other 99 solutions";
}

TEST(SolutionBlobCacheTest, RepeatedGetReusesTheSameObject)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(10, parses);

    auto first  = cache->get(3);
    auto second = cache->get(3);
    EXPECT_EQ(first, second) << "callers must not see two objects for one index";
    EXPECT_EQ(parses->load(), 1) << "second get should hit the memo";
}

TEST(SolutionBlobCacheTest, UnknownIndexReturnsNullWithoutParsing)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(10, parses);

    EXPECT_FALSE(cache->contains(999));
    EXPECT_EQ(cache->get(999), nullptr);
    EXPECT_EQ(parses->load(), 0);
}

TEST(SolutionBlobCacheTest, ParseFailureIsRememberedNotRetried)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(10, parses, /*failEveryParse=*/true);

    EXPECT_EQ(cache->get(5), nullptr);
    EXPECT_EQ(cache->get(5), nullptr);
    EXPECT_EQ(parses->load(), 1)
        << "a corrupt slice must not be re-parsed on every query";
}

TEST(SolutionBlobCacheTest, RejectsPayloadWhoseIndexDisagreesWithTheTable)
{
    // Slice for index 7 holds a payload that decodes to index 9: the file is
    // inconsistent, so the cache must refuse rather than answer under 7.
    std::vector<uint8_t>                      blob{9};
    std::unordered_map<int, FakeCache::Slice> slices;
    slices.emplace(7, FakeCache::Slice(0, 1));

    FakeCache cache(std::move(blob), std::move(slices),
                    [](const uint8_t* data, size_t) {
                        auto rv   = std::make_shared<FakeSolution>();
                        rv->index = static_cast<int>(*data);
                        return rv;
                    });

    EXPECT_EQ(cache.get(7), nullptr);
}

TEST(SolutionBlobCacheTest, MaterializeAllParsesEverythingOnce)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(50, parses);

    cache->materializeAll();
    EXPECT_EQ(cache->materializedCount(), 50u);
    EXPECT_EQ(parses->load(), 50);

    cache->materializeAll();
    EXPECT_EQ(parses->load(), 50) << "second pass should be a no-op";
}

TEST(SolutionBlobCacheTest, CodeObjectFilenameIsStampedOnMaterialization)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(4, parses);

    // Stamped before any parse: applies as solutions come out of the blob.
    cache->setCodeObjectFilename("shard_prefix.co");
    EXPECT_EQ(cache->get(1)->codeObjectFilename, "shard_prefix.co");

    // Stamped after a parse: must reach the already-materialized ones too,
    // since shard loading can publish the name either side of a query.
    auto later = makeFakeCache(4, parses);
    auto early = later->get(2);
    ASSERT_NE(early, nullptr);
    EXPECT_TRUE(early->codeObjectFilename.empty());
    later->setCodeObjectFilename("late.co");
    EXPECT_EQ(early->codeObjectFilename, "late.co");
}

TEST(SolutionBlobCacheTest, IndicesCoversEverySlice)
{
    auto parses  = std::make_shared<std::atomic<int>>(0);
    auto cache   = makeFakeCache(8, parses);
    auto indices = cache->indices();

    std::sort(indices.begin(), indices.end());
    ASSERT_EQ(indices.size(), 8u);
    for(int i = 0; i < 8; i++)
        EXPECT_EQ(indices[i], i);
    EXPECT_EQ(parses->load(), 0) << "listing indices must not parse";
}

// The contract is one retained object per index, not one parse: threads racing
// on the same index may each deserialize it, and the first result published
// wins. See SolutionBlobCache::get(). Several matching table rows commonly wrap
// the same index, so this is the realistic contention case. Run under TSan to
// check the locking.
TEST(SolutionBlobCacheTest, ConcurrentGetOfOneIndexYieldsOneRetainedObjectNotOneParse)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(4, parses);

    constexpr int                             kThreads = 16;
    std::vector<std::thread>                  threads;
    std::vector<std::shared_ptr<FakeSolution>> got(kThreads);

    for(int t = 0; t < kThreads; t++)
        threads.emplace_back([&, t]() { got[t] = cache->get(2); });
    for(auto& thread : threads)
        thread.join();

    for(int t = 0; t < kThreads; t++)
    {
        ASSERT_NE(got[t], nullptr);
        EXPECT_EQ(got[t], got[0]) << "thread " << t << " saw a different object";
    }
    EXPECT_EQ(cache->materializedCount(), 1u);

    // Spelling out the half of the contract the assertions above cannot see:
    // duplicate parses are allowed, one per racing thread at worst, but every
    // one of them beyond the first is discarded rather than retained.
    EXPECT_GE(parses->load(), 1);
    EXPECT_LE(parses->load(), kThreads);
}

TEST(SolutionBlobCacheTest, ConcurrentGetOfDifferentIndicesIsSafe)
{
    auto parses = std::make_shared<std::atomic<int>>(0);
    auto cache  = makeFakeCache(64, parses);

    std::vector<std::thread> threads;
    for(int t = 0; t < 16; t++)
    {
        threads.emplace_back([&, t]() {
            for(int i = t; i < 64; i += 16)
            {
                auto solution = cache->get(i);
                ASSERT_NE(solution, nullptr);
                EXPECT_EQ(solution->index, i);
            }
        });
    }
    for(auto& thread : threads)
        thread.join();

    EXPECT_EQ(cache->materializedCount(), 64u);
    EXPECT_EQ(parses->load(), 64) << "each index should be parsed exactly once";
}

// ---------------------------------------------------------------------------
// 2. Malformed indexed files must fail at load
// ---------------------------------------------------------------------------

namespace
{
    /// Writes an indexed .dat.zlib with a caller-supplied index table and blob.
    ///
    /// The library tree is a single lazy leaf naming `treeIndex`, which is a
    /// genuinely valid tree: a leaf only checks that the index table carries its
    /// index, so it needs no parseable solution in the blob. That matters for the
    /// malformed-table cases below. With a stub tree they would pass even if
    /// table validation were deleted, because reading the tree would fail
    /// anyway; with a valid tree, the table is the only thing left to reject.
    void writeIndexedLibrary(fs::path const&             base,
                             int                         formatVersion,
                             std::vector<int64_t> const& table,
                             std::vector<uint8_t> const& blob,
                             bool                        omitBlob  = false,
                             int64_t                     treeIndex = 0)
    {
        msgpack::sbuffer  buffer;
        msgpack::packer<msgpack::sbuffer> packer(buffer);

        packer.pack_map(omitBlob ? 3 : 4);

        packer.pack(std::string("format_version"));
        packer.pack(formatVersion);

        packer.pack(std::string("solutions_index"));
        packer.pack(table);

        if(!omitBlob)
        {
            packer.pack(std::string("solutions_blob"));
            packer.pack_bin(static_cast<uint32_t>(blob.size()));
            packer.pack_bin_body(reinterpret_cast<const char*>(blob.data()),
                                 static_cast<uint32_t>(blob.size()));
        }

        packer.pack(std::string("library"));
        packer.pack_map(2);
        packer.pack(std::string("type"));
        packer.pack(std::string("Single"));
        packer.pack(std::string("index"));
        packer.pack(static_cast<int>(treeIndex));

        std::vector<uint8_t> raw(buffer.data(), buffer.data() + buffer.size());
        writeFile(fs::path(base.string() + ".zlib"), deflateBytes(raw));
    }
} // namespace

using IndexedLibraryLoadTest = TempDirTest;

// Positive control for every rejection case below: this is the same shape of
// file with a well-formed table, so it must load. Without it, a loader that
// rejected everything would pass the whole group.
TEST_F(IndexedLibraryLoadTest, AcceptsWellFormedIndexedLibrary)
{
    auto base = tmpDir / "TensileLibrary_ok.dat";
    writeIndexedLibrary(base, 2, {0, 0, 1}, {0x01});

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    ASSERT_NE(library, nullptr) << "a well-formed indexed library must load";

    auto* master
        = dynamic_cast<MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>*>(
            library.get());
    ASSERT_NE(master, nullptr);
    ASSERT_NE(master->blobCache, nullptr);
    EXPECT_EQ(master->blobCache->size(), 1u);
    EXPECT_EQ(master->blobCache->materializedCount(), 0u)
        << "loading must not parse the blob";
}

// A tree naming an index the table does not carry has to fail the load, the way
// a dangling reference does on the eager path, rather than surfacing later as a
// query that returns nothing.
TEST_F(IndexedLibraryLoadTest, RejectsTreeReferenceToIndexNotInTable)
{
    auto base = tmpDir / "TensileLibrary_dangling.dat";
    writeIndexedLibrary(base, 2, {0, 0, 1}, {0x01}, /*omitBlob=*/false, /*treeIndex=*/7);

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

// offset + length overflows int64 and wraps negative, so a span check written as
// a sum accepts it and leaves the slice pointing far outside the blob. Loading
// such a file used to succeed and then read wild memory on materialization.
TEST_F(IndexedLibraryLoadTest, RejectsSpanThatOverflowsSignedArithmetic)
{
    constexpr int64_t huge = int64_t(1) << 62;

    auto base = tmpDir / "TensileLibrary_spanoverflow.dat";
    writeIndexedLibrary(base, 2, {0, huge, huge}, {0x01});

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

TEST_F(IndexedLibraryLoadTest, RejectsUnknownFormatVersion)
{
    // A future layout must not be misread as the one this build understands.
    auto base = tmpDir / "TensileLibrary_future.dat";
    writeIndexedLibrary(base, 99, {0, 0, 1}, {0x01});

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

TEST_F(IndexedLibraryLoadTest, RejectsIndexTableThatIsNotTriples)
{
    auto base = tmpDir / "TensileLibrary_ragged.dat";
    writeIndexedLibrary(base, 2, {0, 0}, {0x01});

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

TEST_F(IndexedLibraryLoadTest, RejectsSliceRunningPastTheBlob)
{
    auto base = tmpDir / "TensileLibrary_overrun.dat";
    writeIndexedLibrary(base, 2, {0, 0, 64}, {0x01, 0x02});

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

// An index wider than `int` would be truncated into the slice table, aliasing a
// different solution. The tree here names the truncated value, so a loader
// missing this check would accept the file and resolve that index happily.
TEST_F(IndexedLibraryLoadTest, RejectsIndexTooLargeForInt)
{
    constexpr int64_t aliased = (int64_t(1) << 32) + 5; // narrows to 5

    auto base = tmpDir / "TensileLibrary_wideindex.dat";
    writeIndexedLibrary(base, 2, {aliased, 0, 1}, {0x01}, /*omitBlob=*/false, /*treeIndex=*/5);

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

TEST_F(IndexedLibraryLoadTest, RejectsNegativeFields)
{
    auto base = tmpDir / "TensileLibrary_negative.dat";
    writeIndexedLibrary(base, 2, {0, -8, 1}, {0x01});

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

TEST_F(IndexedLibraryLoadTest, RejectsDuplicateIndex)
{
    auto base = tmpDir / "TensileLibrary_dup.dat";
    // Tree names 5 so that, if the duplicate check were removed, the surviving
    // slice would satisfy the tree and the file would load.
    writeIndexedLibrary(base, 2, {5, 0, 1, 5, 1, 1}, {0x01, 0x02}, /*omitBlob=*/false,
                        /*treeIndex=*/5);

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

TEST_F(IndexedLibraryLoadTest, RejectsMissingBlob)
{
    auto base = tmpDir / "TensileLibrary_noblob.dat";
    writeIndexedLibrary(base, 2, {0, 0, 1}, {}, /*omitBlob=*/true);

    auto library = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    EXPECT_EQ(library, nullptr);
}

// ---------------------------------------------------------------------------
// 3. Parity against a real legacy library, converted in process
// ---------------------------------------------------------------------------

namespace
{
    /// The legacy library the parity tests convert. Registered in
    /// tests/configs/CMakeLists.txt and gunzipped into the test data directory at
    /// configure time, so these run without any manual setup.
    fs::path legacyFixture()
    {
        return TestData::Instance().file("KernelsLite.dat");
    }

    /// Rewrites a legacy {solutions, library} msgpack document into the indexed
    /// layout, re-packing each solution object into the blob. Mirrors what
    /// LibraryIO.writeMsgPackIndexed does on the Python side.
    std::vector<uint8_t> toIndexed(msgpack::object const& legacy)
    {
        std::unordered_map<std::string, msgpack::object> top;
        Serialization::objectToMap(legacy, top);

        auto solutionsIter = top.find("solutions");
        if(solutionsIter == top.end())
            return {};

        auto solutions = solutionsIter->second.as<std::vector<msgpack::object>>();

        std::vector<int64_t> table;
        msgpack::sbuffer     blob;
        for(auto const& solution : solutions)
        {
            std::unordered_map<std::string, msgpack::object> fields;
            Serialization::objectToMap(solution, fields);
            const int64_t index  = fields.at("index").as<int64_t>();
            const size_t  offset = blob.size();
            msgpack::pack(blob, solution);
            table.push_back(index);
            table.push_back(static_cast<int64_t>(offset));
            table.push_back(static_cast<int64_t>(blob.size() - offset));
        }

        msgpack::sbuffer                  out;
        msgpack::packer<msgpack::sbuffer> packer(out);
        const bool hasVersion = top.find("version") != top.end();
        packer.pack_map(hasVersion ? 5 : 4);

        packer.pack(std::string("format_version"));
        packer.pack(2);
        if(hasVersion)
        {
            packer.pack(std::string("version"));
            packer.pack(top.at("version"));
        }
        packer.pack(std::string("solutions_index"));
        packer.pack(table);
        packer.pack(std::string("solutions_blob"));
        packer.pack_bin(static_cast<uint32_t>(blob.size()));
        packer.pack_bin_body(blob.data(), static_cast<uint32_t>(blob.size()));
        packer.pack(std::string("library"));
        packer.pack(top.at("library"));

        return std::vector<uint8_t>(out.data(), out.data() + out.size());
    }
} // namespace

using IndexedLibraryParityTest = TempDirTest;

TEST_F(IndexedLibraryParityTest, IndexedLoadMatchesLegacyLoad)
{
    auto fixture = legacyFixture();
    if(!fs::is_regular_file(fixture))
        GTEST_SKIP() << "no legacy library fixture at " << fixture;

    std::ifstream in(fixture, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.good());
    std::vector<char> bytes(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    auto handle  = msgpack::unpack(bytes.data(), bytes.size());
    auto indexed = toIndexed(handle.get());
    ASSERT_FALSE(indexed.empty()) << "fixture has no solutions array";

    auto indexedBase = tmpDir / "TensileLibrary_indexed.dat";
    writeFile(fs::path(indexedBase.string() + ".zlib"), deflateBytes(indexed));

    auto legacyLib = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(fixture.string());
    if(!legacyLib)
    {
        GTEST_SKIP() << fixture
                     << " does not load; a fixture that predates a required schema field cannot "
                        "be used for parity. Regenerate it (see configs/SolutionLibraries/readme).";
    }
    auto indexedLib
        = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(indexedBase.string());
    ASSERT_NE(indexedLib, nullptr) << "indexed library failed to load";

    auto* legacyMaster
        = dynamic_cast<MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>*>(
            legacyLib.get());
    auto* indexedMaster
        = dynamic_cast<MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>*>(
            indexedLib.get());
    ASSERT_NE(legacyMaster, nullptr);
    ASSERT_NE(indexedMaster, nullptr);

    ASSERT_NE(indexedMaster->blobCache, nullptr) << "indexed load produced no blob cache";
    EXPECT_EQ(indexedMaster->blobCache->size(), legacyMaster->solutions.size());

    // The whole point: the tree is up, but no solution has been parsed.
    EXPECT_EQ(indexedMaster->blobCache->materializedCount(), 0u)
        << "loading an indexed library must not parse any solution";

    // Every index resolves to the same solution the legacy load produced.
    for(auto const& entry : legacyMaster->solutions)
    {
        auto lazy = indexedMaster->resolveSolutionByIndex(entry.first);
        ASSERT_NE(lazy, nullptr) << "index " << entry.first << " did not resolve";
        EXPECT_EQ(lazy->index, entry.second->index);
        EXPECT_EQ(lazy->kernelName, entry.second->kernelName);
        EXPECT_EQ(lazy->solutionName, entry.second->solutionName);
    }
}

TEST_F(IndexedLibraryParityTest, ResolveByIndexWorksWithoutAnyPriorQuery)
{
    // Guards the hipBLASLt algo-index path, which calls getSolutionByIndex
    // without a preceding findBestSolution and used to depend on the shard
    // merge having pre-populated the solutions map.
    auto fixture = legacyFixture();
    if(!fs::is_regular_file(fixture))
        GTEST_SKIP() << "no legacy library fixture at " << fixture;

    std::ifstream in(fixture, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.good());
    std::vector<char> bytes(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    auto handle  = msgpack::unpack(bytes.data(), bytes.size());
    auto indexed = toIndexed(handle.get());
    ASSERT_FALSE(indexed.empty());

    auto base = tmpDir / "TensileLibrary_indexed.dat";
    writeFile(fs::path(base.string() + ".zlib"), deflateBytes(indexed));

    auto lib = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(base.string());
    ASSERT_NE(lib, nullptr);
    auto* master
        = dynamic_cast<MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>*>(
            lib.get());
    ASSERT_NE(master, nullptr);
    ASSERT_NE(master->blobCache, nullptr);

    auto indices = master->blobCache->indices();
    ASSERT_FALSE(indices.empty());

    auto solution = master->resolveSolutionByIndex(indices.front());
    ASSERT_NE(solution, nullptr);
    EXPECT_EQ(solution->index, indices.front());
    EXPECT_EQ(master->blobCache->materializedCount(), 1u)
        << "resolving one index must not materialize the rest";
}

// ---------------------------------------------------------------------------
// 4. A lazy master reaching an indexed shard through a placeholder
// ---------------------------------------------------------------------------
//
// This is how every shipped gfx942 configuration is laid out: a master that
// carries no solutions of its own, a per-arch mapping file, and one placeholder
// per shard. It is also the path where the indexed format has the most to get
// wrong, because a lazily loaded shard has no materialized solutions for the
// parent to merge -- the parent has to adopt the shard's blob cache instead.
//
// A shard can be loaded from either side, and both have to end up with the
// parent able to resolve the shard's indices:
//   * master-first, through MasterSolutionLibrary::loadLibrary via the mapping;
//   * placeholder-first, when a query descends into the placeholder node.

namespace
{
    using Master  = MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>;
    using Caching = CachingLibrary<ContractionProblemGemm, ContractionSolution>;
    using Holder  = PlaceholderLibrary<ContractionProblemGemm, ContractionSolution>;

    constexpr const char* kArch       = "gfx9999";
    constexpr const char* kShardPrefix = "TensileLibrary_shard_gfx9999";

    /// A master with no solutions of its own whose whole tree is one placeholder,
    /// matching the shape of a shipped lazy master.
    void writeLazyMaster(fs::path const& path, std::string const& shardPrefix)
    {
        msgpack::sbuffer                  buffer;
        msgpack::packer<msgpack::sbuffer> packer(buffer);

        packer.pack_map(2);
        packer.pack(std::string("solutions"));
        packer.pack_array(0);
        packer.pack(std::string("library"));
        packer.pack_map(2);
        packer.pack(std::string("type"));
        packer.pack(std::string("Placeholder"));
        packer.pack(std::string("value"));
        packer.pack(shardPrefix);

        writeFile(path, std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size()));
    }

    /// The per-arch mapping file loadLibrary() consults: index -> shard prefix.
    void writeMappingFile(fs::path const& path, std::map<std::string, std::string> const& entries)
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, entries);

        writeFile(path, std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size()));
    }

    Holder* placeholderOf(Master* master)
    {
        auto caching = std::dynamic_pointer_cast<Caching>(master->library);
        if(!caching)
            return nullptr;
        return dynamic_cast<Holder*>(caching->library().get());
    }

    struct PlaceholderIndexedShardTest : public TempDirTest
    {
        std::vector<int> shardIndices;

        fs::path masterPath() const
        {
            return tmpDir / ("TensileLibrary_lazy_" + std::string(kArch) + ".dat");
        }

        /// Lays out a lazy master, its mapping file, and one indexed shard built
        /// from the committed legacy fixture. Returns false when the fixture is
        /// unavailable or predates the current schema, so callers can skip.
        bool layOutLibrary()
        {
            auto fixture = legacyFixture();
            if(!fs::is_regular_file(fixture))
                return false;

            std::ifstream in(fixture, std::ios::binary | std::ios::ate);
            if(!in.good())
                return false;
            std::vector<char> bytes(static_cast<size_t>(in.tellg()));
            in.seekg(0);
            in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));

            auto handle  = msgpack::unpack(bytes.data(), bytes.size());
            auto indexed = toIndexed(handle.get());
            if(indexed.empty())
                return false;

            // Confirm the fixture still loads before building anything on it.
            auto legacy = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(
                fixture.string());
            if(!legacy)
                return false;
            auto* legacyMaster = dynamic_cast<Master*>(legacy.get());
            if(!legacyMaster || legacyMaster->solutions.empty())
                return false;
            for(auto const& entry : legacyMaster->solutions)
                shardIndices.push_back(entry.first);

            writeFile(tmpDir / (std::string(kShardPrefix) + ".dat.zlib"), deflateBytes(indexed));
            writeLazyMaster(masterPath(), kShardPrefix);
            // One range starting at 0, so every index routes to this shard.
            writeMappingFile(tmpDir
                                 / ("TensileLiteLibrary_lazy_" + std::string(kArch)
                                    + "_Mapping.dat"),
                             {{"0", kShardPrefix}});
            return true;
        }
    };
} // namespace

TEST_F(PlaceholderIndexedShardTest, MasterFirstResolvesShardIndexWithNoPriorQuery)
{
    if(!layOutLibrary())
        GTEST_SKIP() << "legacy fixture unavailable or stale; see configs/SolutionLibraries/readme";

    auto lib = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(masterPath().string());
    ASSERT_NE(lib, nullptr);
    auto* master = dynamic_cast<Master*>(lib.get());
    ASSERT_NE(master, nullptr);

    ASSERT_TRUE(master->initLibraryMapping(masterPath().string()));
    EXPECT_TRUE(master->solutions.empty()) << "a lazy master carries no solutions itself";
    EXPECT_EQ(master->blobCache, nullptr);
    EXPECT_TRUE(master->solutionSources.empty()) << "no shard should be loaded yet";

    // The hipBLASLt algo-index path: a lookup with no preceding selection.
    const int index    = shardIndices.front();
    auto      solution = master->resolveSolutionByIndex(index);

    ASSERT_NE(solution, nullptr) << "index " << index << " did not resolve through the shard";
    EXPECT_EQ(solution->index, index);
    EXPECT_EQ(solution->codeObjectFilename.load(), std::string(kShardPrefix) + ".co")
        << "the shard's code object name must reach solutions parsed out of its blob";

    ASSERT_EQ(master->solutionSources.size(), 1u) << "one cache registered per shard";
    auto shardCache = master->solutionSources.begin()->second;
    ASSERT_NE(shardCache, nullptr);
    EXPECT_EQ(shardCache->size(), shardIndices.size());
    EXPECT_EQ(shardCache->materializedCount(), 1u)
        << "resolving one index must not parse the rest of the shard";
}

TEST_F(PlaceholderIndexedShardTest, PlaceholderFirstPublishesItsCacheToTheParent)
{
    if(!layOutLibrary())
        GTEST_SKIP() << "legacy fixture unavailable or stale; see configs/SolutionLibraries/readme";

    auto lib = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(masterPath().string());
    ASSERT_NE(lib, nullptr);
    auto* master = dynamic_cast<Master*>(lib.get());
    ASSERT_NE(master, nullptr);

    // No mapping file consulted here on purpose: the shard arrives because a
    // query descended into the placeholder, not because loadLibrary fetched it.
    auto* placeholder = placeholderOf(master);
    ASSERT_NE(placeholder, nullptr) << "master's tree is not a placeholder";
    ASSERT_TRUE(placeholder->loadPlaceholderLibrary());

    ASSERT_EQ(master->solutionSources.size(), 1u)
        << "the placeholder must publish its shard cache to the owning master";

    const int index    = shardIndices.front();
    auto      solution = master->resolveSolutionByIndex(index);
    ASSERT_NE(solution, nullptr) << "parent could not reach a shard the placeholder loaded";
    EXPECT_EQ(solution->index, index);

    // Both routes have to agree on the object, which is what stops a caller that
    // selects through the tree and re-fetches by index from getting two.
    auto viaCache = master->solutionSources.begin()->second->get(index);
    EXPECT_EQ(solution, viaCache) << "index lookup and the shard's own cache disagree";
}

TEST_F(PlaceholderIndexedShardTest, ShardRegistersOnceWhenReachedFromBothSides)
{
    if(!layOutLibrary())
        GTEST_SKIP() << "legacy fixture unavailable or stale; see configs/SolutionLibraries/readme";

    auto lib = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(masterPath().string());
    ASSERT_NE(lib, nullptr);
    auto* master = dynamic_cast<Master*>(lib.get());
    ASSERT_NE(master, nullptr);
    ASSERT_TRUE(master->initLibraryMapping(masterPath().string()));

    // Master first, then the placeholder for the same shard. Registering the
    // shard twice would retain a second copy of its blob.
    ASSERT_NE(master->resolveSolutionByIndex(shardIndices.front()), nullptr);
    ASSERT_EQ(master->solutionSources.size(), 1u);

    auto* placeholder = placeholderOf(master);
    ASSERT_NE(placeholder, nullptr);
    placeholder->loadPlaceholderLibrary();

    EXPECT_EQ(master->solutionSources.size(), 1u)
        << "a shard reached from both sides must still register exactly once";
}

TEST_F(PlaceholderIndexedShardTest, EnumerationPublishesLoadedShardSolutions)
{
    if(!layOutLibrary())
        GTEST_SKIP() << "legacy fixture unavailable or stale; see configs/SolutionLibraries/readme";

    auto lib = LoadLibraryFile<ContractionProblemGemm, ContractionSolution>(masterPath().string());
    ASSERT_NE(lib, nullptr);
    auto* master = dynamic_cast<Master*>(lib.get());
    ASSERT_NE(master, nullptr);
    ASSERT_TRUE(master->initLibraryMapping(masterPath().string()));
    ASSERT_NE(master->resolveSolutionByIndex(shardIndices.front()), nullptr);

    // What the benchmark client's enumeration iterator depends on: the solutions
    // map it indexes has to be filled, which materializing the caches alone does
    // not do, because leaves resolve through the cache and never touch that map.
    master->materializeAllSolutions();

    EXPECT_EQ(master->solutions.size(), shardIndices.size());
    for(int index : shardIndices)
        EXPECT_NE(master->solutions.find(index), master->solutions.end())
            << "index " << index << " missing after enumeration";
}
