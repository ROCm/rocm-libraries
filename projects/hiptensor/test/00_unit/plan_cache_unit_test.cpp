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

// Plan cache:
//  - the number of cache lines stays within the capacity, and the least recently used line is
//    the one evicted;
//  - the key tells apart problems that differ in device, backend, output tensor E, operators or
//    tensor ranks;
//  - the cache file round-trips, and the cache stays usable after it was written;
//  - files of earlier versions, of another version, device or backend set, damaged files and
//    files with more lines than the cache holds are refused, leaving the cache unchanged;
//  - through the API: contractions keep running after the plan cache was written, and a plan
//    created from a cache line read back from the file computes the right result.
// The contraction results are compared with a plain loop with 64-bit offsets, not with the CPU
// reference. The data are small integers, so every result is exact and compared bit for bit.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "data_types.hpp"
#include "plan_cache.hpp"

namespace
{
    using hiptensor::PlanCache;

    // Any device identity will do: the cache only compares them.
    constexpr char const* archName = "gfx942:sramecc+:xnack-";
    constexpr uint32_t    cuCount  = 100;

    // The backend whose solutions the key tests store.
    std::string const ckBackend = "ck";

    // Offsets in the file: magic, format version, hipTensor major version.
    constexpr std::size_t formatVersionOffset = 4;
    constexpr std::size_t majorVersionOffset  = 8;

    std::string tempFile(char const* name)
    {
        return ::testing::TempDir() + "hiptensor_plan_cache_unit_test_" + name + ".bin";
    }

    std::vector<char> readBytes(std::string const& fileName)
    {
        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(file),
                                 std::istreambuf_iterator<char>());
    }

    void writeBytes(std::string const& fileName, std::vector<char> const& bytes)
    {
        std::ofstream file(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(file.good());
    }

    // Cache lines 1, 2, 3 with uids 101, 102, 103, then line 1 used again, so that the order
    // from least to most recently used is 2, 3, 1.
    void fillThreeLines(PlanCache& cache)
    {
        cache.addCacheLine(1, 101);
        cache.addCacheLine(2, 102);
        cache.addCacheLine(3, 103);
        EXPECT_EQ(cache.querySolutionUidByHash(1), 101u);
    }

    // Reading fileName into a cache holding line 7 must fail with `expected` and keep line 7.
    void expectRefused(std::string const& fileName,
                       hiptensorStatus_t  expected,
                       std::string const& readerArchName = archName,
                       uint32_t           readerCuCount  = cuCount)
    {
        PlanCache reader(readerArchName, readerCuCount);
        reader.addCacheLine(7, 107);

        uint32_t numCachelinesRead = 12345;
        EXPECT_EQ(reader.readFile(fileName.c_str(), &numCachelinesRead), expected);
        EXPECT_EQ(numCachelinesRead, 12345u);
        EXPECT_EQ(reader.getCachelinesNum(), 1u);
        EXPECT_EQ(reader.querySolutionUidByHash(7), 107u);
    }

    std::string writeThreeLineFile(char const* name)
    {
        PlanCache cache(archName, cuCount);
        fillThreeLines(cache);
        auto fileName = tempFile(name);
        EXPECT_EQ(cache.writeFile(fileName.c_str()), HIPTENSOR_STATUS_SUCCESS);
        return fileName;
    }
} // namespace

// ---- Eviction ----

TEST(PlanCacheEvictionTest, SizeStaysWithinCapacity)
{
    PlanCache cache(archName, cuCount);
    cache.resize(4);
    for(PlanCache::HashId hashId = 1; hashId <= 10; hashId++)
    {
        cache.addCacheLine(hashId, 100 + hashId);
        EXPECT_LE(cache.getCachelinesNum(), 4u);
    }

    EXPECT_EQ(cache.getCachelinesNum(), 4u);
    for(PlanCache::HashId hashId = 1; hashId <= 6; hashId++)
    {
        EXPECT_EQ(cache.querySolutionUidByHash(hashId), 0u) << "line " << hashId;
    }
    for(PlanCache::HashId hashId = 7; hashId <= 10; hashId++)
    {
        EXPECT_EQ(cache.querySolutionUidByHash(hashId), 100 + hashId) << "line " << hashId;
    }
}

TEST(PlanCacheEvictionTest, LeastRecentlyUsedLineIsEvicted)
{
    PlanCache cache(archName, cuCount);
    cache.resize(3);
    fillThreeLines(cache);

    // Line 2 is the least recently used.
    cache.addCacheLine(4, 104);

    EXPECT_EQ(cache.getCachelinesNum(), 3u);
    EXPECT_EQ(cache.querySolutionUidByHash(2), 0u);
    EXPECT_EQ(cache.querySolutionUidByHash(1), 101u);
    EXPECT_EQ(cache.querySolutionUidByHash(3), 103u);
    EXPECT_EQ(cache.querySolutionUidByHash(4), 104u);
}

TEST(PlanCacheEvictionTest, UpdatingALineOfAFullCacheEvictsNothing)
{
    PlanCache cache(archName, cuCount);
    cache.resize(2);
    cache.addCacheLine(1, 101);
    cache.addCacheLine(2, 102);
    EXPECT_EQ(cache.querySolutionUidByHash(1), 101u);

    // Line 2 is the least recently used, but no line has to make room.
    cache.addCacheLine(1, 201);

    EXPECT_EQ(cache.getCachelinesNum(), 2u);
    EXPECT_EQ(cache.querySolutionUidByHash(1), 201u);
    EXPECT_EQ(cache.querySolutionUidByHash(2), 102u);
}

TEST(PlanCacheEvictionTest, ResizeKeepsMostRecentlyUsedLines)
{
    PlanCache cache(archName, cuCount);
    fillThreeLines(cache);

    cache.resize(1);

    EXPECT_EQ(cache.getCachelinesNum(), 1u);
    EXPECT_EQ(cache.querySolutionUidByHash(1), 101u);
}

// A copy of the LRU heap owns its elements: emptying the copy leaves the original intact.
TEST(PlanCacheEvictionTest, HeapCopyIsIndependent)
{
    hiptensor::UpdatablePriorityQueue<PlanCache::HashId, PlanCache::UseStamp> heap;
    heap.push(1, 30);
    heap.push(2, 10);
    heap.push(3, 20);
    {
        auto copy = heap;
        while(!copy.empty())
        {
            copy.pop();
        }
        hiptensor::UpdatablePriorityQueue<PlanCache::HashId, PlanCache::UseStamp> assigned;
        assigned.push(9, 1);
        assigned = heap;
        assigned.pop();
    }

    ASSERT_EQ(heap.size(), 3u);
    EXPECT_EQ(heap.top().first, 2u);
    heap.pop();
    EXPECT_EQ(heap.top().first, 3u);
    heap.pop();
    EXPECT_EQ(heap.top().first, 1u);
}

// ---- Key ----

namespace
{
    hiptensorTensorDescriptor tensor(std::vector<std::size_t> lengths,
                                     std::vector<std::size_t> strides)
    {
        hiptensorTensorDescriptor desc{};
        desc.mType                 = HIPTENSOR_R_32F;
        desc.mLengths              = std::move(lengths);
        desc.mStrides              = std::move(strides);
        desc.mAlignmentRequirement = 4;
        return desc;
    }

    // A trinary contraction E[m,n] = A[m,k] * B[k,u] * C[u,n].
    struct TrinaryDesc
    {
        hiptensorTensorDescriptor    a = tensor({4, 5}, {1, 4});
        hiptensorTensorDescriptor    b = tensor({5, 6}, {1, 5});
        hiptensorTensorDescriptor    c = tensor({6, 7}, {1, 6});
        hiptensorTensorDescriptor    e = tensor({4, 7}, {1, 4});
        hiptensorOperationDescriptor op{};

        TrinaryDesc()
        {
            op.mOperationType = HIPTENSOR_CONTRACTION_TRINARY;
            op.mDescA         = &a;
            op.mModeA         = {'m', 'k'};
            op.mDescB         = &b;
            op.mModeB         = {'k', 'u'};
            op.mDescC         = &c;
            op.mModeC         = {'u', 'n'};
            op.mDescE         = &e;
            op.mModeE         = {'m', 'n'};
            op.mOpA = op.mOpB = op.mOpC = op.mOpD = HIPTENSOR_OP_IDENTITY;
            op.mOpAC = op.mOpABC = HIPTENSOR_OP_IDENTITY;
            op.mDescCompute      = HIPTENSOR_COMPUTE_DESC_32F;
        }
    };
} // namespace

TEST(PlanCacheKeyTest, DeviceIsPartOfTheKey)
{
    TrinaryDesc desc;
    PlanCache   cache(archName, cuCount);
    PlanCache   sameDevice(archName, cuCount);
    PlanCache   otherArch("gfx950:sramecc+:xnack-", cuCount);
    PlanCache   otherCuCount(archName, cuCount + 1);

    auto key = cache.getHashID(&desc.op, ckBackend);
    EXPECT_EQ(sameDevice.getHashID(&desc.op, ckBackend), key);
    EXPECT_NE(otherArch.getHashID(&desc.op, ckBackend), key);
    EXPECT_NE(otherCuCount.getHashID(&desc.op, ckBackend), key);
}

TEST(PlanCacheKeyTest, OutputTensorEIsPartOfTheKey)
{
    PlanCache   cache(archName, cuCount);
    TrinaryDesc desc;
    auto        key = cache.getHashID(&desc.op, ckBackend);

    TrinaryDesc otherStrides;
    otherStrides.e = tensor({4, 7}, {7, 1});
    EXPECT_NE(cache.getHashID(&otherStrides.op, ckBackend), key);

    TrinaryDesc otherModes;
    otherModes.e         = tensor({7, 4}, {1, 7});
    otherModes.op.mModeE = {'n', 'm'};
    EXPECT_NE(cache.getHashID(&otherModes.op, ckBackend), key);

    TrinaryDesc otherType;
    otherType.e.mType = HIPTENSOR_R_16F;
    EXPECT_NE(cache.getHashID(&otherType.op, ckBackend), key);
}

TEST(PlanCacheKeyTest, OperatorsArePartOfTheKey)
{
    PlanCache   cache(archName, cuCount);
    TrinaryDesc desc;
    auto        key = cache.getHashID(&desc.op, ckBackend);

    TrinaryDesc opA;
    opA.op.mOpA = HIPTENSOR_OP_SQRT;
    EXPECT_NE(cache.getHashID(&opA.op, ckBackend), key);

    TrinaryDesc opD;
    opD.op.mOpD = HIPTENSOR_OP_SQRT;
    EXPECT_NE(cache.getHashID(&opD.op, ckBackend), key);

    TrinaryDesc opAC;
    opAC.op.mOpAC = HIPTENSOR_OP_ADD;
    EXPECT_NE(cache.getHashID(&opAC.op, ckBackend), key);

    TrinaryDesc opABC;
    opABC.op.mOpABC = HIPTENSOR_OP_MUL;
    EXPECT_NE(cache.getHashID(&opABC.op, ckBackend), key);
}

TEST(PlanCacheKeyTest, BackendIsPartOfTheKey)
{
    TrinaryDesc desc;
    PlanCache   cache(archName, cuCount);

    auto key = cache.getHashID(&desc.op, ckBackend);
    EXPECT_EQ(cache.getHashID(&desc.op, ckBackend), key);
    EXPECT_NE(cache.getHashID(&desc.op, "rocke"), key);

    // A solution cached for one backend is not found for another.
    cache.addCacheLine(key, 101);
    EXPECT_EQ(cache.querySolutionUid(&desc.op, ckBackend), 101u);
    EXPECT_EQ(cache.querySolutionUid(&desc.op, "rocke"), 0u);
}

// A[a,b] with B[c] and A[a] with B[b,c] concatenate to the same lengths, strides and modes.
TEST(PlanCacheKeyTest, RanksArePartOfTheKey)
{
    PlanCache cache(archName, cuCount);

    hiptensorTensorDescriptor    a2 = tensor({2, 3}, {1, 2});
    hiptensorTensorDescriptor    b1 = tensor({4}, {1});
    hiptensorOperationDescriptor first{};
    first.mOperationType = HIPTENSOR_ELEMENTWISE_BINARY;
    first.mDescA         = &a2;
    first.mModeA         = {'a', 'b'};
    first.mDescB         = &b1;
    first.mModeB         = {'c'};

    hiptensorTensorDescriptor    a1 = tensor({2}, {1});
    hiptensorTensorDescriptor    b2 = tensor({3, 4}, {2, 1});
    hiptensorOperationDescriptor second{};
    second.mOperationType = HIPTENSOR_ELEMENTWISE_BINARY;
    second.mDescA         = &a1;
    second.mModeA         = {'a'};
    second.mDescB         = &b2;
    second.mModeB         = {'b', 'c'};

    EXPECT_NE(cache.getHashID(&first, ckBackend), cache.getHashID(&second, ckBackend));
}

// ---- File ----

TEST(PlanCacheFileTest, RoundTripKeepsLinesAndUseOrder)
{
    auto fileName = writeThreeLineFile("round_trip");

    PlanCache reader(archName, cuCount);
    reader.addCacheLine(7, 107);
    uint32_t numCachelinesRead = 0;
    ASSERT_EQ(reader.readFile(fileName.c_str(), &numCachelinesRead), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numCachelinesRead, 3u);
    EXPECT_EQ(reader.getCachelinesNum(), 3u);

    // The lines read replace those of the cache.
    EXPECT_EQ(reader.querySolutionUidByHash(7), 0u);

    // Least recently used first: 2, then 3; line 1 was used last.
    reader.resize(1);
    EXPECT_EQ(reader.querySolutionUidByHash(1), 101u);

    PlanCache again(archName, cuCount);
    ASSERT_EQ(again.readFile(fileName.c_str(), &numCachelinesRead), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(again.querySolutionUidByHash(2), 102u);
    EXPECT_EQ(again.querySolutionUidByHash(3), 103u);

    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, EmptyCacheRoundTrips)
{
    PlanCache cache(archName, cuCount);
    auto      fileName = tempFile("empty");
    ASSERT_EQ(cache.writeFile(fileName.c_str()), HIPTENSOR_STATUS_SUCCESS);

    PlanCache reader(archName, cuCount);
    reader.addCacheLine(7, 107);
    uint32_t numCachelinesRead = 12345;
    ASSERT_EQ(reader.readFile(fileName.c_str(), &numCachelinesRead), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numCachelinesRead, 0u);
    EXPECT_EQ(reader.getCachelinesNum(), 0u);

    std::remove(fileName.c_str());
}

// Writing the file must leave the cache usable: before, the write freed the heap's elements.
TEST(PlanCacheFileTest, CacheIsUsableAfterWrite)
{
    auto fileName = tempFile("use_after_write");
    {
        PlanCache cache(archName, cuCount);
        cache.resize(3);
        fillThreeLines(cache);
        ASSERT_EQ(cache.writeFile(fileName.c_str()), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(cache.writeFile(fileName.c_str()), HIPTENSOR_STATUS_SUCCESS);

        EXPECT_EQ(cache.querySolutionUidByHash(3), 103u);
        cache.addCacheLine(4, 104);
        cache.addCacheLine(5, 105);

        // Lines 2 and then 1 were the least recently used, so lines 4 and 5 evicted them.
        EXPECT_EQ(cache.getCachelinesNum(), 3u);
        EXPECT_EQ(cache.querySolutionUidByHash(1), 0u);
        EXPECT_EQ(cache.querySolutionUidByHash(2), 0u);
        EXPECT_EQ(cache.querySolutionUidByHash(3), 103u);
        EXPECT_EQ(cache.querySolutionUidByHash(4), 104u);
        EXPECT_EQ(cache.querySolutionUidByHash(5), 105u);
    }
    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, MissingFileIsAnIoError)
{
    expectRefused(tempFile("does_not_exist"), HIPTENSOR_STATUS_IO_ERROR);
}

// The layout written by earlier versions: capacity, the lines, then the use times.
TEST(PlanCacheFileTest, FileOfAnEarlierVersionIsRefused)
{
    std::vector<char> bytes;
    auto              append = [&bytes](auto value) {
        char raw[sizeof(value)];
        std::memcpy(raw, &value, sizeof(value));
        bytes.insert(bytes.end(), raw, raw + sizeof(value));
    };
    append(uint32_t{128});
    append(std::size_t{1});
    append(std::size_t{1});
    append(std::size_t{101});
    append(std::size_t{1});
    append(std::size_t{1});
    append(int64_t{0});

    auto fileName = tempFile("earlier_version");
    ASSERT_NO_FATAL_FAILURE(writeBytes(fileName, bytes));
    expectRefused(fileName, HIPTENSOR_STATUS_INVALID_VALUE);
    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, FileOfAnotherVersionIsRefused)
{
    auto fileName = writeThreeLineFile("other_version");
    auto bytes    = readBytes(fileName);
    ASSERT_GT(bytes.size(), majorVersionOffset);

    auto otherFormat = bytes;
    otherFormat[formatVersionOffset] ^= 0x7;
    ASSERT_NO_FATAL_FAILURE(writeBytes(fileName, otherFormat));
    expectRefused(fileName, HIPTENSOR_STATUS_INVALID_VALUE);

    auto otherLibrary = bytes;
    otherLibrary[majorVersionOffset] ^= 0x7;
    ASSERT_NO_FATAL_FAILURE(writeBytes(fileName, otherLibrary));
    expectRefused(fileName, HIPTENSOR_STATUS_INVALID_VALUE);

    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, FileOfAnotherDeviceIsRefused)
{
    auto fileName = writeThreeLineFile("other_device");
    expectRefused(fileName, HIPTENSOR_STATUS_INVALID_VALUE, "gfx950:sramecc+:xnack-", cuCount);
    expectRefused(fileName, HIPTENSOR_STATUS_INVALID_VALUE, archName, cuCount + 1);
    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, FileOfAnotherBackendSetIsRefused)
{
    std::vector<hiptensor::PlanCacheBackend> const backends = {{"ck", 1u}, {"rocke", 1u}};

    PlanCache writer(archName, cuCount, backends);
    fillThreeLines(writer);
    auto fileName = tempFile("backend_set");
    ASSERT_EQ(writer.writeFile(fileName.c_str()), HIPTENSOR_STATUS_SUCCESS);

    std::vector<std::vector<hiptensor::PlanCacheBackend>> const others
        = {{}, {{"ck", 1u}}, {{"ck", 2u}, {"rocke", 1u}}, {{"rocke", 1u}, {"ck", 1u}}};
    for(auto const& other : others)
    {
        PlanCache reader(archName, cuCount, other);
        reader.addCacheLine(7, 107);
        uint32_t numCachelinesRead = 12345;
        EXPECT_EQ(reader.readFile(fileName.c_str(), &numCachelinesRead),
                  HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(numCachelinesRead, 12345u);
        EXPECT_EQ(reader.querySolutionUidByHash(7), 107u);
    }

    PlanCache reader(archName, cuCount, backends);
    uint32_t  numCachelinesRead = 0;
    EXPECT_EQ(reader.readFile(fileName.c_str(), &numCachelinesRead), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numCachelinesRead, 3u);
    EXPECT_EQ(reader.querySolutionUidByHash(2), 102u);

    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, DamagedFileIsRefused)
{
    auto fileName = writeThreeLineFile("damaged");
    auto bytes    = readBytes(fileName);
    ASSERT_GT(bytes.size(), 24u);

    // A changed byte in the last line's uid, just before the checksum.
    auto changed = bytes;
    changed[changed.size() - 9] ^= 0x1;
    ASSERT_NO_FATAL_FAILURE(writeBytes(fileName, changed));
    expectRefused(fileName, HIPTENSOR_STATUS_IO_ERROR);

    auto truncated = bytes;
    truncated.pop_back();
    ASSERT_NO_FATAL_FAILURE(writeBytes(fileName, truncated));
    expectRefused(fileName, HIPTENSOR_STATUS_IO_ERROR);

    auto extended = bytes;
    extended.push_back(0);
    ASSERT_NO_FATAL_FAILURE(writeBytes(fileName, extended));
    expectRefused(fileName, HIPTENSOR_STATUS_IO_ERROR);

    ASSERT_NO_FATAL_FAILURE(
        writeBytes(fileName, std::vector<char>(bytes.begin(), bytes.begin() + 2)));
    expectRefused(fileName, HIPTENSOR_STATUS_IO_ERROR);

    std::remove(fileName.c_str());
}

TEST(PlanCacheFileTest, FileWithMoreLinesThanTheCapacityIsRefused)
{
    auto fileName = writeThreeLineFile("too_many_lines");

    PlanCache reader(archName, cuCount);
    reader.resize(2);
    reader.addCacheLine(7, 107);
    uint32_t numCachelinesRead = 0;
    EXPECT_EQ(reader.readFile(fileName.c_str(), &numCachelinesRead),
              HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE);
    EXPECT_EQ(numCachelinesRead, 3u);
    EXPECT_EQ(reader.getCachelinesNum(), 1u);
    EXPECT_EQ(reader.querySolutionUidByHash(7), 107u);

    // The capacity of 2 is kept.
    reader.addCacheLine(8, 108);
    reader.addCacheLine(9, 109);
    EXPECT_EQ(reader.getCachelinesNum(), 2u);

    std::remove(fileName.c_str());
}

// ---- Through the API ----

namespace
{
    struct HandleGuard
    {
        hiptensorHandle_t handle = nullptr;
        ~HandleGuard()
        {
            if(handle)
            {
                static_cast<void>(hiptensorDestroy(handle));
            }
        }
    };

    struct TensorDescGuard
    {
        hiptensorTensorDescriptor_t desc = nullptr;
        ~TensorDescGuard()
        {
            if(desc)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(desc));
            }
        }
    };

    struct OpDescGuard
    {
        hiptensorOperationDescriptor_t desc = nullptr;
        ~OpDescGuard()
        {
            if(desc)
            {
                static_cast<void>(hiptensorDestroyOperationDescriptor(desc));
            }
        }
    };

    struct PlanGuard
    {
        hiptensorPlan_t plan = nullptr;
        ~PlanGuard()
        {
            if(plan)
            {
                static_cast<void>(hiptensorDestroyPlan(plan));
            }
        }
    };

    struct DeviceBuffer
    {
        void* ptr = nullptr;
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
    };

    // Sets an environment variable for the lifetime of the object, then restores it.
    class ScopedEnvironmentVariable
    {
    public:
        ScopedEnvironmentVariable(char const* name, char const* value)
            : mName(name)
        {
            if(char const* previous = std::getenv(name))
            {
                mPrevious    = previous;
                mHadPrevious = true;
            }
            set(value);
        }

        ~ScopedEnvironmentVariable()
        {
            set(mHadPrevious ? mPrevious.c_str() : nullptr);
        }

        ScopedEnvironmentVariable(ScopedEnvironmentVariable const&)            = delete;
        ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable const&) = delete;

    private:
        void set(char const* value)
        {
#ifdef _WIN32
            _putenv_s(mName.c_str(), value != nullptr ? value : "");
#else
            if(value != nullptr)
            {
                setenv(mName.c_str(), value, 1);
            }
            else
            {
                unsetenv(mName.c_str());
            }
#endif
        }

        std::string mName;
        std::string mPrevious;
        bool        mHadPrevious = false;
    };

    bool planCacheDisabled()
    {
        char const* value = std::getenv("HIPTENSOR_DISABLE_PLAN_CACHE");
        return value != nullptr
               && (std::string(value) == "1" || std::string(value) == "ON"
                   || std::string(value) == "on");
    }

    // D[m,n] = alpha * sum_k A[m,k] * B[n,k], packed column-major, m = n = k = 8.
    constexpr int64_t extent = 8;

    hiptensorStatus_t createDesc(hiptensorHandle_t handle, TensorDescGuard& guard)
    {
        int64_t const lengths[] = {extent, extent};
        int64_t const strides[] = {1, extent};
        return hiptensorCreateTensorDescriptor(
            handle, &guard.desc, 2, lengths, strides, HIPTENSOR_R_32F, 4);
    }

    struct Contraction
    {
        TensorDescGuard descA, descB, descD;
        OpDescGuard     op;
        PlanGuard       plan;
    };

    // Returns false, without failing, when the device has no 32F contraction support.
    bool createContraction(hiptensorHandle_t handle, Contraction& c)
    {
        int32_t const modesA[] = {'m', 'k'};
        int32_t const modesB[] = {'n', 'k'};
        int32_t const modesD[] = {'m', 'n'};
        EXPECT_EQ(createDesc(handle, c.descA), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle, c.descB), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle, c.descD), HIPTENSOR_STATUS_SUCCESS);
        auto status = hiptensorCreateContraction(handle,
                                                 &c.op.desc,
                                                 c.descA.desc,
                                                 modesA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 c.descB.desc,
                                                 modesB,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 c.descD.desc,
                                                 modesD,
                                                 HIPTENSOR_COMPUTE_DESC_32F);
        if(status != HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        }
        return status == HIPTENSOR_STATUS_SUCCESS;
    }

    // A null preference: ALGO_DEFAULT and the default cache mode, PEDANTIC.
    hiptensorStatus_t createPlan(hiptensorHandle_t handle, Contraction& c)
    {
        return hiptensorCreatePlan(handle, &c.plan.plan, c.op.desc, nullptr, 0);
    }

    void contractAndCheck(hiptensorHandle_t handle, Contraction const& c)
    {
        auto const         count = static_cast<std::size_t>(extent * extent);
        std::vector<float> a(count), b(count);
        for(std::size_t i = 0; i < count; i++)
        {
            a[i] = static_cast<float>(static_cast<int>((i * 5 + 2) % 7) - 3);
            b[i] = static_cast<float>(static_cast<int>((i * 3 + 1) % 5) - 2);
        }
        float const        alpha = 2.0f;
        std::vector<float> d(count);
        for(int64_t n = 0; n < extent; n++)
        {
            for(int64_t m = 0; m < extent; m++)
            {
                double acc = 0.0;
                for(int64_t k = 0; k < extent; k++)
                {
                    acc += static_cast<double>(a[static_cast<std::size_t>(m + k * extent)])
                           * static_cast<double>(b[static_cast<std::size_t>(n + k * extent)]);
                }
                d[static_cast<std::size_t>(m + n * extent)] = static_cast<float>(alpha * acc);
            }
        }

        DeviceBuffer devA, devB, devD;
        auto const   bytes = count * sizeof(float);
        ASSERT_EQ(hipMalloc(&devA.ptr, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&devB.ptr, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&devD.ptr, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(devA.ptr, a.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(devB.ptr, b.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(devD.ptr, 0xFF, bytes), hipSuccess);

        ASSERT_EQ(hiptensorContract(handle,
                                    c.plan.plan,
                                    &alpha,
                                    devA.ptr,
                                    devB.ptr,
                                    nullptr,
                                    nullptr,
                                    devD.ptr,
                                    nullptr,
                                    0,
                                    0 /* stream */),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<float> result(count);
        ASSERT_EQ(hipMemcpy(result.data(), devD.ptr, bytes, hipMemcpyDeviceToHost), hipSuccess);
        for(std::size_t i = 0; i < count; i++)
        {
            EXPECT_EQ(result[i], d[i]) << "offset " << i;
        }
    }
} // namespace

TEST(PlanCacheApiTest, InvalidArguments)
{
    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    if(planCacheDisabled())
    {
        GTEST_SKIP() << "the plan cache is disabled";
    }

    auto     fileName          = tempFile("api_arguments");
    uint32_t numCachelinesRead = 0;
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(nullptr, fileName.c_str()),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(handle.handle, nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(nullptr, fileName.c_str(), &numCachelinesRead),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle.handle, nullptr, &numCachelinesRead),
              HIPTENSOR_STATUS_INVALID_VALUE);

    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle.handle, fileName.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle.handle, fileName.c_str(), nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
    std::remove(fileName.c_str());
}

// The arguments are checked also when the plan cache is disabled, and a read then reports that
// it read no line.
TEST(PlanCacheApiTest, InvalidArgumentsWithThePlanCacheDisabled)
{
    ScopedEnvironmentVariable disable("HIPTENSOR_DISABLE_PLAN_CACHE", "ON");

    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

    auto     fileName          = tempFile("api_arguments_disabled");
    uint32_t numCachelinesRead = 7;
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(nullptr, fileName.c_str()),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(handle.handle, nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(nullptr, fileName.c_str(), &numCachelinesRead),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle.handle, nullptr, &numCachelinesRead),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle.handle, fileName.c_str(), nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(numCachelinesRead, 7u);

    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(
                  handle.handle, fileName.c_str(), &numCachelinesRead),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numCachelinesRead, 0u);
}

// Contractions keep running on the handle after its plan cache was written, and the line read
// back on another handle selects a kernel that computes the right result.
TEST(PlanCacheApiTest, ContractionsAfterWriteAndAfterRead)
{
    auto fileName = tempFile("api_round_trip");
    {
        HandleGuard handle;
        ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
        if(planCacheDisabled())
        {
            GTEST_SKIP() << "the plan cache is disabled";
        }

        Contraction c;
        if(!createContraction(handle.handle, c))
        {
            GTEST_SKIP() << "32F contraction is not supported on this device";
        }
        ASSERT_EQ(createPlan(handle.handle, c), HIPTENSOR_STATUS_SUCCESS);

        // The execution adds the plan's kernel to the plan cache.
        ASSERT_NO_FATAL_FAILURE(contractAndCheck(handle.handle, c));
        ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle.handle, fileName.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(contractAndCheck(handle.handle, c));
        ASSERT_NO_FATAL_FAILURE(contractAndCheck(handle.handle, c));
        ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle.handle, fileName.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);
    }

    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);
    uint32_t numCachelinesRead = 0;
    ASSERT_EQ(
        hiptensorHandleReadPlanCacheFromFile(handle.handle, fileName.c_str(), &numCachelinesRead),
        HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numCachelinesRead, 1u);

    Contraction c;
    ASSERT_TRUE(createContraction(handle.handle, c));
    ASSERT_EQ(createPlan(handle.handle, c), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(contractAndCheck(handle.handle, c));

    std::remove(fileName.c_str());
}
