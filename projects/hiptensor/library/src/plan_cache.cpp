/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iterator>
#include <limits>

#include "data_types.hpp"
#include "hash.hpp"
#include "plan_cache.hpp"
#include "util.hpp"

namespace hiptensor
{
    namespace
    {
        // Plan cache file layout, with every integer little-endian:
        //   char[4] magic "HTPC", u32 format version,
        //   u32 hipTensor major, minor and patch version,
        //   u32 length and the bytes of the device architecture name, u32 compute unit count,
        //   u64 line count, then per line the u64 hash key and the u64 solution uid, least
        //   recently used first,
        //   u64 FNV-1a checksum of all the bytes before it.
        // Hash keys and solution uids only hold for the hipTensor version and the device that
        // wrote them, so a file of another version or device is refused.
        constexpr char     planCacheMagic[4]      = {'H', 'T', 'P', 'C'};
        constexpr uint32_t planCacheFormatVersion = 1u;
        constexpr uint32_t libraryMajorVersion    = HIPTENSOR_MAJOR_VERSION;
        constexpr uint32_t libraryMinorVersion    = HIPTENSOR_MINOR_VERSION;
        constexpr uint32_t libraryPatchVersion    = HIPTENSOR_PATCH_VERSION;

        // A longer architecture name is taken as a damaged file.
        constexpr uint32_t maxArchNameLength = 256u;

        constexpr uint64_t fnvOffsetBasis = 14695981039346656037ull;
        constexpr uint64_t fnvPrime       = 1099511628211ull;

        class PlanCacheFileWriter
        {
        public:
            void bytes(const char* data, std::size_t count)
            {
                for(std::size_t i = 0; i < count; i++)
                {
                    mBuffer.push_back(data[i]);
                    mChecksum = (mChecksum ^ static_cast<unsigned char>(data[i])) * fnvPrime;
                }
            }

            void u32(uint32_t value)
            {
                littleEndian(value, 4);
            }

            void u64(uint64_t value)
            {
                littleEndian(value, 8);
            }

            void str(std::string const& value)
            {
                u32(static_cast<uint32_t>(value.size()));
                bytes(value.data(), value.size());
            }

            // Appends the checksum of all the bytes written so far
            void finish()
            {
                uint64_t checksum = mChecksum;
                u64(checksum);
            }

            std::vector<char> const& buffer() const
            {
                return mBuffer;
            }

        private:
            void littleEndian(uint64_t value, int count)
            {
                char data[8];
                for(int i = 0; i < count; i++)
                {
                    data[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
                }
                bytes(data, count);
            }

            std::vector<char> mBuffer;
            uint64_t          mChecksum = fnvOffsetBasis;
        };

        class PlanCacheFileReader
        {
        public:
            explicit PlanCacheFileReader(std::istream& stream)
                : mStream(stream)
            {
            }

            bool bytes(char* data, std::size_t count)
            {
                if(!mStream.read(data, static_cast<std::streamsize>(count)))
                {
                    return false;
                }
                for(std::size_t i = 0; i < count; i++)
                {
                    mChecksum = (mChecksum ^ static_cast<unsigned char>(data[i])) * fnvPrime;
                }
                return true;
            }

            bool u32(uint32_t& value)
            {
                uint64_t result = 0;
                if(!littleEndian(result, 4))
                {
                    return false;
                }
                value = static_cast<uint32_t>(result);
                return true;
            }

            bool u64(uint64_t& value)
            {
                return littleEndian(value, 8);
            }

            // Checksum of all the bytes read so far
            uint64_t checksum() const
            {
                return mChecksum;
            }

        private:
            bool littleEndian(uint64_t& value, int count)
            {
                unsigned char data[8];
                if(!bytes(reinterpret_cast<char*>(data), count))
                {
                    return false;
                }
                value = 0;
                for(int i = 0; i < count; i++)
                {
                    value |= static_cast<uint64_t>(data[i]) << (8 * i);
                }
                return true;
            }

            std::istream& mStream;
            uint64_t      mChecksum = fnvOffsetBasis;
        };
    } // namespace

    /////// Class PlanCache ////////////////////////

    PlanCache::PlanCache(std::string archName, uint32_t cuCount)
        : mArchName(std::move(archName))
        , mCuCount(cuCount)
        , mDeviceHash(Hash{}(Hash{}(mArchName), cuCount))
        , mMaxCachelines(128)
        , mNextUseStamp(0)
    {
    }

    PlanCache::HashId PlanCache::getHashID(hiptensorOperationDescriptor_t desc)
    {
        auto tag             = desc->mTag;
        auto ADataType       = desc->mDescA ? desc->mDescA->mType : hiptensor::NONE_TYPE;
        auto BDataType       = desc->mDescB ? desc->mDescB->mType : hiptensor::NONE_TYPE;
        auto CDataType       = desc->mDescC ? desc->mDescC->mType : hiptensor::NONE_TYPE;
        auto DDataType       = desc->mDescD ? desc->mDescD->mType : hiptensor::NONE_TYPE;
        auto EDataType       = desc->mDescE ? desc->mDescE->mType : hiptensor::NONE_TYPE;
        auto computeType     = desc->mDescCompute;
        auto operationType   = desc->mOperationType;
        auto contractionOpId = desc->mContractionOpId;

        //Get lengths
        std::vector<std::size_t> lengthsA = hiptensor::getTensorLengths(desc->mDescA);
        std::vector<std::size_t> lengthsB = hiptensor::getTensorLengths(desc->mDescB);
        std::vector<std::size_t> lengthsC = hiptensor::getTensorLengths(desc->mDescC);
        std::vector<std::size_t> lengthsD = hiptensor::getTensorLengths(desc->mDescD);
        std::vector<std::size_t> lengthsE = hiptensor::getTensorLengths(desc->mDescE);

        //Get ranks first, so that lengths, strides and modes of different tensors can not be
        //mistaken for one another once they are concatenated
        std::vector<std::size_t> arrLSM = {lengthsA.size(),
                                           lengthsB.size(),
                                           lengthsC.size(),
                                           lengthsD.size(),
                                           lengthsE.size(),
                                           desc->mModeA.size(),
                                           desc->mModeB.size(),
                                           desc->mModeC.size(),
                                           desc->mModeD.size(),
                                           desc->mModeE.size()};

        arrLSM.insert(arrLSM.end(), lengthsA.begin(), lengthsA.end());
        arrLSM.insert(arrLSM.end(), lengthsB.begin(), lengthsB.end());
        arrLSM.insert(arrLSM.end(), lengthsC.begin(), lengthsC.end());
        arrLSM.insert(arrLSM.end(), lengthsD.begin(), lengthsD.end());
        arrLSM.insert(arrLSM.end(), lengthsE.begin(), lengthsE.end());

        //Get strides
        std::vector<std::size_t> stridesA = hiptensor::getTensorStrides(desc->mDescA);
        std::vector<std::size_t> stridesB = hiptensor::getTensorStrides(desc->mDescB);
        std::vector<std::size_t> stridesC = hiptensor::getTensorStrides(desc->mDescC);
        std::vector<std::size_t> stridesD = hiptensor::getTensorStrides(desc->mDescD);
        std::vector<std::size_t> stridesE = hiptensor::getTensorStrides(desc->mDescE);

        arrLSM.insert(arrLSM.end(), stridesA.begin(), stridesA.end());
        arrLSM.insert(arrLSM.end(), stridesB.begin(), stridesB.end());
        arrLSM.insert(arrLSM.end(), stridesC.begin(), stridesC.end());
        arrLSM.insert(arrLSM.end(), stridesD.begin(), stridesD.end());
        arrLSM.insert(arrLSM.end(), stridesE.begin(), stridesE.end());

        //Get modes
        arrLSM.insert(arrLSM.end(), desc->mModeA.begin(), desc->mModeA.end());
        arrLSM.insert(arrLSM.end(), desc->mModeB.begin(), desc->mModeB.end());
        arrLSM.insert(arrLSM.end(), desc->mModeC.begin(), desc->mModeC.end());
        arrLSM.insert(arrLSM.end(), desc->mModeD.begin(), desc->mModeD.end());
        arrLSM.insert(arrLSM.end(), desc->mModeE.begin(), desc->mModeE.end());

        //generate hash ID by {Device, Tag, DatatypeA..E, DatatypeCompute, mOperationType, mContractionOpId, operators, ranks, lengths, strides, modes}
        // A const copy: a non-const std::size_t lvalue would pick Hash's private seed overload
        std::size_t const deviceHash = mDeviceHash;
        PlanCache::HashId hashID     = Hash{}(deviceHash,
                                              tag,
                                              ADataType,
                                              BDataType,
                                              CDataType,
                                              DDataType,
                                              EDataType,
                                              computeType,
                                              operationType,
                                              contractionOpId,
                                              desc->mOpA,
                                              desc->mOpB,
                                              desc->mOpC,
                                              desc->mOpD,
                                              desc->mOpAC,
                                              desc->mOpABC,
                                              arrLSM);

        return hashID;
    }

    PlanCache::Uid PlanCache::querySolutionUid(hiptensorOperationDescriptor_t desc)
    {
        return querySolutionUidByHash(getHashID(desc));
    }

    PlanCache::Uid PlanCache::querySolutionUidByHash(HashId hashId)
    {
        std::scoped_lock lock(mMutex);

        PlanCache::Uid solutionUid = getSolutionID(hashId);
        //Update cache line used time
        if(solutionUid > 0ull)
            updateCachelineTime(hashId);

        return solutionUid;
    }

    void PlanCache::addCacheLine(HashId hashId, Uid sol_id)
    {
        std::scoped_lock lock(mMutex);

        assert(mMaxCachelines > 0);

        //If the table is full and the key is new, then remove the LRU(least-recently-used) record
        if(mPlanCacheLines.find(hashId) == mPlanCacheLines.end())
        {
            while(mPlanCacheLines.size() >= mMaxCachelines && !mPqUidUsedTimes.empty())
            {
                evictLeastRecentlyUsed();
            }
        }

        //Add the cache line
        mPlanCacheLines[hashId] = sol_id;

        //Also add the cache line use to a min heap
        updateCachelineTime(hashId);
    }

    void PlanCache::updateCachelineTime(HashId hashId)
    {
        UseStamp stamp = mNextUseStamp++;
        if(!mPqUidUsedTimes.updateItem(hashId, stamp))
            mPqUidUsedTimes.push(hashId, stamp);

        assert(mPqUidUsedTimes.size() == mPlanCacheLines.size());
    }

    void PlanCache::evictLeastRecentlyUsed()
    {
        std::pair<HashId, UseStamp> lru_item = mPqUidUsedTimes.top();
        mPqUidUsedTimes.pop();

        mPlanCacheLines.erase(lru_item.first);
    }

    void PlanCache::resize(uint32_t numEntries)
    {
        std::scoped_lock lock(mMutex);

        if(numEntries < 1u)
            return;

        mMaxCachelines = numEntries;

        //If the table size exceeds the maximum size, then remove extra LRU(least-recently-used) records
        while(mPqUidUsedTimes.size() > mMaxCachelines)
        {
            evictLeastRecentlyUsed();
        }
    }

    PlanCache::Uid PlanCache::getSolutionID(HashId hashId)
    {
        auto it = mPlanCacheLines.find(hashId);
        if(it != mPlanCacheLines.end())
            return it->second;
        return 0ull;
    }

    hiptensorStatus_t PlanCache::writeFile(const char fileName[])
    {
        PlanCacheFileWriter writer;
        {
            //Serialize under the lock and write the file without it. The bytes do not refer back
            //to the cache, so later uses of the cache can not see the write.
            std::scoped_lock lock(mMutex);

            auto lines = mPqUidUsedTimes.items();
            std::sort(lines.begin(), lines.end(), [](auto const& lhs, auto const& rhs) {
                return lhs.second < rhs.second;
            });

            writer.bytes(planCacheMagic, sizeof(planCacheMagic));
            writer.u32(planCacheFormatVersion);
            writer.u32(libraryMajorVersion);
            writer.u32(libraryMinorVersion);
            writer.u32(libraryPatchVersion);
            writer.str(mArchName);
            writer.u32(mCuCount);
            writer.u64(lines.size());
            for(auto const& line : lines)
            {
                writer.u64(line.first);
                writer.u64(getSolutionID(line.first));
            }
        }
        writer.finish();

        std::ofstream fstream(fileName, std::ios::out | std::ios::binary | std::ios::trunc);
        if(!fstream.is_open())
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }

        fstream.write(writer.buffer().data(), static_cast<std::streamsize>(writer.buffer().size()));
        fstream.close();

        return fstream.fail() ? HIPTENSOR_STATUS_IO_ERROR : HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t PlanCache::readFile(const char fileName[], uint32_t* numCachelinesRead)
    {
        std::ifstream fstream(fileName, std::ios::in | std::ios::binary);
        if(!fstream.is_open())
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }

        PlanCacheFileReader reader(fstream);

        //Files without the header, such as those of earlier hipTensor versions, are refused
        char magic[sizeof(planCacheMagic)];
        if(!reader.bytes(magic, sizeof(magic)))
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }
        if(!std::equal(std::begin(magic), std::end(magic), std::begin(planCacheMagic)))
        {
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        uint32_t formatVersion = 0;
        if(!reader.u32(formatVersion))
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }
        if(formatVersion != planCacheFormatVersion)
        {
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        uint32_t major = 0, minor = 0, patch = 0;
        if(!reader.u32(major) || !reader.u32(minor) || !reader.u32(patch))
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }
        if(major != libraryMajorVersion || minor != libraryMinorVersion
           || patch != libraryPatchVersion)
        {
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        uint32_t archNameLength = 0;
        if(!reader.u32(archNameLength) || archNameLength > maxArchNameLength)
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }
        std::string archName(archNameLength, '\0');
        uint32_t    cuCount = 0;
        if(!reader.bytes(archName.data(), archName.size()) || !reader.u32(cuCount))
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }
        if(archName != mArchName || cuCount != mCuCount)
        {
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        uint64_t lineCount = 0;
        if(!reader.u64(lineCount))
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }

        //Least recently used first
        std::vector<std::pair<HashId, Uid>> lines;
        for(uint64_t i = 0; i < lineCount; i++)
        {
            uint64_t hashId = 0, uid = 0;
            if(!reader.u64(hashId) || !reader.u64(uid))
            {
                return HIPTENSOR_STATUS_IO_ERROR;
            }
            lines.emplace_back(static_cast<HashId>(hashId), static_cast<Uid>(uid));
        }

        uint64_t const expectedChecksum = reader.checksum();
        uint64_t       checksum         = 0;
        if(!reader.u64(checksum) || checksum != expectedChecksum
           || fstream.peek() != std::ifstream::traits_type::eof())
        {
            return HIPTENSOR_STATUS_IO_ERROR;
        }

        std::unordered_map<HashId, Uid> planCacheLines;
        for(auto const& line : lines)
        {
            if(!planCacheLines.insert(line).second)
            {
                return HIPTENSOR_STATUS_IO_ERROR;
            }
        }

        std::scoped_lock lock(mMutex);

        if(lines.size() > mMaxCachelines)
        {
            if(numCachelinesRead != nullptr)
            {
                *numCachelinesRead = static_cast<uint32_t>(
                    std::min<uint64_t>(lines.size(), std::numeric_limits<uint32_t>::max()));
            }
            return HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
        }

        mPlanCacheLines = std::move(planCacheLines);
        mPqUidUsedTimes.clear();
        for(auto const& line : lines)
        {
            mPqUidUsedTimes.push(line.first, mNextUseStamp++);
        }

        if(numCachelinesRead != nullptr)
        {
            *numCachelinesRead = static_cast<uint32_t>(lines.size());
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }

} // namespace hiptensor
