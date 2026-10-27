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

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hiptensor/hiptensor_types.h>

namespace hiptensor
{
    template <typename T>
    auto findSolutionByUid(const std::vector<T*>& candidates, std::size_t Uid)
    {
        T* solPtr = nullptr;
        for(auto sol : candidates)
        {
            if(sol->uid() == Uid)
            {
                solPtr = sol;
                break;
            }
        }
        return solPtr;
    }

    //Updatable priority queue
    template <typename T, typename PriorityType, typename Compare = std::less<PriorityType>>
    class UpdatablePriorityQueue
    {
    public:
        //An element in the queue
        struct Element
        {
            T            key;
            PriorityType priority;
            std::size_t  heapIndex;
        };

        UpdatablePriorityQueue() = default;

        // Copies own their elements: sharing them would let either queue free the other's.
        UpdatablePriorityQueue(const UpdatablePriorityQueue& other)
        {
            copyElements(other);
        }

        UpdatablePriorityQueue& operator=(const UpdatablePriorityQueue& other)
        {
            if(this != &other)
            {
                clear();
                copyElements(other);
            }
            return (*this);
        }

        //Insert a new element
        void push(const T& key, const PriorityType& priority)
        {
            Element* item = new Element{key, priority, mHeapData.size()};
            mHeapData.push_back(item);
            mKeyToElement[key] = mHeapData.back();
            siftUp(mHeapData.size() - 1);
        }

        //Update the priority of an existing element
        bool updateItem(const T& key, const PriorityType& new_priority)
        {
            bool retVal = false;
            auto it     = mKeyToElement.find(key);
            if(it != mKeyToElement.end())
            {
                Element*     elem         = it->second;
                PriorityType old_priority = elem->priority;
                elem->priority            = new_priority;

                if(compare(new_priority, old_priority))
                {
                    siftUp(elem->heapIndex);
                }
                else
                {
                    siftDown(elem->heapIndex);
                }

                retVal = true;
            }

            return retVal;
        }

        const T& topKey() const
        {
            return mHeapData[0]->key;
        }

        const std::pair<T, PriorityType> top() const
        {
            return {mHeapData[0]->key, mHeapData[0]->priority};
        }

        //Remove the top element
        void pop()
        {
            if(empty())
                return;
            mKeyToElement.erase(mHeapData[0]->key);
            std::swap(mHeapData[0], mHeapData.back());
            mHeapData[0]->heapIndex = 0;
            delete mHeapData.back();
            mHeapData.pop_back();
            if(!empty())
            {
                siftDown(0);
            }
        }

        bool empty() const
        {
            return mHeapData.empty();
        }

        std::size_t size() const
        {
            return mHeapData.size();
        }

        //All (key, priority) pairs, in no particular order
        std::vector<std::pair<T, PriorityType>> items() const
        {
            std::vector<std::pair<T, PriorityType>> result;
            result.reserve(mHeapData.size());
            for(auto elem : mHeapData)
                result.emplace_back(elem->key, elem->priority);
            return result;
        }

        //Remove all elements
        void clear()
        {
            for(auto elem : mHeapData)
                delete elem;
            mHeapData.clear();
            mKeyToElement.clear();
        }

        ~UpdatablePriorityQueue()
        {
            clear();
        }

    private:
        std::vector<Element*>           mHeapData;
        std::unordered_map<T, Element*> mKeyToElement;
        Compare                         compare;

        void copyElements(const UpdatablePriorityQueue& other)
        {
            mHeapData.reserve(other.mHeapData.size());
            for(auto elem : other.mHeapData)
            {
                Element* item = new Element(*elem);
                mHeapData.push_back(item);
                mKeyToElement[item->key] = item;
            }
        }

        void siftUp(std::size_t index)
        {
            if(index < 1)
                return;

            std::size_t parent_index = (index - 1) / 2;
            if(compare(mHeapData[index]->priority, mHeapData[parent_index]->priority))
            {
                std::swap(mHeapData[index], mHeapData[parent_index]);
                mHeapData[index]->heapIndex        = index;
                mHeapData[parent_index]->heapIndex = parent_index;
                siftUp(parent_index);
            }
        }

        void siftDown(std::size_t index)
        {
            std::size_t left_child  = 2 * index + 1;
            std::size_t right_child = 2 * index + 2;
            std::size_t largest     = index;

            if(left_child < mHeapData.size()
               && compare(mHeapData[left_child]->priority, mHeapData[largest]->priority))
                largest = left_child;
            if(right_child < mHeapData.size()
               && compare(mHeapData[right_child]->priority, mHeapData[largest]->priority))
                largest = right_child;

            if(largest != index)
            {
                std::swap(mHeapData[largest], mHeapData[index]);
                mHeapData[largest]->heapIndex = largest;
                mHeapData[index]->heapIndex   = index;
                siftDown(largest);
            }
        }
    };

    //! A backend whose solutions a plan cache holds, with the version of its kernel arguments.
    //! Plan cache files record the backends of the cache that wrote them.
    struct PlanCacheBackend
    {
        std::string name;
        uint32_t    kernelAbi;

        bool operator==(PlanCacheBackend const& other) const
        {
            return name == other.name && kernelAbi == other.kernelAbi;
        }
    };

    class PlanCache
    {
    public:
        using HashId = std::size_t;
        using Uid    = std::size_t;
        //Order of last use: a larger stamp is more recent
        using UseStamp = uint64_t;

        //archName and cuCount identify the device: they are part of every key and of the
        //plan cache file, which is only read back on the same kind of device. The file also
        //records the backends, and is only read back by a library with the same backends.
        PlanCache(std::string                   archName,
                  uint32_t                      cuCount,
                  std::vector<PlanCacheBackend> backends = {});
        ~PlanCache() = default;

        //Get hash ID from operation descriptor pointer, for the solutions of a backend
        //Calculate hash key by: hash({device, backend, Tag, DatatypeA..E, DatatypeCompute, mOperationType, mContractionOpId, operators, ranks, lengths, strides, modes})
        HashId getHashID(hiptensorOperationDescriptor_t desc, std::string const& backend);

        //Query solution Uid of a backend through operation descriptor pointer
        Uid querySolutionUid(hiptensorOperationDescriptor_t desc, std::string const& backend);

        //Query solution Uid through hash key, 0 if there is none
        Uid querySolutionUidByHash(HashId hashId);

        //Add a cache record to the solution table
        //If the table is full and the key is new, then remove the LRU(least-recently-used) record first
        void addCacheLine(HashId hashId, Uid sol_id);

        //Function for serialization to disk
        hiptensorStatus_t writeFile(const char fileName[]);

        //Function for deserialization from disk. Replaces the cache lines, keeps the capacity.
        //On success numCachelinesRead is the number of lines read. A file written by another
        //hipTensor version, for another device or by a library with other backends returns
        //HIPTENSOR_STATUS_INVALID_VALUE, and a
        //file with more lines than the capacity returns HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE
        //with numCachelinesRead set to the number of lines it needs. Unreadable or damaged files
        //return HIPTENSOR_STATUS_IO_ERROR. On error the cache is unchanged.
        hiptensorStatus_t readFile(const char fileName[], uint32_t* numCachelinesRead);

        uint32_t getCachelinesNum()
        {
            std::scoped_lock lock(mMutex);
            return mPlanCacheLines.size();
        }

        //Resize Plan Cache
        void resize(uint32_t numEntries);

    private:
        //Device the cached solutions were selected for
        std::string mArchName;
        uint32_t    mCuCount;
        HashId      mDeviceHash;

        //Backends the cached solutions belong to
        std::vector<PlanCacheBackend> mBackends;

        //Max size of the hash table
        uint32_t mMaxCachelines;

        //Stamp given to the next use of a cache line
        UseStamp mNextUseStamp;

        //Hash table to store solution uid for lookup by hash key
        std::unordered_map<HashId, Uid> mPlanCacheLines;

        //Heap to sort the solution uid by last use
        UpdatablePriorityQueue<HashId, UseStamp> mPqUidUsedTimes;

        //Mutex to implement the plan cache in a thread-safe manner
        mutable std::mutex mMutex;

        //Get solution id through hash key
        Uid getSolutionID(HashId hashId);

        //Update cache line used time
        void updateCachelineTime(HashId hashId);

        //Remove the LRU(least-recently-used) record
        void evictLeastRecentlyUsed();
    };

} // namespace hiptensor
