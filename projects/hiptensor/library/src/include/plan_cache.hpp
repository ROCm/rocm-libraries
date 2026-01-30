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
#include <functional>
#include <mutex>
#include <unordered_map>
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

        UpdatablePriorityQueue(const UpdatablePriorityQueue& other)
            : mHeapData(other.mHeapData)
            , mKeyToElement(other.mKeyToElement)
        {
        }

        UpdatablePriorityQueue& operator=(const UpdatablePriorityQueue& other)
        {
            if(this != &other)
            {
                mHeapData     = other.mHeapData;
                mKeyToElement = other.mKeyToElement;
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

        ~UpdatablePriorityQueue()
        {
            for(auto elem : mHeapData)
                delete elem;
            mHeapData.clear();
        }

    private:
        std::vector<Element*>           mHeapData;
        std::unordered_map<T, Element*> mKeyToElement;
        Compare                         compare;

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

    class PlanCache
    {
    public:
        using HashId     = std::size_t;
        using Uid        = std::size_t;
        using time_point = std::chrono::system_clock::time_point;

        PlanCache();
        ~PlanCache() = default;

        //Get hash ID from peration descriptor pointer
        //Calculate hash key by: hash({Tag, DatatypeA, DatatypeB, DatatypeC, DatatypeD, DatatypeCompute, mOperationType, mContractionOpId, lengths, strides, modes})
        HashId getHashID(hiptensorOperationDescriptor_t desc);

        //Query solution Uid through operation descriptor pointer
        Uid querySolutionUid(hiptensorOperationDescriptor_t desc);

        //Add a cache record to the solution table
        //If the table size exceeds the maximum size, then remove one LRU(least-recently-used) record
        void addCacheLine(HashId hashId, Uid sol_id);

        //Function for serialization to disk
        hiptensorStatus_t writeFile(const char fileName[]);

        //Function for deserialization from disk
        hiptensorStatus_t readFile(const char fileName[]);

        uint32_t getCachelinesNum()
        {
            return mPlanCacheLines.size();
        }

        //Resize Plan Cache
        void resize(uint32_t numEntries);

    private:
        //Max size of the hash table
        uint32_t mMaxCachelines;

        //Hash table to store solution uid for lookup by hash key
        std::unordered_map<HashId, Uid> mPlanCacheLines;

        //Heap to sort the solution uid by used time
        UpdatablePriorityQueue<HashId, time_point> mPqUidUsedTimes;

        //Mutex to implement the plan cache in a thread-safe manner
        mutable std::mutex mMutex;

        //Get solution id through hash key
        Uid getSolutionID(HashId hashId);

        //Update cache line used time
        void updateCachelineTime(HashId hashId);
    };

} // namespace hiptensor
