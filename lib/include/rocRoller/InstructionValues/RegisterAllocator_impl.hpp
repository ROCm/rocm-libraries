/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator.hpp>
#include <rocRoller/Utilities/Error.hpp>

// Used for std::iota
#include <numeric>

namespace rocRoller
{
    namespace Register
    {
        inline std::string toString(AllocatorScheme a)
        {
            switch(a)
            {
            case AllocatorScheme::FirstFit:
                return "FirstFit";
            case AllocatorScheme::PerfectFit:
                return "PerfectFit";
            default:
                break;
            }

            Throw<FatalError>("Invalid AllocatorScheme");
        }

        inline std::ostream& operator<<(std::ostream& stream, AllocatorScheme const& a)
        {
            return stream << toString(a);
        }

        inline Allocator::Allocator(Type regType, int count, AllocatorScheme scheme)
            : m_regType(regType)
            , m_registers(count)
            , m_scheme(scheme)
        {
        }

        inline Type Allocator::regType() const
        {
            return m_regType;
        }

        inline std::string Allocator::toString() const
        {
            std::ostringstream msg;

            size_t                                        nextIdx = 0;
            std::map<std::shared_ptr<Allocation>, size_t> allocs;

            for(size_t i = 0; i < m_registers.size(); i++)
            {
                msg << std::setw(4) << i << ": ";
                if(m_registers[i].expired())
                {
                    msg << " (free)";
                }
                else
                {
                    auto myAlloc = m_registers[i].lock();
                    auto iter    = allocs.find(myAlloc);

                    if(iter == allocs.end())
                    {
                        iter = allocs.insert(std::make_pair(myAlloc, nextIdx)).first;
                        nextIdx++;
                    }

                    msg << "(" << iter->second << "): " << iter->first->descriptiveComment("");
                }

                msg << std::endl;
            }

            return msg.str();
        }

        inline void Allocator::allocate(AllocationPtr alloc)
        {
            auto registers = findFree(alloc->registerCount(), alloc->options());

            AssertFatal(!registers.empty(),
                        "No more ",
                        m_regType,
                        " registers!\n",
                        toString(),
                        alloc->options(),
                        ShowValue(alloc->registerCount()));

            allocate(alloc, std::move(registers));
        }

        inline void Allocator::allocate(AllocationPtr alloc, std::vector<int> const& registers)
        {
            for(auto idx : registers)
            {
                m_registers[idx] = alloc;
                m_maxUsed        = std::max(m_maxUsed, idx);
            }

            alloc->setAllocation(shared_from_this(), registers);
        }

        inline void Allocator::allocate(AllocationPtr alloc, std::vector<int>&& registers)
        {
            for(auto idx : registers)
            {
                m_registers[idx] = alloc;
                m_maxUsed        = std::max(m_maxUsed, idx);
            }

            alloc->setAllocation(shared_from_this(), std::move(registers));
        }

        inline bool Allocator::canAllocate(std::shared_ptr<const Allocation> alloc) const
        {
            auto theRegisters = findFree(alloc->registerCount(), alloc->options());
            return !theRegisters.empty();
        }

        template <std::ranges::forward_range T>
        AllocationPtr Allocator::reassign(T const& indices)
        {
            std::vector<int> registers(indices.begin(), indices.end());

            AssertFatal(registers.size() != 0);

            auto firstAlloc = m_registers.at(registers[0]).lock();
            AssertFatal(firstAlloc);

            for(auto index : registers)
                AssertFatal(m_registers.at(index).lock() == firstAlloc);

            auto newAlloc = std::make_shared<Allocation>(
                firstAlloc->m_context.lock(),
                m_regType,
                DataType::Raw32,
                registers.size(),
                Register::AllocationOptions{.contiguousChunkWidth = Register::MANUAL});

            allocate(newAlloc, registers);

            return newAlloc;
        }

        inline std::vector<int> Allocator::findFree(int                      count,
                                                    AllocationOptions const& options) const
        {
            AssertFatal(count > 0, "Invalid register count for findFree", ShowValue(count));

            switch(m_scheme)
            {
            case AllocatorScheme::FirstFit:
                return findFreeFirstFit(count, options);
            case AllocatorScheme::PerfectFit:
                return findFreePerfectFit(count, options);
            default:
                Throw<FatalError>("Allocator scheme not implemented.");
            }
        }

        inline std::vector<int> Allocator::findFreeFirstFit(int                      count,
                                                            AllocationOptions const& options) const
        {
            AssertFatal(count >= 0, "Negative count");
            AssertFatal(options.alignment <= options.contiguousChunkWidth,
                        "Not yet supported",
                        ShowValue(options));

            auto width = options.contiguousChunkWidth;

            std::vector<int> rv;
            rv.reserve(count);

            int idx = 0;

            while(rv.size() < count)
            {
                auto [start, blockSize] = findContiguousRange(idx, width, options, rv);
                idx                     = start;

                if(idx >= 0)
                {
                    std::vector<int> indices(width);
                    std::iota(indices.begin(), indices.end(), idx);
                    rv.insert(rv.begin(), indices.begin(), indices.end());
                }
                else
                {
                    return {};
                }
            }

            return rv;
        }

        inline std::vector<int>
            Allocator::findFreePerfectFit(int count, AllocationOptions const& options) const
        {
            AssertFatal(options.alignment <= options.contiguousChunkWidth,
                        "Not yet supported",
                        ShowValue(options));
            AssertFatal(
                count > 0, "Invalid register count for findFreePerfectFit", ShowValue(count));

            std::vector<int> rv;
            rv.reserve(count);

            // Width of chunks
            auto width = options.contiguousChunkWidth;

            // Remaining number of registers left to handle
            int currentCount = count;

            // Free blocks that can be used. {start, size}
            std::vector<std::pair<int, int>> candidates;

            // Loop through register collection and pick out appropriate free blocks
            int candidateIdx = 0;
            while(candidateIdx >= 0 && candidateIdx < m_registers.size())
            {
                auto candidate = findContiguousRange(candidateIdx, width, options, rv);
                candidateIdx   = candidate.first;
                if(candidateIdx >= 0)
                {
                    candidates.push_back(candidate);
                    candidateIdx += candidate.second;
                }
            }

            while(currentCount > 0)
            {
                // Have we found a place for the current chunk?
                bool found = false;

                // If the last chunk is smaller than the given width, allocate what's left
                width = std::min(width, currentCount);

                for(auto& [idx, blockSize] : candidates)
                {
                    // Check for perfect fit
                    if(blockSize == width)
                    {
                        std::vector<int> indices(width);
                        std::iota(indices.begin(), indices.end(), idx);
                        rv.insert(rv.end(), indices.begin(), indices.end());

                        // Update candidate
                        blockSize = 0;

                        found = true;
                        break;
                    }
                }

                // If a perfect fit was not found, use any other block
                if(!found)
                {
                    for(auto& [idx, blockSize] : candidates)
                    {
                        if(blockSize < width)
                        {
                            continue;
                        }

                        std::vector<int> indices(width);

                        // Try to use the end of this free block
                        int start = align(idx + blockSize - width, options);

                        // Check if chunk is outside of block, or if it runs up against the end of the total number of registers
                        // The equal check in `start + width >= m_registers.size()`
                        // is to avoid increasing register high-water mark by not allocating the last register
                        if(start + width > idx + blockSize || start + width >= m_registers.size())
                        {
                            // Should not use end of block, revert to using beginning
                            start = idx;

                            // Update candidate
                            idx += width;
                        }

                        std::iota(indices.begin(), indices.end(), start);
                        rv.insert(rv.begin(), indices.begin(), indices.end());

                        // Update candidate
                        blockSize -= width;

                        found = true;

                        break;
                    }
                }

                if(!found)
                {
                    // Could not allocate this chunk, full failure
                    return {};
                }

                currentCount -= width;
            }

            return rv;
        }

        inline std::pair<int, int>
            Allocator::findContiguousRange(int                      start,
                                           int                      regCount,
                                           AllocationOptions const& options,
                                           std::vector<int> const&  reservedIndices) const
        {
            AssertFatal(start >= 0 && regCount >= 0, "Negative arguments");

            // The start should always be aligned
            start = align(start, options);

            while(start < m_registers.size())
            {
                // Number of free registers in this block
                int blockSize = 0;

                for(int i = start; i < m_registers.size(); i++)
                {
                    // Check if register is not free, or if it's in our reserved list
                    if(!isFree(i)
                       || std::find(reservedIndices.begin(), reservedIndices.end(), i)
                              != reservedIndices.end()
                       || (regCount > 1 && regCount % options.alignment != 0
                           && i - start == regCount))
                    {
                        break;
                    }

                    blockSize++;
                }

                // If the block is large enough, we have succeeded
                if(blockSize >= regCount)
                    return {start, blockSize};

                // Increment start by block size, or by 1 if in non-free section
                int increment = blockSize > 0 ? blockSize : 1;

                start = align(start + increment, options);
            }

            return {-1, 0};
        }

        constexpr inline int Allocator::align(int start, AllocationOptions const& options)
        {
            AssertFatal(options.alignment <= 1 || options.alignmentPhase < options.alignment,
                        ShowValue(options.alignment),
                        ShowValue(options.alignmentPhase));

            if(options.alignment <= 1)
                return start;

            while((start % options.alignment) != options.alignmentPhase)
                start++;

            return start;
        }

        inline size_t Allocator::size() const
        {
            return m_registers.size();
        }

        inline int Allocator::maxUsed() const
        {
            return m_maxUsed;
        }

        inline int Allocator::useCount() const
        {
            return maxUsed() + 1;
        }

        inline bool Allocator::isFree(int idx) const
        {
            return m_registers[idx].expired();
        }

        inline int Allocator::currentlyFree() const
        {
            int rv = 0;
            for(int idx = 0; idx < size(); idx++)
                if(isFree(idx))
                    rv++;

            return rv;
        }

        inline void Allocator::free(std::vector<int> const& registers)
        {
            for(int idx : registers)
                m_registers[idx].reset();
        }
    }
}
