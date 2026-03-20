// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

        /**
         * @brief Manages candidate free blocks for the PerfectFit allocation strategy.
         *
         * This class encapsulates the list of candidate holes (free blocks) and provides
         * operations to update or remove candidates after allocations.
         */
        class PerfectFitCandidates
        {
        public:
            /// A candidate represents a contiguous free block
            struct Candidate
            {
                int idx; ///< Start index of the block
                int blockSize; ///< Size of the block
            };

            using iterator       = std::vector<Candidate>::iterator;
            using const_iterator = std::vector<Candidate>::const_iterator;

            explicit PerfectFitCandidates(size_t registerCount)
                : m_registerCount(registerCount)
            {
            }

            /// Add a candidate to the list
            void addCandidate(int idx, int blockSize)
            {
                AssertFatal(blockSize > 0 && idx >= 0,
                            "Invalid candidate",
                            ShowValue(idx),
                            ShowValue(blockSize));
                m_candidates.push_back({idx, blockSize});
            }

            /// Check if a candidate is the end candidate (trailing space at end of register pool)
            bool isEndCandidate(const_iterator it) const
            {
                return it->blockSize > 0 && it->idx >= 0
                       && (static_cast<size_t>(it->idx) + static_cast<size_t>(it->blockSize))
                              >= m_registerCount;
            }

            iterator begin()
            {
                return m_candidates.begin();
            }

            iterator end()
            {
                return m_candidates.end();
            }

            const_iterator begin() const
            {
                return m_candidates.begin();
            }

            const_iterator end() const
            {
                return m_candidates.end();
            }

            bool empty() const
            {
                return m_candidates.empty();
            }

            /// Remove a candidate, returns iterator to next element
            iterator remove(iterator it)
            {
                return m_candidates.erase(it);
            }

            /**
             * @brief Update a candidate after allocating from its start.
             *
             * Adjusts idx and blockSize. Removes candidate if remaining size <= 0.
             */
            void updateFromStart(iterator it, int width, AllocationOptions const& options)
            {
                int newIdx = Allocator::align(it->idx + width, options);
                int shrink = newIdx - it->idx;
                it->idx    = newIdx;
                it->blockSize -= shrink;

                if(it->blockSize <= 0)
                    m_candidates.erase(it);
            }

            /**
             * @brief Update a candidate after allocating from its end.
             *
             * Adjusts blockSize. Removes candidate if remaining size <= 0.
             */
            void updateFromEnd(iterator it, int width)
            {
                it->blockSize -= width;
                if(it->blockSize <= 0)
                    m_candidates.erase(it);
            }

        private:
            std::vector<Candidate> m_candidates;
            const size_t           m_registerCount;
        };

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

            auto const chunkWidth = options.contiguousChunkWidth;

            // Gather all candidate blocks
            PerfectFitCandidates candidates(m_registers.size());
            for(int searchStart = 0; static_cast<size_t>(searchStart) < m_registers.size();)
            {
                auto [start, blockSize] = findContiguousRange(searchStart, chunkWidth, options, {});
                if(start < 0)
                    break;

                candidates.addCandidate(start, blockSize);
                searchStart = start + blockSize;
            }

            if(candidates.empty())
                return {};

            // Helper to allocate a chunk starting at the given index
            auto allocateChunk = [&rv](int startIdx, int width) {
                std::vector<int> indices(width);
                std::iota(indices.begin(), indices.end(), startIdx);
                rv.insert(rv.end(), indices.begin(), indices.end());
            };

            // Check if a register is free (not in m_registers and not already planned in rv)
            auto isEffectivelyFree = [this, &rv](int idx) {
                if(!isFree(idx))
                    return false;
                return std::find(rv.begin(), rv.end(), idx) == rv.end();
            };

            int remainingCount = count;

            while(remainingCount > 0)
            {
                // If the last chunk is smaller than chunkWidth, allocate what's left
                int  width = std::min(chunkWidth, remainingCount);
                bool found = false;

                // 1. Look for perfect fit (exact size match)
                for(auto it = candidates.begin(); !found && it != candidates.end(); ++it)
                {
                    if(candidates.isEndCandidate(it) || it->blockSize != width)
                        continue;

                    allocateChunk(it->idx, width);
                    candidates.remove(it);
                    found = true;
                }

                // 2. Look for perfect alignment (no gap created) at start or end of a hole
                for(auto it = candidates.begin(); !found && it != candidates.end(); ++it)
                {
                    if(candidates.isEndCandidate(it) || it->blockSize < width)
                        continue;

                    // Check start: no gap if hole starts at 0 or right after a used register
                    bool startAligned = (it->idx == 0) || !isEffectivelyFree(it->idx - 1);
                    if(startAligned)
                    {
                        allocateChunk(it->idx, width);
                        candidates.updateFromStart(it, width, options);
                        found = true;
                        break;
                    }

                    // Check end: no gap if allocation aligns perfectly at hole's end
                    int  endStart   = it->idx + it->blockSize - width;
                    bool endAligned = (align(endStart, options) == endStart);
                    if(endAligned)
                    {
                        allocateChunk(endStart, width);
                        candidates.updateFromEnd(it, width);
                        found = true;
                    }
                }

                // 3. Allocate at start of a candidate (creates alignment gap)
                for(auto it = candidates.begin(); !found && it != candidates.end(); ++it)
                {
                    if(candidates.isEndCandidate(it) || it->blockSize < width)
                        continue;

                    allocateChunk(it->idx, width);
                    candidates.updateFromStart(it, width, options);
                    found = true;
                }

                // 4. Last resort: allocate at the end of the register space
                if(!found && !candidates.empty())
                {
                    auto it = std::prev(candidates.end());
                    if(candidates.isEndCandidate(it) && it->blockSize >= width)
                    {
                        allocateChunk(it->idx, width);
                        candidates.updateFromStart(it, width, options);
                        found = true;
                    }
                }

                // Could not allocate this chunk, full failure
                if(!found)
                    return {};

                remainingCount -= width;
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
