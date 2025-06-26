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

#include <rocRoller/InstructionValues/Register_fwd.hpp>

#include <string>
#include <vector>

namespace rocRoller
{
    namespace Register
    {
        enum class AllocatorScheme
        {
            FirstFit,
            PerfectFit,

            Count,
        };

        std::string   toString(AllocatorScheme a);
        std::ostream& operator<<(std::ostream&, AllocatorScheme const&);

        class Allocator : public std::enable_shared_from_this<Allocator>
        {
        public:
            Allocator(Type            regType,
                      int             count,
                      AllocatorScheme scheme = AllocatorScheme::PerfectFit);

            Type regType() const;

            size_t size() const;

            /**
             * Get the id of the largest register used
             */
            int maxUsed() const;

            /**
             * Get the number of registers that have been used
             */
            int useCount() const;
            int currentlyFree() const;

            void allocate(AllocationPtr alloc);

            /**
             * Reassigns the registers in `indices` to a new `Allocation` that is returned.
             * The indices must all be currently assigned to the same `Allocation`.
             */
            template <std::ranges::forward_range T>
            AllocationPtr reassign(T const& indices);

            bool canAllocate(std::shared_ptr<const Allocation> alloc) const;

            std::vector<int> findFree(int count, AllocationOptions const& options) const;

            /**
             * @brief Returns the first free range matching the criteria as index and block size. Returns {-1, 0} if no such range exists.
             *
             * @param start The start index to begin investigating. The start of the block will be greater than or equal to this value.
             * @param regCount The number of registers required.  This will be the minimum size of the block.
             * @param options The AllocationOptions
             * @param reservedIndices Indices that can not be considered free for this investigation.
             * @return std::pair<int, int> [blockStartingIndex, blockSize].
             */
            std::pair<int, int> findContiguousRange(int                      start,
                                                    int                      regCount,
                                                    AllocationOptions const& options,
                                                    std::vector<int> const&  reservedIndices
                                                    = {}) const;

            constexpr static int align(int start, AllocationOptions const& options);

            bool isFree(int idx) const;

            std::string toString() const;

            /**
             *  Normally, registers are implicitly freed when their allocation goes out of scope.
             *  This is for manual control only.
             */
            void free(std::vector<int> const& registers);

        private:
            //> Allocate these specific registers.
            void allocate(AllocationPtr alloc, std::vector<int> const& registers);

            //> Allocate these specific registers.
            void allocate(AllocationPtr alloc, std::vector<int>&& registers);

            std::vector<int> findFreeFirstFit(int count, AllocationOptions const& options) const;

            std::vector<int> findFreePerfectFit(int count, AllocationOptions const& options) const;

            AllocatorScheme m_scheme;

            Type m_regType;

            int m_maxUsed = -1;

            std::vector<std::weak_ptr<Allocation>> m_registers;
        };
    }
}

#include <rocRoller/InstructionValues/RegisterAllocator_impl.hpp>
