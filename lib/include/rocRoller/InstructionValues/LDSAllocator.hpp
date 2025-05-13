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

#include <list>

namespace rocRoller
{
    class LDSAllocation;

    /**
     * @brief An LDSAllocator is used to create LDSAllocations. These are
     * used to specify where in local memory data can be written to.
     *
     * The LDSAllocator keeps track of all of the data that has been allocated
     * and deallocated. It does this by keeping track of where memory was
     * last allocated from, as well as keeping track of a list of memory
     * blocks that have been deallocated.
     *
     */
    class LDSAllocator : public std::enable_shared_from_this<LDSAllocator>
    {
    public:
        /**
         * @brief Construct a new LDSAllocator object
         *
         * @param maxAmount The maximum amount of local data that can be allocated
         */
        explicit LDSAllocator(unsigned int maxAmount);

        /**
         * @brief Allocates memory. Returns a LDSAllocation object if succesful.
         *
         * @param size Number of bytes to allocate
         * @param alignment The alignment of allocated data
         * @return std::shared_ptr<LDSAllocation>
         */
        std::shared_ptr<LDSAllocation> allocate(unsigned int size, unsigned int alignment);

        /**
         * @brief Returns memory to the LDSAllocator
         *
         * @param allocation
         */
        void deallocate(std::shared_ptr<LDSAllocation> allocation);

        /**
         * @brief Returns the maximum amount of memory that has been allocated in bytes.
         *
         * @return unsigned int
         */
        unsigned int maxUsed() const;

        /**
         * @brief Returns the amount of memory that is currently in use in bytes.
         *
         * @return unsigned int
         */
        unsigned int currentUsed() const;

    private:
        const unsigned int m_maxAmount;

        unsigned int m_nextAvailable      = 0;
        unsigned int m_currentUsed        = 0;
        unsigned int m_consolidationDepth = 0;
        unsigned int m_maxUsed            = 0;

        std::list<std::shared_ptr<LDSAllocation>> m_freeBlocks;

        /**
         * @brief Updates the max LDS used
         *
         */
        void updateMaxUsed();
    };

    /**
     * @brief An LDSAllocation represents a chunk of local data that has been
     * allocated.
     *
     */
    class LDSAllocation : public std::enable_shared_from_this<LDSAllocation>
    {
        friend class LDSAllocator;

    public:
        /**
         * @brief Construct a new LDSAllocation object
         *
         * @param allocator The LDSAllocator that was used to allocate this
         * @param size The number of bytes to be allocated
         * @param offset The offset, in bytes, from the beginning of local data
         */
        LDSAllocation(std::shared_ptr<LDSAllocator> allocator,
                      unsigned int                  size,
                      unsigned int                  offset);

        /**
         * @brief Construct a new LDSAllocation object
         *
         * @param size The number of bytes to be allocated
         * @param offset The offset, in bytes, from the beginning of local data
         */
        LDSAllocation(unsigned int size, unsigned int offset);
        ~LDSAllocation();

        /**
         * @brief Returns the amount of bytes allocated
         *
         * @return unsigned int
         */
        unsigned int size() const;

        /**
         * @brief Returns the offset, in bytes, from the beginning of local data
         *
         * @return unsigned int
         */
        unsigned int offset() const;

        /**
         * @brief Return a string representing an LDSAllocation
         *
         * @return std::string
         */
        std::string toString() const;

        bool intersects(LDSAllocation const& other) const;
        bool intersects(std::shared_ptr<LDSAllocation> const& other) const;

    private:
        /**
         * @brief Set the amount of bytes allocated
         *
         * @param size
         */
        void setSize(unsigned int size);

        /**
         * @brief Set the offset size(in bytes)
         *
         * @param size
         */
        void setOffset(unsigned int offset);

        unsigned int                m_size;
        unsigned int                m_offset;
        std::weak_ptr<LDSAllocator> m_allocator;

        std::shared_ptr<LDSAllocation> copyForAllocator() const;
    };
}
