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

#include <rocRoller/InstructionValues/LDSAllocator.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <iostream>
#include <sstream>

namespace rocRoller
{
    LDSAllocator::LDSAllocator(unsigned int maxAmount)
        : m_maxAmount(maxAmount)
    {
    }

    unsigned int LDSAllocator::maxUsed() const
    {
        return m_maxUsed;
    }

    unsigned int LDSAllocator::currentUsed() const
    {
        return m_currentUsed;
    }

    void LDSAllocator::updateMaxUsed()
    {
        if(m_maxUsed < m_nextAvailable)
        {
            m_maxUsed = m_nextAvailable;
        }
    }

    std::shared_ptr<LDSAllocation> LDSAllocator::allocate(unsigned int size, unsigned int alignment)
    {
        unsigned int                   alignedSize = RoundUpToMultiple(size, alignment);
        std::shared_ptr<LDSAllocation> result;

        AssertFatal(size == alignedSize,
                    "Can only allocate aligned sizes\n",
                    ShowValue(size),
                    ShowValue(alignedSize));

        // Look for available free blocks that can hold the requested amount of memory.
        for(auto it = m_freeBlocks.begin(); it != m_freeBlocks.end(); it++)
        {
            auto block = *it;

            // If the start of free block's alignment doesn't match, skip it
            auto alignedOffset = RoundUpToMultiple(block->offset(), alignment);
            if(block->offset() != alignedOffset)
                continue;

            if(block->size() >= alignedSize)
            {
                result = std::make_shared<LDSAllocation>(
                    shared_from_this(), alignedSize, block->offset());

                // If block is the exact amount requested, it can be removed from the m_freeBlocks list.
                // Otherwise, the correct amount of space should be taken from the block.
                if(block->size() == alignedSize)
                {
                    m_consolidationDepth++;
                    m_freeBlocks.erase(it);
                    m_consolidationDepth--;
                }
                else
                {
                    block->setSize(block->size() - alignedSize);
                    block->setOffset(block->offset() + alignedSize);
                }

                m_currentUsed += alignedSize;
                updateMaxUsed();
                return result;
            }
        }

        // If there are no free blocks that can hold size, allocate more memory from m_nextAvailable
        auto alignedOffset = RoundUpToMultiple(m_nextAvailable, alignment);
        AssertFatal(alignedOffset + alignedSize <= m_maxAmount,
                    "Attempting to allocate more Local Data than is available",
                    ShowValue(alignedOffset),
                    ShowValue(alignedSize),
                    ShowValue(m_maxAmount));

        if(alignedOffset > m_nextAvailable)
        {
            auto gap = std::make_shared<LDSAllocation>(
                shared_from_this(), alignedOffset - m_nextAvailable, m_nextAvailable);
            m_freeBlocks.push_back(gap);
        }

        auto allocation
            = std::make_shared<LDSAllocation>(shared_from_this(), alignedSize, alignedOffset);

        m_nextAvailable = alignedOffset + alignedSize;
        m_currentUsed += alignedSize;

        updateMaxUsed();
        return allocation;
    }

    void LDSAllocator::deallocate(std::shared_ptr<LDSAllocation> allocation)
    {
        unsigned int allocation_end = allocation->offset() + allocation->size();
        bool         deallocated    = false;

        // Blocks should be returned in sorted order.
        auto it = m_freeBlocks.begin();
        while(it != m_freeBlocks.end() && !deallocated)
        {
            auto block = *it;
            // Make sure block hasn't already been deallocated
            AssertFatal(
                !(allocation->offset() <= block->offset() && allocation_end > block->offset()),
                ShowValue(allocation->offset()),
                ShowValue(allocation_end),
                ShowValue(block->offset()),
                ShowValue(allocation->size()),
                allocation->toString(),
                "\nLocal memory allocation has already been freed.");

            // Can allocation be added to the end of the current block?
            if(block->offset() + block->size() == allocation->offset())
            {
                // Can allocation also be added to the beginning of the next block as well?
                // If so, block and the next block can be merged together.
                auto next = std::next(it);
                if(next != m_freeBlocks.end() && allocation_end == (*next)->offset())
                {
                    block->setSize(block->size() + allocation->size() + (*next)->size());
                    m_consolidationDepth++;
                    m_freeBlocks.erase(next);
                    m_consolidationDepth--;
                }
                else
                {
                    block->setSize(block->size() + allocation->size());
                }

                deallocated = true;
            }
            // Can allocation be added to the beginning of the current block?
            else if(allocation_end == block->offset())
            {
                block->setSize(block->size() + allocation->size());
                block->setOffset(allocation->offset());
                deallocated = true;
            }
            // Should new block be added
            else if(allocation->offset() < block->offset())
            {
                m_freeBlocks.insert(it, allocation);
                deallocated = true;
            }

            it++;
        }

        // If allocation wasn't added to m_freeBlocks, insert it at the end
        if(!deallocated)
            m_freeBlocks.push_back(allocation);

        if(m_consolidationDepth == 0)
        {
            AssertFatal(m_currentUsed >= allocation->size(),
                        ShowValue(m_currentUsed),
                        ShowValue(allocation->size()));

            m_currentUsed -= allocation->size();

            // Remove last block if it's on the edge of free memory
            if(!m_freeBlocks.empty()
               && m_nextAvailable == m_freeBlocks.back()->offset() + m_freeBlocks.back()->size())
            {
                m_consolidationDepth += 1;
                m_nextAvailable -= m_freeBlocks.back()->size();
                m_freeBlocks.pop_back();
                m_consolidationDepth -= 1;
            }
        }
    }

    LDSAllocation::LDSAllocation(std::shared_ptr<LDSAllocator> allocator,
                                 unsigned int                  size,
                                 unsigned int                  offset)
        : m_size(size)
        , m_offset(offset)
        , m_allocator(allocator)
    {
    }

    LDSAllocation::LDSAllocation(unsigned int size, unsigned int offset)
        : m_size(size)
        , m_offset(offset)
    {
    }

    LDSAllocation::~LDSAllocation()
    {
        if(m_allocator.lock())
            m_allocator.lock()->deallocate(copyForAllocator());
    }

    std::shared_ptr<LDSAllocation> LDSAllocation::copyForAllocator() const
    {
        return std::make_shared<LDSAllocation>(m_size, m_offset);
    }

    unsigned int LDSAllocation::size() const
    {
        return m_size;
    }

    unsigned int LDSAllocation::offset() const
    {
        return m_offset;
    }

    void LDSAllocation::setSize(unsigned int size)
    {
        m_size = size;
    }

    void LDSAllocation::setOffset(unsigned int offset)
    {
        m_offset = offset;
    }

    std::string LDSAllocation::toString() const
    {
        std::stringstream ss;
        ss << "(Offset: " << m_offset << ", Size: " << m_size << ")";
        return ss.str();
    }

    bool LDSAllocation::intersects(LDSAllocation const& other) const
    {
        auto myMax    = m_offset + m_size;
        auto otherMax = other.m_offset + other.m_size;

        return (myMax > other.m_offset && myMax <= otherMax)
               || (otherMax > m_offset && otherMax <= myMax);
    }

    bool LDSAllocation::intersects(std::shared_ptr<LDSAllocation> const& other) const
    {
        AssertFatal(other);

        return intersects(*other);
    }
}
