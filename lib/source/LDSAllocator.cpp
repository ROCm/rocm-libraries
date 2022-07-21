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
        return m_nextAvailable;
    }

    unsigned int LDSAllocator::currentUsed() const
    {
        return m_currentUsed;
    }

    std::shared_ptr<LDSAllocation> LDSAllocator::allocate(unsigned int size, unsigned int alignment)
    {
        unsigned int                   newOffset;
        unsigned int                   alignedSize = RoundUpToMultiple(size, alignment);
        std::shared_ptr<LDSAllocation> result;

        // Look for available free blocks that can hold the requested amount of memory.
        for(auto it = freeBlocks.begin(); it != freeBlocks.end(); it++)
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

                // If block is the exact amount requested, it can be removed from the freeBlocks list.
                // Otherwise, the correct amount of space should be taken from the block.
                if(block->size() == alignedSize)
                {
                    m_consolidationDepth++;
                    freeBlocks.erase(it);
                    m_consolidationDepth--;
                }
                else
                {
                    block->setSize(block->size() - alignedSize);
                    block->setOffset(block->offset() + alignedSize);
                }

                m_currentUsed += alignedSize;
                return result;
            }
        }

        // If there are no free blocks that can hold size, allocate more memory from m_nextAvailable
        auto alignedOffset = RoundUpToMultiple(m_nextAvailable, alignment);
        AssertRecoverable(alignedOffset + alignedSize <= m_maxAmount,
                          "Attempting to allocate more Local Data than is available");

        if(alignedOffset > m_nextAvailable)
        {
            auto gap = std::make_shared<LDSAllocation>(
                shared_from_this(), alignedOffset - m_nextAvailable, m_nextAvailable);
            freeBlocks.push_back(gap);
        }

        auto allocation
            = std::make_shared<LDSAllocation>(shared_from_this(), alignedSize, alignedOffset);

        m_nextAvailable = alignedOffset + alignedSize;
        m_currentUsed += alignedSize;

        return allocation;
    }

    void LDSAllocator::deallocate(std::shared_ptr<LDSAllocation> allocation)
    {
        unsigned int allocation_end = allocation->offset() + allocation->size();
        bool         deallocated    = false;

        // Blocks should be returned in sorted order.
        auto it = freeBlocks.begin();
        while(it != freeBlocks.end() && !deallocated)
        {
            auto block = *it;
            // Make sure block hasn't already been deallocated
            AssertFatal(
                !(allocation->offset() <= block->offset() && allocation_end > block->offset()),
                "Local memory allocation has already been freed.");
            // Can allocation be added to the end of the current block?
            if(block->offset() + block->size() == allocation->offset())
            {
                // Can allocation also be added to the beginning of the next block as well?
                // If so, block and the next block can be merged together.
                auto next = std::next(it);
                if(next != freeBlocks.end() && allocation_end == (*next)->offset())
                {
                    block->setSize(block->size() + allocation->size() + (*next)->size());
                    m_consolidationDepth++;
                    freeBlocks.erase(next);
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
                freeBlocks.insert(it, allocation);
                deallocated = true;
            }

            it++;
        }

        // If allocation wasn't added to freeBlocks, insert it at the end
        if(!deallocated)
            freeBlocks.push_back(allocation);

        if(m_consolidationDepth == 0)
            m_currentUsed -= allocation->size();
    }

    LDSAllocation::LDSAllocation(std::shared_ptr<LDSAllocator> allocator,
                                 unsigned int                  size,
                                 unsigned int                  offset)
        : m_allocator(allocator)
        , m_size(size)
        , m_offset(offset)
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
}
