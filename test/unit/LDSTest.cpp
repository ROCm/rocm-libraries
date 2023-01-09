#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "DataTypes/DataTypes.hpp"
#include "GenericContextFixture.hpp"
#include "InstructionValues/Register.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class LDSTest : public GenericContextFixture
    {
    };

    TEST_F(LDSTest, LDSAllocationTest)
    {
        auto lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 1000);

        auto lds2 = Register::Value::AllocateLDS(m_context, DataType::Int32, 300);

        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 0, Size: 4000)");
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 4000, Size: 1200)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 5200);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 5200);

        lds1.reset();
        // Free blocks: [0, 4000]

        // Can fit in free block, so offset starts at 0
        lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 500);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 0, Size: 2000)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 3200);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 5200);
        // Free blocks: [2000, 4000]

        // Doesn't fit in free block, so needs to allocate from new memory; alignment gap is 48
        auto lds3 = Register::Value::AllocateLDS(m_context, DataType::Int32, 600, 128);
        EXPECT_EQ(lds3->toString(), "LDS:(Offset: 5248, Size: 2432)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 5632);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7632 + 48);

        // Does fit in free block, so no new memory needed
        auto lds4 = Register::Value::AllocateLDS(m_context, DataType::Int32, 500);
        EXPECT_EQ(lds4->toString(), "LDS:(Offset: 2000, Size: 2000)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 7632);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7632 + 48);

        lds2.reset();
        lds4.reset();

        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4432);
        // Free blocks [2000, 5200]

        lds2 = Register::Value::AllocateLDS(m_context, DataType::Int32, 600);
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 2000, Size: 2400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 6832);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7632 + 48);

        lds3.reset();
        lds1.reset();

        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2400);
        // Free blocks [0, 2000], [4400, 7632]

        // Doesn't fit in first free block, but does fit in second
        lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 600);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 4400, Size: 2400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4800);
        // Free blocks [0, 2000], [6800, 7632]

        // Fits in the first free block
        lds4 = Register::Value::AllocateLDS(m_context, DataType::Int32, 100);
        EXPECT_EQ(lds4->toString(), "LDS:(Offset: 0, Size: 400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 5200);
        // Free blocks [400, 2000], [6800, 7632]

        // Exercise merging two free blocks together
        lds2.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2800);
        // Free blocks [400, 4400], [6800, 7632]
        lds1.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 400);
        // Free blocks [400, 7632]

        // Should fit in the free block
        lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 1800);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 400, Size: 7200)");

        lds1.reset();
        // Free blocks [400, 7632]

        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 400);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7632 + 48);

        // Try different alignment
        lds2 = Register::Value::AllocateLDS(m_context, DataType::Int32, 512, 128);
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 7680, Size: 2048)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2448);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 9728);

        // Try over allocating
        auto maxLDS = m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
        EXPECT_THROW(Register::Value::AllocateLDS(m_context, DataType::Int32, maxLDS / 4),
                     FatalError);
    }

    TEST_F(LDSTest, LDSAllocationDoubleFreeTest)
    {
        auto allocator = m_context->ldsAllocator();
        auto alloc1    = allocator->allocate(100, 8);
        allocator->deallocate(alloc1);
        EXPECT_THROW(allocator->deallocate(alloc1), FatalError);
    }
}
