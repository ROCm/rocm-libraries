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

#include "GenericContextFixture.hpp"
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

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

        // Doesn't fit in free block, so needs to allocate from new memory
        auto lds3 = Register::Value::AllocateLDS(m_context, DataType::Half, 1200, 2);
        EXPECT_EQ(lds3->toString(), "LDS:(Offset: 5200, Size: 2400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 5600);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7600);

        // Does fit in free block, so no new memory needed
        auto lds4 = Register::Value::AllocateLDS(m_context, DataType::Int32, 500);
        EXPECT_EQ(lds4->toString(), "LDS:(Offset: 2000, Size: 2000)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 7600);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7600);

        lds2.reset();
        lds4.reset();

        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4400);
        // Free blocks [2000, 5200]

        lds2 = Register::Value::AllocateLDS(m_context, DataType::Int32, 600);
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 2000, Size: 2400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 6800);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7600);

        lds3.reset();
        lds1.reset();

        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2400);
        // Free blocks [0, 2000]

        // Doesn't fit in first free block, append to end
        lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 600);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 4400, Size: 2400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4800);
        // Free blocks [0, 2000]

        // Fits in the first free block
        lds4 = Register::Value::AllocateLDS(m_context, DataType::Int32, 100);
        EXPECT_EQ(lds4->toString(), "LDS:(Offset: 0, Size: 400)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 5200);
        // Free blocks [400, 2000]

        // Exercise merging two free blocks together
        lds2.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2800);
        // Free blocks [400, 4400]
        lds1.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 400);
        // Free blocks: None

        lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 1800);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 400, Size: 7200)");

        lds1.reset();
        // Free blocks: None

        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 400);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7600);

        // Try different alignment
        lds2 = Register::Value::AllocateLDS(m_context, DataType::Half, 1024, 2);
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 400, Size: 2048)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2448);
        EXPECT_EQ(m_context->ldsAllocator()->maxUsed(), 7600);

        // Try over allocating
        auto maxLDS = m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
        EXPECT_THROW(Register::Value::AllocateLDS(m_context, DataType::Int32, maxLDS / 4),
                     FatalError);
    }

    TEST_F(LDSTest, LDSAlignment)
    {
        auto lds1 = Register::Value::AllocateLDS(m_context, DataType::Half, 1, 2);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 0, Size: 2)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 2);

        auto lds2 = Register::Value::AllocateLDS(m_context, DataType::Half, 2, 2);
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 2, Size: 4)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 6);

        auto lds3 = Register::Value::AllocateLDS(m_context, DataType::Int32, 1, 4);
        EXPECT_EQ(lds3->toString(), "LDS:(Offset: 8, Size: 4)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 10);

        lds1.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 8);

        lds3.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4);

        lds2.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 0);
        // All free
    }

    TEST_F(LDSTest, LDSCurrentUsage)
    {
        auto lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 1000);
        EXPECT_EQ(lds1->toString(), "LDS:(Offset: 0, Size: 4000)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4000);

        auto lds2 = Register::Value::AllocateLDS(m_context, DataType::Int32, 1000);
        EXPECT_EQ(lds2->toString(), "LDS:(Offset: 4000, Size: 4000)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 8000);

        auto lds3 = Register::Value::AllocateLDS(m_context, DataType::Half, 10, 2);
        EXPECT_EQ(lds3->toString(), "LDS:(Offset: 8000, Size: 20)");
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 8020);

        lds1.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4020);

        lds3.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 4000);

        lds2.reset();
        EXPECT_EQ(m_context->ldsAllocator()->currentUsed(), 0);
        // All free
    }

    TEST_F(LDSTest, LDSOverAllocate)
    {
        auto maxLDS = m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
        EXPECT_THROW(Register::Value::AllocateLDS(m_context, DataType::Int32, 1 + (maxLDS / 4), 4),
                     FatalError);
    }

    TEST_F(LDSTest, LDSAllocationDoubleFreeTest)
    {
        auto allocator = m_context->ldsAllocator();
        auto alloc1    = allocator->allocate(100, 4);
        allocator->deallocate(alloc1);
        EXPECT_THROW(allocator->deallocate(alloc1), FatalError);
    }
}
