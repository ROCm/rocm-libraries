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

#include <rocRoller/InstructionValues/RegisterAllocator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"

using namespace rocRoller;

namespace RegisterAllocatorTest
{
    class RegisterAllocatorTest : public GenericContextFixture
    {
    protected:
        void SetUp() override
        {
            m_kernelOptions->maxACCVGPRs = 50;
            m_kernelOptions->maxSGPRs    = 50;
            m_kernelOptions->maxVGPRs    = 50;

            GenericContextFixture::SetUp();
        }
    };

    TEST_F(RegisterAllocatorTest, WeakPtrBehaviour)
    {
        std::weak_ptr<int> test;
        EXPECT_EQ(true, test.expired());

        auto sh = std::make_shared<int>(4);
        test    = sh;

        EXPECT_EQ(false, test.expired());

        sh.reset();

        EXPECT_EQ(true, test.expired());
    }

    TEST_F(RegisterAllocatorTest, SimpleBasicScheme)
    {
        auto allocator = std::make_shared<Register::Allocator>(
            Register::Type::Scalar, 10, Register::AllocatorScheme::FirstFit);

        auto allocOpts      = Register::AllocationOptions::FullyContiguous();
        allocOpts.alignment = 1;

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->regType(), Register::Type::Scalar);
        EXPECT_EQ(allocator->size(), 10);
        EXPECT_EQ(allocator->currentlyFree(), 10);

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 1, allocOpts);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 10);

        auto [idx, blockSize] = allocator->findContiguousRange(0, 1, alloc0->options());
        EXPECT_EQ(0, idx);

        allocator->allocate(alloc0);

        EXPECT_EQ(std::vector{0}, alloc0->registerIndices());
        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(0, allocator->maxUsed());
        EXPECT_EQ(1, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 9);

        auto [idx2, blockSize2] = allocator->findContiguousRange(0, 1, alloc0->options());
        EXPECT_EQ(1, idx2);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 3, allocOpts);

        EXPECT_EQ((std::vector{1, 2, 3}),
                  allocator->findFree(alloc1->registerCount(), alloc1->options()));
        allocator->allocate(alloc1);

        EXPECT_EQ((std::vector{1, 2, 3}), alloc1->registerIndices());
        EXPECT_EQ(false, allocator->isFree(1));
        EXPECT_EQ(false, allocator->isFree(2));
        EXPECT_EQ(false, allocator->isFree(3));
        EXPECT_EQ(true, allocator->isFree(4));
        EXPECT_EQ(3, allocator->maxUsed());
        EXPECT_EQ(4, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 6);

        alloc0.reset();
        EXPECT_EQ(true, allocator->isFree(0));
        EXPECT_EQ(3, allocator->maxUsed());
        EXPECT_EQ(4, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 7);

        Register::AllocationOptions options;
        options.alignment            = 1;
        options.contiguousChunkWidth = 1;

        auto alloc2 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 3, options);

        allocator->allocate(alloc2);

        auto indices = alloc2->registerIndices();
        std::sort(indices.begin(), indices.end());

        EXPECT_EQ((std::vector{0, 4, 5}), indices);
        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(false, allocator->isFree(4));
        EXPECT_EQ(false, allocator->isFree(5));
        EXPECT_EQ(true, allocator->isFree(6));
        EXPECT_EQ(allocator->currentlyFree(), 4);

        auto alloc3 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 5, options);

        EXPECT_EQ(false, allocator->canAllocate(alloc3));

        alloc1->free();
        EXPECT_EQ(true, allocator->isFree(1));
        EXPECT_EQ(true, allocator->isFree(2));
        EXPECT_EQ(true, allocator->isFree(3));
        EXPECT_EQ(5, allocator->maxUsed());
        EXPECT_EQ(6, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 7);
    }

    TEST_F(RegisterAllocatorTest, SimplePerfectFitScheme)
    {
        auto allocator = std::make_shared<Register::Allocator>(
            Register::Type::Scalar, 10, Register::AllocatorScheme::PerfectFit);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->regType(), Register::Type::Scalar);
        EXPECT_EQ(allocator->size(), 10);
        EXPECT_EQ(allocator->currentlyFree(), 10);

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 10);

        EXPECT_EQ(0, allocator->findContiguousRange(0, 1, alloc0->options()).first);

        allocator->allocate(alloc0);

        EXPECT_EQ(std::vector{0}, alloc0->registerIndices());
        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(0, allocator->maxUsed());
        EXPECT_EQ(1, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 9);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 3);

        allocator->allocate(alloc1);

        EXPECT_EQ((std::vector{3, 2, 1}), alloc1->registerIndices());
        EXPECT_EQ(false, allocator->isFree(1));
        EXPECT_EQ(false, allocator->isFree(2));
        EXPECT_EQ(false, allocator->isFree(3));
        EXPECT_EQ(true, allocator->isFree(4));
        EXPECT_EQ(3, allocator->maxUsed());
        EXPECT_EQ(4, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 6);

        alloc0.reset();
        EXPECT_EQ(true, allocator->isFree(0));
        EXPECT_EQ(3, allocator->maxUsed());
        EXPECT_EQ(4, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 7);

        Register::AllocationOptions options;
        options.contiguousChunkWidth = 1;

        auto alloc2 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 3, options);

        allocator->allocate(alloc2);

        EXPECT_EQ((std::vector{5, 4, 0}), alloc2->registerIndices());
        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(false, allocator->isFree(4));
        EXPECT_EQ(false, allocator->isFree(5));
        EXPECT_EQ(true, allocator->isFree(6));
        EXPECT_EQ(allocator->currentlyFree(), 4);

        auto alloc3 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 5, options);

        EXPECT_EQ(false, allocator->canAllocate(alloc3));

        alloc1->free();
        EXPECT_EQ(true, allocator->isFree(1));
        EXPECT_EQ(true, allocator->isFree(2));
        EXPECT_EQ(true, allocator->isFree(3));
        EXPECT_EQ(5, allocator->maxUsed());
        EXPECT_EQ(6, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 7);
    }

    TEST_F(RegisterAllocatorTest, MaxReg)
    {
        std::vector<Register::Type> regTypes = {
            Register::Type::Accumulator,
            Register::Type::Vector,
            Register::Type::Scalar,
        };

        ASSERT_LE(m_context->kernelOptions()->maxACCVGPRs, 50);
        ASSERT_LE(m_context->kernelOptions()->maxVGPRs, 50);
        ASSERT_LE(m_context->kernelOptions()->maxSGPRs, 50);

        for(auto regType : regTypes)
        {
            EXPECT_NO_THROW({ createRegisters(regType, DataType::Float, 5); });
            EXPECT_THROW({ createRegisters(regType, DataType::Float, 55); }, FatalError);
        }
    }

    TEST_F(RegisterAllocatorTest, PerfectFit)
    {
        auto allocator = std::make_shared<Register::Allocator>(
            Register::Type::Scalar, 16, Register::AllocatorScheme::PerfectFit);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->regType(), Register::Type::Scalar);
        EXPECT_EQ(allocator->size(), 16);
        EXPECT_EQ(allocator->currentlyFree(), 16);

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 2);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 4);

        {
            auto alloc2 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 4);
            auto alloc3 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 2);
            allocator->allocate(alloc2);
            allocator->allocate(alloc0);
            allocator->allocate(alloc3);
            allocator->allocate(alloc1);

            //[XXXX,XXXX,XXXX,OOOO]
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 4);
        }

        //[OOOO,XXOO,XXXX,OOOO]
        EXPECT_EQ(true, allocator->isFree(0));
        EXPECT_EQ(true, allocator->isFree(1));
        EXPECT_EQ(true, allocator->isFree(2));
        EXPECT_EQ(true, allocator->isFree(3));
        EXPECT_EQ(false, allocator->isFree(4));
        EXPECT_EQ(false, allocator->isFree(5));
        EXPECT_EQ(true, allocator->isFree(6));
        EXPECT_EQ(true, allocator->isFree(7));
        EXPECT_EQ(false, allocator->isFree(8));
        EXPECT_EQ(false, allocator->isFree(9));
        EXPECT_EQ(false, allocator->isFree(10));
        EXPECT_EQ(false, allocator->isFree(11));
        EXPECT_EQ(11, allocator->maxUsed());
        EXPECT_EQ(12, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 10);

        {
            auto allocFit = std::make_shared<Register::Allocation>(
                m_context,
                Register::Type::Scalar,
                DataType::Float,
                2,
                Register::AllocationOptions::FullyContiguous());
            allocator->allocate(allocFit);

            //[OOOO,XXXX,XXXX,OOOO]
            EXPECT_EQ(true, allocator->isFree(0));
            EXPECT_EQ(true, allocator->isFree(1));
            EXPECT_EQ(false, allocator->isFree(6));
            EXPECT_EQ(false, allocator->isFree(7));
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 8);
        }
    }

    TEST_F(RegisterAllocatorTest, EndOfBlockAlloc)
    {
        auto allocator = std::make_shared<Register::Allocator>(
            Register::Type::Scalar, 16, Register::AllocatorScheme::PerfectFit);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->regType(), Register::Type::Scalar);
        EXPECT_EQ(allocator->size(), 16);
        EXPECT_EQ(allocator->currentlyFree(), 16);

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 1);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 4);

        {
            auto alloc2 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 1);
            auto alloc3 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 6);
            allocator->allocate(alloc0);
            allocator->allocate(alloc2);
            allocator->allocate(alloc3);
            allocator->allocate(alloc1);

            //[XXXX,XXXX,XXXX,OOOO]
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 4);
        }

        //[XOOO,OOOO,XXXX,OOOO]

        EXPECT_EQ(false, allocator->isFree(0));
        EXPECT_EQ(true, allocator->isFree(1));
        EXPECT_EQ(true, allocator->isFree(2));
        EXPECT_EQ(true, allocator->isFree(3));
        EXPECT_EQ(true, allocator->isFree(4));
        EXPECT_EQ(true, allocator->isFree(5));
        EXPECT_EQ(true, allocator->isFree(6));
        EXPECT_EQ(true, allocator->isFree(7));
        EXPECT_EQ(false, allocator->isFree(8));
        EXPECT_EQ(false, allocator->isFree(9));
        EXPECT_EQ(false, allocator->isFree(10));
        EXPECT_EQ(false, allocator->isFree(11));
        EXPECT_EQ(11, allocator->maxUsed());
        EXPECT_EQ(12, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 11);

        {
            Register::AllocationOptions opt;
            opt.alignment            = 2;
            opt.contiguousChunkWidth = Register::FULLY_CONTIGUOUS;

            auto allocEnd = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 2, opt);
            allocator->allocate(allocEnd);

            //[XOOO,OOXX,XXXX,OOOO]
            EXPECT_EQ(false, allocator->isFree(0));
            EXPECT_EQ(true, allocator->isFree(1));
            EXPECT_EQ(true, allocator->isFree(2));
            EXPECT_EQ(true, allocator->isFree(3));
            EXPECT_EQ(false, allocator->isFree(6));
            EXPECT_EQ(false, allocator->isFree(7));
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 9);
        }
    }

    TEST_F(RegisterAllocatorTest, MinContiguity)
    {
        auto test_scheme = [&](Register::AllocatorScheme scheme) {
            {
                auto allocator
                    = std::make_shared<Register::Allocator>(Register::Type::Scalar, 16, scheme);
                Register::AllocationOptions opt{.contiguousChunkWidth = 2, .alignment = 2};

                auto alloc0 = std::make_shared<Register::Allocation>(
                    m_context, Register::Type::Scalar, DataType::Float, 6, opt);
                allocator->allocate(alloc0);

                EXPECT_EQ((std::vector{4, 5, 2, 3, 0, 1}), alloc0->registerIndices());
            }
            {
                auto allocator
                    = std::make_shared<Register::Allocator>(Register::Type::Scalar, 16, scheme);
                Register::AllocationOptions opt{.contiguousChunkWidth = 3, .alignment = 2};

                auto alloc0 = std::make_shared<Register::Allocation>(
                    m_context, Register::Type::Scalar, DataType::Float, 12, opt);
                allocator->allocate(alloc0);

                auto expected = scheme == Register::AllocatorScheme::PerfectFit
                                    ? std::vector{0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}
                                    : std::vector{12, 13, 14, 8, 9, 10, 4, 5, 6, 0, 1, 2};
                EXPECT_EQ(expected, alloc0->registerIndices());
            }
            {
                auto allocator
                    = std::make_shared<Register::Allocator>(Register::Type::Scalar, 17, scheme);
                Register::AllocationOptions opt{.contiguousChunkWidth = 5, .alignment = 3};

                auto alloc0 = std::make_shared<Register::Allocation>(
                    m_context, Register::Type::Scalar, DataType::Float, 15, opt);
                allocator->allocate(alloc0);

                auto expected
                    = scheme == Register::AllocatorScheme::PerfectFit
                          ? std::vector{0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16}
                          : std::vector{12, 13, 14, 15, 16, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4};
                EXPECT_EQ(expected, alloc0->registerIndices());
            }
            {
                auto allocator
                    = std::make_shared<Register::Allocator>(Register::Type::Scalar, 16, scheme);
                Register::AllocationOptions opt{.contiguousChunkWidth = 1, .alignment = 1};

                auto alloc0 = std::make_shared<Register::Allocation>(
                    m_context, Register::Type::Scalar, DataType::Float, 6, opt);
                allocator->allocate(alloc0);

                EXPECT_EQ((std::vector{5, 4, 3, 2, 1, 0}), alloc0->registerIndices());
            }
        };

        test_scheme(Register::AllocatorScheme::PerfectFit);
        test_scheme(Register::AllocatorScheme::FirstFit);
    }

    TEST_F(RegisterAllocatorTest, Contiguity)
    {
        auto allocator = std::make_shared<Register::Allocator>(
            Register::Type::Scalar, 16, Register::AllocatorScheme::PerfectFit);

        EXPECT_EQ(-1, allocator->maxUsed());
        EXPECT_EQ(0, allocator->useCount());
        EXPECT_EQ(allocator->regType(), Register::Type::Scalar);
        EXPECT_EQ(allocator->size(), 16);
        EXPECT_EQ(allocator->currentlyFree(), 16);

        EXPECT_EQ(allocator->findContiguousRange(0, 1, {.contiguousChunkWidth = 1}).first, 0);
        EXPECT_EQ(allocator->findContiguousRange(0, 1, {.contiguousChunkWidth = 1}).second, 16);

        auto alloc0 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 2);

        auto alloc1 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 1);

        auto alloc2 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 1);

        auto alloc3 = std::make_shared<Register::Allocation>(
            m_context, Register::Type::Scalar, DataType::Float, 2);

        {
            auto alloc4 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 2);

            auto alloc5 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 1);

            auto alloc6 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 1);

            auto alloc7 = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 2);

            allocator->allocate(alloc4);
            allocator->allocate(alloc0);
            allocator->allocate(alloc5);
            allocator->allocate(alloc1);
            allocator->allocate(alloc6);
            allocator->allocate(alloc2);
            allocator->allocate(alloc7);
            allocator->allocate(alloc3);

            //[XXXX,XXXX,XXXX,OOOO]
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 4);
        }

        //[OOXX,OXOX,OOXX,OOOO]
        EXPECT_EQ(true, allocator->isFree(0));
        EXPECT_EQ(true, allocator->isFree(1));
        EXPECT_EQ(false, allocator->isFree(2));
        EXPECT_EQ(false, allocator->isFree(3));
        EXPECT_EQ(true, allocator->isFree(4));
        EXPECT_EQ(false, allocator->isFree(5));
        EXPECT_EQ(true, allocator->isFree(6));
        EXPECT_EQ(false, allocator->isFree(7));
        EXPECT_EQ(true, allocator->isFree(8));
        EXPECT_EQ(true, allocator->isFree(9));
        EXPECT_EQ(false, allocator->isFree(10));
        EXPECT_EQ(false, allocator->isFree(11));
        EXPECT_EQ(11, allocator->maxUsed());
        EXPECT_EQ(12, allocator->useCount());
        EXPECT_EQ(allocator->currentlyFree(), 10);

        {
            Register::AllocationOptions opt;
            opt.contiguousChunkWidth = 1;

            auto allocContig = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 4, opt);
            allocator->allocate(allocContig);

            //[XXXX,XXXX,OOXX,OOOO]
            EXPECT_EQ(false, allocator->isFree(0));
            EXPECT_EQ(false, allocator->isFree(1));
            EXPECT_EQ(false, allocator->isFree(4));
            EXPECT_EQ(false, allocator->isFree(6));
            EXPECT_EQ(true, allocator->isFree(8));
            EXPECT_EQ(true, allocator->isFree(9));
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 6);
        }

        //[OOXX,OXOX,OOXX,OOOO]

        {
            Register::AllocationOptions opt;
            opt.contiguousChunkWidth = 2;

            auto allocContig = std::make_shared<Register::Allocation>(
                m_context, Register::Type::Scalar, DataType::Float, 4, opt);
            allocator->allocate(allocContig);

            //[XXXX,OXOX,XXXX,OOOO]
            EXPECT_EQ(false, allocator->isFree(0));
            EXPECT_EQ(false, allocator->isFree(1));
            EXPECT_EQ(true, allocator->isFree(4));
            EXPECT_EQ(true, allocator->isFree(6));
            EXPECT_EQ(false, allocator->isFree(8));
            EXPECT_EQ(false, allocator->isFree(9));
            EXPECT_EQ(11, allocator->maxUsed());
            EXPECT_EQ(12, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 6);
        }

        //[OOXX,OXOX,OOXX,OOOO]

        {
            EXPECT_EQ(allocator->findContiguousRange(0, 4, {.contiguousChunkWidth = 4}).first, 12);
            EXPECT_EQ(allocator->findContiguousRange(0, 4, {.contiguousChunkWidth = 4}).second, 4);
            std::vector<int> freeReg = {12, 13, 14, 15};
            EXPECT_EQ(allocator->findFree(4, {.contiguousChunkWidth = 4}), freeReg);

            auto allocContig = std::make_shared<Register::Allocation>(
                m_context,
                Register::Type::Scalar,
                DataType::Float,
                4,
                Register::AllocationOptions::FullyContiguous());
            allocator->allocate(allocContig);

            //[OOXX,OXOX,OOXX,XXXX]
            EXPECT_EQ(true, allocator->isFree(0));
            EXPECT_EQ(true, allocator->isFree(1));
            EXPECT_EQ(true, allocator->isFree(4));
            EXPECT_EQ(true, allocator->isFree(6));
            EXPECT_EQ(true, allocator->isFree(8));
            EXPECT_EQ(true, allocator->isFree(9));
            EXPECT_EQ(15, allocator->maxUsed());
            EXPECT_EQ(16, allocator->useCount());
            EXPECT_EQ(allocator->currentlyFree(), 6);
        }
    }

    class ARCH_RegisterAllocatorTest : public GPUContextFixture
    {
    };

    TEST_P(ARCH_RegisterAllocatorTest, GPU_AlignedSGPR)
    {
        auto       k             = m_context->kernel();
        auto       arch          = m_context->targetArchitecture();
        auto const wavefrontSize = arch.GetCapability(GPUCapability::DefaultWavefrontSize);

        k->setKernelName("AlignedSGPR");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument({"a", DataType::Int32});
        k->addArgument({"b", DataType::Int32});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto s_c = Register::Value::Placeholder(m_context,
                                                    Register::Type::Scalar,
                                                    (wavefrontSize == 64) ? DataType::Bool64
                                                                          : DataType::Bool32,
                                                    1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield s_c->allocate();
            co_yield v_a->allocate();
            co_yield v_b->allocate();

            // this will trip "invalid register alignment" if s_c
            // isn't aligned properly on some archs
            co_yield_(Instruction("v_cmp_ge_i32", {s_c}, {v_a, v_b}, {}, ""));
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);
    }

    INSTANTIATE_TEST_SUITE_P(ARCH_RegisterAllocatorTests,
                             ARCH_RegisterAllocatorTest,
                             supportedISATuples());

}
