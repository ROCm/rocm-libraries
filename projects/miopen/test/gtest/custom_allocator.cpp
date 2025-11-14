/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <stdexcept>

#include "get_handle.hpp"

namespace {

enum class AllocatorTestType
{
    TestAllocator,
    TestNullAllocator,
    TestDeallocator,
    TestDeallocator2
};

struct AllocatorTestCase
{
    AllocatorTestType test_type;
};

std::vector<AllocatorTestCase> GetAllocatorTestCases()
{
    return {{AllocatorTestType::TestAllocator},
            {AllocatorTestType::TestNullAllocator},
            {AllocatorTestType::TestDeallocator},
            {AllocatorTestType::TestDeallocator2}};
}

struct GPU_CustomAllocator_FP32 : public ::testing::TestWithParam<AllocatorTestCase>
{
    void SetUp() override
    {
        auto&& handle = get_handle();
        buffer        = handle.Create(size);
    }

    void TearDown() override
    {
        // Reset allocator to default after each test
        auto&& handle = get_handle();
        handle.SetAllocator(nullptr, nullptr, nullptr);
    }

    miopen::Allocator::ManageDataPtr buffer;
    static constexpr int size = 42;
};

TEST_P(GPU_CustomAllocator_FP32, Test)
{
    const auto& test_case = this->GetParam();
    auto&& handle         = get_handle();

    switch(test_case.test_type)
    {
    case AllocatorTestType::TestAllocator: {
        handle.SetAllocator(
            +[](void*, std::size_t n) -> void* {
                EXPECT_EQ(n, size);
                throw "Called allocator"; // NOLINT
            },
            nullptr,
            nullptr);
        miopen::Allocator::ManageDataPtr p = nullptr;
        EXPECT_THROW({ p = handle.Create(size); }, const char*);
        break;
    }
    case AllocatorTestType::TestNullAllocator: {
        handle.SetAllocator(
            +[](void*, std::size_t n) -> void* {
                EXPECT_EQ(n, size);
                return nullptr;
            },
            nullptr,
            nullptr);
        miopen::Allocator::ManageDataPtr p = nullptr;
        EXPECT_THROW({ p = handle.Create(size); }, std::exception);
        break;
    }
    case AllocatorTestType::TestDeallocator: {
        handle.SetAllocator(
            +[](void* ctx, std::size_t n) -> void* {
                EXPECT_EQ(n, size);
                return reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx)->get();
            },
            +[](void* ctx, void* data) {
                auto b = reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx);
                EXPECT_EQ(data, b->get());
                *b = nullptr;
            },
            &buffer);
        miopen::Allocator::ManageDataPtr p = handle.Create(size);
        EXPECT_EQ(p.get(), buffer.get());
        p = nullptr;
        EXPECT_EQ(p, nullptr);
        EXPECT_EQ(buffer, nullptr);
        break;
    }
    case AllocatorTestType::TestDeallocator2: {
        handle.SetAllocator(
            +[](void* ctx, std::size_t n) -> void* {
                EXPECT_EQ(n, size);
                return reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx)->get();
            },
            +[](void* ctx, void* data) {
                auto b = reinterpret_cast<miopen::Allocator::ManageDataPtr*>(ctx);
                EXPECT_EQ(data, b->get());
            },
            &buffer);
        miopen::Allocator::ManageDataPtr p = handle.Create(size);
        EXPECT_EQ(p.get(), buffer.get());
        p = nullptr;
        EXPECT_EQ(p, nullptr);
        EXPECT_NE(buffer, nullptr);
        break;
    }
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CustomAllocator_FP32,
                         testing::ValuesIn(GetAllocatorTestCases()));

} // namespace
