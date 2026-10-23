// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdlib>
#include <new>

#include "../src/detail/blocked_gemm.hpp"
#ifdef HOSTNUMERICS_TEST_BLAS
#include "../src/backends/blas_transforming.hpp"
#endif

namespace {
// This executable owns the allocation override; other test executables retain
// their normal allocator. Arm only after the complete invocation exists.
thread_local bool failNextAllocation = false;

template <typename Function>
void withAllocationFailure(Function&& function) {
    struct Reset {
        ~Reset() {
            failNextAllocation = false;
        }
    } reset;
    failNextAllocation = true;
    function();
}
}  // namespace

void* operator new(std::size_t size) {
    if (failNextAllocation) {
        failNextAllocation = false;
        throw std::bad_alloc();
    }
    if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(size);
    } catch (...) {
        return nullptr;
    }
}

void operator delete(void* memory, const std::nothrow_t&) noexcept {
    std::free(memory);
}

TEST(GemmResourceFailure, ValidationFailurePropagatesInsteadOfBecomingUnsupported) {
    using namespace roc::hostnumerics;
    const detail::GemmInvocation request(Tensor(ScalarType::Float32, Shape{2, 2}),
                                         Tensor(ScalarType::Float32, Shape{2, 2}),
                                         Tensor(ScalarType::Float32, Shape{2, 2}));
    EXPECT_TRUE(detail::queryGemmSupport(request, GemmBackend::Blocked));
    EXPECT_THROW(withAllocationFailure(
                     [&] { (void)detail::queryGemmSupport(request, GemmBackend::Blocked); }),
                 std::bad_alloc);
    EXPECT_THROW(withAllocationFailure([&] { (void)detail::queryBlockedGemmSupport(request); }),
                 std::bad_alloc);
    EXPECT_THROW(
        withAllocationFailure([&] { (void)detail::executeGemm(request, GemmBackend::Blocked); }),
        std::bad_alloc);
#ifdef HOSTNUMERICS_TEST_BLAS
    EXPECT_THROW(withAllocationFailure([&] { (void)BlasGemmBackend{}.querySupport(request); }),
                 std::bad_alloc);
    EXPECT_THROW(withAllocationFailure([&] { (void)queryTransformingBlasGemmSupport(request); }),
                 std::bad_alloc);
    EXPECT_THROW(withAllocationFailure(
                     [&] { (void)detail::executeBlasGemm(request, GemmBackend::Automatic); }),
                 std::bad_alloc);
#endif
}

TEST(GemmResourceFailure, CheckedGeometryDoesNotAllocateButOwningFactoriesCan) {
    using namespace roc::hostnumerics;
    const Shape shape{2, 3};
    const Layout layout(shape, {3, 1}, 0);
    const std::array<size_t, 2> order{0, 1};
    const std::array<size_t, 2> valid{1, 2}, invalid{2, 0};
    withAllocationFailure([&] {
        const auto offset = layout.elementOffset(valid);
        const auto failure = layout.elementOffset(invalid);
        const auto bytes = detail::getStorageBytesForLayout(ScalarType::Float32, layout);
        const bool noAllocation = failNextAllocation;
        failNextAllocation = false;
        ASSERT_TRUE(offset);
        EXPECT_EQ(*offset, 5);
        ASSERT_FALSE(failure);
        EXPECT_EQ(failure.error().error, detail::LayoutError::CoordinateOutOfRange);
        ASSERT_TRUE(bytes);
        EXPECT_EQ(*bytes, 24);
        EXPECT_TRUE(noAllocation);
    });
    EXPECT_THROW(withAllocationFailure([&] { (void)Layout::getContiguousLayout(shape, order); }),
                 std::bad_alloc);
}
