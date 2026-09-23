// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Counts global operator new calls to prove copying an array Value shares its
// storage. Replacing operator new applies to the whole executable and turns off
// AddressSanitizer's new/delete mismatch checks there, so this file builds into
// its own small test binary rather than into hipdnn_plugin_sdk_tests.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <optional>

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

using V = jexpr::Value;

namespace
{

/// Per-thread state behind the replaced global allocation functions below.
///
/// Counting is off unless a scope arms it, and it is per-thread, so
/// allocations made by GoogleTest or another thread never land in a
/// measurement.
struct AllocationCounterState
{
    std::size_t newCalls = 0;
    bool armed = false;
};

AllocationCounterState& allocationCounterState()
{
    thread_local AllocationCounterState s_state;
    return s_state;
}

/// Arms the counter for its scope. The destructor disarms it, so an assertion
/// that fails inside the scope cannot leave counting on.
class ArmedAllocationCounter
{
public:
    ArmedAllocationCounter()
    {
        AllocationCounterState& state = allocationCounterState();
        state.newCalls = 0;
        state.armed = true;
    }
    ~ArmedAllocationCounter()
    {
        allocationCounterState().armed = false;
    }
    ArmedAllocationCounter(const ArmedAllocationCounter&) = delete;
    ArmedAllocationCounter(ArmedAllocationCounter&&) = delete;
    ArmedAllocationCounter& operator=(const ArmedAllocationCounter&) = delete;
    ArmedAllocationCounter& operator=(ArmedAllocationCounter&&) = delete;
};

/// Run `op` and report how many calls to the global operator new it made.
/// `op` must not use a GoogleTest macro: a failing assertion allocates, and
/// that allocation would be counted as if the measured code had made it.
template <typename Op>
std::size_t allocationsDuring(const Op& op)
{
    const ArmedAllocationCounter armed;
    op();
    return allocationCounterState().newCalls;
}

} // namespace

// These forward to malloc/free and only touch the counter while a scope is
// armed on the calling thread. Over-aligned allocation is left to the default
// implementation, which pairs with the default aligned deallocation.
void* operator new(std::size_t size)
{
    AllocationCounterState& state = allocationCounterState();
    if(state.armed)
    {
        ++state.newCalls;
    }
    // malloc(0) may return nullptr, which operator new must never do.
    void* const memory = std::malloc(size != 0 ? size : 1);
    if(memory == nullptr)
    {
        throw std::bad_alloc();
    }
    return memory;
}
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}
void operator delete(void* memory) noexcept
{
    std::free(memory);
}
void operator delete[](void* memory) noexcept
{
    ::operator delete(memory);
}
void operator delete(void* memory, std::size_t /*size*/) noexcept
{
    ::operator delete(memory);
}
void operator delete[](void* memory, std::size_t /*size*/) noexcept
{
    ::operator delete(memory);
}

// A copy of an array Value shares its immutable storage, so passing a Value
// around by value costs a reference-count update, not an allocation, however
// deeply it nests. An allocation count is deterministic, so it can be exact.
TEST(TestJsonValueAllocations, CopyingAnArrayValueAllocatesNothing)
{
    const V original(V::Array{V(1), V(V::Array{V(2), V(3)})});
    // Copy once before arming the counter, so the thread-local counter state
    // and anything else a first copy touches is already initialised.
    EXPECT_EQ(V(original), original);

    std::optional<V> copyConstructed;
    V copyAssigned;
    V nestedCopy;
    V copyOfCopy;
    const std::size_t allocations = allocationsDuring([&] {
        copyConstructed.emplace(original);
        copyAssigned = original;
        nestedCopy = original.asArray()[1];
        copyOfCopy = copyAssigned;
    });

    EXPECT_EQ(allocations, 0U);
    EXPECT_EQ(*copyConstructed, original);
    EXPECT_EQ(copyAssigned, original);
    EXPECT_EQ(nestedCopy, original.asArray()[1]);
    EXPECT_EQ(copyOfCopy, original);
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
