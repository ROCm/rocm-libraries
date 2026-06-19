// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "GpuInputSlotSet.hpp"

namespace
{
    using TensileLite::Client::GpuInputSlot;
    using TensileLite::Client::GpuInputSlotSet;

    std::shared_ptr<TensileLite::ProblemInputs> makeSentinelInputs(uintptr_t value)
    {
        return {reinterpret_cast<TensileLite::ProblemInputs*>(value), [](auto*) {}};
    }

    void populateSlot(GpuInputSlot& slot, uintptr_t ptrValue, uintptr_t batchValue, uintptr_t inputsValue)
    {
        slot.ptrs.push_back(reinterpret_cast<void*>(ptrValue));
        slot.batchPtrs.push_back(reinterpret_cast<void**>(batchValue));
        slot.cachedInputs = makeSentinelInputs(inputsValue);
    }
} // namespace

TEST(GpuInputSlotSet, SlotsAreIndependentAndClearFromKeepsSlot0)
{
    GpuInputSlotSet<3> slots;

    populateSlot(slots.at(0), 0x10, 0x20, 0x30);
    populateSlot(slots.at(1), 0x40, 0x50, 0x60);
    populateSlot(slots.at(2), 0x70, 0x80, 0x90);

    EXPECT_TRUE(slots.at(0).populated());
    EXPECT_TRUE(slots.populated(1));
    EXPECT_TRUE(slots.populated(2));
    EXPECT_NE(slots.at(0).cachedInputs, slots.at(1).cachedInputs);
    EXPECT_NE(slots.at(0).cachedInputs, slots.at(2).cachedInputs);
    EXPECT_NE(slots.at(0).ptrs.front(), slots.at(1).ptrs.front());
    EXPECT_NE(slots.at(0).ptrs.front(), slots.at(2).ptrs.front());
    EXPECT_NE(slots.at(0).batchPtrs.front(), slots.at(1).batchPtrs.front());
    EXPECT_NE(slots.at(0).batchPtrs.front(), slots.at(2).batchPtrs.front());

    slots.clearFrom(1);

    EXPECT_TRUE(slots.at(0).populated());
    EXPECT_TRUE(slots.at(0).cachedInputs);
    EXPECT_FALSE(slots.at(1).populated());
    EXPECT_FALSE(slots.at(1).cachedInputs);
    EXPECT_FALSE(slots.at(2).populated());
    EXPECT_FALSE(slots.at(2).cachedInputs);
    EXPECT_EQ(slots.at(0).batchPtrs.size(), 1u);
    EXPECT_EQ(slots.at(0).ptrs.size(), 1u);
}

TEST(GpuInputSlotSet, OutOfRangeAccessThrows)
{
    GpuInputSlotSet<3> slots;

    EXPECT_THROW(slots.at(3), std::out_of_range);
    EXPECT_THROW(slots.clear(3), std::out_of_range);
    EXPECT_THROW(slots.populated(3), std::out_of_range);
}
