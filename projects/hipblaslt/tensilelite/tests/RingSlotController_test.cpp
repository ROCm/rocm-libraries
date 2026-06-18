// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RingSlotController.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>

using TensileLite::Client::RingSlotController;

TEST(RingSlotController, PrimeDoesNotOverfill)
{
    for(size_t count = 1; count <= 3; ++count)
    {
        RingSlotController controller(count);

        for(size_t attempt = 0; attempt < count + 3; ++attempt)
        {
            size_t const beforeAvailableSlots = controller.availableSlots();
            size_t const beforeActiveSlot     = controller.activeSlot();
            bool const   beforeNeedsBarrier   = controller.needsCopyBarrier();
            auto const    target = controller.nextPrimeSlot();

            EXPECT_EQ(controller.availableSlots(), beforeAvailableSlots);
            EXPECT_EQ(controller.activeSlot(), beforeActiveSlot);
            EXPECT_EQ(controller.needsCopyBarrier(), beforeNeedsBarrier);

            controller.markSlotPrimed();

            if(target)
                EXPECT_EQ(controller.availableSlots(), beforeAvailableSlots + 1);
            else
                EXPECT_EQ(controller.availableSlots(), beforeAvailableSlots);

            EXPECT_LE(controller.availableSlots(), count - 1);
            if(beforeAvailableSlots == count - 1)
                EXPECT_EQ(controller.availableSlots(), beforeAvailableSlots);
        }

        EXPECT_EQ(controller.nextPrimeSlot(), std::nullopt);
        EXPECT_EQ(controller.hasAvailableSlot(), count > 1);
        EXPECT_EQ(controller.activeSlot(), 0u);
        EXPECT_FALSE(controller.needsCopyBarrier());
        EXPECT_EQ(controller.availableSlots(), count - 1);
    }
}

TEST(RingSlotController, AdvanceConsumesAvailableSlot)
{
    RingSlotController controller(3);

    ASSERT_TRUE(controller.nextPrimeSlot().has_value());
    controller.markSlotPrimed();
    ASSERT_TRUE(controller.nextPrimeSlot().has_value());
    controller.markSlotPrimed();
    ASSERT_EQ(controller.availableSlots(), 2u);

    auto active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(*active, 1u);
    EXPECT_EQ(controller.activeSlot(), 1u);
    EXPECT_EQ(controller.availableSlots(), 1u);
}

TEST(RingSlotController, AdvanceWrapsModuloActiveBufferCount)
{
    RingSlotController controller(3);

    controller.markSlotPrimed();
    auto active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(*active, 1u);
    controller.markBarrierWaited();

    controller.markSlotPrimed();
    active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(*active, 2u);
    controller.markBarrierWaited();

    controller.markSlotPrimed();
    active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(*active, 0u);
    EXPECT_EQ(controller.activeSlot(), 0u);
    EXPECT_EQ(controller.availableSlots(), 0u);
}

TEST(RingSlotController, AdvanceMarksCopyBarrierRequired)
{
    RingSlotController controller(3);

    ASSERT_FALSE(controller.needsCopyBarrier());
    controller.markSlotPrimed();
    ASSERT_EQ(controller.availableSlots(), 1u);

    auto active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(*active, 1u);
    EXPECT_EQ(controller.activeSlot(), 1u);
    EXPECT_EQ(controller.availableSlots(), 0u);
    EXPECT_TRUE(controller.needsCopyBarrier());
}

TEST(RingSlotController, SingleSlotNeverPrimesOrAdvances)
{
    RingSlotController controller(1);

    EXPECT_EQ(controller.activeBufferCount(), 1u);
    EXPECT_EQ(controller.activeSlot(), 0u);
    EXPECT_EQ(controller.availableSlots(), 0u);
    EXPECT_FALSE(controller.hasAvailableSlot());
    EXPECT_FALSE(controller.needsCopyBarrier());
    EXPECT_FALSE(controller.hasPendingWork());
    EXPECT_EQ(controller.nextPrimeSlot(), std::nullopt);

    controller.markSlotPrimed();
    EXPECT_EQ(controller.availableSlots(), 0u);
    EXPECT_EQ(controller.nextPrimeSlot(), std::nullopt);
    EXPECT_FALSE(controller.hasAvailableSlot());

    EXPECT_EQ(controller.advance(), std::nullopt);
    EXPECT_EQ(controller.activeSlot(), 0u);
    EXPECT_EQ(controller.availableSlots(), 0u);
    EXPECT_FALSE(controller.needsCopyBarrier());
    EXPECT_FALSE(controller.hasPendingWork());
}

TEST(RingSlotController, MarkBarrierWaitedClearsBarrier)
{
    RingSlotController controller(3);

    controller.markSlotPrimed();
    auto active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_TRUE(controller.needsCopyBarrier());

    controller.markBarrierWaited();
    EXPECT_FALSE(controller.needsCopyBarrier());
    EXPECT_EQ(controller.activeSlot(), 1u);
    EXPECT_EQ(controller.availableSlots(), 0u);

    controller.markBarrierWaited();
    EXPECT_FALSE(controller.needsCopyBarrier());
    EXPECT_EQ(controller.activeSlot(), 1u);
    EXPECT_EQ(controller.availableSlots(), 0u);
}

TEST(RingSlotController, CancelResetsToSlotZero)
{
    RingSlotController controller(3);

    controller.markSlotPrimed();
    controller.markSlotPrimed();
    auto active = controller.advance();
    ASSERT_TRUE(active.has_value());
    EXPECT_EQ(*active, 1u);
    EXPECT_EQ(controller.availableSlots(), 1u);
    EXPECT_TRUE(controller.needsCopyBarrier());
    EXPECT_TRUE(controller.hasPendingWork());

    controller.cancel();
    EXPECT_EQ(controller.activeSlot(), 0u);
    EXPECT_EQ(controller.availableSlots(), 0u);
    EXPECT_FALSE(controller.needsCopyBarrier());
    EXPECT_FALSE(controller.hasPendingWork());

    auto next = controller.nextPrimeSlot();
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(*next, 1u);
}
