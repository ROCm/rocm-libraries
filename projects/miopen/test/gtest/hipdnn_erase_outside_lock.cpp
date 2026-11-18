// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for src/private/erase_outside_lock.hpp, which releases the hipDNN state of a
// MIOpen handle without holding the map lock. No hipDNN install and no GPU are needed.

#include <gtest/gtest.h>

#include "../../src/private/erase_outside_lock.hpp"

#include <memory>
#include <unordered_map>

namespace {

// std::mutex cannot say whether it is held, so the tests use one that can.
struct TrackingMutex
{
    void lock() { held = true; }
    void unlock() { held = false; }
    bool held = false;
};

struct Observation
{
    bool destroyed                = false;
    bool heldWhenDestroyed        = true;
    std::size_t sizeWhenDestroyed = 0;
};

// The stored value: on destruction it records the state of the mutex and the map.
struct Probe
{
    using Map = std::unordered_map<int, std::unique_ptr<Probe>>;

    Probe(const TrackingMutex& mutex, const Map& map, Observation& observation)
        : mutex_(mutex), map_(map), observation_(observation)
    {
    }
    ~Probe()
    {
        observation_.destroyed         = true;
        observation_.heldWhenDestroyed = mutex_.held;
        observation_.sizeWhenDestroyed = map_.size();
    }

    const TrackingMutex& mutex_;
    const Map& map_;
    Observation& observation_;
};

} // namespace

// The destructor of a hipDNN handle's state can end in a call that takes the same lock, so
// it must run after the lock is released.
TEST(CPU_HipdnnEraseOutsideLock_NONE, DestroysTheValueAfterReleasingTheMutex)
{
    TrackingMutex mutex;
    Probe::Map map;
    Observation observation;
    map[1] = std::make_unique<Probe>(mutex, map, observation);

    miopen::wrapper::EraseDestroyingOutsideLock(map, mutex, 1);

    EXPECT_TRUE(observation.destroyed);
    EXPECT_FALSE(observation.heldWhenDestroyed);
    EXPECT_FALSE(mutex.held);
}

TEST(CPU_HipdnnEraseOutsideLock_NONE, TheEntryIsGoneBeforeTheValueIsDestroyed)
{
    TrackingMutex mutex;
    Probe::Map map;
    Observation observation;
    Observation other;
    map[1] = std::make_unique<Probe>(mutex, map, observation);
    map[2] = std::make_unique<Probe>(mutex, map, other);

    miopen::wrapper::EraseDestroyingOutsideLock(map, mutex, 1);

    // A destructor that looks the handle up again must not find it.
    EXPECT_EQ(observation.sizeWhenDestroyed, 1u);
    EXPECT_EQ(map.count(1), 0u);
    EXPECT_EQ(map.count(2), 1u);
    EXPECT_FALSE(other.destroyed);
}

TEST(CPU_HipdnnEraseOutsideLock_NONE, AMissingKeyChangesNothing)
{
    TrackingMutex mutex;
    Probe::Map map;
    Observation observation;
    map[1] = std::make_unique<Probe>(mutex, map, observation);

    miopen::wrapper::EraseDestroyingOutsideLock(map, mutex, 2);

    EXPECT_EQ(map.size(), 1u);
    EXPECT_FALSE(observation.destroyed);
    EXPECT_FALSE(mutex.held);
}
