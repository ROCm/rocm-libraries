// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for src/private/erase_outside_lock.hpp and src/private/insert_outside_lock.hpp,
// which create and release the hipDNN state of a MIOpen handle without holding the map lock.
// The tests use no hipDNN and no GPU, but like every test binary built on main_hip.cpp (and
// miopen_gtest) they report a failure when no device is visible.

#include <gtest/gtest.h>

#include "../../src/private/erase_outside_lock.hpp"
#include "../../src/private/insert_outside_lock.hpp"

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

// A state that lost the race to be the handle's state is destroyed by the same plugin code, so
// it too must be destroyed after the lock is released.
TEST(CPU_HipdnnEraseOutsideLock_NONE, DestroysALosingCandidateAfterReleasingTheMutex)
{
    TrackingMutex mutex;
    Probe::Map map;
    Observation winner;
    Observation loser;
    map[1] = std::make_unique<Probe>(mutex, map, winner);

    const bool inserted = miopen::wrapper::InsertDestroyingLoserOutsideLock(
        map, mutex, 1, std::make_unique<Probe>(mutex, map, loser));

    EXPECT_FALSE(inserted);
    EXPECT_TRUE(loser.destroyed);
    EXPECT_FALSE(loser.heldWhenDestroyed);
    EXPECT_FALSE(mutex.held);
    EXPECT_FALSE(winner.destroyed);
    EXPECT_EQ(map.size(), 1u);
}

TEST(CPU_HipdnnEraseOutsideLock_NONE, StoresACandidateWhenTheKeyIsFree)
{
    TrackingMutex mutex;
    Probe::Map map;
    Observation observation;

    auto candidate           = std::make_unique<Probe>(mutex, map, observation);
    const Probe* const probe = candidate.get();
    const bool inserted =
        miopen::wrapper::InsertDestroyingLoserOutsideLock(map, mutex, 1, std::move(candidate));

    EXPECT_TRUE(inserted);
    EXPECT_FALSE(observation.destroyed);
    EXPECT_FALSE(mutex.held);
    ASSERT_EQ(map.count(1), 1u);
    EXPECT_EQ(map.at(1).get(), probe);
}
