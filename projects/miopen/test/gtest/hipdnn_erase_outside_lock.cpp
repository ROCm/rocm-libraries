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

#include <cstddef>
#include <memory>
#include <new>
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

// Throws std::bad_alloc from the Nth allocation after `failIn` is set (1 is the next one), to
// reach the points where a map insert can fail part-way.
int failIn = 0;

template <typename T>
struct FailingAllocator
{
    using value_type = T;

    FailingAllocator() = default;
    template <typename U>
    FailingAllocator(const FailingAllocator<U>&)
    {
    }

    T* allocate(std::size_t n)
    {
        if(failIn > 0 && --failIn == 0)
            throw std::bad_alloc();
        return std::allocator<T>().allocate(n);
    }
    void deallocate(T* p, std::size_t n) { std::allocator<T>().deallocate(p, n); }

    template <typename U>
    bool operator==(const FailingAllocator<U>&) const
    {
        return true;
    }
    template <typename U>
    bool operator!=(const FailingAllocator<U>&) const
    {
        return false;
    }
};

struct LockProbe
{
    LockProbe(const TrackingMutex& mutex, Observation& observation)
        : mutex_(mutex), observation_(observation)
    {
    }
    ~LockProbe()
    {
        observation_.destroyed         = true;
        observation_.heldWhenDestroyed = mutex_.held;
    }

    const TrackingMutex& mutex_;
    Observation& observation_;
};

} // namespace

// The destructor of a hipDNN handle's state can end in a call that takes the same lock, so
// it must run after the lock is released.
TEST(CPU_HipdnnEraseOutsideLock_NONE, DestroysTheValueAfterReleasingTheMutex)
{
    TrackingMutex mutex;
    Observation observation;
    Probe::Map map;
    map[1] = std::make_unique<Probe>(mutex, map, observation);

    miopen::wrapper::EraseDestroyingOutsideLock(map, mutex, 1);

    EXPECT_TRUE(observation.destroyed);
    EXPECT_FALSE(observation.heldWhenDestroyed);
    EXPECT_FALSE(mutex.held);
}

TEST(CPU_HipdnnEraseOutsideLock_NONE, TheEntryIsGoneBeforeTheValueIsDestroyed)
{
    TrackingMutex mutex;
    Observation observation;
    Observation other;
    Probe::Map map;
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
    Observation observation;
    Probe::Map map;
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
    Observation winner;
    Observation loser;
    Probe::Map map;
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
    Observation observation;
    Probe::Map map;

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

// A throwing allocation must not make the insert destroy the candidate under the lock. The
// failing allocation is walked across every allocation the insert makes, so the test does not
// depend on how the map grows.
TEST(CPU_HipdnnEraseOutsideLock_NONE, AnInsertThatThrowsNeverDestroysTheCandidateUnderTheLock)
{
    using FailingMap =
        std::unordered_map<int,
                           std::unique_ptr<LockProbe>,
                           std::hash<int>,
                           std::equal_to<int>,
                           FailingAllocator<std::pair<const int, std::unique_ptr<LockProbe>>>>;
    int threw = 0;
    for(int allocation = 1; allocation <= 4; ++allocation)
    {
        TrackingMutex mutex;
        Observation observation;
        FailingMap map;
        failIn = allocation;
        try
        {
            miopen::wrapper::InsertDestroyingLoserOutsideLock(
                map, mutex, 1, std::make_unique<LockProbe>(mutex, observation));
        }
        catch(const std::bad_alloc&)
        {
            ++threw;
        }
        failIn = 0;

        EXPECT_FALSE(mutex.held) << "allocation " << allocation;
        if(observation.destroyed)
            EXPECT_FALSE(observation.heldWhenDestroyed) << "allocation " << allocation;
    }
    EXPECT_GT(threw, 0);
}
