// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <mutex>
#include <utility>

namespace miopen {
namespace wrapper {

// Stores `candidate` under `key` unless the key is taken, and returns whether it was stored.
// A losing candidate is destroyed after the lock is released, because its destructor can take
// `mutex` again (see erase_outside_lock.hpp).
//
// The key goes in with an empty value and the candidate is moved in afterwards. With
// `try_emplace(key, std::move(candidate))`, a throw while growing the bucket array would destroy
// the candidate under the lock.
template <typename Map, typename Mutex, typename Key>
bool InsertDestroyingLoserOutsideLock(Map& map,
                                      Mutex& mutex,
                                      const Key& key,
                                      typename Map::mapped_type candidate)
{
    typename Map::mapped_type loser;
    {
        const std::lock_guard<Mutex> lock(mutex);
        const auto result = map.try_emplace(key);
        if(result.second)
        {
            result.first->second = std::move(candidate);
            return true;
        }
        loser = std::move(candidate);
    }
    return false;
}

} // namespace wrapper
} // namespace miopen
