// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <mutex>
#include <utility>

namespace miopen {
namespace wrapper {

// Removes `key` from `map` under `mutex` and destroys the value only after releasing it. A
// value's destructor can take `mutex` again: a handle's hipDNN state ends in the plugin's
// handle destructor, which for a plugin built against the public libMIOpen.so calls
// miopenDestroy, which looks the handle up in the same map. Destroying under the lock would
// deadlock.
template <typename Map, typename Mutex, typename Key>
void EraseDestroyingOutsideLock(Map& map, Mutex& mutex, const Key& key)
{
    typename Map::mapped_type doomed;
    {
        const std::lock_guard<Mutex> lock(mutex);
        const auto found = map.find(key);
        if(found == map.end())
            return;
        doomed = std::move(found->second);
        map.erase(found);
    }
}

} // namespace wrapper
} // namespace miopen
