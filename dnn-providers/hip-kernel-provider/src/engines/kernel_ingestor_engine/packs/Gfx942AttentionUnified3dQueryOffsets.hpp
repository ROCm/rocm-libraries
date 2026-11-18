// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @brief A grow-only host buffer holding `[0, 1, ..., capacity - 1]` as int32, served as a
 *        prefix: any `count <= capacity` reads the first `count` entries.
 *
 * Source: the strided entry of `run_unified_attention_torch` takes caller-owned int32
 * `cu_seqlens_q[B+1] = [0, 1, ..., B]`; a dense hipDNN graph has no such tensor, so the
 * engine copies a prefix of this buffer into its workspace on every launch
 * (hipMemcpyAsync, handle stream).
 *
 * Lifetime: blocks are NEVER freed and NEVER written after they are published. Growing
 * allocates a larger block, fills it, then publishes it; older blocks stay valid (kept in
 * @c _blocks), so an asynchronous copy still reading one is unaffected. No destructor
 * synchronization is needed for the same reason. Memory held is bounded by roughly twice
 * the largest batch ever served.
 *
 * Thread safety: acquire() is guarded by a mutex; the returned pointer may be read by any
 * thread or stream afterwards.
 *
 * The allocator is injected so the grow logic is testable without HIP; the engine passes
 * hipHostMalloc (pinned, so hipMemcpyAsync does not stall the host).
 */
class Gfx942GrowOnlyArange
{
public:
    /// Returns @p bytes of host memory, or nullptr on failure. Never freed.
    using Allocate = void* (*)(size_t bytes);

    /// Smallest block allocated, in entries (B up to 255).
    static constexpr size_t MIN_CAPACITY = 256;

    explicit Gfx942GrowOnlyArange(Allocate allocate)
        : _allocate(allocate)
    {
    }

    Gfx942GrowOnlyArange(const Gfx942GrowOnlyArange&) = delete;
    Gfx942GrowOnlyArange& operator=(const Gfx942GrowOnlyArange&) = delete;

    /// A pointer whose first @p count entries are `0 .. count-1`. Valid forever.
    const int32_t* acquire(size_t count)
    {
        if(count == 0 || count > static_cast<size_t>(INT32_MAX))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 unified 3d: cu_seqlens_q needs 1..INT32_MAX entries");
        }
        const std::lock_guard<std::mutex> lock(_mutex);
        if(count > _capacity)
        {
            const size_t capacity = std::min(std::max({count, 2 * _capacity, MIN_CAPACITY}),
                                             static_cast<size_t>(INT32_MAX));
            auto* block = static_cast<int32_t*>(_allocate(capacity * sizeof(int32_t)));
            if(block == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_ALLOC_FAILED,
                    "gfx942 unified 3d: cannot allocate pinned host memory for cu_seqlens_q");
            }
            for(size_t i = 0; i < capacity; ++i)
            {
                block[i] = static_cast<int32_t>(i);
            }
            _blocks.push_back(block);
            _capacity = capacity;
        }
        return _blocks.back();
    }

    size_t capacity() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _capacity;
    }

    /// Every block ever allocated, oldest first (all still valid).
    std::vector<const int32_t*> blocks() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return {_blocks.begin(), _blocks.end()};
    }

private:
    Allocate _allocate;
    mutable std::mutex _mutex;
    size_t _capacity = 0;
    std::vector<int32_t*> _blocks;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine
