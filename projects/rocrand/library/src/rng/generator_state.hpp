// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/// \file
/// \brief Generator position kept in memory of the system.
///
/// Generators that would otherwise keep their position (a counter or an offset) on the host
/// and pass it to the generation kernel by value keep it in a `generator_state` instead.
/// Every block of the generation kernel reads the position at its start, and the last block
/// to finish advances it (host generators advance it in a separate single-thread kernel).
/// When the generate call is captured into a HIP graph, every launch of the graph therefore
/// continues from where the previous launch (or eager call) stopped, like generators whose
/// engines live in device memory.

#ifndef ROCRAND_RNG_GENERATOR_STATE_H_
#define ROCRAND_RNG_GENERATOR_STATE_H_

#include "config_types.hpp"
#include "system.hpp"

#include <rocrand/rocrand.h>

#include <hip/hip_runtime.h>

#include <utility>

namespace rocrand_impl::host
{

template<class State>
struct generator_state_storage
{
    State state;
    // Number of blocks of the running generation kernel that have finished reading the state.
    unsigned int finished_blocks;
};

/// \brief Kernel argument through which a generation kernel reads and advances the state.
template<class State, class Update>
struct generator_state_access
{
    generator_state_storage<State>* storage;
    Update                          update;

    __host__ __device__ __forceinline__
    State load() const
    {
        return storage->state;
    }

    /// \brief Must be called by every thread of the generation kernel after it has finished
    /// using the state. The last block to call it applies \p update to the state.
    ///
    /// Host generators run blocks in parallel on the host, they apply the update in a separate
    /// kernel instead (see `generator_state::update`).
    __host__ __device__ __forceinline__
    void advance([[maybe_unused]] dim3 thread_idx, [[maybe_unused]] dim3 grid_dim) const
    {
#ifdef __HIP_DEVICE_COMPILE__
        // Every thread of the block has used its copy of the state.
        __syncthreads();
        if(thread_idx.x == 0 && thread_idx.y == 0 && thread_idx.z == 0)
        {
            const unsigned int blocks = grid_dim.x * grid_dim.y * grid_dim.z;
            __threadfence();
            if(atomicAdd(&storage->finished_blocks, 1u) == blocks - 1)
            {
                // Every block has read the state, the next kernel will read the updated one.
                __threadfence();
                update(storage->state);
                storage->finished_blocks = 0;
            }
        }
#endif
    }
};

template<class State>
struct store_generator_state
{
    template<host::target_arch Arch = host::target_arch::unknown>
    __host__ __device__ __forceinline__
    static void generate(dim3 /*block_idx*/,
                         dim3 /*thread_idx*/,
                         dim3 /*grid_dim*/,
                         dim3 /*block_dim*/,
                         generator_state_storage<State>* storage,
                         State                           value)
    {
        storage->state           = value;
        storage->finished_blocks = 0;
    }
};

template<class State, class Update>
struct update_generator_state
{
    template<host::target_arch Arch = host::target_arch::unknown>
    __host__ __device__ __forceinline__
    static void generate(dim3 /*block_idx*/,
                         dim3 /*thread_idx*/,
                         dim3 /*grid_dim*/,
                         dim3 /*block_dim*/,
                         generator_state_storage<State>* storage,
                         Update                          update)
    {
        update(storage->state);
    }
};

/// \brief Update that discards \p count values of an engine.
struct discard_engine
{
    unsigned long long count;

    template<class Engine>
    __host__ __device__ __forceinline__
    void operator()(Engine& engine) const
    {
        engine.discard(count);
    }
};

/// \brief Update that advances an offset by \p count.
struct advance_offset
{
    unsigned long long count;

    __host__ __device__ __forceinline__
    void operator()(unsigned long long& offset) const
    {
        offset += count;
    }
};

/// \brief Returns whether \p stream is being captured into a HIP graph.
inline rocrand_status is_stream_capturing(hipStream_t stream, bool& capturing)
{
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &capture_status) != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    capturing = capture_status != hipStreamCaptureStatusNone;
    return ROCRAND_STATUS_SUCCESS;
}

/// \brief Allocates \p new_count elements for engines, reusing \p ptr if it already holds
/// \p old_count elements and \p old_count equals \p new_count.
///
/// Reusing the memory keeps the pointer held by graphs that captured an earlier generate call
/// valid; such a graph then continues from the re-initialized engines. Freeing memory
/// synchronizes the device, so outside stream capture the device is synchronized before
/// reusing it too, as work that may still use the old engines must finish first.
template<class System, class T>
rocrand_status
    reuse_or_allocate(hipStream_t stream, T*& ptr, const size_t old_count, const size_t new_count)
{
    if(ptr != nullptr && old_count == new_count)
    {
        bool                 capturing;
        const rocrand_status status = is_stream_capturing(stream, capturing);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        if(!capturing && hipDeviceSynchronize() != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        return ROCRAND_STATUS_SUCCESS;
    }
    if(ptr != nullptr)
    {
        System::free(ptr);
        ptr = nullptr;
    }
    return System::alloc(&ptr, new_count);
}

/// \brief Owns one \p State in memory of \p System (device memory for device generators).
template<class System, class State>
class generator_state
{
public:
    generator_state() = default;

    generator_state(const generator_state&) = delete;

    generator_state(generator_state&& other) noexcept
        : m_state(std::exchange(other.m_state, nullptr))
    {}

    generator_state& operator=(const generator_state&) = delete;

    generator_state& operator=(generator_state&& other) noexcept
    {
        if(this != &other)
        {
            free();
            m_state = std::exchange(other.m_state, nullptr);
        }
        return *this;
    }

    ~generator_state()
    {
        free();
    }

    /// \brief Sets the state to \p value, allocating it on first use.
    ///
    /// The memory is allocated once and reused, so a graph that captured an earlier generate
    /// call keeps a valid pointer and continues from the new state. Outside stream capture,
    /// waits for work that may still read the old state, and for the new state to be stored,
    /// so the generator may be used on another stream afterwards.
    rocrand_status store(hipStream_t stream, const State& value)
    {
        if(m_state == nullptr)
        {
            const rocrand_status status = System::alloc(&m_state, 1);
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                m_state = nullptr;
                return status;
            }
        }

        host::target_arch target_arch;
        if(host::get_device_arch(stream, target_arch) != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        bool           capturing;
        rocrand_status status = is_stream_capturing(stream, capturing);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        if(!capturing && hipDeviceSynchronize() != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        status = System::template launch<store_generator_state<State>>(target_arch,
                                                                       dim3(1),
                                                                       dim3(1),
                                                                       0,
                                                                       stream,
                                                                       m_state,
                                                                       value);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        if(!capturing && hipStreamSynchronize(stream) != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    /// \brief Returns the kernel argument through which the generation kernel reads the state
    /// and applies \p update to it.
    template<class Update>
    generator_state_access<State, Update> access(Update update) const
    {
        return generator_state_access<State, Update>{m_state, update};
    }

    /// \brief Applies \p update to the state after the generation kernel, for host generators.
    /// Device generation kernels apply it themselves (see `generator_state_access::advance`).
    template<class Update>
    rocrand_status update(host::target_arch target_arch, hipStream_t stream, Update update)
    {
        if constexpr(System::is_device())
        {
            (void)target_arch;
            (void)stream;
            (void)update;
            return ROCRAND_STATUS_SUCCESS;
        }
        else
        {
            return System::template launch<update_generator_state<State, Update>>(target_arch,
                                                                                  dim3(1),
                                                                                  dim3(1),
                                                                                  0,
                                                                                  stream,
                                                                                  m_state,
                                                                                  update);
        }
    }

private:
    void free()
    {
        if(m_state != nullptr)
        {
            System::free(m_state);
            m_state = nullptr;
        }
    }

    generator_state_storage<State>* m_state = nullptr;
};

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_GENERATOR_STATE_H_
