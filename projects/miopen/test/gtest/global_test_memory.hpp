// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_GLOBAL_TEST_MEMORY_HPP
#define GUARD_MIOPEN_GLOBAL_TEST_MEMORY_HPP

#include <cstdint>

class bfloat16;

namespace half_float {
class half;
} // namespace half_float

using hipStream_t = struct ihipStream_t*;

namespace test::gtest::global_buffer {

/**
 * @brief Lightweight view over a subrange of a shared global device allocation used by tests.
 *
 * Instances are returned by @ref GetDeviceBuffer and point into an internal, process-wide
 * device buffer that is grown on demand. The underlying allocation is owned elsewhere;
 * this struct is a non-owning handle.
 *
 * Typical usage:
 * - call @ref EnsureBufferSize for the total bytes needed by a test,
 * - call @ref GetDeviceBuffer repeatedly to carve out chunks,
 * - call @ref ReleaseBuffer between tests to reset the internal offset.
 *
 * @note No synchronization is performed; intended for single-threaded test allocation patterns.
 */
struct DeviceBufferObject
{
    /// Base device pointer of this chunk (non-owning).
    void* ptr_ = nullptr;
    /// Size of this chunk in bytes.
    std::size_t size_bytes_ = 0;
    /// Host mirror pointer for this chunk (non-owning).
    void* host_mirror_ = nullptr;

    hipStream_t stream_ = nullptr;

    /**
     * @brief Fill the chunk with a scalar value.
     *
     * Uses a fast memset path when @p val is zero; otherwise launches a small fill kernel.
     *
     * @tparam T Element type to interpret the buffer as.
     * @param val Value to write to every element.
     * @return Copy of *this for chaining.
     */
    template <typename T>
    DeviceBufferObject fill(T val) const;

    /**
     * @brief Fill the chunk with deterministic pseudo-random values.
     *
     * Generates element i from (seed, i) so results are independent of launch configuration.
     *
     * @tparam T Element type to interpret the buffer as.
     * @param seed Seed used for deterministic generation.
     * @return Copy of *this for chaining.
     */
    template <typename T>
    DeviceBufferObject randomize(unsigned long long seed = 12345678) const;

    void HostMirror() const;

    void HostMirrorAsync() const;
    bool HostMirrorReady() const;
    void HostMirrorWait() const;
};

/**
 * @brief Ensure the shared global device buffer is at least @p size_in_bytes.
 *
 * If the requested size exceeds the current capacity, the internal allocation is freed and
 * replaced with a larger one.
 */
void EnsureBufferSize(std::size_t size_in_bytes);

/**
 * @brief Allocate a chunk of @p size_in_bytes from the shared global device buffer.
 *
 * Returns a non-owning view pointing into the current buffer and advances the internal offset.
 * Call @ref ReleaseBuffer to reset the offset between tests.
 */
DeviceBufferObject GetDeviceBuffer(std::size_t size_in_bytes);

/**
 * @brief Reset the internal allocation cursor to the beginning of the global buffer.
 *
 * Does not free device memory; only resets the chunk allocator offset.
 */
void ReleaseBuffer();

void FreeBuffer();

// Explicit instantiations provided in the corresponding .cpp to keep templates out of headers.
extern template DeviceBufferObject DeviceBufferObject::fill<double>(double) const;
extern template DeviceBufferObject DeviceBufferObject::randomize<double>(unsigned long long) const;

extern template DeviceBufferObject DeviceBufferObject::fill<float>(float) const;
extern template DeviceBufferObject DeviceBufferObject::randomize<float>(unsigned long long) const;

extern template DeviceBufferObject DeviceBufferObject::fill<int8_t>(int8_t) const;
extern template DeviceBufferObject DeviceBufferObject::randomize<int8_t>(unsigned long long) const;

extern template DeviceBufferObject DeviceBufferObject::fill<int32_t>(int32_t) const;
extern template DeviceBufferObject DeviceBufferObject::randomize<int32_t>(unsigned long long) const;

extern template DeviceBufferObject DeviceBufferObject::fill<bfloat16>(bfloat16) const;
extern template DeviceBufferObject
DeviceBufferObject::randomize<bfloat16>(unsigned long long) const;

extern template DeviceBufferObject
    DeviceBufferObject::fill<half_float::half>(half_float::half) const;
extern template DeviceBufferObject
DeviceBufferObject::randomize<half_float::half>(unsigned long long) const;

} // namespace test::gtest::global_buffer

#endif // GUARD_MIOPEN_GLOBAL_TEST_MEMORY_HPP
