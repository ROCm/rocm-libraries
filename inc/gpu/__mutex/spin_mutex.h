#ifndef __GPU___MUTEX_SPIN_MUTEX_H__
#define __GPU___MUTEX_SPIN_MUTEX_H__

#include "gpu/__config"

#include "hip/hip_runtime.h"
#include <cassert>
#include <cstdint>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::mutex
//====================================================================================================================//

// TODO: add a scope paramter
class _LIBGPU_TYPE_VIS spin_mutex {
    // HIP requires that (gridDim * blockDim) < 2^32
    enum : uint64_t { INVALID_OWNER = -1ULL };
    uint64_t owner = INVALID_OWNER; // stores the owner's blockId

  public:
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR spin_mutex() = default;

    __device__ spin_mutex(const spin_mutex &) = delete;
    __device__ spin_mutex &operator=(const spin_mutex &) = delete;
    __device__ _LIBGPU_HIDE_FROM_ABI ~spin_mutex() = default;

    __device__ void lock() {
        const uint64_t myBlockId = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;
        // The read operation here by atomicCAS counts as an atomic acquire operation, which the 'release fence'
        // operation in unlock() synchronizes-with.
        for (uint64_t ownerBlockId = atomicCAS(&owner, INVALID_OWNER, myBlockId); ownerBlockId != INVALID_OWNER;
             ownerBlockId = atomicCAS(&owner, INVALID_OWNER, myBlockId)) {
            // Since execution only continues past the loop once ALL threads in a wave have exited the loop,
            // we need to prevent an entire wave from spinning waiting for one of it's own threads to release the lock.
            //
            // Technically this is more strict than we need to be, because we're doing this at a block level and not a
            // wave level, but that's fine.
            assert(ownerBlockId != myBlockId && "Deadlock detected: Tried to acquire a lock already owned by someone in the same block");
        }
    }
    /** Note that prior calls to lock() do NOT synchronize-with try_lock if it returns false! In otherwords, there is
     * no memory order relationship between lock and try_lock or between try_lock and itself. Only unlock() establishes
     * any memory order relationship. (see https://en.cppreference.com/w/cpp/thread/mutex/try_lock)
     */
    __device__ bool try_lock() _NOEXCEPT {
        const uint64_t myBlockId = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;
        // The read operation here by atomicCAS counts as an atomic acquire operation, which the 'release fence'
        // operation in unlock() synchronizes-with.
        return atomicCAS(&owner, INVALID_OWNER, myBlockId) == INVALID_OWNER;
    }
    __device__ void unlock() _NOEXCEPT {
        // Create a 'release fence' operation which synchronizes-with the atomic acquire operation in lock() and
        // try_lock(). This establishes the synchronizes-with relationship required by the C++ standard:
        //  - Unlock "synchronizes-with any subsequent lock operation that obtains ownership of the same mutex"
        //    (see https://en.cppreference.com/w/cpp/thread/mutex/unlock)
        // In other words, we make sure all writes in the critical section are visible to other threads when they
        // acquire the mutex.
        __threadfence();
        // This counts as the atomic store operation 'X' described in "Fence-atomic synchronization" at
        // https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence.
        [[maybe_unused]] uint64_t oldOwner = atomicExch(&owner, INVALID_OWNER);
        assert(oldOwner == blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z);
    }
};

} // namespace gpu

#endif // __GPU___MUTEX_SPIN_MUTEX_H__
