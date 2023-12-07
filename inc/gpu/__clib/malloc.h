#ifndef __GPU___MEMORY_MALLOC_H__
#define __GPU___MEMORY_MALLOC_H__

#include "hip/hip_runtime_api.h"
#include <cstddef>

#include "gpu/__support/hip_check.h"

namespace gpu {

namespace internal {
// NOTE: NOT STATIC so that there is only one copy of the static local variable inside!
inline hipStream_t &getEnqueingStream() {
    // TODO: investigate using hipExtStreamCreateWithCUMask for this
    static hipStream_t enqueingStream = []() -> hipStream_t {
        hipStream_t s;
        __LIBGPU_HIP_CHECK__(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        return s;
    }();
    return enqueingStream;
}
}

inline __host__ void *malloc(std::size_t size) {
    void *ptr;
    __LIBGPU_HIP_CHECK__(hipMallocAsync(&ptr, size, internal::getEnqueingStream()));
    __LIBGPU_HIP_CHECK__(hipStreamSynchronize(internal::getEnqueingStream()));
    return ptr;
}

inline __host__ void free(void* ptr) {
    __LIBGPU_HIP_CHECK__(hipFreeAsync(ptr, internal::getEnqueingStream()));
}

} // namespace gpu


#endif // __GPU___MEMORY_MALLOC_H__
