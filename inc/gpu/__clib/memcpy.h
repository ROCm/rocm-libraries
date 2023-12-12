#ifndef __GPU___CLIB_MEMCPY_H__
#define __GPU___CLIB_MEMCPY_H__

#include "hip/hip_runtime_api.h"
#include <cstddef>

namespace gpu {

inline __host__ __device__ void *memcpy(void *dest, const void *src, std::size_t count) {
    unsigned char *d = reinterpret_cast<unsigned char *>(dest);
    const unsigned char *s = reinterpret_cast<const unsigned char *>(src);
    for (; count > 0; ++d, ++s, --count) {
        *d = *s;
    }
    return dest;
}

// TODO: Should we provide a version that enables host-to-device, device-to-host, etc.?
}

#endif // __GPU___CLIB_MEMCPY_H__
