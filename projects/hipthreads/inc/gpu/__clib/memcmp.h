#ifndef __GPU___CLIB_MEMCMP_H__
#define __GPU___CLIB_MEMCMP_H__

#include "hip/hip_runtime_api.h"
#include <cstddef>

namespace gpu {

inline __host__ __device__ int memcmp(const void *lhs, const void *rhs, std::size_t count) {
    for (const unsigned char *l = reinterpret_cast<const unsigned char *>(lhs),
                             *r = reinterpret_cast<const unsigned char *>(rhs);
         count > 0; ++l, ++r, --count) {
        if (*l != *r) {
            return (*l - *r);
        }
    }
    return 0;
}

}

#endif // __GPU___CLIB_MEMCMP_H__
