// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " file: " << __FILE__ \
                      << " line: " << __LINE__ << std::endl;                                \
            exit(error);                                                                    \
        }                                                                                   \
    }

ROCPRIM_HOST
inline rocprim::native_half half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const rocprim::native_half*>(&x);
}

struct half_less
{
    ROCPRIM_HOST_DEVICE
    inline bool
        operator()(const rocprim::half& a, const rocprim::half& b) const
    {
#if __HIP_DEVICE_COMPILE__
        return a < b;
#else
        return half_to_native(a) < half_to_native(b);
#endif
    }
};
