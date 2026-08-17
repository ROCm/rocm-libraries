#ifndef __HIP_HELPERS_H__
#define __HIP_HELPERS_H__

#include "hip/hip_runtime_api.h"
#include <iostream>

#define __HIP_CHECK__(cmd)                                                                         \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#endif // __HIP_HELPERS_H__
