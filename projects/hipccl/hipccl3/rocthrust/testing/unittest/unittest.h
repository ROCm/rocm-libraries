#pragma once

// this is the only header included by unittests
// it pulls in all the others used for unittesting

#include <unittest/assertions.h>
#include <unittest/meta.h>
#include <unittest/random.h>
#include <unittest/testframework.h>
#include <unittest/special_types.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
    #define THRUST_DEVICE_BACKEND hip
    #define THRUST_DEVICE_BACKEND_DETAIL hip_rocprim
    #define SPECIALIZE_DEVICE_RESOURCE_NAME(name) hip##name
#elif THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
    #define THRUST_DEVICE_BACKEND cuda
    #define THRUST_DEVICE_BACKEND_DETAIL cuda_cub
    #define SPECIALIZE_DEVICE_RESOURCE_NAME(name) cuda##name
#endif

