#pragma once

#include <hip/hip_runtime.h>

struct LaunchParams
{
    dim3 grid;
    dim3 block_size;
    size_t dynamic_shared_bytes = 0;
};
