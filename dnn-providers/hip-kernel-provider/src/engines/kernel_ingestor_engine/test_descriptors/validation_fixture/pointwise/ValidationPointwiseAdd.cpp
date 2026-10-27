// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The three kernels of the validation fixture, selected by build defines:
//
//   HIP_PLUGIN_VALIDATION_WRONG=0  c[0] = a[0] + b[0], the correct result.
//   HIP_PLUGIN_VALIDATION_WRONG=1  c[0] = a[0] - b[0], a deliberately wrong result.
//   HIP_PLUGIN_SPIN_CYCLES         iterations of a busy loop before the store. The correct
//                                  kernels spin so that the wrong kernel, which does not,
//                                  is reliably the fastest candidate when benchmarking
//                                  ranks by time alone.
//
// HIP_PLUGIN_POINTWISE_BLOCK_SIZE is unused here but must still reach the compiler: it is
// what makes the block-size variants distinct (source, build) inputs.
//
// The loop counter is volatile, so the compiler cannot remove the loop; it uses no
// architecture builtin and compiles for every target the fixture packs for. The include
// is explicit because the packager drives clang directly with `-x hip` and no runtime
// preamble.

#include <hip/hip_runtime.h>

extern "C" __global__ void ValidationPointwiseAdd(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                                  const HIP_PLUGIN_POINTWISE_TYPE* b,
                                                  HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        volatile unsigned int counter = 0;
        for(unsigned int i = 0; i < HIP_PLUGIN_SPIN_CYCLES; ++i)
        {
            counter = counter + 1;
        }

#if HIP_PLUGIN_VALIDATION_WRONG
        c[0] = a[0] - b[0];
#else
        c[0] = a[0] + b[0];
#endif
    }
}
