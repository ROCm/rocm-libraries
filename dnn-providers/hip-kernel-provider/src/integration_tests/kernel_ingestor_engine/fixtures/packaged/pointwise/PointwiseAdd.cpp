// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Elementwise add over a single-element tensor, compiled ahead of time by the packager
// rather than at plan-build time. Byte-identical in behaviour to the embedded_source
// kernel the shipped pointwise engine compiles, so the packaged engine's numeric result
// is comparable against the same CPU reference.
//
// HIP_PLUGIN_POINTWISE_BLOCK_SIZE is unused here but must still reach the compiler: it is
// what makes two block-size variants distinct (source, build) inputs, and therefore two
// distinct blobs under two toc_keys in the archive.
//
// The include is explicit, unlike the shipped kernel's: CMake's HIP language force-includes
// the runtime header, but the packager drives clang directly with `-x hip` and no such
// preamble, so blockIdx and threadIdx would otherwise be undeclared.
//
// PointwiseAddSecondSymbol is named by no descriptor and launched by nothing. It is here
// because a toc_key identifies a (source, build) pair, so the only way two symbols can
// share one -- and therefore one hipModule_t -- is to sit in one translation unit. It is
// the subject of TestKpackKernelLoader.TwoSymbolsResolveAgainstOneModule, which resolves
// both entry points against a single toc_key and asserts they came from the same module.
// Deleting it as dead code deletes that test's only evidence.

#include <hip/hip_runtime.h>

extern "C" __global__ void PointwiseAdd(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                        const HIP_PLUGIN_POINTWISE_TYPE* b,
                                        HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}

extern "C" __global__ void PointwiseAddSecondSymbol(const HIP_PLUGIN_POINTWISE_TYPE* a,
                                                    const HIP_PLUGIN_POINTWISE_TYPE* b,
                                                    HIP_PLUGIN_POINTWISE_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
