// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The kernel behind the pointwise-add descriptor pack. Deliberately the
// narrowest useful operation: elementwise add over a single-element tensor.
//
// HIP_PLUGIN_POINTWISE_ADD_TYPE is the element type, supplied per compilation from the
// kernel descriptor's dtype metadata. HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE is the block
// size the descriptor selected; the kernel does not read it, because one element needs
// one thread. It exists so the ingestor's ranking, knob reporting, and per-kernel
// dispatch have a metadata field that genuinely varies between kernels and genuinely
// reaches the compiler, rather than a field that only tests observe.

extern "C" __global__ void PointwiseAdd(const HIP_PLUGIN_POINTWISE_ADD_TYPE* a,
                                        const HIP_PLUGIN_POINTWISE_ADD_TYPE* b,
                                        HIP_PLUGIN_POINTWISE_ADD_TYPE* c)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        c[0] = a[0] + b[0];
    }
}
