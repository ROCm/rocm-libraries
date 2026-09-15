// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Minimal bundle source for the hiprtc_file config fixtures. Its body is
// irrelevant: no fixture compiles it. What matters is that it EXISTS, because
// staging is a file copy and the emitter refuses a descriptor naming a file
// its bundle does not hold.

extern "C" __global__ void FixtureKernel(float* y, unsigned int count)
{
    const unsigned int index = blockIdx.x * FIXTURE_BLOCK_SIZE + threadIdx.x;
    if(index < count)
    {
        y[index] = FIXTURE_DTYPE(0);
    }
}
