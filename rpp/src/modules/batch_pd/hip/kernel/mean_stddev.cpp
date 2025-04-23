#include <hip/hip_runtime.h>
extern "C" __global__ void sum (  const unsigned char* input,
                     long *partialSums)
{
    uint local_id = hipThreadIdx_x;
    uint group_size = hipBlockDim_x;
    __shared__ int localSums[256];

    localSums[local_id] = (long)input[hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x];

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        __syncthreads();
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }
    if (local_id == 0)
    partialSums[hipBlockIdx_x] = localSums[0];
}

extern "C" __global__ void mean_stddev (  const unsigned char* input,
                     float *partial_mean_sum,
                    const float mean)
{
    uint local_id = hipThreadIdx_x;
    uint group_size = hipBlockDim_x;
    __shared__ float localSums[256];

    localSums[local_id] = (float)input[hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x];

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            __syncthreads();
            if(stride == group_size/2)
                localSums[local_id] = sqrt((localSums[local_id] - mean) * (localSums[local_id] - mean)) + sqrt((localSums[local_id + stride] - mean ) * (localSums[local_id + stride] - mean));
            else
                localSums[local_id] += localSums[local_id + stride];
        }
        __syncthreads();
    }
    if (local_id == 0)
    partial_mean_sum[hipBlockIdx_x] = localSums[0];
}