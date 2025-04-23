#include <hip/hip_runtime.h>
extern "C" __global__ void min_loc (  const unsigned char* input,
                     unsigned char *partial_min,
                     unsigned char *partial_min_location)
{
    uint local_id = hipThreadIdx_x;
    uint group_size = hipBlockDim_x;
    __shared__ unsigned char localMins[256];
    __shared__ unsigned char localMinsLocation[256];

    localMins[local_id] = input[hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x];
    localMinsLocation[local_id] = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            __syncthreads();
            if(localMins[local_id] > localMins[local_id + stride])
            {
                localMins[local_id] = localMins[local_id + stride];
                localMinsLocation[local_id] = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x + stride;
            }
        }
        __syncthreads();
    }
    if (local_id == 0)
    {
        partial_min[hipBlockIdx_x] = localMins[0];
        partial_min_location[hipBlockIdx_x] = localMinsLocation[0];
    }
}                        
extern "C" __global__ void max_loc (  const unsigned char* input,
                     unsigned char *partial_max,
                     unsigned char *partial_max_location)
{
    uint local_id = hipThreadIdx_x;
    uint group_size = hipBlockDim_x;
    __shared__ unsigned char localmaxs[256];
    __shared__ unsigned char localmaxsLocation[256];

    localmaxs[local_id] = input[hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x];
    localmaxsLocation[local_id] = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            __syncthreads();
            if(localmaxs[local_id] > localmaxs[local_id + stride])
            {
                localmaxs[local_id] = localmaxs[local_id + stride];
                localmaxsLocation[local_id] = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x + stride;
            }
        }
        __syncthreads();
    }
    if (local_id == 0)
    {
        partial_max[hipBlockIdx_x] = localmaxs[0];
        partial_max_location[hipBlockIdx_x] = localmaxsLocation[0];
    }
}    