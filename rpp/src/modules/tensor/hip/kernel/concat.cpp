#include "hip_tensor_executors.hpp"

template <typename T>
__global__ void concat_generic_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint *srcTensor1Strides,
                                          uint *srcTensor2Strides,
                                          T *dstPtr,
                                          uint *dstStrides,
                                          uint axis,
                                          uint numDims,
                                          Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_x >= dstStrides[0])
        return;

    uint *roi = roiTensor + id_z * numDims * 2;
    uint *begin = roi;
    uint *length = &roi[numDims];
    uint dstIdx = (id_z * *dstStrides++);
    uint srcIdx1 = (id_z * *srcTensor1Strides++);
    uint srcIdx2 = (id_z * *srcTensor2Strides++);
    uint coords[RPPT_MAX_DIMS];

    uint temp = id_x;
    for (int i = 0; i < numDims; i++)
    {
        coords[i] = temp / dstStrides[i];
        temp %= dstStrides[i];
        if(i < axis)
        {
            dstIdx += coords[i] * dstStrides[i];
            srcIdx1 += (coords[i] + begin[i]) * srcTensor1Strides[i];
            srcIdx2 += (coords[i] + begin[i]) * srcTensor2Strides[i];
        }
        else if(i == axis)
        {
            if(coords[i] < length[i])
            {
                dstIdx += coords[i] * dstStrides[i];
                srcIdx1 += (coords[i] + begin[i]) * srcTensor1Strides[i];
            }
            else
            {
                uint shifted_coord = coords[i] - length[i];
                dstIdx += coords[i] * dstStrides[i];
                srcIdx2 += (shifted_coord + begin[i]) * srcTensor2Strides[i];
            }
        }
        else
        {
            dstIdx += coords[i] * dstStrides[i];
            srcIdx1 += coords[i] * srcTensor1Strides[i];
            srcIdx2 += coords[i] * srcTensor2Strides[i];
        }
    }

    // Write to output tensor
    if(coords[axis] < length[axis])
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
    else
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
}

template <typename T>
__global__ void concat_2d_hip_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     uint *srcTensor1Strides,
                                     uint *srcTensor2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *dims,
                                     Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(id_x >= dims[1] || id_y >= dims[0])
        return;

    uint dstIdx = id_y * dstStrides[1] + id_x;
    uint srcIdx1 = id_y * srcTensor1Strides[1] + id_x;
    uint srcIdx2 = id_y * srcTensor2Strides[1] + id_x;

    d_float8 src_f8, src2_f8;
    if((dims[1] - id_x) >= 8)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx + srcTensor1Strides[1], &src_f8);
    }
    else
    {
        for(int i = 0; i < (dims[1] - id_x); i++)
        {
            dstPtr[dstIdx + i] = srcPtr1[srcIdx1 + i];
            dstPtr[dstIdx + srcTensor1Strides[1] + i] = srcPtr2[srcIdx2 + i];
        }
    }
}

template <typename T>
__global__ void concat_3d_hip_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     uint *srcTensor1Strides,
                                     uint *srcTensor2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *dims,
                                     Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_x >= dims[2] || id_y >= dims[1] || id_z >= dims[0])
        return;

    uint dstIdx = id_z * dstStrides[1] + id_y * dstStrides[2] + id_x;
    uint srcIdx1 = id_z * srcTensor1Strides[1] + id_y * srcTensor1Strides[2] + id_x;
    uint srcIdx2 = id_z * srcTensor2Strides[1] + id_y * srcTensor2Strides[2] + id_x;

    d_float8 src_f8, src2_f8;
    if((dims[2] - id_x) >= 8)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx + srcTensor1Strides[2], &src_f8);
    }
    else
    {
        for(int i = 0; i < (dstStrides[2] - id_x); i++)
        {
            dstPtr[dstIdx + i] = srcPtr1[srcIdx1 + i];
            dstPtr[dstIdx + srcTensor1Strides[2] + i] = srcPtr2[srcIdx2 + i];
        }
    }
}

template <typename T>
RppStatus hip_exec_concat_tensor(T *srcPtr1,
                                 RpptGenericDescPtr srcPtr1GenericDescPtr,
                                 T *srcPtr2,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr,
                                 T *dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32u axis,
                                 Rpp32u *roiTensor,
                                 rpp::Handle& handle)
{
    int globalThreads_x = dstGenericDescPtr->strides[0];
    int globalThreads_y = 1;
    int globalThreads_z = dstGenericDescPtr->dims[0];

    int numDims = dstGenericDescPtr->numDims - 1;

    if(numDims == 2)
    {
        if(axis == 0)
        {
            srcPtr1GenericDescPtr->strides[1] = srcPtr1GenericDescPtr->strides[0];
            srcPtr1GenericDescPtr->strides[0] = 1;
            srcPtr2GenericDescPtr->strides[1] = srcPtr2GenericDescPtr->strides[0];
            srcPtr2GenericDescPtr->strides[0] = 1;
            dstGenericDescPtr->strides[1] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = 1;
        }
        else if(axis == 1)
        {
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[2];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[2];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[2];
        }
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32u *begin = roi;
            Rpp32u *length = &roi[numDims];
            Rpp32u *dims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
            if(axis == 0)
            {
                dims[0] = 1;
                dims[1] = length[0] * length[1];
            }
            else if(axis == 1)
            {
                dims[0] = length[0];
                dims[1] = length[1];
            }
            globalThreads_x = dims[1];
            globalThreads_y = dims[0];
            hipLaunchKernelGGL(concat_2d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * dims[0] * dims[1]),
                               srcPtr2 + (batchCount * dims[0] * dims[1]),
                               srcPtr1GenericDescPtr->strides,
                               srcPtr2GenericDescPtr->strides,
                               dstPtr + (batchCount * dims[0] * dims[1] * 2),
                               dstGenericDescPtr->strides,
                               dims,
                               roi);
        }
    }
    else if(numDims == 3)
    {
        if(axis == 0)
        {
            srcPtr1GenericDescPtr->strides[2] = srcPtr1GenericDescPtr->strides[0];
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[1] = 1;
            srcPtr2GenericDescPtr->strides[2] = srcPtr2GenericDescPtr->strides[0];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        else if(axis == 1)
        {
            srcPtr1GenericDescPtr->strides[2] = srcPtr1GenericDescPtr->strides[1];
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[1] = 1;
            srcPtr2GenericDescPtr->strides[2] = srcPtr2GenericDescPtr->strides[1];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[1];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32u *begin = roi;
            Rpp32u *length = &roi[numDims];
            Rpp32u *dims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
            if(axis == 0)
            {
                dims[0] = dims[1] = 1;
                dims[2] = length[0] * length[1] * length[2];
            }
            else if(axis == 1)
            {
                dims[0] = 1;
                dims[1] = length[0];
                dims[2] = length[1] * length[2];
            }
            else if(axis == 2)
            {
                dims[0] = length[0];
                dims[1] = length[1];
                dims[2] = length[2];
            }
            globalThreads_x = dims[2];
            globalThreads_y = dims[1];
            globalThreads_z = dims[0];
            hipLaunchKernelGGL(concat_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * dims[0] * dims[1] * dims[2]),
                               srcPtr2 + (batchCount * dims[0] * dims[1] * dims[2]),
                               srcPtr1GenericDescPtr->strides,
                               srcPtr2GenericDescPtr->strides,
                               dstPtr + (batchCount * dims[0] * dims[1] * dims[2] * 2),
                               dstGenericDescPtr->strides,
                               dims,
                               &roiTensor[batchCount * 6]);
        }
    }
    else
    {
        hipLaunchKernelGGL(concat_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       srcPtr1GenericDescPtr->strides,
                       srcPtr2GenericDescPtr->strides,
                       dstPtr,
                       dstGenericDescPtr->strides,
                       axis,
                       dstGenericDescPtr->numDims - 1,
                       roiTensor);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_concat_tensor<Rpp8u>(Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp32u,
                                                 Rpp32u*,
                                                 rpp::Handle&);

template RppStatus hip_exec_concat_tensor<half>(half*,
                                                RpptGenericDescPtr,
                                                half*,
                                                RpptGenericDescPtr,
                                                half*,
                                                RpptGenericDescPtr,
                                                Rpp32u,
                                                Rpp32u*,
                                                rpp::Handle&);

template RppStatus hip_exec_concat_tensor<Rpp32f>(Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32u,
                                                  Rpp32u*,
                                                  rpp::Handle&);

template RppStatus hip_exec_concat_tensor<Rpp8s>(Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp32u,
                                                 Rpp32u*,
                                                 rpp::Handle&);
