/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_tensor_executors.hpp"

// Vectorized dst->src mapping
template <typename T>
__global__ void transpose_generic_hip_tensor(T *srcPtr,
                                             uint *srcStrides,
                                             T *dstPtr,
                                             uint *dstStrides,
                                             uint *dstDims,
                                             uint tensorDims,
                                             uint *permTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(id_x >= dstStrides[0])
        return;

    int maxLength = dstStrides[0];
    int xDiff = maxLength - (maxLength & ~7);    // difference between maxLength and alignedLength. (alignedLength = maxLength & ~7)

    // Point dstIdx and srcIdx to be at the start of given input tensor in batch
    uint dstIdx = (id_y * *dstStrides++);        // post-increment dstStrides pointer by 1 to exclude outermost batch-dimension stride (for example exclude nStride in an NCDHW tensor)
    uint srcIdx = (id_y * *srcStrides++);        // post-increment srcStrides pointer by 1 to exclude outermost batch-dimension stride (for example exclude nStride in an NCDHW tensor)

    d_uint8 dstCoords[RPPT_MAX_DIMS], srcIdxs;
    uint4 idx0123 = make_uint4(id_x, id_x + 1, id_x + 2, id_x + 3);                  // get idx for elements 0, 1, 2, 3 in the 8-element vectorized kernel
    uint4 idx4567 = make_uint4(id_x + 4, id_x + 5, id_x + 6, id_x + 7);              // get idx for elements 4, 5, 6, 7 in the 8-element vectorized kernel
    srcIdxs.ui4[0] = srcIdxs.ui4[1] = make_uint4(srcIdx, srcIdx, srcIdx, srcIdx);    // create 8-element vectorized srcIdxs

    // Compute 8 dstCoords given idx0123 and idx4567, corresponding to the 8 srcCoords processed in a thread
    for (int i = 0; i < tensorDims; i++)
    {
        dstCoords[i].ui4[0] = (idx0123 / dstStrides[i]) % dstDims[i];                // transpose 4 srcCoords using idx0123 to 4 dstCoords in dstCoords[i].ui4[0] for the ith tensor dimension
        dstCoords[i].ui4[1] = (idx4567 / dstStrides[i]) % dstDims[i];                // transpose 4 srcCoords using idx4567 to 4 dstCoords in dstCoords[i].ui4[1] for the ith tensor dimension
    }

    // Compute corresponding 8 srcIdxs given id_x
    for (int i = 0; i < tensorDims; i++)
    {
        uint4 srcStrides_ui4 = static_cast<uint4>(srcStrides[permTensor[permTensor[i]]]);
        srcIdxs.ui4[0] += (dstCoords[permTensor[i]].ui4[0] * srcStrides_ui4);        // incrementally adding respective (coordinate value * stride) to get srcIdxs for 0, 1, 2, 3 elements
        srcIdxs.ui4[1] += (dstCoords[permTensor[i]].ui4[1] * srcStrides_ui4);        // incrementally adding respective (coordinate value * stride) to get srcIdxs for 4, 5, 6, 7 elements
        dstIdx += (dstCoords[i].ui1[0] * dstStrides[i]);
    }

    // Move srcIdx to access next input tensor once id_x goes beyond present tensor
    if((id_x + 8) > maxLength)
        for(int i = xDiff; i < 8; i++)
            srcIdxs.ui1[i] += maxLength;

    // Load corresponding 8 src pixels from computed src idx values
    d_float8 dst_f8;
    dst_f8.f1[0] = static_cast<float>(srcPtr[srcIdxs.ui1[0]]);
    dst_f8.f1[1] = static_cast<float>(srcPtr[srcIdxs.ui1[1]]);
    dst_f8.f1[2] = static_cast<float>(srcPtr[srcIdxs.ui1[2]]);
    dst_f8.f1[3] = static_cast<float>(srcPtr[srcIdxs.ui1[3]]);
    dst_f8.f1[4] = static_cast<float>(srcPtr[srcIdxs.ui1[4]]);
    dst_f8.f1[5] = static_cast<float>(srcPtr[srcIdxs.ui1[5]]);
    dst_f8.f1[6] = static_cast<float>(srcPtr[srcIdxs.ui1[6]]);
    dst_f8.f1[7] = static_cast<float>(srcPtr[srcIdxs.ui1[7]]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
RppStatus hip_exec_transpose_tensor(T *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    T *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u *permTensor,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    // Check for feasibility of direct copy from input to output if no permutation detected
    bool copyInput = true;
    for(int i = 0; i < dstGenericDescPtr->numDims - 1; i++)
        copyInput *= (permTensor[i] == i);

    if (copyInput)
    {
        CHECK_RETURN_STATUS(hipMemcpyAsync(dstPtr, srcPtr, dstGenericDescPtr->dims[0] * dstGenericDescPtr->strides[0] * sizeof(T), hipMemcpyDeviceToDevice, handle.GetStream()));
    }
    else
    {
        int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
        int globalThreads_y = dstGenericDescPtr->dims[0];
        int globalThreads_z = 1;

        hipLaunchKernelGGL(transpose_generic_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcGenericDescPtr->strides,
                           dstPtr,
                           dstGenericDescPtr->strides,
                           dstGenericDescPtr->dims + 1,
                           dstGenericDescPtr->numDims - 1,
                           permTensor);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_transpose_tensor<Rpp8u>(Rpp8u*,
                                                    RpptGenericDescPtr,
                                                    Rpp8u*,
                                                    RpptGenericDescPtr,
                                                    Rpp32u*,
                                                    Rpp32u*,
                                                    rpp::Handle&);

template RppStatus hip_exec_transpose_tensor<half>(half*,
                                                  RpptGenericDescPtr,
                                                  half*,
                                                  RpptGenericDescPtr,
                                                  Rpp32u*,
                                                  Rpp32u*,
                                                  rpp::Handle&);

template RppStatus hip_exec_transpose_tensor<Rpp32f>(Rpp32f*,
                                                     RpptGenericDescPtr,
                                                     Rpp32f*,
                                                     RpptGenericDescPtr,
                                                     Rpp32u*,
                                                     Rpp32u*,
                                                     rpp::Handle&);

template RppStatus hip_exec_transpose_tensor<Rpp8s>(Rpp8s*,
                                                    RpptGenericDescPtr,
                                                    Rpp8s*,
                                                    RpptGenericDescPtr,
                                                    Rpp32u*,
                                                    Rpp32u*,
                                                    rpp::Handle&);
