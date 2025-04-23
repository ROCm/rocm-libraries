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

template <typename T>
__global__ void fill_value_ncdhw_hip_tensor(T *dstPtr,
                                            uint3 dstStridesCDH,
                                            int channels,
                                            uint3 dstDimsDHW,
                                            T *fillValue)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= dstDimsDHW.x) || (id_y >= dstDimsDHW.y) || (id_x >= dstDimsDHW.z))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;
    d_float8 val_f8;
    val_f8.f4[0] = (float4)(*fillValue);
    val_f8.f4[1] = val_f8.f4[0];
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        dstIdx += dstStridesCDH.x;
    }
}

template <typename T>
__global__ void fill_value_ndhwc_hip_tensor(T *dstPtr,
                                            uint2 dstStridesDH,
                                            uint3 dstDimsDHW,
                                            T *fillValue)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= dstDimsDHW.x) || (id_y >= dstDimsDHW.y) || (id_x >= dstDimsDHW.z))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;
    d_float24 val_f24;
    val_f24.f4[0] = (float4)(*fillValue);
    val_f24.f4[1] = val_f24.f4[0];
    val_f24.f4[2] = val_f24.f4[0];
    val_f24.f4[3] = val_f24.f4[0];
    val_f24.f4[4] = val_f24.f4[0];
    val_f24.f4[5] = val_f24.f4[0];
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

template <typename T>
__global__ void slice_ncdhw_hip_tensor(T *srcPtr,
                                       uint3 srcStridesCDH,
                                       T *dstPtr,
                                       uint3 dstStridesCDH,
                                       int channels,
                                       uint3 validShapeDHW)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= validShapeDHW.x) || (id_y >= validShapeDHW.y) || (id_x >= validShapeDHW.z))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesCDH.y) + (id_y * srcStridesCDH.z) + id_x;
    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;

    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

template <typename T>
__global__ void slice_ndhwc_hip_tensor(T *srcPtr,
                                       uint2 srcStridesDH,
                                       T *dstPtr,
                                       uint2 dstStridesDH,
                                       uint3 validShapeDHW)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= validShapeDHW.x) || (id_y >= validShapeDHW.y) || (id_x >= validShapeDHW.z))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesDH.x) + (id_y * srcStridesDH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + (id_x * 3);

    d_float24 val_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &val_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

template <typename T>
RppStatus hip_exec_fill_value_tensor(T *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     Rpp32s *anchorTensor,
                                     Rpp32s *shapeTensor,
                                     T *fillValue,
                                     Rpp32u *roiTensor,
                                     rpp::Handle& handle,
                                     Rpp32u numDims)
{
    if (numDims == 4)
    {
        // set the dimsOrder and globalthreads values required for NDHWC layout
        Rpp32s dimsOrder[3] = {0, 1, 2};
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];                   // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];                   // D - depth (z direction)

        // change the dimsOrder and globalthreads values if layout is NCDHW
        if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
        {
            dimsOrder[0] = 1;  // depth
            dimsOrder[1] = 2;  // height
            dimsOrder[2] = 3;  // width
            globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
            globalThreads_y = dstGenericDescPtr->dims[3];               // H - height (y direction)
            globalThreads_z = dstGenericDescPtr->dims[2];               // D - depth (z direction)
        }
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxDepth = std::min(shape[dimsOrder[0]], length[dimsOrder[0]] - anchor[dimsOrder[0]]);
            Rpp32u maxHeight = std::min(shape[dimsOrder[1]], length[dimsOrder[1]] - anchor[dimsOrder[1]]);
            Rpp32u maxWidth = std::min(shape[dimsOrder[2]], length[dimsOrder[2]] - anchor[dimsOrder[2]]);

            // checking if padding is required
            bool needPadding = (((anchor[dimsOrder[0]] + shape[dimsOrder[0]]) > length[dimsOrder[0]]) ||
                                ((anchor[dimsOrder[1]] + shape[dimsOrder[1]]) > length[dimsOrder[1]]) ||
                                ((anchor[dimsOrder[2]] + shape[dimsOrder[2]]) > length[dimsOrder[2]]));

            // if needPadding is set, launch kernel for filling the padded region with fill value specified
            if (needPadding && dstGenericDescPtr->layout == RpptLayout::NCDHW)
            {
                hipLaunchKernelGGL(fill_value_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint3(maxDepth, maxHeight, maxWidth),
                                   fillValue);
            }
            else if (needPadding && dstGenericDescPtr->layout == RpptLayout::NDHWC)
            {
                hipLaunchKernelGGL(fill_value_ndhwc_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                   make_uint3(maxDepth, maxHeight, maxWidth),
                                   fillValue);
            }
        }
    }
    else if (numDims == 3)
    {
        // set the dimsOrder and globalthreads values required for NHWC layout
        Rpp32s dimsOrder[2] = {0, 1};
        int globalThreads_x = (dstGenericDescPtr->strides[1] / 3 + 7) >> 3; // W - width  (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[1];                   // H - height (y direction)
        int globalThreads_z = 1;

        // change the dimsOrder and globalthreads values if layout is NCHW
        if (dstGenericDescPtr->layout == RpptLayout::NCHW)
        {
            dimsOrder[0] = 1;  // height
            dimsOrder[1] = 2;  // width
            globalThreads_x = (dstGenericDescPtr->strides[2] + 7) >> 3; // W - width  (x direction) - vectorized for 8 element loads/stores per channel
            globalThreads_y = dstGenericDescPtr->dims[2];               // H - height (y direction)
            globalThreads_z = 1;
        }

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxHeight = std::min(shape[dimsOrder[0]], length[dimsOrder[0]] - anchor[dimsOrder[0]]);
            Rpp32u maxWidth = std::min(shape[dimsOrder[1]], length[dimsOrder[1]] - anchor[dimsOrder[1]]);

            // check if padding is needed
            bool needPadding = (((anchor[dimsOrder[0]] + shape[dimsOrder[0]]) > length[dimsOrder[0]]) ||
                                ((anchor[dimsOrder[1]] + shape[dimsOrder[1]]) > length[dimsOrder[1]]));

            // launch kernel for filling the padded region with fill value specified
            if (needPadding && dstGenericDescPtr->layout == RpptLayout::NCHW)
            {
                hipLaunchKernelGGL(fill_value_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), globalThreads_z),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                   0,
                                   handle.GetStream(),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint3(dstGenericDescPtr->strides[1], 0, dstGenericDescPtr->strides[2]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint3(1, shape[1], shape[2]),
                                   fillValue);
            }
            else if (needPadding && dstGenericDescPtr->layout == RpptLayout::NHWC)
            {
                hipLaunchKernelGGL(fill_value_ndhwc_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), globalThreads_z),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                   0,
                                   handle.GetStream(),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint2(1, dstGenericDescPtr->strides[1]),
                                   make_uint3(1, maxHeight, maxWidth),
                                   fillValue);
            }
        }
    }
    else if (numDims == 2)
    {
        // NHW
        int globalThreads_x = (dstGenericDescPtr->strides[1] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[1];               // H - height (y direction)
        int globalThreads_z = 1;

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxHeight = std::min(shape[0], length[0] - anchor[0]);
            Rpp32u maxWidth = std::min(shape[1], length[1] - anchor[1]);

            // check if padding is needed
            bool needPadding = (((anchor[0] + shape[0]) > length[0]) ||
                                ((anchor[1] + shape[1]) > length[1]));

            // launch kernel for filling the padded region with fill value specified
            if (needPadding)
            {
                hipLaunchKernelGGL(fill_value_ncdhw_hip_tensor,
                                    dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                    0,
                                    handle.GetStream(),
                                    dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                    make_uint3(0, 0, dstGenericDescPtr->strides[1]),
                                    1,
                                    make_uint3(1, shape[0], shape[1]),
                                    fillValue);
            }
        }
    }
    else if (numDims == 1)
    {
        int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = 1;
        int globalThreads_z = 1;

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxLength = std::min(shape[0], length[0] - anchor[0]);

            // check if padding is needed
            bool needPadding = ((anchor[0] + shape[0]) > length[0]);

            // launch kernel for filling the padded region with fill value specified
            if (needPadding)
            {
                hipLaunchKernelGGL(fill_value_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, 1, 1),
                                   0,
                                   handle.GetStream(),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint3(0, 0, 1),
                                   1,
                                   make_uint3(1, 1, shape[0]),
                                   fillValue);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_slice_tensor(T *srcPtr,
                                RpptGenericDescPtr srcGenericDescPtr,
                                T *dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr,
                                Rpp32s *anchorTensor,
                                Rpp32s *shapeTensor,
                                T *fillValue,
                                bool enablePadding,
                                Rpp32u *roiTensor,
                                rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims

    /* if enabledPadding is set to true, launch kernel to fill the output buffers with fill value specified.
    This will be only done if shapeTensor[d] > roiTensor[d] where d is the dimension*/
    if (enablePadding)
    {
        hip_exec_fill_value_tensor(dstPtr,
                                   dstGenericDescPtr,
                                   anchorTensor,
                                   shapeTensor,
                                   fillValue,
                                   roiTensor,
                                   handle,
                                   numDims);
        hipStreamSynchronize(handle.GetStream());
    }

    if(numDims == 4)
    {
        // set the dimsOrder and globalthreads values required for NDHWC layout
        Rpp32s dimsOrder[3] = {0, 1, 2};
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];                   // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];                   // D - depth (z direction)

        // change the dimsOrder and globalthreads values if layout is NCDHW
        if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
        {
            dimsOrder[0] = 1;  // depth
            dimsOrder[1] = 2;  // height
            dimsOrder[2] = 3;  // width
            globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
            globalThreads_y = dstGenericDescPtr->dims[3];               // H - height (y direction)
            globalThreads_z = dstGenericDescPtr->dims[2];               // D - depth (z direction)
        }

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxDepth = std::min(shape[dimsOrder[0]], length[dimsOrder[0]] - anchor[dimsOrder[0]]);
            Rpp32u maxHeight = std::min(shape[dimsOrder[1]], length[dimsOrder[1]] - anchor[dimsOrder[1]]);
            Rpp32u maxWidth = std::min(shape[dimsOrder[2]], length[dimsOrder[2]] - anchor[dimsOrder[2]]);
            if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
            {
                T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[1] * srcGenericDescPtr->strides[2] + anchor[2] * srcGenericDescPtr->strides[3] + anchor[3];
                T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);
                hipLaunchKernelGGL(slice_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtrTemp,
                                   make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                                   dstPtrTemp,
                                   make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint3(maxDepth, maxHeight, maxWidth));
            }
            else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
            {
                T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[0] * srcGenericDescPtr->strides[1] + anchor[1] * srcGenericDescPtr->strides[2] + anchor[2];
                T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);
                hipLaunchKernelGGL(slice_ndhwc_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtrTemp,
                                   make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                                   dstPtrTemp,
                                   make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                   make_uint3(maxDepth, maxHeight, maxWidth));
            }
        }
    }
    else if (numDims == 3)
    {
        // set the dimsOrder and globalthreads values required for NHWC layout
        Rpp32s dimsOrder[2] = {0, 1};
        int globalThreads_x = (dstGenericDescPtr->strides[1] / 3 + 7) >> 3; // W - width  (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[1];                   // H - height (y direction)
        int globalThreads_z = 1;

        // change the dimsOrder and globalthreads values if layout is NCHW
        if (dstGenericDescPtr->layout == RpptLayout::NCHW)
        {
            dimsOrder[0] = 1;  // height
            dimsOrder[1] = 2;  // width
            globalThreads_x = (dstGenericDescPtr->strides[2] + 7) >> 3; // W - width  (x direction) - vectorized for 8 element loads/stores per channel
            globalThreads_y = dstGenericDescPtr->dims[2];               // H - height (y direction)
            globalThreads_z = 1;
        }

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxHeight = std::min(shape[dimsOrder[0]], length[dimsOrder[0]] - anchor[dimsOrder[0]]);
            Rpp32u maxWidth = std::min(shape[dimsOrder[1]], length[dimsOrder[1]] - anchor[dimsOrder[1]]);
            if (dstGenericDescPtr->layout == RpptLayout::NCHW)
            {
                T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[1] * srcGenericDescPtr->strides[2] + anchor[2];
                T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);
                hipLaunchKernelGGL(slice_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                   0,
                                   handle.GetStream(),
                                   srcPtrTemp,
                                   make_uint3(srcGenericDescPtr->strides[1], 0, srcGenericDescPtr->strides[2]),
                                   dstPtrTemp,
                                   make_uint3(dstGenericDescPtr->strides[1], 0, dstGenericDescPtr->strides[2]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint3(1, maxHeight, maxWidth));
            }
            else if (dstGenericDescPtr->layout == RpptLayout::NHWC)
            {
                T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[0] * srcGenericDescPtr->strides[1] + anchor[1];
                T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);
                hipLaunchKernelGGL(slice_ndhwc_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), globalThreads_z),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                   0,
                                   handle.GetStream(),
                                   srcPtrTemp,
                                   make_uint2(1, srcGenericDescPtr->strides[1]),
                                   dstPtrTemp,
                                   make_uint2(1, dstGenericDescPtr->strides[1]),
                                   make_uint3(1, maxHeight, maxWidth));
            }
        }
    }
    else if (numDims == 2)
    {
        // NHW
        int globalThreads_x = (dstGenericDescPtr->strides[1] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[1];               // H - height (y direction)
        int globalThreads_z = 1;
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxHeight = std::min(shape[0], length[0] - anchor[0]);
            Rpp32u maxWidth = std::min(shape[1], length[1] - anchor[1]);
            T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[0] * srcGenericDescPtr->strides[2] + anchor[1];
            T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);

            hipLaunchKernelGGL(slice_ncdhw_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                               0,
                               handle.GetStream(),
                               srcPtrTemp,
                               make_uint3(0, 0, srcGenericDescPtr->strides[1]),
                               dstPtrTemp,
                               make_uint3(0, 0, dstGenericDescPtr->strides[1]),
                               1,
                               make_uint3(1, maxHeight, maxWidth));
        }
    }
    else if (numDims == 1)
    {
        int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = 1;
        int globalThreads_z = 1;
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxLength = std::min(shape[0], length[0] - anchor[0]);
            T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[0];
            T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);

            hipLaunchKernelGGL(slice_ncdhw_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtrTemp,
                               make_uint3(0, 0, 1),
                               dstPtrTemp,
                               make_uint3(0, 0, 1),
                               1,
                               make_uint3(1, 1, maxLength));
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_slice_tensor<Rpp32f>(Rpp32f*,
                                                 RpptGenericDescPtr,
                                                 Rpp32f*,
                                                 RpptGenericDescPtr,
                                                 Rpp32s*,
                                                 Rpp32s*,
                                                 Rpp32f*,
                                                 bool,
                                                 Rpp32u*,
                                                 rpp::Handle&);

template RppStatus hip_exec_slice_tensor<Rpp8u>(Rpp8u*,
                                                RpptGenericDescPtr,
                                                Rpp8u*,
                                                RpptGenericDescPtr,
                                                Rpp32s*,
                                                Rpp32s*,
                                                Rpp8u*,
                                                bool,
                                                Rpp32u*,
                                                rpp::Handle&);
