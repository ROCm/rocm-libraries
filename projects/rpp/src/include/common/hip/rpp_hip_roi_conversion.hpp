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

#ifndef RPP_HIP_ROI_CONVERSION_H
#define RPP_HIP_ROI_CONVERSION_H

#include <hip/hip_runtime.h>

// Both conversion helpers below read the caller's ROI tensor and write the converted result into
// a scratch buffer owned by the handle (mem.mgpu.roiConversionScratchHip), instead of mutating the
// caller's buffer in place. The returned RpptROIPtr (pointing at the scratch buffer) is what the
// rest of the calling function should use for its kernel launches.

// LTRB to XYWH

[[maybe_unused]] static __global__ void roi_conversion_ltrb_to_xywh(int* roiTensorPtrSrc,
                                                                    int* roiTensorPtrDst) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 4;

    int4 roi_i4 = *(int4*)&roiTensorPtrSrc[id_x];
    roi_i4.z -= (roi_i4.x - 1);
    roi_i4.w -= (roi_i4.y - 1);
    *(int4*)&roiTensorPtrDst[id_x] = roi_i4;
}

[[maybe_unused]] static RpptROIPtr hip_exec_roi_conversion_ltrb_to_xywh(RpptROIPtr roiTensorPtrSrc,
                                                                        rpp::Handle& handle) {
    RpptROIPtr roiTensorPtrDst = handle.GetInitHandle()->mem.mgpu.roiConversionScratchHip;

    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(roi_conversion_ltrb_to_xywh,
                       dim3(ceil((float)globalThreads_x / localThreads_x),
                            ceil((float)globalThreads_y / localThreads_y),
                            ceil((float)globalThreads_z / localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z), 0, handle.GetStream(),
                       (int*)roiTensorPtrSrc, (int*)roiTensorPtrDst);

    hipError_t status = hipGetLastError();
    if (status != hipSuccess)
        fprintf(stderr, "HIP kernel launch error: returned %d at %s:%d", status, __FILE__,
                __LINE__);

    return roiTensorPtrDst;
}

// XYWH to LTRB

[[maybe_unused]] static __global__ void roi_conversion_xywh_to_ltrb(int* roiTensorPtrSrc,
                                                                    int* roiTensorPtrDst) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 4;

    int4 roi_i4 = *(int4*)&roiTensorPtrSrc[id_x];
    roi_i4.z += (roi_i4.x - 1);
    roi_i4.w += (roi_i4.y - 1);
    *(int4*)&roiTensorPtrDst[id_x] = roi_i4;
}

[[maybe_unused]] static RpptROIPtr hip_exec_roi_conversion_xywh_to_ltrb(RpptROIPtr roiTensorPtrSrc,
                                                                        rpp::Handle& handle) {
    RpptROIPtr roiTensorPtrDst = handle.GetInitHandle()->mem.mgpu.roiConversionScratchHip;

    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(roi_conversion_xywh_to_ltrb,
                       dim3(ceil((float)globalThreads_x / localThreads_x),
                            ceil((float)globalThreads_y / localThreads_y),
                            ceil((float)globalThreads_z / localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z), 0, handle.GetStream(),
                       (int*)roiTensorPtrSrc, (int*)roiTensorPtrDst);

    hipError_t status = hipGetLastError();
    if (status != hipSuccess)
        fprintf(stderr, "HIP kernel launch error: returned %d at %s:%d", status, __FILE__,
                __LINE__);

    return roiTensorPtrDst;
}

#endif  // RPP_HIP_ROI_CONVERSION_H
