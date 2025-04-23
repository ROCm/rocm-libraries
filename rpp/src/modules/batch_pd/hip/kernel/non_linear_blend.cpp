#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ float gaussian(int x, int y, float std_dev)
{
    float res, pi = 3.14;
    res = 1 / (2 * pi * std_dev * std_dev);
    float exp1, exp2;
    exp1 = -(x * x) / (2 * std_dev * std_dev);
    exp2 = -(y * y) / (2 * std_dev * std_dev);
    exp1 = exp1 + exp2;
    exp1 = exp(exp1);
    res *= exp1;
    return res;
}

extern "C" __global__ void non_linear_blend_batch(unsigned char *input1,
                                                  unsigned char *input2,
                                                  unsigned char *output,
                                                  float *std_dev,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned int *height,
                                                  unsigned int *width,
                                                  unsigned int *max_width,
                                                  unsigned long long *batch_index,
                                                  const unsigned int channel,
                                                  unsigned int *src_inc,
                                                  unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                                  const int in_plnpkdind,
                                                  const int out_plnpkdind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        int x = (id_x - (width[id_z] >> 1));
        int y = (id_y - (height[id_z] >> 1));
        float gaussianvalue = gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);

        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = gaussianvalue * input1[src_pix_idx] + (1 - gaussianvalue) * input2[src_pix_idx];
            src_pix_idx += src_inc[id_z];
            dst_pix_idx += dst_inc[id_z];
        }
    }
    else
    {
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = 0;
            dst_pix_idx += dst_inc[id_z];
        }
    }
}

// extern "C" __global__ void non_linear_blend_batch_fp16(
//     half *input1, half *input2, half *output,
//     float *std_dev, int *xroi_begin, int *xroi_end,
//     int *yroi_begin, int *yroi_end,
//     unsigned int *height, unsigned int *width,
//     unsigned int *max_width, unsigned long *batch_index,
//     const unsigned int channel, unsigned int *src_inc,
//     unsigned int *dst_inc, // use width * height for pln and 1 for pkd
//     const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   int indextmp = 0;
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

//   if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
//       (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
//     int x = (id_x - (width[id_z] >> 1));
//     int y = (id_y - (height[id_z] >> 1));
//     float gaussianvalue =
//         gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pix_idx] = (half) gaussianvalue * input1[src_pix_idx] +
//                             (1 - gaussianvalue) * input2[src_pix_idx];
//       src_pix_idx += src_inc[id_z];
//       dst_pix_idx += dst_inc[id_z];
//     }
//   } else {
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pix_idx] = 0;
//       dst_pix_idx += dst_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void non_linear_blend_batch_fp32(float *input1,
                                                       float *input2,
                                                       float *output,
                                                       float *std_dev,
                                                       unsigned int *xroi_begin,
                                                       unsigned int *xroi_end,
                                                       unsigned int *yroi_begin,
                                                       unsigned int *yroi_end,
                                                       unsigned int *height,
                                                       unsigned int *width,
                                                       unsigned int *max_width,
                                                       unsigned long long *batch_index,
                                                       const unsigned int channel,
                                                       unsigned int *src_inc,
                                                       unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                                       const int in_plnpkdind,
                                                       const int out_plnpkdind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        int x = (id_x - (width[id_z] >> 1));
        int y = (id_y - (height[id_z] >> 1));
        float gaussianvalue = gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = gaussianvalue * input1[src_pix_idx] + (1 - gaussianvalue) * input2[src_pix_idx];
            src_pix_idx += src_inc[id_z];
            dst_pix_idx += dst_inc[id_z];
        }
    }
    else
    {
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = 0;
            dst_pix_idx += dst_inc[id_z];
        }
    }
}

extern "C" __global__ void non_linear_blend_batch_int8(signed char *input1,
                                                       signed char *input2,
                                                       signed char *output,
                                                       float *std_dev,
                                                       unsigned int *xroi_begin,
                                                       unsigned int *xroi_end,
                                                       unsigned int *yroi_begin,
                                                       unsigned int *yroi_end,
                                                       unsigned int *height,
                                                       unsigned int *width,
                                                       unsigned int *max_width,
                                                       unsigned long long *batch_index,
                                                       const unsigned int channel,
                                                       unsigned int *src_inc,
                                                       unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                                       const int in_plnpkdind,
                                                       const int out_plnpkdind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        int x = (id_x - (width[id_z] >> 1));
        int y = (id_y - (height[id_z] >> 1));
        float gaussianvalue = gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = gaussianvalue * input1[src_pix_idx] + (1 - gaussianvalue) * input2[src_pix_idx];
            src_pix_idx += src_inc[id_z];
            dst_pix_idx += dst_inc[id_z];
        }
    }
    else
    {
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = 0;
            dst_pix_idx += dst_inc[id_z];
        }
    }
}

RppStatus hip_exec_non_linear_blend_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(non_linear_blend_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       dstPtr,
                       handle_obj->mem.mgpu.floatArr[0].floatmem,
                       handle_obj->mem.mgpu.roiPoints.x,
                       handle_obj->mem.mgpu.roiPoints.roiWidth,
                       handle_obj->mem.mgpu.roiPoints.y,
                       handle_obj->mem.mgpu.roiPoints.roiHeight,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       tensor_info._in_channels,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_non_linear_blend_batch_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(non_linear_blend_batch_fp16,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr1,
//                        srcPtr2,
//                        dstPtr,
//                        handle_obj->mem.mgpu.floatArr[0].floatmem,
//                        handle_obj->mem.mgpu.roiPoints.x,
//                        handle_obj->mem.mgpu.roiPoints.roiWidth,
//                        handle_obj->mem.mgpu.roiPoints.y,
//                        handle_obj->mem.mgpu.roiPoints.roiHeight,
//                        handle_obj->mem.mgpu.srcSize.height,
//                        handle_obj->mem.mgpu.srcSize.width,
//                        handle_obj->mem.mgpu.maxSrcSize.width,
//                        handle_obj->mem.mgpu.srcBatchIndex,
//                        tensor_info._in_channels,
//                        handle_obj->mem.mgpu.inc,
//                        handle_obj->mem.mgpu.dstInc,
//                        in_plnpkdind,
//                        out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_non_linear_blend_batch_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(non_linear_blend_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       dstPtr,
                       handle_obj->mem.mgpu.floatArr[0].floatmem,
                       handle_obj->mem.mgpu.roiPoints.x,
                       handle_obj->mem.mgpu.roiPoints.roiWidth,
                       handle_obj->mem.mgpu.roiPoints.y,
                       handle_obj->mem.mgpu.roiPoints.roiHeight,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       tensor_info._in_channels,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_non_linear_blend_batch_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(non_linear_blend_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       dstPtr,
                       handle_obj->mem.mgpu.floatArr[0].floatmem,
                       handle_obj->mem.mgpu.roiPoints.x,
                       handle_obj->mem.mgpu.roiPoints.roiWidth,
                       handle_obj->mem.mgpu.roiPoints.y,
                       handle_obj->mem.mgpu.roiPoints.roiHeight,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       tensor_info._in_channels,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);
    return RPP_SUCCESS;
}