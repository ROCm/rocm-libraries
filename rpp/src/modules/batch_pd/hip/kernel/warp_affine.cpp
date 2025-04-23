#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

extern "C" __global__ void warp_affine_pln(unsigned char *srcPtr,
                                           unsigned char *dstPtr,
                                           float *affine,
                                           const unsigned int source_height,
                                           const unsigned int source_width,
                                           const unsigned int dest_height,
                                           const unsigned int dest_width,
                                           const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int xc = id_x - source_width/2;
    int yc = id_y - source_height/2;

    int k ;
    int l ;

    k = (int)((affine[0] * xc )+ (affine[1] * yc)) + affine[2];
    l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];

    k = k + source_width/2;
    l = l + source_height/2;

    if (l < source_height && l >=0 && k < source_width && k >=0)
    {
        dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = srcPtr[(id_z * source_height * source_width) + (l * source_width) + k];
    }
    else
    {
        dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = 0;
    }
}

extern "C" __global__ void warp_affine_pkd(unsigned char *srcPtr,
                                           unsigned char *dstPtr,
                                           float* affine,
                                           const unsigned int source_height,
                                           const unsigned int source_width,
                                           const unsigned int dest_height,
                                           const unsigned int dest_width,
                                           const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int xc = id_x - source_width / 2;
    int yc = id_y - source_height / 2;

    int k;
    int l;

    k = (int)((affine[0] * xc )+ (affine[1] * yc)) + affine[2];
    l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];

    k = k + source_width/2;
    l = l + source_height/2;

    if (l < source_height && l >=0 && k < source_width && k >=0)
    {
        dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] = srcPtr[id_z + (channel * l * source_width) + (channel * k)];
    }
    else
    {
        dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] = 0;
    }
}

extern "C" __global__ void warp_affine_batch(unsigned char *srcPtr,
                                             unsigned char *dstPtr,
                                             float *affine,
                                             unsigned int *source_height,
                                             unsigned int *source_width,
                                             unsigned int *dest_height,
                                             unsigned int *dest_width,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned int *max_source_width,
                                             unsigned int *max_dest_width,
                                             unsigned long long *source_batch_index,
                                             unsigned long long *dest_batch_index,
                                             const unsigned int channel,
                                             unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                             unsigned int *dest_inc,
                                             const int in_plnpkdind, // use 1 pln 3 for pkd
                                             const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    int indextmp = 0;
    int xc = id_x - (dest_width[id_z] >> 1);
    int yc = id_y - (dest_height[id_z] >> 1);
    int affine_index = id_z * 6;
    int k = (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) + affine[affine_index + 2];
    int l = (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) + affine[affine_index + 5];
    k = k + (source_width[id_z] >> 1);
    l = l + (source_height[id_z] >> 1);

    if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] && (k >= xroi_begin[id_z]))
    {
        unsigned long src_pixIdx, dst_pixIdx;
        src_pixIdx = source_batch_index[id_z] + (k + l * max_source_width[id_z]) * in_plnpkdind;
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
            src_pixIdx += source_inc[id_z];
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        unsigned long dst_pixIdx;
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void warp_affine_batch_fp32(float *srcPtr,
                                                  float *dstPtr,
                                                  float *affine,
                                                  unsigned int *source_height,
                                                  unsigned int *source_width,
                                                  unsigned int *dest_height,
                                                  unsigned int *dest_width,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned int *max_source_width,
                                                  unsigned int *max_dest_width,
                                                  unsigned long long *source_batch_index,
                                                  unsigned long long *dest_batch_index,
                                                  const unsigned int channel,
                                                  unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                  unsigned int *dest_inc,
                                                  const int in_plnpkdind, // use 1 pln 3 for pkd
                                                  const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    int indextmp = 0;
    int xc = id_x - (dest_width[id_z] >> 1);
    int yc = id_y - (dest_height[id_z] >> 1);
    int affine_index = id_z * 6;

    int k = (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) + affine[affine_index + 2];
    int l = (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) + affine[affine_index + 5];
    k = k + (source_width[id_z] >> 1);
    l = l + (source_height[id_z] >> 1);

    if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] && (k >= xroi_begin[id_z]))
    {
        unsigned long src_pixIdx, dst_pixIdx;
        src_pixIdx = source_batch_index[id_z] + (k + l * max_source_width[id_z]) * in_plnpkdind;
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
            src_pixIdx += source_inc[id_z];
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        unsigned long dst_pixIdx;
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

// extern "C" __global__ void warp_affine_batch_fp16(
//     half *srcPtr, half *dstPtr, float *affine,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *xroi_begin, unsigned int *xroi_end,
//     unsigned int *yroi_begin, unsigned int *yroi_end,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width,
//     unsigned long long *source_batch_index,
//     unsigned long long *dest_batch_index, const unsigned int channel,
//     unsigned int *source_inc, unsigned int *dest_inc, // use width * height for pln and 1 for pkd
//     const int in_plnpkdind, const int out_plnpkdind) // use 1 pln 3 for pkd
// {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   int indextmp = 0;
//   unsigned long src_pixIdx = 0, dst_pixIdx = 0;
//   int xc = id_x - (dest_width[id_z] >> 1);
//   int yc = id_y - (dest_height[id_z] >> 1);
//   int affine_index = id_z * 6;

//   int k =
//       (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) +
//       affine[affine_index + 2];
//   int l =
//       (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) +
//       affine[affine_index + 5];
//   k = k + (source_width[id_z] >> 1);
//   l = l + (source_height[id_z] >> 1);

//   if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
//       (k >= xroi_begin[id_z])) {
//     src_pixIdx = source_batch_index[id_z] +
//                  (k + l * max_source_width[id_z]) * in_plnpkdind;
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
//       src_pixIdx += source_inc[id_z];
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }

//   else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void warp_affine_batch_int8(signed char *srcPtr,
                                                  signed char *dstPtr,
                                                  float *affine,
                                                  unsigned int *source_height,
                                                  unsigned int *source_width,
                                                  unsigned int *dest_height,
                                                  unsigned int *dest_width,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned int *max_source_width,
                                                  unsigned int *max_dest_width,
                                                  unsigned long long *source_batch_index,
                                                  unsigned long long *dest_batch_index,
                                                  const unsigned int channel,
                                                  unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                  unsigned int *dest_inc,
                                                  const int in_plnpkdind, // use 1 pln 3 for pkd
                                                  const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    int indextmp = 0;
    int xc = id_x - (dest_width[id_z] >> 1);
    int yc = id_y - (dest_height[id_z] >> 1);
    int affine_index = id_z * 6;

    int k = (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) + affine[affine_index + 2];
    int l = (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) + affine[affine_index + 5];
    k = k + (source_width[id_z] >> 1);
    l = l + (source_height[id_z] >> 1);

    if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] && (k >= xroi_begin[id_z]))
    {
        unsigned long src_pixIdx, dst_pixIdx;
        src_pixIdx = source_batch_index[id_z] + (k + l * max_source_width[id_z]) * in_plnpkdind;
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
            src_pixIdx += source_inc[id_z];
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        unsigned long dst_pixIdx;
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = -128;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

RppStatus hip_exec_warp_affine_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(warp_affine_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       affine,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       tensor_info._in_channels,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_warp_affine_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // hipLaunchKernelGGL(warp_affine_batch_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    affine,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.x,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.y,
    //                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
    //                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
    //                    handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
    //                    tensor_info._in_channels,
    //                    handle.GetInitHandle()->mem.mgpu.inc,
    //                    handle.GetInitHandle()->mem.mgpu.dstInc,
    //                    in_plnpkdind,
    //                    out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_warp_affine_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(warp_affine_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       affine,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       tensor_info._in_channels,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_warp_affine_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(warp_affine_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       affine,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       tensor_info._in_channels,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}