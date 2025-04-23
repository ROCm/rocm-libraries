#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void crop_mirror_normalize_batch(unsigned char *input,
                                                       unsigned char *output,
                                                       unsigned int *dst_height,
                                                       unsigned int *dst_width,
                                                       unsigned int *src_width,
                                                       unsigned int *start_x,
                                                       unsigned int *start_y,
                                                       float *mean,
                                                       float *std_dev,
                                                       unsigned int *flip,
                                                       unsigned int *max_src_width,
                                                       unsigned int *max_dst_width,
                                                       unsigned long long *src_batch_index,
                                                       unsigned long long *dst_batch_index,
                                                       const unsigned int channel,
                                                       unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                                       unsigned int *dst_inc,
                                                       const int in_plnpkdind, // use 1 pln 3 for pkd
                                                       const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    const float local_mean = mean[id_z];
    const float local_std_dev = std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx;

    if (local_flip == 1)
    {
        src_pixIdx = src_batch_index[id_z] + ((dst_width[id_z] - 1 - id_x + start_x[id_z]) + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }
    else
    {
        src_pixIdx = src_batch_index[id_z] + (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }

    unsigned long dst_pixIdx = dst_batch_index[id_z] + (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;

    if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (unsigned char) ((float)input[src_pixIdx] - local_mean) / local_std_dev;
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }
    }
    else
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0;
            dst_pixIdx += dst_inc[id_z];
        }
    }
}

// extern "C" __global__ void crop_mirror_normalize_batch_fp16(
//     half *input,
//     half *output,
//     unsigned int *dst_height, unsigned int *dst_width,
//     unsigned int *src_width, unsigned int *start_x,
//     unsigned int *start_y, float *mean,
//     float *std_dev, unsigned int *flip,
//     unsigned int *max_src_width, unsigned int *max_dst_width,
//     unsigned long long *src_batch_index,
//     unsigned long long *dst_batch_index, const unsigned int channel,
//     // const unsigned int batch_size,
//     unsigned int *src_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dst_inc,
//     const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//   int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//   int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   int indextmp = 0;
//   const half local_mean = (half)mean[id_z];
//   const half local_std_dev = (half)std_dev[id_z];
//   const unsigned int local_flip = flip[id_z];
//   unsigned long  src_pixIdx;
//   if (local_flip == 1) {
//     src_pixIdx = src_batch_index[id_z] +
//                  ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
//                   (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
//   }
//   else{
//      src_pixIdx = src_batch_index[id_z] +
//       (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
//   }
//   unsigned long dst_pixIdx =
//       dst_batch_index[id_z] +
//       (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
//   if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pixIdx] = (input[src_pixIdx] - local_mean) / local_std_dev;
//       src_pixIdx += src_inc[id_z];
//       dst_pixIdx += dst_inc[id_z];
//     }
//   } else {
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pixIdx] = 0;
//       dst_pixIdx += dst_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void crop_mirror_normalize_batch_int8(signed char *input,
                                                            signed char *output,
                                                            unsigned int *dst_height,
                                                            unsigned int *dst_width,
                                                            unsigned int *src_width,
                                                            unsigned int *start_x,
                                                            unsigned int *start_y,
                                                            float *mean,
                                                            float *std_dev,
                                                            unsigned int *flip,
                                                            unsigned int *max_src_width,
                                                            unsigned int *max_dst_width,
                                                            unsigned long long *src_batch_index,
                                                            unsigned long long *dst_batch_index,
                                                            const unsigned int channel,
                                                            unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                                            unsigned int *dst_inc,
                                                            const int in_plnpkdind, // use 1 pln 3 for pkd
                                                            const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    const float local_mean = mean[id_z];
    const float local_std_dev = std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx;

    if (local_flip == 1)
    {
        src_pixIdx = src_batch_index[id_z] + ((dst_width[id_z] - 1 - id_x + start_x[id_z]) + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }
    else
    {
        src_pixIdx = src_batch_index[id_z] + (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }

    unsigned long dst_pixIdx = dst_batch_index[id_z] + (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;

    if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (signed char) ((input[src_pixIdx] - local_mean) / local_std_dev);
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }
    }
    else
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = -128;
            dst_pixIdx += dst_inc[id_z];
        }
    }
}

extern "C" __global__ void crop_mirror_normalize_batch_fp32(float *input,
                                                            float *output,
                                                            unsigned int *dst_height,
                                                            unsigned int *dst_width,
                                                            unsigned int *src_width,
                                                            unsigned int *start_x,
                                                            unsigned int *start_y,
                                                            float *mean,
                                                            float *std_dev,
                                                            unsigned int *flip,
                                                            unsigned int *max_src_width,
                                                            unsigned int *max_dst_width,
                                                            unsigned long long *src_batch_index,
                                                            unsigned long long *dst_batch_index,
                                                            const unsigned int channel,
                                                            unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                                            unsigned int *dst_inc,
                                                            const int in_plnpkdind, // use 1 pln 3 for pkd
                                                            const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    const float local_mean = mean[id_z];
    const float local_std_dev = std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx;

    if (local_flip == 1)
    {
        src_pixIdx = src_batch_index[id_z] + ((dst_width[id_z] - 1 - id_x + start_x[id_z]) + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }
    else
    {
        src_pixIdx = src_batch_index[id_z] + (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }

    unsigned long dst_pixIdx = dst_batch_index[id_z] + (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;

    if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (input[src_pixIdx] - local_mean) / local_std_dev;
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }
    }
    else
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0;
            dst_pixIdx += dst_inc[id_z];
        }
    }
}

// extern "C" __global__ void crop_mirror_normalize_batch_u8_fp16(
//     unsigned char *input,
//     half *output,
//     unsigned int *dst_height, unsigned int *dst_width,
//     unsigned int *src_width, unsigned int *start_x,
//     unsigned int *start_y, float *mean,
//     float *std_dev, unsigned int *flip,
//     unsigned int *max_src_width, unsigned int *max_dst_width,
//     unsigned long long *src_batch_index,
//     unsigned long long *dst_batch_index, const unsigned int channel,
//     // const unsigned int batch_size,
//     unsigned int *src_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dst_inc,
//     const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//   int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//   int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   int indextmp = 0;
//   const float local_mean    = mean[id_z];
//   const float local_std_dev = std_dev[id_z];
//   const unsigned int local_flip = flip[id_z];
//   unsigned long src_pixIdx;
//   if (local_flip == 1) {
//     src_pixIdx = src_batch_index[id_z] +
//                  ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
//                   (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
//   }
//   else{
//      src_pixIdx = src_batch_index[id_z] +
//       (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
//   }
//   unsigned long dst_pixIdx =
//       dst_batch_index[id_z] +
//       (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
//   if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pixIdx] = (half)((input[src_pixIdx] - local_mean) / 255.0 * local_std_dev);
//       src_pixIdx += src_inc[id_z];
//       dst_pixIdx += dst_inc[id_z];
//     }
//   } else {
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pixIdx] = 0.0;
//       dst_pixIdx += dst_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void crop_mirror_normalize_batch_u8_fp32(unsigned char *input,
                                                               float *output,
                                                               unsigned int *dst_height,
                                                               unsigned int *dst_width,
                                                               unsigned int *src_width,
                                                               unsigned int *start_x,
                                                               unsigned int *start_y,
                                                               float *mean,
                                                               float *std_dev,
                                                               unsigned int *flip,
                                                               unsigned int *max_src_width,
                                                               unsigned int *max_dst_width,
                                                               unsigned long long *src_batch_index,
                                                               unsigned long long *dst_batch_index,
                                                               const unsigned int channel,
                                                               unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                                               unsigned int *dst_inc,
                                                               const int in_plnpkdind, // use 1 pln 3 for pkd
                                                               const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    const float local_mean = mean[id_z];
    const float local_std_dev = std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx;

    if (local_flip == 1)
    {
        src_pixIdx = src_batch_index[id_z] + ((dst_width[id_z] - 1 - id_x + start_x[id_z]) + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }
    else
    {
        src_pixIdx = src_batch_index[id_z] + (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }

    unsigned long dst_pixIdx = dst_batch_index[id_z] + (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;

    if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (float)((input[src_pixIdx] - local_mean) / 255.0 * local_std_dev);
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }
    }
    else
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0.0;
            dst_pixIdx += dst_inc[id_z];
        }
    }
}

extern "C" __global__ void crop_mirror_normalize_batch_u8_int8(unsigned char *input,
                                                               signed char *output,
                                                               unsigned int *dst_height,
                                                               unsigned int *dst_width,
                                                               unsigned int *src_width,
                                                               unsigned int *start_x,
                                                               unsigned int *start_y,
                                                               float *mean,
                                                               float *std_dev,
                                                               unsigned int *flip,
                                                               unsigned int *max_src_width,
                                                               unsigned int *max_dst_width,
                                                               unsigned long long *src_batch_index,
                                                               unsigned long long *dst_batch_index,
                                                               const unsigned int channel,
                                                               unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                                               unsigned int *dst_inc,
                                                               const int in_plnpkdind, // use 1 pln 3 for pkd
                                                               const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    const float local_mean = mean[id_z];
    const float local_std_dev = std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx;

    if (local_flip == 1)
    {
        src_pixIdx = src_batch_index[id_z] + ((dst_width[id_z] - 1 - id_x + start_x[id_z]) + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }
    else
    {
        src_pixIdx = src_batch_index[id_z] + (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
    }

    unsigned long dst_pixIdx = dst_batch_index[id_z] + (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;

    if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (signed char)((input[src_pixIdx] - 128 - local_mean ) /  local_std_dev);
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }
    }
    else
    {
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0.0;
            dst_pixIdx += dst_inc[id_z];
        }
    }
}

RppStatus hip_exec_crop_mirror_normalize_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(crop_mirror_normalize_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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

RppStatus hip_exec_crop_mirror_normalize_batch_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // hipLaunchKernelGGL(crop_mirror_normalize_batch_u8_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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

RppStatus hip_exec_crop_mirror_normalize_batch_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(crop_mirror_normalize_batch_u8_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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

RppStatus hip_exec_crop_mirror_normalize_batch_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(crop_mirror_normalize_batch_u8_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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

RppStatus hip_exec_crop_mirror_normalize_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // hipLaunchKernelGGL(crop_mirror_normalize_batch_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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

RppStatus hip_exec_crop_mirror_normalize_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(crop_mirror_normalize_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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

RppStatus hip_exec_crop_mirror_normalize_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(crop_mirror_normalize_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
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