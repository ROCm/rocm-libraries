#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void glitch_batch(unsigned char *input,
                                        unsigned char *output,
                                        unsigned int *x_offset_r,
                                        unsigned int *y_offset_r,
                                        unsigned int *x_offset_g,
                                        unsigned int *y_offset_g,
                                        unsigned int *x_offset_b,
                                        unsigned int *y_offset_b,
                                        unsigned int *xroi_begin,
                                        unsigned int *xroi_end,
                                        unsigned int *yroi_begin,
                                        unsigned int *yroi_end,
                                        unsigned int *height,
                                        unsigned int *width,
                                        unsigned int *max_width,
                                        unsigned long long *batch_index,
                                        const unsigned int channel,
                                        unsigned int *inc, // use width * height for pln and 1 for pkd
                                        unsigned int *dstinc, // use width * height for pln and 1 for pkd
                                        int in_plnpkdind, // use 1 pln 3 for pkd
                                        int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    output[dst_pix_idx] = input[src_pix_idx];
    output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
    output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

    int x_r, x_g, x_b, y_r, y_g, y_b;

    // R
    x_r = (id_x + x_offset_r[id_z]);
    y_r = (id_y + y_offset_r[id_z]);

    // G
    x_g = (id_x + x_offset_g[id_z]);
    y_g = (id_y + y_offset_g[id_z]);

    // B
    x_b = (id_x + x_offset_b[id_z]);
    y_b = (id_y + y_offset_b[id_z]);

    // R
    if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) && (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
    {
        unsigned char R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        indextmp = indextmp + 1;
        output[dst_pix_idx] = R;
        dst_pix_idx += dstinc[id_z];
    }

    // G
    if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) && (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
    {
        unsigned char G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        indextmp = indextmp + 1;
        output[dst_pix_idx] = G;
        dst_pix_idx += dstinc[id_z];
    }

    // B
    if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) && (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
    {
        unsigned char B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        output[dst_pix_idx] = B;
    }
}

// extern "C" __global__ void glitch_batch_fp16(
//     half *input, half *output,
//     unsigned int *x_offset_r, unsigned int *y_offset_r,
//     unsigned int *x_offset_g, unsigned int *y_offset_g,
//     unsigned int *x_offset_b, unsigned int *y_offset_b,
//     unsigned int *xroi_begin, unsigned int *xroi_end, unsigned int *yroi_begin,
//     unsigned int *yroi_end, unsigned int *height,
//     unsigned int *width, unsigned int *max_width,
//     unsigned long long *batch_index, const unsigned int channel,
//     unsigned int *inc,    // use width * height for pln and 1 for pkd
//     unsigned int *dstinc, // use width * height for pln and 1 for pkd
//     int in_plnpkdind,              // use 1 pln 3 for pkd
//     int out_plnpkdind) {

//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//   int indextmp = 0;
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

//   output[dst_pix_idx] = input[src_pix_idx];
//   output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
//   output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

//   int x_r, x_g, x_b, y_r, y_g, y_b;
//   // R
//   x_r = (id_x + x_offset_r[id_z]);
//   y_r = (id_y + y_offset_r[id_z]);

//   // G
//   x_g = (id_x + x_offset_g[id_z]);
//   y_g = (id_y + y_offset_g[id_z]);

//   // B
//   x_b = (id_x + x_offset_b[id_z]);
//   y_b = (id_y + y_offset_b[id_z]);

//   // R
//   if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) &&
//       (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
//   {
//     half R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind +
//               indextmp * inc[id_z]];
//     indextmp = indextmp + 1;
//     output[dst_pix_idx] = R;
//     dst_pix_idx += dstinc[id_z];
//   }

//   // G
//   if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) &&
//       (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
//   {
//     half G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind +
//               indextmp * inc[id_z]];
//     indextmp = indextmp + 1;
//     output[dst_pix_idx] = G;
//     dst_pix_idx += dstinc[id_z];
//   }

//   // B
//   if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) &&
//       (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
//   {
//     half B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind +
//               indextmp * inc[id_z]];
//     output[dst_pix_idx] = B;
//   }
// }

extern "C" __global__ void glitch_batch_fp32(float *input,
                                             float *output,
                                             unsigned int *x_offset_r,
                                             unsigned int *y_offset_r,
                                             unsigned int *x_offset_g,
                                             unsigned int *y_offset_g,
                                             unsigned int *x_offset_b,
                                             unsigned int *y_offset_b,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned int *height,
                                             unsigned int *width,
                                             unsigned int *max_width,
                                             unsigned long long *batch_index,
                                             const unsigned int channel,
                                             unsigned int *inc, // use width * height for pln and 1 for pkd
                                             unsigned int *dstinc, // use width * height for pln and 1 for pkd
                                             int in_plnpkdind, // use 1 pln 3 for pkd
                                             int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    output[dst_pix_idx] = input[src_pix_idx];
    output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
    output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

    int x_r, x_g, x_b, y_r, y_g, y_b;

    // R
    x_r = (id_x + x_offset_r[id_z]);
    y_r = (id_y + y_offset_r[id_z]);

    // G
    x_g = (id_x + x_offset_g[id_z]);
    y_g = (id_y + y_offset_g[id_z]);

    // B
    x_b = (id_x + x_offset_b[id_z]);
    y_b = (id_y + y_offset_b[id_z]);

    // R
    if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) && (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
    {
        float R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        indextmp = indextmp + 1;
        output[dst_pix_idx] = R;
        dst_pix_idx += dstinc[id_z];
    }

    // G
    if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) && (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
    {
        float G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        indextmp = indextmp + 1;
        output[dst_pix_idx] = G;
        dst_pix_idx += dstinc[id_z];
    }

    // B
    if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) && (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
    {
        float B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        output[dst_pix_idx] = B;
    }
}

extern "C" __global__ void glitch_batch_int8(signed char *input,
                                             signed char *output,
                                             unsigned int *x_offset_r,
                                             unsigned int *y_offset_r,
                                             unsigned int *x_offset_g,
                                             unsigned int *y_offset_g,
                                             unsigned int *x_offset_b,
                                             unsigned int *y_offset_b,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned int *height,
                                             unsigned int *width,
                                             unsigned int *max_width,
                                             unsigned long long *batch_index,
                                             const unsigned int channel,
                                             unsigned int *inc,    // use width * height for pln and 1 for pkd
                                             unsigned int *dstinc, // use width * height for pln and 1 for pkd
                                             int in_plnpkdind,              // use 1 pln 3 for pkd
                                             int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int indextmp = 0;
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    output[dst_pix_idx] = input[src_pix_idx];
    output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
    output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

    int x_r, x_g, x_b, y_r, y_g, y_b;

    // R
    x_r = (id_x + x_offset_r[id_z]);
    y_r = (id_y + y_offset_r[id_z]);

    // G
    x_g = (id_x + x_offset_g[id_z]);
    y_g = (id_y + y_offset_g[id_z]);

    // B
    x_b = (id_x + x_offset_b[id_z]);
    y_b = (id_y + y_offset_b[id_z]);


    // R
    if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) && (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
    {
        char R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        indextmp = indextmp + 1;
        output[dst_pix_idx] = R;
        dst_pix_idx += dstinc[id_z];
    }

    // G
    if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) && (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
    {
        char G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        indextmp = indextmp + 1;
        output[dst_pix_idx] = G;
        dst_pix_idx += dstinc[id_z];
    }

    // B
    if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) && (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
    {
        char B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind + indextmp * inc[id_z]];
        output[dst_pix_idx] = B;
    }
}

RppStatus hip_exec_glitch_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(glitch_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.uintArr[0].uintmem,
                       handle_obj->mem.mgpu.uintArr[1].uintmem,
                       handle_obj->mem.mgpu.uintArr[2].uintmem,
                       handle_obj->mem.mgpu.uintArr[3].uintmem,
                       handle_obj->mem.mgpu.uintArr[4].uintmem,
                       handle_obj->mem.mgpu.uintArr[5].uintmem,
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

RppStatus hip_exec_glitch_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(glitch_batch_fp16,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
                    //    dstPtr,
                    //    handle_obj->mem.mgpu.uintArr[0].uintmem,
                    //    handle_obj->mem.mgpu.uintArr[1].uintmem,
                    //    handle_obj->mem.mgpu.uintArr[2].uintmem,
                    //    handle_obj->mem.mgpu.uintArr[3].uintmem,
                    //    handle_obj->mem.mgpu.uintArr[4].uintmem,
                    //    handle_obj->mem.mgpu.uintArr[5].uintmem,
                    //    handle_obj->mem.mgpu.roiPoints.x,
                    //    handle_obj->mem.mgpu.roiPoints.roiWidth,
                    //    handle_obj->mem.mgpu.roiPoints.y,
                    //    handle_obj->mem.mgpu.roiPoints.roiHeight,
                    //    handle_obj->mem.mgpu.srcSize.height,
                    //    handle_obj->mem.mgpu.srcSize.width,
                    //    handle_obj->mem.mgpu.maxSrcSize.width,
                    //    handle_obj->mem.mgpu.srcBatchIndex,
                    //    tensor_info._in_channels,
                    //    handle_obj->mem.mgpu.inc,
                    //    handle_obj->mem.mgpu.dstInc,
                    //    in_plnpkdind,
                    //    out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_glitch_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(glitch_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.uintArr[0].uintmem,
                       handle_obj->mem.mgpu.uintArr[1].uintmem,
                       handle_obj->mem.mgpu.uintArr[2].uintmem,
                       handle_obj->mem.mgpu.uintArr[3].uintmem,
                       handle_obj->mem.mgpu.uintArr[4].uintmem,
                       handle_obj->mem.mgpu.uintArr[5].uintmem,
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

RppStatus hip_exec_glitch_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(glitch_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.uintArr[0].uintmem,
                       handle_obj->mem.mgpu.uintArr[1].uintmem,
                       handle_obj->mem.mgpu.uintArr[2].uintmem,
                       handle_obj->mem.mgpu.uintArr[3].uintmem,
                       handle_obj->mem.mgpu.uintArr[4].uintmem,
                       handle_obj->mem.mgpu.uintArr[5].uintmem,
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