#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void color_cast_batch(unsigned char *input,
                                            unsigned char *output,
                                            unsigned char *user_input_r,  //which color to cast for red
                                            unsigned char *user_input_g,  //which color to cast for green
                                            unsigned char *user_input_b,  //which color to cast for blue
                                            float *alpha,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc,  // use width * height for pln and 1 for pkd
                                            unsigned int *dstinc , // use width * height for pln and 1 for pkd
                                            int in_plnpkdind,         // use 1 pln 3 for pkd
                                            int out_plnpkdind)
{

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned char user_input[3] = {user_input_r[id_z], user_input_g[id_z], user_input_b[id_z]};
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
    float alphatmp=alpha[id_z];

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for (int indextmp = channel - 1; indextmp >= 0; indextmp--)
        {
            unsigned char input_pixel1 = input[src_pix_idx];
            unsigned char input_pixel2 = user_input[indextmp];
            output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);
            src_pix_idx += inc[id_z];
            dst_pix_idx += dstinc[id_z];
        }
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = 0;
            dst_pix_idx += dstinc[id_z];
        }
    }
}

// extern "C" __global__ void color_cast_batch_fp16(
//     half *input,
//     half *output,
//     unsigned char* user_input_r,  //which color to cast for red
//     unsigned char* user_input_g,  //which color to cast for green
//     unsigned char* user_input_b,  //which color to cast for blue
//     float *alpha,
//     int *xroi_begin,
//     int *xroi_end,
//     int *yroi_begin,
//     int *yroi_end,
//     unsigned int *height,
//     unsigned int *width,
//     unsigned int *max_width,
//     unsigned long *batch_index,
//     const unsigned int channel,
//     unsigned int *inc,  // use width * height for pln and 1 for pkd
//     unsigned int *dstinc , // use width * height for pln and 1 for pkd
//     int in_plnpkdind,         // use 1 pln 3 for pkd
//     int out_plnpkdind
// ) {

//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//   half user_input[3]={(half)(user_input_r[id_z]/255.0 ),(half)(user_input_g[id_z]/255.0) ,(half)(user_input_b[id_z]/255.0) };
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

//   float alphatmp=alpha[id_z];

//   if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
//       (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {

//     for (int indextmp = channel - 1; indextmp >= 0; indextmp--) {
//       half input_pixel1 = input[src_pix_idx];
//       half input_pixel2 = user_input[indextmp];
//       output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

//       src_pix_idx += inc[id_z];
//       dst_pix_idx += dstinc[id_z];
//     }
//   } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pix_idx] = 0;
//       dst_pix_idx += dstinc[id_z];
//     }
//   }
// }

extern "C" __global__ void color_cast_batch_fp32(float *input,
                                                 float *output,
                                                 unsigned char *user_input_r,  //which color to cast for red
                                                 unsigned char *user_input_g,  //which color to cast for green
                                                 unsigned char *user_input_b,  //which color to cast for blue
                                                 float *alpha,
                                                 unsigned int *xroi_begin,
                                                 unsigned int *xroi_end,
                                                 unsigned int *yroi_begin,
                                                 unsigned int *yroi_end,
                                                 unsigned int *height,
                                                 unsigned int *width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 const unsigned int channel,
                                                 unsigned int *inc,  // use width * height for pln and 1 for pkd
                                                 unsigned int *dstinc , // use width * height for pln and 1 for pkd
                                                 int in_plnpkdind,         // use 1 pln 3 for pkd
                                                 int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float divisor = 255;
    float user_input[3] = {user_input_r[id_z] / divisor, user_input_g[id_z] / divisor, user_input_b[id_z] / divisor};
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
    float alphatmp=alpha[id_z];

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for (int indextmp = channel - 1; indextmp >= 0; indextmp--)
        {
            float input_pixel1 = input[src_pix_idx];
            float input_pixel2 = user_input[indextmp];
            output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);
            src_pix_idx += inc[id_z];
            dst_pix_idx += dstinc[id_z];
        }
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = 0;
            dst_pix_idx += dstinc[id_z];
        }
    }
}

extern "C" __global__ void color_cast_batch_int8(signed char *input,
                                                 signed char *output,
                                                 unsigned char* user_input_r,  //which color to cast for red
                                                 unsigned char* user_input_g,  //which color to cast for green
                                                 unsigned char* user_input_b,  //which color to cast for blue
                                                 float *alpha,
                                                 unsigned int *xroi_begin,
                                                 unsigned int *xroi_end,
                                                 unsigned int *yroi_begin,
                                                 unsigned int *yroi_end,
                                                 unsigned int *height,
                                                 unsigned int *width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 const unsigned int channel,
                                                 unsigned int *inc,  // use width * height for pln and 1 for pkd
                                                 unsigned int *dstinc , // use width * height for pln and 1 for pkd
                                                 int in_plnpkdind,         // use 1 pln 3 for pkd
                                                 int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int subtrahend = 128;
    char user_input[3] = {(char) (user_input_r[id_z] - subtrahend), (char) (user_input_g[id_z] - subtrahend), (char) (user_input_b[id_z] - subtrahend)};
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    float alphatmp = alpha[id_z];

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for (int indextmp = channel - 1; indextmp >= 0; indextmp--)
        {
            char input_pixel1 = input[src_pix_idx];
            char input_pixel2 = user_input[indextmp];
            output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);
            src_pix_idx += inc[id_z];
            dst_pix_idx += dstinc[id_z];
        }
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_idx] = 0;
            dst_pix_idx += dstinc[id_z];
        }
    }
}

RppStatus hip_exec_color_cast_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(color_cast_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.ucharArr[0].ucharmem,
                       handle_obj->mem.mgpu.ucharArr[1].ucharmem,
                       handle_obj->mem.mgpu.ucharArr[2].ucharmem,
                       handle_obj->mem.mgpu.floatArr[3].floatmem,
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

RppStatus hip_exec_color_cast_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(color_cast_batch_fp16,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
                    //    dstPtr,
                    //    handle_obj->mem.mgpu.ucharArr[0].ucharmem,
                    //    handle_obj->mem.mgpu.ucharArr[1].ucharmem,
                    //    handle_obj->mem.mgpu.ucharArr[2].ucharmem,
                    //    handle_obj->mem.mgpu.floatArr[3].floatmem,
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

RppStatus hip_exec_color_cast_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(color_cast_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.ucharArr[0].ucharmem,
                       handle_obj->mem.mgpu.ucharArr[1].ucharmem,
                       handle_obj->mem.mgpu.ucharArr[2].ucharmem,
                       handle_obj->mem.mgpu.floatArr[3].floatmem,
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

RppStatus hip_exec_color_cast_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(color_cast_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.ucharArr[0].ucharmem,
                       handle_obj->mem.mgpu.ucharArr[1].ucharmem,
                       handle_obj->mem.mgpu.ucharArr[2].ucharmem,
                       handle_obj->mem.mgpu.floatArr[3].floatmem,
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