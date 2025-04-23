#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void water_batch(unsigned char *input,
                                       unsigned char *output,
                                       float *ampl_x,
                                       float *ampl_y,
                                       float *freq_x,
                                       float *freq_y,
                                       float *phase_x,
                                       float *phase_y,
                                       unsigned int *xroi_begin,
                                       unsigned int *xroi_end,
                                       unsigned int *yroi_begin,
                                       unsigned int *yroi_end,
                                       unsigned int *height,
                                       unsigned int *width,
                                       unsigned int *max_height,
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

    float water_wave_x, water_wave_y;

    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];

    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int) water_wave_x + (int) water_wave_y * max_width[id_z]) * in_plnpkdind;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                output[dst_pix_id] = input[src_pix_id];
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_id] = input[dst_pix_id];
            dst_pix_id += dst_inc[id_z];
        }
    }
}

extern "C" __global__ void water_batch_fp32(float *input,
                                            float *output,
                                            float *ampl_x,
                                            float *ampl_y,
                                            float *freq_x,
                                            float *freq_y,
                                            float *phase_x,
                                            float *phase_y,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_height,
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

    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];

    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int) water_wave_x + (int) water_wave_y * max_width[id_z]) * in_plnpkdind;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                output[dst_pix_id] = input[src_pix_id];
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_id] = input[dst_pix_id];
            dst_pix_id += dst_inc[id_z];
        }
    }
}

// extern "C" __global__ void water_batch_fp16(  half* input,
//                             half* output,
//                             float *ampl_x,
//                             float *ampl_y,
//                             float *freq_x,
//                             float *freq_y,
//                             float *phase_x,
//                             float *phase_y,
//                             int *xroi_begin,
//                             int *xroi_end,
//                             int *yroi_begin,
//                             int *yroi_end,
//                             unsigned int *height,
//                             unsigned int *width,
//                             unsigned int *max_height,
//                             unsigned int *max_width,
//                             unsigned long *batch_index,
//                             const unsigned int channel,
//                             unsigned int *src_inc,
//                             unsigned int *dst_inc, // use width * height for pln and 1 for pkd
//                             const int in_plnpkdind,
//                             const int out_plnpkdind // use 1 pln 3 for pkd
//                             )
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//     half valuergb;
//     float water_wave_x, water_wave_y;
//     float ampl_x_temp = ampl_x[id_z];
//     float ampl_y_temp = ampl_y[id_z];
//     float freq_x_temp = freq_x[id_z];
//     float freq_y_temp = freq_y[id_z];
//     float phase_x_temp = phase_x[id_z];
//     float phase_y_temp = phase_y[id_z];
//     int img_width = width[id_z];

//     int img_height = height[id_z];
//     int indextmp=0;
//     long dst_pix_id = 0;
//     dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
//     long src_pix_id = 0;
//     water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
//     water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
//     src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
//     if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
//     {
//         for(indextmp = 0; indextmp < channel; indextmp++)
//         {
//             if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
//             {
//                 valuergb = input[src_pix_id];
//                 output[dst_pix_id] = valuergb;
//             }
//             dst_pix_id += dst_inc[id_z];
//             src_pix_id += src_inc[id_z];
//         }
//     }
//     else if((id_x < width[id_z] ) && (id_y < height[id_z])){
//             for(indextmp = 0; indextmp < channel; indextmp++){
//                 output[dst_pix_id] = input[dst_pix_id];
//                 dst_pix_id += dst_inc[id_z];
//             }
//         }
// }

extern "C" __global__ void water_batch_int8(signed char *input,
                                            signed char *output,
                                            float *ampl_x,
                                            float *ampl_y,
                                            float *freq_x,
                                            float *freq_y,
                                            float *phase_x,
                                            float *phase_y,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_height,
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

    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];

    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int) water_wave_x + (int) water_wave_y * max_width[id_z]) * in_plnpkdind;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                output[dst_pix_id] = input[src_pix_id];
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pix_id] = input[dst_pix_id];
            dst_pix_id += dst_inc[id_z];
        }
    }
}

RppStatus hip_exec_water_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(water_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.floatArr[0].floatmem,
                       handle_obj->mem.mgpu.floatArr[1].floatmem,
                       handle_obj->mem.mgpu.floatArr[2].floatmem,
                       handle_obj->mem.mgpu.floatArr[3].floatmem,
                       handle_obj->mem.mgpu.floatArr[4].floatmem,
                       handle_obj->mem.mgpu.floatArr[5].floatmem,
                       handle_obj->mem.mgpu.roiPoints.x,
                       handle_obj->mem.mgpu.roiPoints.roiWidth,
                       handle_obj->mem.mgpu.roiPoints.y,
                       handle_obj->mem.mgpu.roiPoints.roiHeight,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.height,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       tensor_info._in_channels,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_water_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(water_batch_fp16,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
                        //   dstPtr,
                        //   handle_obj->mem.mgpu.floatArr[0].floatmem,
                        //   handle_obj->mem.mgpu.floatArr[1].floatmem,
                        //   handle_obj->mem.mgpu.floatArr[2].floatmem,
                        //   handle_obj->mem.mgpu.floatArr[3].floatmem,
                        //   handle_obj->mem.mgpu.floatArr[4].floatmem,
                        //   handle_obj->mem.mgpu.floatArr[5].floatmem,
                        //   handle_obj->mem.mgpu.roiPoints.x,
                        //   handle_obj->mem.mgpu.roiPoints.roiWidth,
                        //   handle_obj->mem.mgpu.roiPoints.y,
                        //   handle_obj->mem.mgpu.roiPoints.roiHeight,
                        //   handle_obj->mem.mgpu.srcSize.height,
                        //   handle_obj->mem.mgpu.srcSize.width,
                        //   handle_obj->mem.mgpu.maxSrcSize.height,
                        //   handle_obj->mem.mgpu.maxSrcSize.width,
                        //   handle_obj->mem.mgpu.srcBatchIndex,
                        //   tensor_info._in_channels,
                        //   handle_obj->mem.mgpu.inc,
                        //   handle_obj->mem.mgpu.dstInc,
                        //   in_plnpkdind,
                        //   out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_water_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(water_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.floatArr[0].floatmem,
                       handle_obj->mem.mgpu.floatArr[1].floatmem,
                       handle_obj->mem.mgpu.floatArr[2].floatmem,
                       handle_obj->mem.mgpu.floatArr[3].floatmem,
                       handle_obj->mem.mgpu.floatArr[4].floatmem,
                       handle_obj->mem.mgpu.floatArr[5].floatmem,
                       handle_obj->mem.mgpu.roiPoints.x,
                       handle_obj->mem.mgpu.roiPoints.roiWidth,
                       handle_obj->mem.mgpu.roiPoints.y,
                       handle_obj->mem.mgpu.roiPoints.roiHeight,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.height,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       tensor_info._in_channels,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_water_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(water_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle_obj->mem.mgpu.floatArr[0].floatmem,
                       handle_obj->mem.mgpu.floatArr[1].floatmem,
                       handle_obj->mem.mgpu.floatArr[2].floatmem,
                       handle_obj->mem.mgpu.floatArr[3].floatmem,
                       handle_obj->mem.mgpu.floatArr[4].floatmem,
                       handle_obj->mem.mgpu.floatArr[5].floatmem,
                       handle_obj->mem.mgpu.roiPoints.x,
                       handle_obj->mem.mgpu.roiPoints.roiWidth,
                       handle_obj->mem.mgpu.roiPoints.y,
                       handle_obj->mem.mgpu.roiPoints.roiHeight,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.height,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       tensor_info._in_channels,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);
    return RPP_SUCCESS;
}