
#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void look_up_table_pkd(unsigned char *input,
                                             unsigned char *output,
                                             unsigned char *lutPtr,
                                             const unsigned int height,
                                             const unsigned int width,
                                             const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int index = input[pixIdx] * channel + id_z;
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;
}

extern "C" __global__ void look_up_table_pln(unsigned char *input,
                                             unsigned char *output,
                                             unsigned char *lutPtr,
                                             const unsigned int height,
                                             const unsigned int width,
                                             const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int index = input[pixIdx] + id_z * 256;
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;
}

extern "C" __global__ void look_up_table_batch(unsigned char *input,
                                               unsigned char *output,
                                               unsigned char *lutPtr,
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
                                               const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x < width[id_z] && id_y < height[id_z])
    {
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

        if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            for (int indextmp = 0; indextmp < channel; indextmp++)
            {
                int luptrIndex = (id_z * channel * 256) + (input[pixIdx] * plnpkdindex);
                output[pixIdx] = saturate_8u((int) lutPtr[luptrIndex]);
                pixIdx += inc[id_z];
            }
        }
        else if ((id_x < width[id_z]) && (id_y < height[id_z]))
        {
            for (int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}

extern "C" __global__ void lut_batch(unsigned char *input,
                                     unsigned char *output,
                                     unsigned char *lutPtr,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long long *batch_index,
                                     const unsigned int channel,
                                     unsigned int *inc,
                                     unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                     const int in_pln_pkd_ind,
                                     const int out_pln_pkd_ind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x < width[id_z] && id_y < height[id_z])
    {
        long in_pix_index = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_pln_pkd_ind;
        long out_pix_index = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_pln_pkd_ind;
        int luptrIndex = id_z << 8;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            int lutIndex = luptrIndex + input[in_pix_index];
            output[out_pix_index] = lutPtr[lutIndex];
            in_pix_index += inc[id_z];
            out_pix_index += dst_inc[id_z];
        }
    }
}

extern "C" __global__ void lut_batch_int8(signed char *input,
                                          signed char *output,
                                          signed char *lutPtr,
                                          unsigned int *height,
                                          unsigned int *width,
                                          unsigned int *max_width,
                                          unsigned long long *batch_index,
                                          const unsigned int channel,
                                          unsigned int *inc,
                                          unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                                          const int in_pln_pkd_ind,
                                          const int out_pln_pkd_ind) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x < width[id_z] && id_y < height[id_z])
    {
        long in_pix_index = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_pln_pkd_ind;
        long out_pix_index = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_pln_pkd_ind;
        int luptrIndex = id_z << 8;

        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            int lutIndex = luptrIndex + input[in_pix_index] + 128;
            output[out_pix_index] = lutPtr[lutIndex];
            in_pix_index += inc[id_z];
            out_pix_index += dst_inc[id_z];
        }
    }
}

RppStatus hip_exec_look_up_table_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp8u *hipLutPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(look_up_table_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       hipLutPtr,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_lut_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp8u* lut, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(lut_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       lut,
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

RppStatus hip_exec_lut_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp8s* lut, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(lut_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       lut,
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