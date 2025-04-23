#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void erase_batch(unsigned char *input,
                                       unsigned char *output,
                                       unsigned int *box_info,
                                       unsigned char *colors,
                                       unsigned int *box_offset,
                                       unsigned int *no_of_boxes,
                                       unsigned int *src_height,
                                       unsigned int *src_width,
                                       unsigned int *max_width,
                                       unsigned long long *batch_index,
                                       unsigned int *src_inc,
                                       unsigned int *dst_inc,
                                       const int in_plnpkdind,
                                       const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uchar3 pixel;
    uint l_box_offset = box_offset[id_z];
    bool is_erase = false;

    for (int i = 0; i < no_of_boxes[id_z]; i++)
    {
        int temp = (l_box_offset + i) * 4;
        if (id_x >= box_info[temp] && id_x < box_info[temp + 2] && id_y >= box_info[temp + 1] && id_y < box_info[temp + 3])
        {
            is_erase = true;
            temp = (l_box_offset + i) * 3;
            pixel.x = colors[temp];
            pixel.y = colors[temp + 1];
            pixel.z = colors[temp + 2];
            break;
        }
    }
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if (is_erase == true)
    {
        output[dst_pix_idx] = pixel.x;
        dst_pix_idx += dst_inc[id_z];
        output[dst_pix_idx] = pixel.y;
        dst_pix_idx += dst_inc[id_z];
        output[dst_pix_idx] = pixel.z;
    }
    else
    {
        output[dst_pix_idx] = input[src_pix_idx];
        dst_pix_idx += dst_inc[id_z];
        src_pix_idx += src_inc[id_z];
        output[dst_pix_idx] = input[src_pix_idx];
        dst_pix_idx += dst_inc[id_z];
        src_pix_idx += src_inc[id_z];
        output[dst_pix_idx] = input[src_pix_idx];
    }
}

extern "C" __global__ void erase_pln1_batch(unsigned char *input,
                                            unsigned char *output,
                                            unsigned int *box_info,
                                            unsigned char *colors,
                                            unsigned int *box_offset,
                                            unsigned int *no_of_boxes,
                                            unsigned int *src_height,
                                            unsigned int *src_width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            unsigned int *src_inc,
                                            unsigned int *dst_inc,
                                            const int in_plnpkdind,
                                            const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uchar pixel;
    uint l_box_offset = box_offset[id_z];
    bool is_erase = false;

    for (int i = 0; i < no_of_boxes[id_z]; i++)
    {
        int temp = (l_box_offset + i) * 4;
        if (id_x >= box_info[temp] && id_x < box_info[temp + 2] && id_y >= box_info[temp + 1] && id_y < box_info[temp + 3])
        {
            is_erase = true;
            temp = (l_box_offset + i);
            pixel = colors[temp];
            break;
        }
    }

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if (is_erase == true)
    {
        output[dst_pix_idx] = pixel;
    }
    else
    {
        output[dst_pix_idx] = input[src_pix_idx];
    }
}

extern "C" __global__ void erase_batch_int8(signed char *input,
                                            signed char *output,
                                            unsigned int *box_info,
                                            signed char *colors,
                                            unsigned int *box_offset,
                                            unsigned int *no_of_boxes,
                                            unsigned int *src_height,
                                            unsigned int *src_width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            unsigned int *src_inc,
                                            unsigned int *dst_inc,
                                            const int in_plnpkdind,
                                            const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    char3 pixel;
    uint l_box_offset = box_offset[id_z];
    bool is_erase = false;

    for (int i = 0; i < no_of_boxes[id_z]; i++)
    {
        int temp = (l_box_offset + i) * 4;
        if (id_x >= box_info[temp] && id_x < box_info[temp + 2] && id_y >= box_info[temp + 1] && id_y < box_info[temp + 3])
        {
            is_erase = true;
            temp = (l_box_offset + i) * 3;
            pixel.x = colors[temp];
            pixel.y = colors[temp + 1];
            pixel.z = colors[temp + 2];
            break;
        }
    }

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if (is_erase == true)
    {
        output[dst_pix_idx] = pixel.x;
        dst_pix_idx += dst_inc[id_z];
        output[dst_pix_idx] = pixel.y;
        dst_pix_idx += dst_inc[id_z];
        output[dst_pix_idx] = pixel.z;
    }
    else
    {
        output[dst_pix_idx] = input[src_pix_idx];
        dst_pix_idx += dst_inc[id_z];
        src_pix_idx += src_inc[id_z];
        output[dst_pix_idx] = input[src_pix_idx];
        dst_pix_idx += dst_inc[id_z];
        src_pix_idx += src_inc[id_z];
        output[dst_pix_idx] = input[src_pix_idx];
    }
}

extern "C" __global__ void erase_pln1_batch_int8(char *input,
                                                 char *output,
                                                 unsigned int *box_info,
                                                 char *colors,
                                                 unsigned int *box_offset,
                                                 unsigned int *no_of_boxes,
                                                 unsigned int *src_height,
                                                 unsigned int *src_width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 unsigned int *src_inc,
                                                 unsigned int *dst_inc,
                                                 const int in_plnpkdind,
                                                 const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    char pixel;
    uint l_box_offset = box_offset[id_z];
    bool is_erase = false;

    for (int i = 0; i < no_of_boxes[id_z]; i++)
    {
        int temp = (l_box_offset + i) * 4;
        if (id_x >= box_info[temp] && id_x < box_info[temp + 2] && id_y >= box_info[temp + 1] && id_y < box_info[temp + 3])
        {
            is_erase = true;
            temp = (l_box_offset + i);
            pixel = colors[temp];
            break;
        }
    }

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if (is_erase == true)
    {
        output[dst_pix_idx] = pixel;
    }
    else
    {
        output[dst_pix_idx] = input[src_pix_idx];
    }
}

extern "C" __global__ void erase_batch_fp32(float *input,
                                            float *output,
                                            unsigned int *box_info,
                                            float *colors,
                                            unsigned int *box_offset,
                                            unsigned int *no_of_boxes,
                                            unsigned int *src_height,
                                            unsigned int *src_width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            unsigned int *src_inc,
                                            unsigned int *dst_inc,
                                            const int in_plnpkdind,
                                            const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float3 pixel;
    uint l_box_offset = box_offset[id_z];
    bool is_erase = false;

    for (int i = 0; i < no_of_boxes[id_z]; i++)
    {
        int temp = (l_box_offset + i) * 4;
        if (id_x >= box_info[temp] && id_x < box_info[temp + 2] && id_y >= box_info[temp + 1] && id_y < box_info[temp + 3])
        {
            is_erase = true;
            temp = (l_box_offset + i) * 3;
            pixel.x = colors[temp];
            pixel.y = colors[temp + 1];
            pixel.z = colors[temp + 2];
            break;
        }
    }
    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if (is_erase == true)
    {
        output[dst_pix_idx] = pixel.x;
        dst_pix_idx += dst_inc[id_z];
        output[dst_pix_idx] = pixel.y;
        dst_pix_idx += dst_inc[id_z];
        output[dst_pix_idx] = pixel.z;
    }
    else
    {
        output[dst_pix_idx] = input[src_pix_idx];
        dst_pix_idx += dst_inc[id_z];
        src_pix_idx += src_inc[id_z];
        output[dst_pix_idx] = input[src_pix_idx];
        dst_pix_idx += dst_inc[id_z];
        src_pix_idx += src_inc[id_z];
        output[dst_pix_idx] = input[src_pix_idx];
    }
}

extern "C" __global__ void erase_pln1_batch_fp32(float *input,
                                                 float *output,
                                                 unsigned int *box_info,
                                                 float *colors,
                                                 unsigned int *box_offset,
                                                 unsigned int *no_of_boxes,
                                                 unsigned int *src_height,
                                                 unsigned int *src_width,
                                                 unsigned int *max_width,
                                                 unsigned long long *batch_index,
                                                 unsigned int *src_inc,
                                                 unsigned int *dst_inc,
                                                 const int in_plnpkdind,
                                                 const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float pixel;
    uint l_box_offset = box_offset[id_z];
    bool is_erase = false;

    for (int i = 0; i < no_of_boxes[id_z]; i++)
    {
        int temp = (l_box_offset + i) * 4;
        if (id_x >= box_info[temp] && id_x < box_info[temp + 2] && id_y >= box_info[temp + 1] && id_y < box_info[temp + 3])
        {
            is_erase = true;
            temp = (l_box_offset + i);
            pixel = colors[temp];
            break;
        }
    }

    unsigned long src_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
    unsigned long dst_pix_idx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

    if (is_erase == true)
    {
        output[dst_pix_idx] = pixel;
    }
    else
    {
        output[dst_pix_idx] = input[src_pix_idx];
    }
}

// extern "C" __global__ void erase_batch_fp16(
//     half *input, half *output,
//     unsigned int *box_info, half *colors,
//     unsigned int *box_offset, unsigned int *no_of_boxes,
//     unsigned int *src_height, unsigned int *src_width,
//     unsigned int *max_width, unsigned long *batch_index,
//     unsigned int *src_inc, unsigned int *dst_inc,
//     const int in_plnpkdind, const int out_plnpkdind) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   half3 pixel;
//   uint l_box_offset = box_offset[id_z];
//   bool is_erase = false;
//   for (int i = 0; i < no_of_boxes[id_z]; i++) {
//     int temp = (l_box_offset + i) * 4;
//     if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
//         id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
//       is_erase = true;
//       temp = (l_box_offset + i) * 3;
//       pixel.x = colors[temp];
//       pixel.y = colors[temp + 1];
//       pixel.z = colors[temp + 2];
//       break;
//     }
//   }
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

//   if (is_erase == true)
//   {
//     output[dst_pix_idx] = pixel.x;
//     dst_pix_idx += dst_inc[id_z];
//     output[dst_pix_idx] = pixel.y;
//     dst_pix_idx += dst_inc[id_z];
//     output[dst_pix_idx] = pixel.z;
//   }
//   else
//   {
//     output[dst_pix_idx] = input[src_pix_idx];
//     dst_pix_idx += dst_inc[id_z];
//     src_pix_idx += src_inc[id_z];
//     output[dst_pix_idx] = input[src_pix_idx];
//     dst_pix_idx += dst_inc[id_z];
//     src_pix_idx += src_inc[id_z];
//     output[dst_pix_idx] = input[src_pix_idx];
//   }
// }

// extern "C" __global__ void erase_pln1_batch_fp16(
//     half *input, half *output,
//     unsigned int *box_info, half *colors,
//     unsigned int *box_offset, unsigned int *no_of_boxes,
//     unsigned int *src_height, unsigned int *src_width,
//     unsigned int *max_width, unsigned long *batch_index,
//     unsigned int *src_inc, unsigned int *dst_inc,
//     const int in_plnpkdind, const int out_plnpkdind) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   half pixel;
//   uint l_box_offset = box_offset[id_z];
//   bool is_erase = false;
//   for (int i = 0; i < no_of_boxes[id_z]; i++) {
//     int temp = (l_box_offset + i) * 4;
//     if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
//         id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
//       is_erase = true;
//       temp = (l_box_offset + i);
//       pixel = colors[temp];
//       break;
//     }
//   }
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

//   if (is_erase == true)
//   {
//     output[dst_pix_idx] = pixel;
//   }
//   else
//   {
//     output[dst_pix_idx] = input[src_pix_idx];
//   }
// }

RppStatus hip_exec_erase_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32u* anchor_box_info, Rpp8u* colors, rpp::Handle &handle, Rpp32u* box_offset, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(erase_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       anchor_box_info,
                       colors,
                       box_offset,
                       handle_obj->mem.mgpu.uintArr[0].uintmem,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_erase_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, Rpp32u* anchor_box_info, Rpp16f* colors, rpp::Handle &handle, Rpp32u* box_offset, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(erase_batch_fp16,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
                        //   dstPtr,
                        //   anchor_box_info,
                        //   colors,
                        //   box_offset,
                        //   handle_obj->mem.mgpu.uintArr[0].uintmem,
                        //   handle_obj->mem.mgpu.srcSize.height,
                        //   handle_obj->mem.mgpu.srcSize.width,
                        //   handle_obj->mem.mgpu.maxSrcSize.width,
                        //   handle_obj->mem.mgpu.srcBatchIndex,
                        //   handle_obj->mem.mgpu.inc,
                        //   handle_obj->mem.mgpu.dstInc,
                        //   in_plnpkdind,
                        //   out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_erase_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, Rpp32u* anchor_box_info, Rpp32f* colors, rpp::Handle &handle, Rpp32u* box_offset, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(erase_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       anchor_box_info,
                       colors,
                       box_offset,
                       handle_obj->mem.mgpu.uintArr[0].uintmem,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_erase_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp32u* anchor_box_info, Rpp8s* colors, rpp::Handle &handle, Rpp32u* box_offset, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();

    hipLaunchKernelGGL(erase_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       anchor_box_info,
                       colors,
                       box_offset,
                       handle_obj->mem.mgpu.uintArr[0].uintmem,
                       handle_obj->mem.mgpu.srcSize.height,
                       handle_obj->mem.mgpu.srcSize.width,
                       handle_obj->mem.mgpu.maxSrcSize.width,
                       handle_obj->mem.mgpu.srcBatchIndex,
                       handle_obj->mem.mgpu.inc,
                       handle_obj->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);
    return RPP_SUCCESS;
}