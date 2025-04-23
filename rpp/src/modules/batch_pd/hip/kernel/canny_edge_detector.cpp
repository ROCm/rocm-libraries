#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define RPPABS(a) ((a < 0) ? (-a) : (a))
#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void canny_ced_pln3_to_pln1(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int IPpixIdx = id_x + id_y * width;
    int ch = height * width;
    float value = ((input[IPpixIdx] + input[IPpixIdx + ch] + input[IPpixIdx + ch * 2]) / 3);
    output[IPpixIdx] = (unsigned char)value;
}

extern "C" __global__ void canny_ced_pkd3_to_pln1(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int OPpixIdx = id_x + id_y * width;
    int IPpixIdx = id_x * channel + id_y * width * channel;
    float value = (input[IPpixIdx] + input[IPpixIdx + 1] + input[IPpixIdx + 2]) / 3;
    output[OPpixIdx] = (unsigned char)value;
}

extern "C" __global__ void canny_ced_pln1_to_pln3(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int IPpixIdx = id_x + id_y * width;
    int ch = height * width;
    output[IPpixIdx] = input[IPpixIdx];
    output[IPpixIdx + ch] = input[IPpixIdx];
    output[IPpixIdx + ch * 2] = input[IPpixIdx];
}

extern "C" __global__ void canny_ced_pln1_to_pkd3(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx = id_x * channel + id_y * width * channel;
    output[OPpixIdx] = input[IPpixIdx];
    output[OPpixIdx + 1] = input[IPpixIdx];
    output[OPpixIdx + 2] = input[IPpixIdx];
}

extern "C" __global__ void ced_pln3_to_pln1_batch(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel,
                                                  const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    int ch = height * width;
    float value = ((input[IPpixIdx] + input[IPpixIdx + ch] + input[IPpixIdx + ch * 2]) / 3);
    output[OPpixIdx] = (unsigned char)value;
}

extern "C" __global__ void ced_pkd3_to_pln1_batch(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel,
                                                  const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x * (unsigned long)channel + (unsigned long)id_y * (unsigned long)width * (unsigned long)channel;
    float value = (input[IPpixIdx] + input[IPpixIdx + 1] + input[IPpixIdx + 2]) / 3;
    output[OPpixIdx] = (unsigned char)value;
}

extern "C" __global__ void ced_pln1_to_pln3_batch(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel,
                                                  const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    unsigned long IPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    int ch = height * width;
    output[OPpixIdx] = input[IPpixIdx];
    output[OPpixIdx + ch] = input[IPpixIdx];
    output[OPpixIdx + ch * 2] = input[IPpixIdx];
}

extern "C" __global__ void ced_pln1_to_pkd3_batch(unsigned char *input,
                                                  unsigned char *output,
                                                  const unsigned int height,
                                                  const unsigned int width,
                                                  const unsigned int channel,
                                                  const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    unsigned long IPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x * (unsigned long)channel + (unsigned long)id_y * (unsigned long)width * (unsigned long)channel;
    output[OPpixIdx] = input[IPpixIdx];
    output[OPpixIdx + 1] = input[IPpixIdx];
    output[OPpixIdx + 2] = input[IPpixIdx];
}

extern "C" __global__ void ced_non_max_suppression(unsigned char *input,
                                                   unsigned char *input1,
                                                   unsigned char *input2,
                                                   unsigned char *output,
                                                   const unsigned int height,
                                                   const unsigned int width,
                                                   const unsigned int channel,
                                                   const unsigned char min,
                                                   const unsigned char max)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    float gradient = atan((float)input1[pixIdx] / (float)input2[pixIdx]);
    unsigned char pixel1, pixel2;

    if (RPPABS(gradient) > 1.178097)
    {
        if(id_x != 0)
            pixel1 = input[pixIdx - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1)
            pixel2 = input[pixIdx + 1];
        else
            pixel2 = 0;
    }
    else if (gradient > 0.392699)
    {
        if(id_x != 0 && id_y !=0)
            pixel1 = input[pixIdx - width - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1 && id_y != height - 1)
            pixel2 = input[pixIdx + width + 1];
        else
            pixel2 = 0;
    }
    else if (gradient < -0.392699)
    {
        if(id_x != width - 1 && id_y !=0)
            pixel1 = input[pixIdx - width + 1];
        else
            pixel1 = 0;

        if(id_x != 0 && id_y != height - 1)
            pixel2 = input[pixIdx + width - 1];
        else
            pixel2 = 0;
    }
    else
    {
        if(id_y != 0)
            pixel1 = input[pixIdx - width];
        else
            pixel1 = 0;

        if(id_y != height - 1)
            pixel2 = input[pixIdx + width];
        else
            pixel2 = 0;
    }

    if(input[pixIdx] >= pixel1 && input[pixIdx] >= pixel2)
    {
        if(input[pixIdx] >= max)
            output[pixIdx] = 255;
        else if(input[pixIdx] <= min)
            output[pixIdx] = 0;
        else
            output[pixIdx] = 128;
    }
    else
        output[pixIdx] = 0;
}

extern "C" __global__ void ced_non_max_suppression_batch(unsigned char *input,
                                                         unsigned char *input1,
                                                         unsigned char *input2,
                                                         unsigned char *output,
                                                         const unsigned int height,
                                                         const unsigned int width,
                                                         const unsigned int channel,
                                                         const unsigned char min,
                                                         const unsigned char max)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    float gradient = atan((float)input1[pixIdx] / (float)input2[pixIdx]);
    unsigned char pixel1, pixel2;

    if (RPPABS(gradient) > 1.178097)
    {
        if(id_x != 0)
            pixel1 = input[pixIdx - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1)
            pixel2 = input[pixIdx + 1];
        else
            pixel2 = 0;
    }
    else if (gradient > 0.392699)
    {
        if(id_x != 0 && id_y !=0)
            pixel1 = input[pixIdx - width - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1 && id_y != height - 1)
            pixel2 = input[pixIdx + width + 1];
        else
            pixel2 = 0;
    }
    else if (gradient < -0.392699)
    {
        if(id_x != width - 1 && id_y !=0)
            pixel1 = input[pixIdx - width + 1];
        else
            pixel1 = 0;

        if(id_x != 0 && id_y != height - 1)
            pixel2 = input[pixIdx + width - 1];
        else
            pixel2 = 0;
    }
    else
    {
        if(id_y != 0)
            pixel1 = input[pixIdx - width];
        else
            pixel1 = 0;

        if(id_y != height - 1)
            pixel2 = input[pixIdx + width];
        else
            pixel2 = 0;
    }

    if(input[pixIdx] >= pixel1 && input[pixIdx] >= pixel2)
    {
        if(input[pixIdx] >= max)
            output[pixIdx] = 255;
        else if(input[pixIdx] <= min)
            output[pixIdx] = 0;
        else
            output[pixIdx] = 128;
    }
    else
        output[pixIdx] = 0;
}

extern "C" __global__ void canny_edge(unsigned char *input,
                                      unsigned char *output,
                                      const unsigned int height,
                                      const unsigned int width,
                                      const unsigned int channel,
                                      const unsigned char min,
                                      const unsigned char max)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int pixIdx = id_y * width + id_x + id_z * width * height;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    if(input[pixIdx] == 0 || input[pixIdx] == 255)
    {
        output[pixIdx] = input[pixIdx];
    }
    else
    {
        for(int i = -1; i <= 1; i++)
        {
            for(int j = -1; j <= 1; j++)
            {
                if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
                {
                    unsigned int index = pixIdx + j + (i * width);
                    if(input[index] == 255)
                    {
                        output[pixIdx] = 255;
                        break;
                    }
                }
            }
        }
    }
}

extern "C" __global__ void canny_edge_batch(unsigned char *input,
                                            unsigned char *output,
                                            const unsigned int height,
                                            const unsigned int width,
                                            const unsigned int channel,
                                            const unsigned char min,
                                            const unsigned char max,
                                            const unsigned long batchIndex,
                                            const unsigned int originalChannel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    unsigned long IPpixIdx, OPpixIdx;
    IPpixIdx = (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;

    if(originalChannel == 1)
    {
        OPpixIdx = (unsigned long)batchIndex + (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
    }
    else
    {
        OPpixIdx = (unsigned long)IPpixIdx;
    }

    if(input[IPpixIdx] == 0 || input[IPpixIdx] == 255)
    {
        output[OPpixIdx] = input[IPpixIdx];
    }
    else
    {
        for(int i = -1; i <= 1; i++)
        {
            for(int j = -1; j <= 1; j++)
            {
                if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
                {
                    unsigned long index = (unsigned long)IPpixIdx + (unsigned long)j + ((unsigned long)i * (unsigned long)width);
                    if(input[index] == 255)
                    {
                        output[OPpixIdx] = 255;
                        break;
                    }
                }
            }
        }
    }
}

RppStatus hip_exec_canny_ced_pln3_to_pln1(Rpp8u *srcPtr, Rpp8u *gsin, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(canny_ced_pln3_to_pln1,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       gsin,
                       height,
                       width,
                       channel);

    return RPP_SUCCESS;
}

RppStatus hip_exec_canny_ced_pkd3_to_pln1(Rpp8u *srcPtr, Rpp8u *gsin, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(canny_ced_pkd3_to_pln1,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       gsin,
                       height,
                       width,
                       channel);

    return RPP_SUCCESS;
}

RppStatus hip_exec_canny_ced_pln1_to_pkd3(Rpp8u *gsout, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(canny_ced_pln1_to_pkd3,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       gsout,
                       dstPtr,
                       height,
                       width,
                       channel);

    return RPP_SUCCESS;
}

RppStatus hip_exec_canny_ced_pln1_to_pln3(Rpp8u *gsout, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(canny_ced_pln1_to_pln3,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       gsout,
                       dstPtr,
                       height,
                       width,
                       channel);

    return RPP_SUCCESS;
}

RppStatus hip_exec_ced_non_max_suppression(Rpp8u *srcPtr, Rpp8u *sobelX, Rpp8u *sobelY, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(ced_non_max_suppression,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       sobelX,
                       sobelY,
                       dstPtr,
                       height,
                       width,
                       channel,
                       handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                       handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);

    return RPP_SUCCESS;
}

RppStatus hip_exec_canny_edge(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(canny_edge,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       height,
                       width,
                       channel,
                       handle.GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem[i],
                       handle.GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem[i]);

    return RPP_SUCCESS;
}