#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void laplacian_image_pyramid_pkd(unsigned char *input,
                                                       unsigned char *output,
                                                       const unsigned int height,
                                                       const unsigned int width,
                                                       const unsigned int channel,
                                                       float *kernelArray,
                                                       const unsigned int kernelHeight,
                                                       const unsigned int kernelWidth)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel + id_z;

    int boundx = (kernelWidth - 1) / 2;
    int boundy = (kernelHeight - 1) / 2;
    int sum = 0;
    int counter = 0;

    for(int i = -boundy; i <= boundy; i++)
    {
        for(int j = -boundx; j <= boundx; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = input[pixIdx] - saturate_8u(sum);
}

extern "C" __global__ void laplacian_image_pyramid_pln(unsigned char *input,
                                                       unsigned char *output,
                                                       const unsigned int height,
                                                       const unsigned int width,
                                                       const unsigned int channel,
                                                       float *kernelArray,
                                                       const unsigned int kernelHeight,
                                                       const unsigned int kernelWidth)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * width + id_x + id_z * width * height;

    int boundx = (kernelWidth - 1) / 2;
    int boundy = (kernelHeight - 1) / 2;
    int sum = 0;
    int counter = 0;

    for(int i = -boundy; i <= boundy; i++)
    {
        for(int j = -boundx; j <= boundx; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = input[pixIdx] - saturate_8u(sum);
}

extern "C" __global__ void gaussian_image_pyramid_pkd_batch(unsigned char *input,
                                                            unsigned char *output,
                                                            const unsigned int height,
                                                            const unsigned int width,
                                                            const unsigned int channel,
                                                            float *kernelArray,
                                                            const unsigned int kernelHeight,
                                                            const unsigned int kernelWidth,
                                                            const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel || id_x % 2 != 0 || id_y % 2 != 0)
    {
        return;
    }

    unsigned long pixIdx = batchIndex + id_y * channel * width + id_x * channel + id_z;
    unsigned long outPixIdx = (id_y / 2) * channel * width + (id_x / 2) * channel + id_z;

    int boundx = (kernelWidth - 1) / 2;
    int boundy = (kernelHeight - 1) / 2;
    int sum = 0;
    int counter = 0;
    unsigned long index = 0;

    for(int i = -boundy; i <= boundy; i++)
    {
        for(int j = -boundx; j <= boundx; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width -1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                index = (unsigned long)pixIdx + ((unsigned long)j * (unsigned long)channel) + ((unsigned long)i * (unsigned long)width * (unsigned long)channel);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = saturate_8u(sum);
}

extern "C" __global__ void gaussian_image_pyramid_pln_batch(unsigned char *input,
                                                            unsigned char *output,
                                                            const unsigned int height,
                                                            const unsigned int width,
                                                            const unsigned int channel,
                                                            float *kernelArray,
                                                            const unsigned int kernelHeight,
                                                            const unsigned int kernelWidth,
                                                            const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel || id_x % 2 != 0 || id_y % 2 != 0)
    {
        return;
    }

    unsigned long pixIdx = batchIndex + id_y * width + id_x + id_z * width * height;
    unsigned long outPixIdx =  (id_y / 2) * width + (id_x / 2) + id_z * width * height;

    int boundx = (kernelWidth - 1) / 2;
    int boundy = (kernelHeight - 1) / 2;
    int sum = 0;
    int counter = 0;

    for(int i = -boundy; i <= boundy; i++)
    {
        for(int j = -boundx; j <= boundx; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned long index = (unsigned long)pixIdx + (unsigned long)j + ((unsigned long)i * (unsigned long)width);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = saturate_8u(sum);
}

extern "C" __global__ void laplacian_image_pyramid_pkd_batch(unsigned char *input,
                                                             unsigned char *output,
                                                             const unsigned int height,
                                                             const unsigned int width,
                                                             const unsigned int channel,
                                                             float *kernelArray,
                                                             const unsigned int kernelHeight,
                                                             const unsigned int kernelWidth,
                                                             const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int outPixIdx = batchIndex + id_y * channel * width + id_x * channel + id_z;

    int boundx = (kernelWidth - 1) / 2;
    int boundy = (kernelHeight - 1) / 2;
    int sum = 0;
    int counter = 0;

    for(int i = -boundy; i <= boundy; i++)
    {
        for(int j = -boundx; j <= boundx; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned long index = (unsigned long)pixIdx + ((unsigned long)j * (unsigned long)channel) + ((unsigned long)i * (unsigned long)width * (unsigned long)channel);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = input[pixIdx] - saturate_8u(sum);
}

extern "C" __global__ void laplacian_image_pyramid_pln_batch(unsigned char *input,
                                                             unsigned char *output,
                                                             const unsigned int height,
                                                             const unsigned int width,
                                                             const unsigned int channel,
                                                             float *kernelArray,
                                                             const unsigned int kernelHeight,
                                                             const unsigned int kernelWidth,
                                                             const unsigned long batchIndex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel)
    {
        return;
    }

    int pixIdx = (id_z * width * height) + (id_y * width) + id_x;
    int outPixIdx = batchIndex + (id_z * width * height) + (id_y * width) + id_x;

    int boundx = (kernelWidth - 1) / 2;
    int boundy = (kernelHeight - 1) / 2;
    int sum = 0;
    int counter = 0;

    for(int i = -boundy; i <= boundy; i++)
    {
        for(int j = -boundx; j <= boundx; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = input[pixIdx] - saturate_8u(sum);
}

RppStatus hip_exec_gaussian_image_pyramid_pkd_batch(Rpp8u *srcPtr, Rpp8u *srcPtr1, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    int globalThreads_y = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
    int globalThreads_z = channel;

    hipLaunchKernelGGL(gaussian_image_pyramid_pkd_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcPtr1,
                       max_height,
                       max_width,
                       channel,
                       kernelArray,
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       batchIndex);

    return RPP_SUCCESS;
}

RppStatus hip_exec_gaussian_image_pyramid_pln_batch(Rpp8u *srcPtr, Rpp8u *srcPtr1, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    int globalThreads_y = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
    int globalThreads_z = channel;

    hipLaunchKernelGGL(gaussian_image_pyramid_pln_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcPtr1,
                       max_height,
                       max_width,
                       channel,
                       kernelArray,
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       batchIndex);

    return RPP_SUCCESS;
}

RppStatus hip_exec_laplacian_image_pyramid_pkd_batch(Rpp8u *srcPtr1, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    int globalThreads_y = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
    int globalThreads_z = channel;

    hipLaunchKernelGGL(laplacian_image_pyramid_pkd_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       dstPtr,
                       max_height,
                       max_width,
                       channel,
                       kernelArray,
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       batchIndex);

    return RPP_SUCCESS;
}

RppStatus hip_exec_laplacian_image_pyramid_pln_batch(Rpp8u *srcPtr1, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    int globalThreads_y = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
    int globalThreads_z = channel;

    hipLaunchKernelGGL(laplacian_image_pyramid_pln_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       dstPtr,
                       max_height,
                       max_width,
                       channel,
                       kernelArray,
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       handle.GetInitHandle()->mem.mcpu.uintArr[1].uintmem[i],
                       batchIndex);

    return RPP_SUCCESS;
}