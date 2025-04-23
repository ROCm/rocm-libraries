#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void harris_corner_detector_strength(unsigned char *sobelX,
                                                           unsigned char *sobelY,
                                                           float *output,
                                                           const unsigned int height,
                                                           const unsigned int width,
                                                           const unsigned int channel,
                                                           const unsigned int kernelSize,
                                                           const float kValue,
                                                           const float threshold)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    float sumXX = 0, sumYY = 0, sumXY = 0, det = 0, trace = 0, pixel = 0;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int bound = (kernelSize - 1) / 2;

    for(int i = -bound; i <= bound; i++)
    {
        for(int j = -bound; j <= bound; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sumXX += (sobelX[index] * sobelX[index]);
                sumYY += (sobelY[index] * sobelY[index]);
                sumXY += (sobelX[index] * sobelY[index]);
            }
        }
    }

    det = (sumXX * sumYY) - (sumXY * sumXY);
    trace = sumXX + sumYY;
    pixel = (det) - (kValue * trace * trace);

    if (pixel > threshold)
    {
        output[pixIdx] = pixel;
    }
    else
    {
        output[pixIdx] = 0;
    }
}

extern "C" __global__ void harris_corner_detector_nonmax_supression(float *input,
                                                                    float *output,
                                                                    const unsigned int height,
                                                                    const unsigned int width,
                                                                    const unsigned int channel,
                                                                    const unsigned int kernelSize)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int bound = (kernelSize - 1) / 2;
    float pixel = input[pixIdx];

    for(int i = -bound; i <= bound; i++)
    {
        for(int j = -bound; j <= bound; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                if(input[index] > pixel)
                {
                    return;
                }
            }
        }
    }

    output[pixIdx] = input[pixIdx];
}

extern "C" __global__ void harris_corner_detector_pln(unsigned char *input,
                                                      float *inputFloat,
                                                      const unsigned int height,
                                                      const unsigned int width,
                                                      const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int pixIdx = id_y * width + id_x;
    if (id_x >= width || id_y >= height || id_z >= channel || inputFloat[pixIdx] == 0)
    {
        return;
    }

    unsigned int kernelSize = 3;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                if(channel == 3)
                {
                    input[index] = 0;
                    input[index + height * width] = 0;
                    input[index + height * width * 2] = 255;
                }
                else if(channel == 1)
                {
                    input[index] = 255;
                }
            }
        }
    }
}

extern "C" __global__ void harris_corner_detector_pkd(unsigned char *input,
                                                      float *inputFloat,
                                                      const unsigned int height,
                                                      const unsigned int width,
                                                      const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int pixIdx = id_y * width + id_x;

    if (id_x >= width || id_y >= height || id_z >= channel || inputFloat[pixIdx] == 0)
    {
        return;
    }

    pixIdx = id_y * channel * width + id_x * channel;

    unsigned int kernelSize = 3;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound; i <= bound; i++)
    {
        for(int j = -bound; j <= bound; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                input[index] = 0;
                input[index+1] = 0;
                input[index+2] = 255;
            }
        }
    }
}

RppStatus hip_exec_harris_corner_detector_strength(Rpp8u *sobelX, Rpp8u *sobelY, Rpp32f *dstFloat, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(harris_corner_detector_strength,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       sobelX,
                       sobelY,
                       dstFloat,
                       height,
                       width,
                       channel,
                       handle.GetInitHandle()->mem.mcpu.uintArr[2].uintmem[i],
                       handle.GetInitHandle()->mem.mcpu.floatArr[3].floatmem[i],
                       handle.GetInitHandle()->mem.mcpu.floatArr[4].floatmem[i]);

    return RPP_SUCCESS;
}

RppStatus hip_exec_harris_corner_detector_nonmax_supression(Rpp32f *input, Rpp32f *output, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(harris_corner_detector_nonmax_supression,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       input,
                       output,
                       height,
                       width,
                       channel,
                       handle.GetInitHandle()->mem.mcpu.uintArr[5].uintmem[i]);

    return RPP_SUCCESS;
}

RppStatus hip_exec_harris_corner_detector_pkd(Rpp8u *input, Rpp32f *inputFloat, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(harris_corner_detector_pkd,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       input,
                       inputFloat,
                       height,
                       width,
                       channel);

    return RPP_SUCCESS;
}

RppStatus hip_exec_harris_corner_detector_pln(Rpp8u *input, Rpp32f *inputFloat, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(harris_corner_detector_pln,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       input,
                       inputFloat,
                       height,
                       width,
                       channel);

    return RPP_SUCCESS;
}