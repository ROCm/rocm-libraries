#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

extern "C" __global__ void custom_convolution_pkd(unsigned char *input,
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

    if (id_x >= width || id_y >= height || id_z >= channel)
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
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height - 1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = saturate_8u(sum);
}

extern "C" __global__ void custom_convolution_pln(unsigned char *input,
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

    if (id_x >= width || id_y >= height || id_z >= channel)
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
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height - 1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernelArray[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = saturate_8u(sum);
}


extern "C" __global__ void custom_convolution_batch(unsigned char *input,
                                                    unsigned char *output,
                                                    float *kernelValue,
                                                    const unsigned int kHeight,
                                                    const unsigned int kWidth,
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

    long pixIdx = 0;
    int temp;
    int boundy = (kHeight - 1) / 2;
    int boundx = (kWidth - 1) / 2;
    float sumR = 0, sumG = 0, sumB = 0;
    int kernelIndex = kHeight * kWidth * id_z;
    int counter = kernelIndex;

    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            for(int i = -boundy; i <= boundy; i++)
            {
                for(int j = -boundx; j <= boundx; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] - 1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        sumR += (float)input[index] * kernelValue[counter];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            sumG += (float)input[index] * kernelValue[counter];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            sumB += (float)input[index] * kernelValue[counter];
                        }
                    }
                    counter++;
                }
            }
            output[pixIdx] = saturate_8u(sumR);
            if(channel == 3)
            {
                output[pixIdx + inc[id_z]] = saturate_8u(sumG);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(sumB);
            }
        }
        else if((id_x < width[id_z]) && (id_y < height[id_z]))
        {
            for(int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}

RppStatus hip_exec_custom_convolution_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *d_kernel, RppiSize kernelSize, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(custom_convolution_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       d_kernel,
                       kernelSize.height,
                       kernelSize.width,
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