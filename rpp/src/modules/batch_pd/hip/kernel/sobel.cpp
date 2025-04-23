#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ unsigned int power_sobel(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
    {
        sum += sum * a;
    }
    return sum;
}

__device__ int calcSobelx(int a[3][3])
{
    int gx[3][3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += a[i][j] * gx[i][j];
        }
    }
    return sum;
}

__device__ int calcSobely(int a[3][3])
{
    int gy[3][3]={-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += a[i][j] * gy[i][j];
        }
    }
    return sum;
}

extern "C" __global__ void sobel_pkd(unsigned char *input,
                                     unsigned char *output,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int channel,
                                     const unsigned int sobelType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int value = 0;
    int value1 = 0;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int a[3][3];

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height - 1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power_sobel(value, 2);
        value1 = power_sobel(value1, 2);
        value = sqrt((float)(value + value1));
        output[pixIdx] = saturate_8u(value);
    }

    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}

extern "C" __global__ void sobel_pln(unsigned char *input,
                                     unsigned char *output,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int channel,
                                     const unsigned int sobelType)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int value = 0;
    int value1 = 0;
    int a[3][3];

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height - 1)
            {
                unsigned int index = pixIdx + j + (i * width);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power_sobel(value, 2);
        value1 = power_sobel(value1, 2);
        value = sqrt((float)(value + value1));
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}


extern "C" __global__ void sobel_batch(unsigned char *input,
                                       unsigned char *output,
                                       unsigned int *sobelType,
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

    unsigned char valuer, valuer1, valueg, valueg1, valueb, valueb1;
    int sobelTypeTemp = sobelType[id_z];
    int indextmp = 0;
    long pixIdx = 0;
    // printf("%d", id_x);
    int value = 0;
    int value1 = 0;
    int r[3][3],g[3][3],b[3][3];

    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int i = -1; i <= 1; i++)
        {
            for(int j = -1; j <= 1; j++)
            {
                if(id_x != 0 && id_x != width[id_z] - 1 && id_y != 0 && id_y != height[id_z] - 1)
                {
                    unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                    r[i+1][j+1] = input[index];
                    if(channel == 3)
                    {
                        index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                        g[i+1][j+1] = input[index];
                        index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                        b[i+1][j+1] = input[index];
                    }
                }
                else
                {
                    r[i+1][j+1] = 0;
                    if(channel == 3)
                    {
                        g[i+1][j+1] = 0;
                        b[i+1][j+1] = 0;
                    }
                }
            }
        }

        if(sobelType[id_z] == 2)
        {
            value = calcSobelx(r);
            value1 = calcSobely(r);
            value = power_sobel(value, 2);
            value1 = power_sobel(value1, 2);
            value = sqrt((float)(value + value1));
            output[pixIdx] = saturate_8u(value);
            if(channel == 3)
            {
                value = calcSobelx(g);
                value1 = calcSobely(g);
                value = power_sobel(value, 2);
                value1 = power_sobel(value1, 2);
                value = sqrt((float)(value + value1));
                output[pixIdx + inc[id_z]] = saturate_8u(value);
                value = calcSobelx(b);
                value1 = calcSobely(b);
                value = power_sobel(value, 2);
                value1 = power_sobel(value1, 2);
                value = sqrt((float)(value + value1));
                output[pixIdx + inc[id_z] * 2] = saturate_8u(value);
            }
        }
        if(sobelType[id_z] == 1)
        {
            value = calcSobely(r);
            output[pixIdx] = saturate_8u(value);
            if(channel == 3)
            {
                value = calcSobely(g);
                output[pixIdx + inc[id_z]] = saturate_8u(value);
                value = calcSobely(g);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(value);
            }
        }
        if(sobelType[id_z] == 0)
        {
            value = calcSobelx(r);
            output[pixIdx] = saturate_8u(value);
            if(channel == 3)
            {
                value = calcSobelx(g);
                output[pixIdx + inc[id_z]] = saturate_8u(value);
                value = calcSobelx(g);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(value);
            }
        }
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[pixIdx] = input[pixIdx];
            pixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_sobel_pln(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32u sobelType)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(sobel_pln,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       height,
                       width,
                       channel,
                       sobelType);

    return RPP_SUCCESS;
}

RppStatus hip_exec_sobel_pkd(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32u sobelType)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(sobel_pkd,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       height,
                       width,
                       channel,
                       sobelType);

    return RPP_SUCCESS;
}

RppStatus hip_exec_sobel_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(sobel_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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