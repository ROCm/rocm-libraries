#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ float gaussian_vignette(int x,int y, float stdDev)
{
    float res, pi = 3.14;
    res = 1 / (2 * pi * stdDev * stdDev);
    float exp1, exp2;
    exp1 = - (x * x) / (2 * stdDev * stdDev);
    exp2 = - (y * y) / (2 * stdDev * stdDev);
    exp1 = exp1 + exp2;
    exp1 = exp(exp1);
    res *= exp1;
	return res;
}

extern "C" __global__ void vignette_pkd(unsigned char *input,
                                        unsigned char *output,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int channel,
                                        const float stdDev)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int x = (id_x - (width / 2));
    int y = (id_y - (height / 2));
    float gaussianvalue = gaussian_vignette(x, y, stdDev) / gaussian_vignette(0.0, 0.0, stdDev);
    float res = ((float) input[pixIdx]) * gaussianvalue;
    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void vignette_pln(unsigned char *input,
                                        unsigned char *output,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int channel,
                                        const float stdDev)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_z * width * height + id_y * width + id_x;
    int x = (id_x - (width / 2));
    int y = (id_y - (height / 2));
    float gaussianvalue = gaussian_vignette(x, y, stdDev) / gaussian_vignette(0.0, 0.0, stdDev);
    float res = ((float) input[pixIdx]) * gaussianvalue;
    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void vignette_batch(unsigned char *input,
                                          unsigned char *output,
                                          float *stdDev,
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

    float tempstdDev = stdDev[id_z];
    int indextmp = 0;
    unsigned long pixIdx = 0;
    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= 0) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {
        int x = (id_x - (width[id_z] / 2));
        int y = (id_y - (height[id_z] / 2));
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            float gaussianvalue = gaussian_vignette(x, y, tempstdDev) / gaussian_vignette(0.0, 0.0, tempstdDev);
            float res = ((float) input[pixIdx]) * gaussianvalue;
            output[pixIdx] = saturate_8u((int) res);
            pixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_vignette_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(vignette_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}