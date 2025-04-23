#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ double gaussian(double x, double sigmaI)
{
    double a = 2.0;
    return exp(-(pow(x, a))/(2 * pow(sigmaI, a))) / (2 * M_PI * pow(sigmaI, a));
}

__device__ double distance(int x1, int y1, int x2, int y2)
{
    double d_x = x2-x1;
    double d_y = y2-y1;
    double a = 2.0;
    double dis = sqrt(pow(d_x,a) + pow(d_y,a));
    return dis;
}

extern "C" __global__ void bilateral_filter_planar(const unsigned char *input,
                                                   unsigned char *output,
                                                   const unsigned int height,
                                                   const unsigned int width,
                                                   const unsigned int channel,
                                                   const unsigned int filterSize,
                                                   const double sigmaI,
                                                   const double sigmaS)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_x + id_y * width + id_z * width * height;
    int hfFiltSz = filterSize / 2;

    if ((id_x < hfFiltSz) || (id_y < hfFiltSz) || (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)))
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    double sum = 0.0;
    double w_sum = 0.0;
    for (int ri = (-1 * hfFiltSz), rf = 0; (ri <= hfFiltSz) && (rf < filterSize); ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0; (ci <= hfFiltSz) && (cf < filterSize); ci++, cf++)
        {
            const int idxI = pixIdx + ri + ci * width;
            double gi = gaussian(input[idxI] - input[pixIdx], sigmaI);
            double dis = distance(id_y, id_x, id_y+ri, id_x+ci);
            double gs = gaussian(dis, sigmaS);
            double w = gi * gs;
            sum += input[idxI] * w;
            w_sum += w;
        }
    }
    int res = sum / w_sum;
    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void bilateral_filter_packed(const unsigned char *input,
                                                   unsigned char *output,
                                                   const unsigned int height,
                                                   const unsigned int width,
                                                   const unsigned int channel,
                                                   const unsigned int filterSize,
                                                   const double sigmaI,
                                                   const double sigmaS)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_x * channel + id_y * width * channel + id_z;
    int hfFiltSz = filterSize / 2;

    if ((id_x < hfFiltSz) || (id_y < hfFiltSz) || (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)))
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    double sum = 0.0;
    double w_sum = 0.0;

    for (int ri = (-1 * hfFiltSz), rf = 0; (ri <= hfFiltSz) && (rf < filterSize); ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz), cf = 0; (ci <= hfFiltSz) && (cf < filterSize); ci++, cf++)
        {
            const int idxI = pixIdx + ri * channel + ci * width * channel;
            double gi = gaussian(input[idxI] - input[pixIdx], sigmaI);
            double dis = distance(id_y, id_x, id_y + (ri * channel), id_x + (ci * channel));
            double gs = gaussian(dis, sigmaS);
            double w = gi * gs;
            sum += input[idxI] * w;
            w_sum += w;
        }
    }
    int res = sum / w_sum;
    output[pixIdx] = saturate_8u(res);
}


extern "C" __global__ void bilateral_filter_batch(unsigned char *input,
                                                  unsigned char *output,
                                                  unsigned int *kernelSize,
                                                  double *sigmaS,
                                                  double *sigmaI,
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

    int kernelSizeTemp = kernelSize[id_z];
    int indextmp = 0;
    long pixIdx = 0;
    int bound = (kernelSizeTemp - 1) / 2;

    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

        if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            double sum1 = 0.0;
            double w_sum1 = 0.0;
            double sum2 = 0.0;
            double w_sum2 = 0.0;
            double sum3 = 0.0;
            double w_sum3 = 0.0;

            for(int i = -bound; i <= bound; i++)
            {
                for(int j = -bound; j <= bound; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] - 1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        double gi1 = gaussian(input[index] - input[pixIdx], sigmaI[id_z]);
                        double dis1 = distance(id_y, id_x, id_y + (i * plnpkdindex), id_x + (j * plnpkdindex));
                        double gs1 = gaussian(dis1,sigmaS[id_z]);
                        double w1 = gi1 * gs1;
                        sum1 += input[index] * w1;
                        w_sum1 += w1;

                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            double gi2 = gaussian(input[index] - input[pixIdx], sigmaI[id_z]);
                            double dis2 = distance(id_y, id_x, id_y + (i * plnpkdindex), id_x + (j * plnpkdindex));
                            double gs2 = gaussian(dis2, sigmaS[id_z]);
                            double w2 = gi2 * gs2;
                            sum2 += input[index] * w2;
                            w_sum2 += w2;

                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            double gi3 = gaussian(input[index] - input[pixIdx], sigmaI[id_z]);
                            double dis3 = distance(id_y, id_x, id_y + (i * plnpkdindex) , id_x + (j * plnpkdindex));
                            double gs3 = gaussian(dis3, sigmaS[id_z]);
                            double w3 = gi3 * gs3;
                            sum3 += input[index] * w3;
                            w_sum3 += w3;
                        }
                    }
                }
            }

            int res1 = sum1 / w_sum1;
            int res2 = sum2 / w_sum2;
            int res3 = sum3 / w_sum3;
            output[pixIdx] = saturate_8u(res1);

            if(channel == 3)
            {
                output[pixIdx + inc[id_z]] = saturate_8u(res2);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(res3);
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
}

RppStatus hip_exec_bilateral_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(bilateral_filter_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.doubleArr[1].doublemem,
                       handle.GetInitHandle()->mem.mgpu.doubleArr[2].doublemem,
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