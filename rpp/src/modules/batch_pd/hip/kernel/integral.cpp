#include <hip/hip_runtime.h>
extern "C" __global__ void integral_pkd_col(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_x * channel + id_z;
    
    output[pixIdx] = 0;

    for(int i = id_z; i <= id_x * channel + id_z ; i += channel)
    {
        output[pixIdx] += input[i];
    }
}

extern "C" __global__ void integral_pln_col(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_z * height * width + id_x;

    output[pixIdx] = 0;

    for(int i = (id_z * height * width) ; i <= (id_z * height * width + id_x) ; i++)
    {
        output[pixIdx] += input[i];
    }
}

extern "C" __global__ void integral_pkd_row(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_x * channel * width + id_z;
    
    output[pixIdx] = 0;

    for(int i = id_z; i <= id_x * channel * width + id_z ; i += width * channel)
    {
        output[pixIdx] += input[i];
    }
}

extern "C" __global__ void integral_pln_row(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_z * height * width + id_x * width;

    output[pixIdx] = 0;

    for(int i = (id_z * height * width) ; i <= (id_z * height * width + id_x * width) ; i += width)
    {
        output[pixIdx] += input[i];
    }
}

extern "C" __global__ void integral_up_pln(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = ((width * (loop + 1 )) - (id_x * width) + (id_x + 1)) + (id_z * height * width);
    int A, B, C;
    A = pixIdx - width - 1;
    B = pixIdx - width;
    C = pixIdx - 1;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}

extern "C" __global__ void integral_low_pln(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = ((width * (height - 1)) + (id_x + loop + 1) - (id_x * width) + 1) + (id_z * height * width);
    int A, B, C;
    A = pixIdx - width - 1;
    B = pixIdx - width;
    C = pixIdx - 1;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}

extern "C" __global__ void integral_up_pkd(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = (width * channel * (loop + 1)) - (id_x * channel * width) + (id_x * channel + channel) + id_z;
    int A, B, C;
    A = pixIdx - width * channel - channel;
    B = pixIdx - width * channel;
    C = pixIdx - channel;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}

extern "C" __global__ void integral_low_pkd(  unsigned char* input,
                             unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = (width * channel * (height - 1)) - (id_x * channel * width) + (id_x * channel + (loop + 1) * channel) + channel + id_z;
    int A, B, C;
    A = pixIdx - width * channel - channel;
    B = pixIdx - width * channel;
    C = pixIdx - channel;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}