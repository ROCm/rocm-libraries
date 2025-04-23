#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

extern "C" __global__ void reconstruction_laplacian_image_pyramid_pkd(   unsigned char* input1,
                     unsigned char* input2,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int height2,
                    const unsigned int width2,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x * channel + id_y * width * channel + id_z;
    output[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
}

extern "C" __global__ void reconstruction_laplacian_image_pyramid_pln(   unsigned char* input1,
                     unsigned char* input2,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int height2,
                    const unsigned int width2,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * height * width;
    output[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
}