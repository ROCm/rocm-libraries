#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

__kernel void reconstruction_laplacian_image_pyramid_pkd(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int height2,
                    const unsigned int width2,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x * channel + id_y * width * channel + id_z;
    output[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
}

__kernel void reconstruction_laplacian_image_pyramid_pln(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int height2,
                    const unsigned int width2,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * height * width;
    output[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
}