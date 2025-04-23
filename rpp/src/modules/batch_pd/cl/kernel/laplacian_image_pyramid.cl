#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void gaussian_image_pyramid_pkd_batch( __global unsigned char* input,
                                                __global unsigned char* output,
                                                const unsigned int height,
                                                const unsigned int width,
                                                const unsigned int channel,
                                                __global float* kernal,
                                                const unsigned int kernalheight,
                                                const unsigned int kernalwidth,
                                                const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel || id_x % 2 != 0 || id_y % 2 != 0) return;

    unsigned long pixIdx = batchIndex + id_y * channel * width + id_x * channel + id_z;
    unsigned long outPixIdx = (id_y / 2) * channel * width + (id_x / 2) * channel + id_z;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    unsigned long index = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width -1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                index = (unsigned long)pixIdx + ((unsigned long)j * (unsigned long)channel) + ((unsigned long)i * (unsigned long)width * (unsigned long)channel);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = saturate_8u(sum);
}

__kernel void gaussian_image_pyramid_pln_batch( __global unsigned char* input,
                                                __global unsigned char* output,
                                                const unsigned int height,
                                                const unsigned int width,
                                                const unsigned int channel,
                                                __global float* kernal,
                                                const unsigned int kernalheight,
                                                const unsigned int kernalwidth,
                                                const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel || id_x % 2 != 0 || id_y % 2 != 0) return;

    unsigned long pixIdx = batchIndex + id_y * width + id_x + id_z * width * height;
    unsigned long outPixIdx =  (id_y / 2) * width + (id_x / 2) + id_z * width * height;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned long index = (unsigned long)pixIdx + (unsigned long)j + ((unsigned long)i * (unsigned long)width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = saturate_8u(sum); 
}

__kernel void laplacian_image_pyramid_pkd_batch(__global unsigned char* input,
                                                __global unsigned char* output,
                                                const unsigned int height,
                                                const unsigned int width,
                                                const unsigned int channel,
                                                __global float* kernal,
                                                const unsigned int kernalheight,
                                                const unsigned int kernalwidth,
                                                const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel) return;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int outPixIdx = batchIndex + id_y * channel * width + id_x * channel + id_z;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned long index = (unsigned long)pixIdx + ((unsigned long)j * (unsigned long)channel) + ((unsigned long)i * (unsigned long)width * (unsigned long)channel);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = input[pixIdx] - saturate_8u(sum); 
}

__kernel void laplacian_image_pyramid_pln_batch(__global unsigned char* input,
                                                __global unsigned char* output,
                                                const unsigned int height,
                                                const unsigned int width,
                                                const unsigned int channel,
                                                __global float* kernal,
                                                const unsigned int kernalheight,
                                                const unsigned int kernalwidth,
                                                const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel) return;

    int pixIdx = (id_z * width * height) + (id_y * width) + id_x;
    int outPixIdx = batchIndex + (id_z * width * height) + (id_y * width) + id_x;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = input[pixIdx] - saturate_8u(sum); 
}

__kernel void laplacian_image_pyramid_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel) return;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = input[pixIdx] - saturate_8u(sum);
}

__kernel void laplacian_image_pyramid_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
if (id_x >= ceil((float)(width / 2)) || id_y >= ceil((float)(height / 2)) || id_z >= channel) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = input[pixIdx] - saturate_8u(sum); 
}