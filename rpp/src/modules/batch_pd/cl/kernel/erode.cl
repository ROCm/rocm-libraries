#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void erode_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                if(input[index] < pixel)
                {
                    pixel = input[index];
                }
            }
        }
    }
    output[pixIdx] = pixel;
}

__kernel void erode_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                if(input[index] < pixel)
                {
                    pixel = input[index];
                }
            }
        }
    }
    output[pixIdx] = pixel;    
}

__kernel void erode_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned int *kernelSize,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int kernelSizeTemp = kernelSize[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int temp;
    // printf("%d", id_x);
    int value = 0;
    int value1 =0;
    unsigned char r = 0, g = 0, b = 0;
    int checkR = 0, checkB = 0, checkG = 0;
    int bound = (kernelSizeTemp - 1) / 2;
    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    r = input[pixIdx];
    if(id_x < width[id_z] && id_y < height[id_z])
    {    
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        r = input[pixIdx];
        if(channel == 3)
        {
            g = input[pixIdx + inc[id_z]];
            b = input[pixIdx + inc[id_z] * 2];
        }  
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {   
            for(int i = -bound ; i <= bound ; i++)
            {
                for(int j = -bound ; j <= bound ; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        if(r > input[index])
                            r = input[index];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            if(g > input[index])
                                g = input[index];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            if(b > input[index])
                                b = input[index];
                        }
                    }
                }
            }
            output[pixIdx] = r;
            if(channel == 3)
            {
                output[pixIdx + inc[id_z]] = g;
                output[pixIdx + inc[id_z] * 2] = b;
            }
        }
        else if((id_x < width[id_z] ) && (id_y < height[id_z]))
        {
            for(indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}