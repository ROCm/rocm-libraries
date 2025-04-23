#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
unsigned int xorshift(int pixid) {
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
	return res;
}
__kernel void jitter_pkd(  __global unsigned char* input,
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

    
    int pixIdx = id_y * channel * width + id_x * channel;
    int nhx = xorshift(pixIdx) % (kernelSize);
    int nhy = xorshift(pixIdx) % (kernelSize);
    int bound = (kernelSize - 1) / 2;
    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
    {
        int index = ((id_y - bound) * channel * width) + ((id_x - bound) * channel) + (nhy * channel * width) + (nhx * channel);
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + i] = input[index + i];  
        }
    }
    else 
    {
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + i] = input[pixIdx + i];  
        }
    }
}

__kernel void jitter_pln(  __global unsigned char* input,
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

    
    int pixIdx = id_y * width + id_x;
    int channelPixel = height * width;
    int nhx = xorshift(pixIdx) % (kernelSize);
    int nhy = xorshift(pixIdx) % (kernelSize);
    int bound = (kernelSize - 1) / 2;
    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
    {
        int index = ((id_y - bound) * width) + (id_x - bound) + (nhy * width) + (nhx);
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + (height * width * i)] = input[index + (height * width * i)];  
        }
    }
    else 
    {
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + (height * width * i)] = input[pixIdx + (height * width * i)];  
        }
    }
}


__kernel void jitter_batch(  __global unsigned char* input,
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
    int bound = (kernelSizeTemp - 1) / 2;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {   
            int nhx = xorshift(pixIdx) % (kernelSizeTemp);
            int nhy = xorshift(pixIdx) % (kernelSizeTemp);
            if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height[id_z] - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width[id_z] - 1)
            {
                int index = batch_index[id_z] + ((((id_y - bound) * max_width[id_z]) + (id_x - bound)) * plnpkdindex) + (((nhy * max_width[id_z]) + (nhx)) * plnpkdindex);
                for(int i = 0 ; i < channel ; i++)
                {
                    output[pixIdx] = input[index]; 
                    pixIdx += inc[id_z]; 
                    index += inc[id_z];
                }
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