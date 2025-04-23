#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
unsigned int xorshift(int pixid) {
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
	return res;
}
__kernel void gaussian(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const float mean,
                    const float sigma,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    float res = input1[pixIdx] + input2[pixIdx];
    output[pixIdx] = saturate_8u(res);
}

__kernel void snp_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int pixelDistance
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y * width * channel + id_x * channel;
    int rand;
    
    if(pixIdx % pixelDistance == 0 )
    {
        int rand_id = xorshift(pixIdx) % (int)(60 * pixelDistance);
        rand_id -= rand_id % 3;
        rand = (rand_id % 2) ? 0 : 255;
        for(int i = 0 ; i < channel ; i++)
            output[pixIdx + i + rand_id] = rand;
    }
}




__kernel void snp_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int pixelDistance
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y * width + id_x;
    int channelSize = width * height;
    
    int rand;
    
    if(pixIdx % pixelDistance == 0 )
    {
        int rand_id = xorshift(pixIdx) % (int)(60 * pixelDistance);
        rand_id -= rand_id % 3;
        rand = (rand_id % 2) ? 0 : 255;
        for(int i = 0 ; i < channel ; i++)
            output[pixIdx + channelSize * i + rand_id] = rand;
    }
}

__kernel void noise_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float *noiseProbability,
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
    float probTemp = noiseProbability[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int rand;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {   
            float noisePixel = probTemp * (float)(width[id_z] * height[id_z]);
            float pixelDistance = 1.0;
            pixelDistance /=  probTemp;
            if(((pixIdx - batch_index[id_z]) % (int)pixelDistance) == 0)
            {
                int rand_id = xorshift(pixIdx) % (int)(60 * pixelDistance);
                rand = (rand_id % 2) ? 0 : 255;
                rand_id = rand_id % (int)pixelDistance;
                rand_id -= rand_id % 3;
                rand_id = rand_id * plnpkdindex;
                for(indextmp = 0; indextmp < channel; indextmp++)
                {
                    output[pixIdx + rand_id] = rand;
                    pixIdx += inc[id_z];
                }
            }
            else
            {
                for(indextmp = 0; indextmp < channel; indextmp++)
                {
                    output[pixIdx] = input[pixIdx];
                    pixIdx += inc[id_z];
                }

            }
        }
    }

    else {
        pixIdx = batch_index[id_z]   + (id_x  + id_y * max_width[id_z] ) * plnpkdindex;
        for(indextmp = 0; indextmp < channel; indextmp++){
            output[pixIdx] = 0;
            pixIdx +=  inc[id_z];
        }
    }
}