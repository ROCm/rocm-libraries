__kernel void convert_bit_depth_u8s8(  __global unsigned char* input,
                            __global char* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    output[pixIdx] = (char)(input[pixIdx] - 128);

}
__kernel void convert_bit_depth_u8u16(  __global unsigned char* input,
                            __global unsigned short* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;
    
    output[pixIdx] = (unsigned short)(input[pixIdx] * 257);
}
__kernel void convert_bit_depth_u8s16(  __global unsigned char* input,
                            __global short* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;
    
    output[pixIdx] = (short)((input[pixIdx] * 257) - 32768);
}

__kernel void convert_bit_depth_batch_u8s8(  __global unsigned char* input,
                                    __global char* output,
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
    int indextmp=0;
    unsigned long pixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[pixIdx] = (char)(input[pixIdx] - 128);
            pixIdx += inc[id_z];
        }
    }
}

__kernel void convert_bit_depth_batch_u8u16(  __global unsigned char* input,
                                    __global unsigned short* output,
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
    int indextmp=0;
    unsigned long pixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[pixIdx] = (unsigned short)(input[pixIdx] * 257);
            pixIdx += inc[id_z];
        }
    }
}

__kernel void convert_bit_depth_batch_u8s16(  __global unsigned char* input,
                                    __global short* output,
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
    int indextmp=0;
    unsigned long pixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[pixIdx] = (short)((input[pixIdx] * 257) - 32768);
            pixIdx += inc[id_z];
        }
    }
}