__kernel void channel_combine_pln(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* input3,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx1 = IPpixIdx;
    int OPpixIdx2 = IPpixIdx + width * height;
    int OPpixIdx3 = IPpixIdx + 2 * width * height;

    output[OPpixIdx1] = input1[IPpixIdx];
    output[OPpixIdx2] = input2[IPpixIdx];
    output[OPpixIdx3] = input3[IPpixIdx];
}
__kernel void channel_combine_pkd(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* input3,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx = IPpixIdx * channel;
    output[OPpixIdx] = input1[IPpixIdx];
    output[OPpixIdx + 1] = input2[IPpixIdx];
    output[OPpixIdx + 2] = input3[IPpixIdx];
}

__kernel void channel_combine_batch(  __global unsigned char* input1,
                                    __global unsigned char* input2,
                                    __global unsigned char* input3,
                                    __global unsigned char* output,
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
    unsigned long pixIdx = 0, InPixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex;
        InPixIdx = (batch_index[id_z] / 3) + (id_x  + id_y * max_width[id_z]);
        output[pixIdx] = input1[InPixIdx];
        output[pixIdx + inc[id_z]] = input2[InPixIdx];
        output[pixIdx + inc[id_z] * 2] = input3[InPixIdx];
    }
}