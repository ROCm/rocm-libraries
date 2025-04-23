__kernel void channel_extract_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int extractChannelNumber
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int OPpixIdx = id_x + id_y * width;
    int IPpixIdx = OPpixIdx + extractChannelNumber * width * height;
    output[OPpixIdx] = input[IPpixIdx];
}
__kernel void channel_extract_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int extractChannelNumber
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int OPpixIdx = id_y * width + id_x ;
    int IPpixIdx = id_y * width * channel + id_x * channel + extractChannelNumber;
    output[OPpixIdx] = input[IPpixIdx];
}

__kernel void channel_extract_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned int* channelNumber,
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
    int tempchannelNumber = channelNumber[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0, outPixIdx = 0;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex + tempchannelNumber;
        outPixIdx = (batch_index[id_z] / 3) + (id_x  + id_y * max_width[id_z]);
        output[outPixIdx] = input[pixIdx];
    }
}