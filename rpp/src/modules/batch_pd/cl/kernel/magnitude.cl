#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void magnitude(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
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

    float res = sqrt((float)(pow((int)input1[pixIdx],2.0) + pow((int)input2[pixIdx],2.0)));
    output[pixIdx] = (int)saturate_8u(res);
}


__kernel void magnitude_batch(  __global unsigned char* input1,
                                    __global unsigned char* input2,
                                    __global unsigned char* output,
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
    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
        (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
        for (indextmp = 0; indextmp < channel; indextmp++) {
            output[pixIdx] = saturate_8u(sqrt((float)(input1[pixIdx]*input1[pixIdx] + input2[pixIdx]*input2[pixIdx])));
            pixIdx += inc[id_z];
        }
    } else if ((id_x < width[id_z]) && (id_y < height[id_z])) {
        for (indextmp = 0; indextmp < channel; indextmp++) {
            output[pixIdx] = input1[pixIdx];
            pixIdx += inc[id_z];
        }
    }
}