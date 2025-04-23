#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void accumulate(  __global unsigned char* input1,
                            __global unsigned char* input2,
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

    int res = input1[pixIdx] + input2[pixIdx];
    input1[pixIdx] = saturate_8u(res);
}

__kernel void accumulate_weighted(  __global unsigned char* input1,
                            __global unsigned char* input2,
                            const float alpha,
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

    int res = (1 - alpha) * input1[pixIdx] + alpha * input2[pixIdx];
    input1[pixIdx] = saturate_8u(res);
}

__kernel void accumulate_squared(  __global unsigned char* input,
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

    int res = (int)pow((float)input[pixIdx],(float)2);
    input[pixIdx] = saturate_8u(res);
}

__kernel void accumulate_batch(  __global unsigned char* input1,
                                    __global unsigned char* input2,
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

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            input1[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
            pixIdx += inc[id_z];
        }
    }
}

unsigned char accumulate_weight_formula( unsigned char input_pixel1, unsigned char input_pixel2, float alpha){
    return saturate_8u(alpha * input_pixel1 + (1 - alpha) * input_pixel2);
}


__kernel void accumulate_weighted_batch(  __global unsigned char* input1,
                                    __global unsigned char* input2,
                                    __global float *alpha,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int  *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    unsigned char valuergb1, valuergb2;
    float alphatmp = alpha[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb1 = input1[pixIdx];
            valuergb2 = input2[pixIdx];
            input1[pixIdx] = accumulate_weight_formula(valuergb1, valuergb2, alphatmp);
            pixIdx += inc[id_z];
        }
    }
}

__kernel void accumulate_squared_batch(  __global unsigned char* input,
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

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            int pixel = saturate_8u(input[pixIdx] * input[pixIdx]);
            input[pixIdx] = saturate_8u(pixel);
            pixIdx += inc[id_z];
        }
    }
}