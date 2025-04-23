#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void temperature_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const int modificationValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;
    int pixIdx = id_x + id_y * width;
    int c = width * height;

    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);
    if( channel > 1)
    {
        output[pixIdx + c] = input[pixIdx + c];
        res = input[pixIdx + c * 2] - modificationValue;
        output[pixIdx + c * 2] = saturate_8u(res);
    }
}
__kernel void temperature_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const int modificationValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y * width * channel + id_x * channel;

    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);

    output[pixIdx + 1] = input[pixIdx + 1];

    res = input[pixIdx+2] - modificationValue;
    output[pixIdx+2] = saturate_8u(res);
}

unsigned char temperature(unsigned char input, unsigned char value, int RGB){
    if(RGB == 0)
        return saturate_8u((short)input + (short)value);
    else if(RGB == 1)
        return (input);
    else
        return saturate_8u((short)input - (short)value);
}

__kernel void color_temperature_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global int *modificationValue,
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
    unsigned char valuergb;
    unsigned char modificationValuetmp = modificationValue[id_z];
    int indextmp=0;
    long pixIdx = 0;
    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(indextmp = channel - 1; indextmp >= 0; indextmp--){
            valuergb = input[pixIdx];
            output[pixIdx] = temperature(valuergb , modificationValuetmp, indextmp);
            pixIdx += inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
}