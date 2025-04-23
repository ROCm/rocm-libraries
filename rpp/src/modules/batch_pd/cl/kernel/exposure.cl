#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

unsigned char exposure_value(unsigned char input, float value){
    return saturate_8u(input * pow(2,value));
}
__kernel void exposure(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float value
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int res = input[pixIdx] * pow(2,value);
    output[pixIdx] = saturate_8u(res);
}

__kernel void exposure_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float *value,
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
    float valuetmp = value[id_z];
    int indextmp=0;
    long pixIdx = 0;
    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb = input[pixIdx];
            output[pixIdx] = exposure_value(valuergb , valuetmp);
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