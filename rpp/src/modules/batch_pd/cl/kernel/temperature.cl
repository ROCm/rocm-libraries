#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void temperature_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float *value,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    unsigned char valuergb;
    float tmpvalue = value[id_z];
    int indextmp=0;
    int pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
            valuergb = input[pixIdx];
            output[pixIdx] = saturate_8u(valuergb + tmpvalue);
            if(channel > 1){
                pixIdx += inc[id_z];
                valuergb = input[pixIdx];
                output[pixIdx] = valuergb;
                pixIdx += inc[id_z];
                valuergb = input[pixIdx];
                output[pixIdx] = saturate_8u(valuergb + tmpvalue);
            }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
}