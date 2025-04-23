#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void temperature_batch(   unsigned char* input,
                                     unsigned char* output,
                                     float *value,
                                     int *xroi_begin,
                                     int *xroi_end,
                                     int *yroi_begin,
                                     int *yroi_end,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *batch_index,
                                    const unsigned int channel,
                                     unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
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