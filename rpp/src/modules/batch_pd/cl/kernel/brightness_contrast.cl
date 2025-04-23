#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

unsigned char brightness( unsigned char input_pixel, double alpha, double beta){
    return saturate_8u(alpha * input_pixel + beta);
}
unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, 
                        unsigned int height, unsigned channel){
 return ( id_x + id_y * width + id_z * width * height);
}
__kernel void brightness_contrast(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    const float alpha,
                                    const int beta,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = get_pln_index(id_x, id_y, id_z, width, height, channel);

    int res = input[pixIdx] * alpha + beta;
    output[pixIdx] = saturate_8u(res);
}

__kernel void brightness_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float *alpha,
                                    __global float *beta,
                                    __global unsigned int *xroi_begin,
                                    __global unsigned int *xroi_end,
                                    __global unsigned int *yroi_begin,
                                    __global unsigned int *yroi_end,
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
    float alphatmp = alpha[id_z], betatmp = beta[id_z];
    int indextmp=0;
    long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb = input[pixIdx];
            output[pixIdx] = brightness(valuergb , alphatmp , betatmp);
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
