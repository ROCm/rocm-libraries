#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void water_batch(  __global unsigned char* input,
                            __global unsigned char* output,
                            __global float *ampl_x,
                            __global float *ampl_y,
                            __global float *freq_x,
                            __global float *freq_y,
                            __global float *phase_x,
                            __global float *phase_y,
                            __global int *xroi_begin,
                            __global int *xroi_end,
                            __global int *yroi_begin,
                            __global int *yroi_end,
                            __global unsigned int *height,
                            __global unsigned int *width,
                            __global unsigned int *max_height,
                            __global unsigned int *max_width,
                            __global unsigned long *batch_index,
                            const unsigned int channel,
                            __global unsigned int *src_inc,
                            __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    unsigned char valuergb;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    int img_width = width[id_z];

    int img_height = height[id_z];
    int indextmp=0;
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                valuergb = input[src_pix_id];
                output[dst_pix_id] = valuergb;
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}

__kernel void water_batch_fp32(  __global float* input,
                            __global float* output,
                            __global float *ampl_x,
                            __global float *ampl_y,
                            __global float *freq_x,
                            __global float *freq_y,
                            __global float *phase_x,
                            __global float *phase_y,
                            __global int *xroi_begin,
                            __global int *xroi_end,
                            __global int *yroi_begin,
                            __global int *yroi_end,
                            __global unsigned int *height,
                            __global unsigned int *width,
                            __global unsigned int *max_height,
                            __global unsigned int *max_width,
                            __global unsigned long *batch_index,
                            const unsigned int channel,
                            __global unsigned int *src_inc,
                            __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    float valuergb;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    int img_width = width[id_z];

    int img_height = height[id_z];
    int indextmp=0;
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                valuergb = input[src_pix_id];
                output[dst_pix_id] = valuergb;
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}

__kernel void water_batch_fp16(  __global half* input,
                            __global half* output,
                            __global float *ampl_x,
                            __global float *ampl_y,
                            __global float *freq_x,
                            __global float *freq_y,
                            __global float *phase_x,
                            __global float *phase_y,
                            __global int *xroi_begin,
                            __global int *xroi_end,
                            __global int *yroi_begin,
                            __global int *yroi_end,
                            __global unsigned int *height,
                            __global unsigned int *width,
                            __global unsigned int *max_height,
                            __global unsigned int *max_width,
                            __global unsigned long *batch_index,
                            const unsigned int channel,
                            __global unsigned int *src_inc,
                            __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    half valuergb;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    int img_width = width[id_z];

    int img_height = height[id_z];
    int indextmp=0;
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                valuergb = input[src_pix_id];
                output[dst_pix_id] = valuergb;
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}

__kernel void water_batch_int8(  __global char* input,
                            __global char* output,
                            __global float *ampl_x,
                            __global float *ampl_y,
                            __global float *freq_x,
                            __global float *freq_y,
                            __global float *phase_x,
                            __global float *phase_y,
                            __global int *xroi_begin,
                            __global int *xroi_end,
                            __global int *yroi_begin,
                            __global int *yroi_end,
                            __global unsigned int *height,
                            __global unsigned int *width,
                            __global unsigned int *max_height,
                            __global unsigned int *max_width,
                            __global unsigned long *batch_index,
                            const unsigned int channel,
                            __global unsigned int *src_inc,
                            __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    char valuergb;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    int img_width = width[id_z];

    int img_height = height[id_z];
    int indextmp=0;
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                valuergb = input[src_pix_id];
                output[dst_pix_id] = valuergb;
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}