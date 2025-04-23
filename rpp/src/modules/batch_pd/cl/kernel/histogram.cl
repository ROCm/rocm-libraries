#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#define round(value) ( (value - (int)(value)) >=0.5 ? (value + 1) : (value))
#define MAX_SIZE 64

kernel
void partial_histogram_pln(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 ;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 ;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x  + id_y * width ;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + width * height];
        unsigned char pixelB = input[pixId + 2 * width * height];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[pixelG]);
        atomic_inc(&tmp_histogram[pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}

kernel
void partial_histogram_pkd(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 ;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 ;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x * channel + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + 1];
        unsigned char pixelB = input[pixId + 2];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[pixelG]);
        atomic_inc(&tmp_histogram[pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}

kernel
void partial_histogram_batch(__global unsigned char* input,
                                    __global unsigned int *histogramPartial,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int num_groups,// For partial histogram indexing// try out a better way
                                    const unsigned int channel,
                                    const unsigned int batch_size,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    ){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    int hist_index = num_groups * id_z * 256;
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256;
    unsigned int pixId;
    local uint tmp_histogram [256 * MAX_SIZE];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 ;
    int indx = 0;
    //printf("%d",id_z);
    int temp_index = id_z * 256;
    do
    {
        if (tid < j)
        tmp_histogram[ temp_index +indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((id_z < batch_size) && (id_x < width[id_z]) && (id_y < height[id_z]))
    {
        pixId =  batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex  ;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + inc[id_z]];
        unsigned char pixelB = input[pixId + 2 * inc[id_z]];
        atomic_inc(&tmp_histogram[temp_index + pixelR]);
        atomic_inc(&tmp_histogram[temp_index + pixelG]);
        atomic_inc(&tmp_histogram[temp_index + pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[hist_index + group_indx + tid] = tmp_histogram[temp_index + tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[hist_index + group_indx + indx + tid] = tmp_histogram[ temp_index + indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
     // printf("tmp hist%d", histogramPartial[hist_index + group_indx + tid]);

}

kernel
void partial_histogram_semibatch(__global unsigned char* input,
                                    __global unsigned int *histogramPartial,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int max_width,
                                     const unsigned long batch_index,
                                     const unsigned  int hist_index,// For partial histogram indexing// try out a better way
                                    const unsigned int channel,
                                     const unsigned int inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    ){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);

    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 ;
    int indx = 0;
    //printf("%d",id_z);
    do
    {
        if (tid < j)
        tmp_histogram[ indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( (id_x < width) && (id_y < height))
    {
        pixId =  batch_index + (id_x  + id_y * max_width ) * plnpkdindex  ;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + inc];
        unsigned char pixelB = input[pixId + 2 * inc];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[pixelG]);
        atomic_inc(&tmp_histogram[pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[hist_index + group_indx + tid] = tmp_histogram[ tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[hist_index + group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
      //printf("tmp hist%d", histogramPartial[hist_index + group_indx + tid]);
      //printf("batch index %lu", batch_index);
}



kernel void
histogram_sum_partial(global unsigned int *histogramPartial,
                      global unsigned int *histogram,
                      const unsigned int num_groups)
{
    int  tid = (int)get_global_id(0);
    int  group_indx;
    int  n = num_groups;

    local uint tmp_histogram[256];
    tmp_histogram[tid] = histogramPartial[tid];  
  
    group_indx = 256;
    while (--n > 1)
    {
        tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[group_indx + tid];
        group_indx += 256; 
    }
    histogram[tid] = tmp_histogram[tid];

}

kernel void
histogram_sum_partial_batch(global unsigned int *histogramPartial,
                      global unsigned int *histogram,
                      const unsigned int batch_size,
                      const unsigned int num_groups,
                      const unsigned int channel)
{
    int  tid = (int)get_global_id(0);
    int  bid = (int)get_global_id(1); //With respect batching
    int  group_indx;
    int  n = num_groups;
    local uint tmp_histogram[256];
    unsigned int hist_index = num_groups * bid * 256;
    group_indx = 256;
    if(bid < batch_size){
        tmp_histogram[tid] = histogramPartial[hist_index + tid];    
         while (--n > 1)
            {
                tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[hist_index + group_indx + tid];
                group_indx += 256; 
            }
            histogram[256 * bid + tid] = tmp_histogram[tid];
    }

}

kernel void
histogram_equalize_pln(global unsigned char *input,
                   global unsigned char *output,
                   global unsigned int *cum_histogram,
                   const unsigned int width,
                   const unsigned int height,
                   const unsigned int channel
                   )
{
    float normalize_factor = 255.0 / (height * width * channel);
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int id_z = get_global_id(2);
    unsigned pixId;
    pixId = id_x  + id_y * width + id_z * height * width;
    output[pixId] = cum_histogram[input[pixId]] * (normalize_factor);
}

kernel void
histogram_equalize_pkd(global unsigned char *input,
                   global unsigned char *output,
                   global unsigned int *cum_histogram,
                   const unsigned int width,
                   const unsigned int height,
                   const unsigned int channel
                   )
{
    float normalize_factor = 255.0 / (height * width * channel);
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int id_z = get_global_id(2);
    unsigned pixId;
    pixId = id_x * channel + id_y * width * channel + id_z;
    output[pixId] = round(cum_histogram[input[pixId]] * (normalize_factor));
}


kernel void
histogram_equalize_batch(__global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned int *cum_histogram,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    const unsigned int batch_size,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                   )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    float normalize_factor = 255.0 / (height[id_z] * width[id_z] * channel);
    int indextmp=0;
    unsigned long pixIdx = 0;
    
    if(id_x < width[id_z] && id_y < height[id_z])
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        for(indextmp = 0; indextmp < channel; indextmp++){
                //output[pixIdx] = cum_histogram[ 256 * id_z + input[pixIdx]] * (normalize_factor);
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
    }
}