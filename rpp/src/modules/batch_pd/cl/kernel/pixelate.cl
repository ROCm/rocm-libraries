#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void pixelate_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    if (x * 7 >= width || y * 7 >= height || c >= channel) return;

    y = y * 7;
    x = x * 7;
    int sum = 0;
    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width && y + i >= 0 && x + j >= 0)
            {    
                sum += input[((y + i) * width * channel + (x + j) * channel + c)];
            }
        }
    }
    sum /= 49;

    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width)
            {    
                output[((y + i) * width * channel + (x + j) * channel + c)] = saturate_8u(sum);
            }
        }
    }
    
}

__kernel void pixelate_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    if (x * 7 >= width || y * 7 >= height || c >= channel) return;

    y = y * 7;
    x = x * 7;
    int sum = 0;

    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width && y + i >= 0 && x + j >= 0)
            {    
                sum += input[(y + i) * width + (x + j) + c * height * width];
            }
        }
    }
    sum /= 49;

    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width)
            {    
                output[(y + i) * width + (x + j) + c * height * width] = saturate_8u(sum);
            }
        }
    }
}


__kernel void pixelate_batch(  __global unsigned char* input,
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
    long pixIdx = 0;
    int x,y;
    if(id_x * 7 < width[id_z] && id_y * 7 < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        if((id_y * 7 >= yroi_begin[id_z] ) && (id_y * 7 <= yroi_end[id_z]) && (id_x * 7 >= xroi_begin[id_z]) && (id_x * 7 <= xroi_end[id_z]))
        {  
            y = id_y * 7;
            x = id_x * 7;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            for(int i = 0 ; i < 7 ; i++)
            {
                for(int j = 0 ; j < 7 ; j++)
                {
                    if(y + i < height[id_z] && x + j < width[id_z] && y + i >= 0 && x + j >= 0)
                    {    
                        sum1 += input[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex];
                        if(channel == 3)
                        {
                            sum2 += input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]];
                            sum3 += input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2];
                        }
                    }
                }
            }
            sum1 /= 49;
            sum3 /= 49;
            sum2 /= 49;
            for(int i = 0 ; i < 7 ; i++)
            {
                for(int j = 0 ; j < 7 ; j++)
                {
                    if(y + i < height[id_z] && x + j < width[id_z])
                    {    
                        output[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex] = saturate_8u(sum1);
                        if(channel == 3)
                        {
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]] = saturate_8u(sum2);
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2] = saturate_8u(sum3);
                        }
                    }
                }
            }
        }
        else if((id_x * 7 < width[id_z] ) && (id_y * 7 < height[id_z]))
        {
            y = id_y * 7;
            x = id_x * 7;
            for(int i = 0 ; i < 7 ; i++)
            {
                for(int j = 0 ; j < 7 ; j++)
                {
                    if(y + i < height[id_z] && x + j < width[id_z])
                    {    
                        output[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex] = input[batch_index[id_z] + (((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex];
                        if(channel == 3)
                        {
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]] = input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z]];
                            output[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2] =  input[batch_index[id_z] + ((((y + i) * max_width[id_z]) + (x + j)) * plnpkdindex) + inc[id_z] * 2];
                        }
                    }
                }
            }
        }
    }
}