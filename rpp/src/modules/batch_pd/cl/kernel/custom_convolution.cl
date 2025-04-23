#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void custom_convolution_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = saturate_8u(sum);
}

__kernel void custom_convolution_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = saturate_8u(sum); 
}


__kernel void custom_convolution_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float *kernelValue,
                                    const unsigned int kHeight,
                                    const unsigned int kWidth,
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
    int temp;
    int boundy = (kHeight - 1) / 2;
    int boundx = (kWidth - 1) / 2;
    float sumR = 0, sumG = 0, sumB = 0;
    int kernelIndex = kHeight * kWidth * id_z;
    int counter = kernelIndex;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {   
            for(int i = -boundy ; i <= boundy ; i++)
            {
                for(int j = -boundx ; j <= boundx ; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        sumR += (float)input[index] * kernelValue[counter];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            sumG += (float)input[index] * kernelValue[counter];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            sumB += (float)input[index] * kernelValue[counter];
                        }
                    }
                    counter++;
                }
            }
            
            output[pixIdx] = saturate_8u(sumR);
            if(channel == 3)
            {
                output[pixIdx + inc[id_z]] = saturate_8u(sumG);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(sumB);
            }
        }
        else if((id_x < width[id_z] ) && (id_y < height[id_z]))
        {
            for(indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}