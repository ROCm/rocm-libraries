#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define SIZE 7*7

/*__kernel void median_filter_pkd*/

__kernel void median_filter_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;

    int c[SIZE];
    int counter = 0;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j > 1 && id_x + j < width - 1 && id_y + i > 1 && id_y + i < height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                c[counter] = input[index];
            }
            else
                c[counter] = 0;
            counter++;
        }
    }
    int pos;
    int max = 0;
    for (int i = 0; i < counter; i++)          
    {
        for (int j = i; j < counter; j++)  
        {
            if (max < c[j]) 
            {
                max = c[j];
                pos = j;
            }
        }
        max = 0;
        int temp = c[pos];
        c[pos] = c[i];
        c[i] = temp;
    }
    counter = kernelSize * bound + bound + 1;
    output[pixIdx] = c[counter];
}

__kernel void median_filter_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    int pixIdx = id_y * width + id_x + id_z * width * height;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int c[SIZE];
    int counter = 0;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j > 0 && id_x + j < width - 1 && id_y + i > 0 && id_y + i < height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                c[counter] = input[index];
            }
            else
                c[counter] = 0;
            counter++;
        }
    }
    int pos;
    int max = 0;
    for (int i = 0; i < counter; i++)          
    {
        for (int j = i; j < counter; j++)  
        {
            if (max < c[j]) 
            {
                max = c[j];
                pos = j;
            }
        }
        max = 0;
        int temp = c[pos];
        c[pos] = c[i];
        c[i] = temp;
    }
    counter = kernelSize * bound + bound + 1;
    output[pixIdx] = c[counter];
    output[pixIdx] = input[pixIdx];
}

__kernel void median_filter_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned int *kernelSize,
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
    unsigned char valuer,valuer1,valueg,valueg1,valueb,valueb1;
    int kernelSizeTemp = kernelSize[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int temp;
    // printf("%d", id_x);
    int value = 0;
    int value1 =0;
    int counter = 0;
    int r[SIZE], g[SIZE], b[SIZE], maxR = 0, maxG = 0, maxB = 0, posR, posG, posB;
    int bound = (kernelSizeTemp - 1) / 2;
    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(int i = -bound ; i <= bound ; i++)
        {
            for(int j = -bound ; j <= bound ; j++)
            {
                if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                {
                    unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                    r[counter] = input[index];
                    if(channel == 3)
                    {
                        index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                        g[counter] = input[index];
                        index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                        b[counter] = input[index];
                    }
                }
                else
                {
                    r[counter] = 0;
                    if(channel == 3)
                    {
                        g[counter] = 0;
                        b[counter] = 0;
                    }
                }
                counter++;
            }
        }

        for (int i = 0; i < counter; i++)          
        {
            posB = i;
            posG = i;
            posR = i;
            for (int j = i; j < counter; j++)  
            {
                if (maxR < r[j]) 
                {
                    maxR = r[j];
                    posR = j;
                }
                if (maxG < g[j]) 
                {
                    maxG = g[j];
                    posG = j;
                }
                if (maxB < b[j]) 
                {
                    maxB = b[j];
                    posB = j;
                }
            }
            maxR = 0;
            maxG = 0;
            maxB = 0;

            int temp;

            temp = r[posR];
            r[posR] = r[i];
            r[i] = temp;

            temp = g[posG];
            g[posG] = g[i];
            g[i] = temp;

            temp = b[posB];
            b[posB] = b[i];
            b[i] = temp;
        }

        counter = kernelSizeTemp * bound + bound + 1;
        output[pixIdx] = r[counter];
        if(channel == 3)
        {
            output[pixIdx + inc[id_z]] = g[counter];
            output[pixIdx + inc[id_z] * 2] = b[counter];
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

