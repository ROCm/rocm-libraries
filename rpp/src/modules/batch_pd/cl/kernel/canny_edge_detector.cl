#define RPPABS(a) ((a < 0) ? (-a) : (a))
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void canny_ced_pln3_to_pln1(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int ch = height * width;
    float value = ((input[IPpixIdx] + input[IPpixIdx + ch] + input[IPpixIdx + ch * 2]) / 3);
    output[IPpixIdx] = (unsigned char)value ;
}
__kernel void canny_ced_pkd3_to_pln1(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int OPpixIdx = id_x + id_y * width;
    int IPpixIdx = id_x * channel + id_y * width * channel;
    float value = (input[IPpixIdx] + input[IPpixIdx + 1] + input[IPpixIdx + 2]) / 3;
    output[OPpixIdx] = (unsigned char)value ;
}
__kernel void ced_pln3_to_pln1_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    int ch = height * width;
    float value = ((input[IPpixIdx] + input[IPpixIdx + ch] + input[IPpixIdx + ch * 2]) / 3);
    output[OPpixIdx] = (unsigned char)value ;
}

__kernel void ced_pkd3_to_pln1_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x * (unsigned long)channel + (unsigned long)id_y * (unsigned long)width * (unsigned long)channel;
    float value = (input[IPpixIdx] + input[IPpixIdx + 1] + input[IPpixIdx + 2]) / 3;
    output[OPpixIdx] = (unsigned char)value ;
}

unsigned int power(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
        sum += sum * a;
    return sum;
}

int calcSobelx(int (*a)[3])
{
    int gx[3][3]={-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sum = 0;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += a[i][j] * gx[i][j];
        }
    }
    return sum;
}
int calcSobely(int (*a)[3])
{
    int gy[3][3]={-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int sum = 0;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += a[i][j] * gy[i][j];
        }
    }
    return sum;
}

__kernel void ced_non_max_suppression(  __global unsigned char* input,
                    __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned char min,
                    const unsigned char max
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    float gradient = atan((float)input1[pixIdx] / (float)input2[pixIdx]);
    unsigned char pixel1, pixel2;

    if (RPPABS(gradient) > 1.178097)
    {
        if(id_x != 0)
            pixel1 = input[pixIdx - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1)
            pixel2 = input[pixIdx + 1];
        else
            pixel2 = 0;
    }
    else if (gradient > 0.392699)
    {
        if(id_x != 0 && id_y !=0)
            pixel1 = input[pixIdx - width - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1 && id_y != height - 1)
            pixel2 = input[pixIdx + width + 1];
        else
            pixel2 = 0;
    }
    else if (gradient < -0.392699)
    {
        if(id_x != width - 1 && id_y !=0)
            pixel1 = input[pixIdx - width + 1];
        else
            pixel1 = 0;

        if(id_x != 0 && id_y != height - 1)
            pixel2 = input[pixIdx + width - 1];
        else
            pixel2 = 0;
    }
    else
    {
        if(id_y != 0)
            pixel1 = input[pixIdx - width];
        else
            pixel1 = 0;

        if(id_y != height - 1)
            pixel2 = input[pixIdx + width];
        else
            pixel2 = 0;
    }

    if(input[pixIdx] >= pixel1 && input[pixIdx] >= pixel2)
    {
        if(input[pixIdx] >= max)
            output[pixIdx] = 255;
        else if(input[pixIdx] <= min)
            output[pixIdx] = 0;
        else
            output[pixIdx] = 128;
    }
    else
        output[pixIdx] = 0;
}

__kernel void ced_non_max_suppression_batch(  __global unsigned char* input,
                    __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned char min,
                    const unsigned char max
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    float gradient = atan((float)input1[pixIdx] / (float)input2[pixIdx]);
    unsigned char pixel1, pixel2;

    if (RPPABS(gradient) > 1.178097)
    {
        if(id_x != 0)
            pixel1 = input[pixIdx - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1)
            pixel2 = input[pixIdx + 1];
        else
            pixel2 = 0;
    }
    else if (gradient > 0.392699)
    {
        if(id_x != 0 && id_y !=0)
            pixel1 = input[pixIdx - width - 1];
        else
            pixel1 = 0;

        if(id_x != width - 1 && id_y != height - 1)
            pixel2 = input[pixIdx + width + 1];
        else
            pixel2 = 0;
    }
    else if (gradient < -0.392699)
    {
        if(id_x != width - 1 && id_y !=0)
            pixel1 = input[pixIdx - width + 1];
        else
            pixel1 = 0;

        if(id_x != 0 && id_y != height - 1)
            pixel2 = input[pixIdx + width - 1];
        else
            pixel2 = 0;
    }
    else
    {
        if(id_y != 0)
            pixel1 = input[pixIdx - width];
        else
            pixel1 = 0;

        if(id_y != height - 1)
            pixel2 = input[pixIdx + width];
        else
            pixel2 = 0;
    }

    if(input[pixIdx] >= pixel1 && input[pixIdx] >= pixel2)
    {
        if(input[pixIdx] >= max)
            output[pixIdx] = 255;
        else if(input[pixIdx] <= min)
            output[pixIdx] = 0;
        else
            output[pixIdx] = 128;
    }
    else
        output[pixIdx] = 0;
}

__kernel void canny_edge(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned char min,
                    const unsigned char max
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    int pixIdx = id_y * width + id_x + id_z * width * height;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int value = 0;
    int value1 =0;
    if(input[pixIdx] == 0 || input[pixIdx] == 255)
    {
        output[pixIdx] = input[pixIdx];
    }
    else
    {
        for(int i = -1 ; i <= 1 ; i++)
        {
            for(int j = -1 ; j <= 1 ; j++)
            {
                if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
                {
                    unsigned int index = pixIdx + j + (i * width);
                    if(input[index] == 255)
                    {
                        output[pixIdx] = 255;
                        break;
                    }
                }
            }
        }
    }
}

__kernel void canny_edge_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned char min,
                    const unsigned char max,
                    const unsigned long batchIndex,
                    const unsigned int originalChannel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    unsigned long IPpixIdx, OPpixIdx;
    IPpixIdx = (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x ;
    if(originalChannel == 1)
        OPpixIdx = (unsigned long)batchIndex + (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
    else
        OPpixIdx = (unsigned long)IPpixIdx;

    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int value = 0;
    int value1 =0;
    if(input[IPpixIdx] == 0 || input[IPpixIdx] == 255)
    {
        output[OPpixIdx] = input[IPpixIdx];
    }
    else
    {
        for(int i = -1 ; i <= 1 ; i++)
        {
            for(int j = -1 ; j <= 1 ; j++)
            {
                if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
                {
                    unsigned long index = (unsigned long)IPpixIdx + (unsigned long)j + ((unsigned long)i * (unsigned long)width);
                    if(input[index] == 255)
                    {
                        output[OPpixIdx] = 255;
                        break;
                    }
                }
            }
        }
    }
}

__kernel void ced_pln1_to_pln3_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    unsigned long IPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    int ch = height * width;
    output[OPpixIdx] = input[IPpixIdx];
    output[OPpixIdx + ch] = input[IPpixIdx];
    output[OPpixIdx + ch * 2] = input[IPpixIdx];
}

__kernel void ced_pln1_to_pkd3_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    unsigned long IPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x * (unsigned long)channel + (unsigned long)id_y * (unsigned long)width * (unsigned long)channel;
    output[OPpixIdx] = input[IPpixIdx];
    output[OPpixIdx + 1] = input[IPpixIdx];
    output[OPpixIdx + 2] = input[IPpixIdx];
}
__kernel void canny_ced_pln1_to_pln3(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int ch = height * width;
    output[IPpixIdx] = input[IPpixIdx];
    output[IPpixIdx + ch] = input[IPpixIdx];
    output[IPpixIdx + ch * 2] = input[IPpixIdx];
}
__kernel void canny_ced_pln1_to_pkd3(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx = id_x * channel + id_y * width * channel;
    output[OPpixIdx] = input[IPpixIdx];
    output[OPpixIdx + 1] = input[IPpixIdx];
    output[OPpixIdx + 2] = input[IPpixIdx];
}