#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define RPPINRANGE(a, x, y) ((a >= x) && (a <= y) ? 1 : 0)
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPABS(a) ((a < 0) ? (-a) : (a))
#define RPPISGREATER(pixel, value)  ((pixel > value) ? 1 : 0)
#define RPPISLESSER(pixel, value)  ((pixel < value) ? -1 : 0)

__kernel void ced_pln3_to_pln1(  __global unsigned char* input,
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
__kernel void ced_pkd3_to_pln1(  __global unsigned char* input,
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

__kernel void fast_corner_detector(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned char threshold,
                    const unsigned int numOfPixels
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y * width + id_x;

    unsigned int bCValues[16], bCOutputMin[16], bCOutputMax[16];
    bCValues[0] = (id_y - 3 >= 0) ? (input[pixIdx + width * (-3)]) : 0;
    bCValues[1] = (id_y - 3 >= 0 && id_x + 1 < width) ? (input[pixIdx - width * 3 + 1]) : 0;
    bCValues[2] = (id_y - 2 >= 0 && id_x + 2 < width) ? (input[pixIdx - width * 2 + 2]) : 0;
    bCValues[3] = (id_y - 1 >= 0 && id_x + 3 < width) ? (input[pixIdx - width  + 3]) : 0;
    bCValues[4] = (id_x + 3 < width) ? (input[pixIdx + 3]) : 0;
    bCValues[5] = (id_y + 1 < height && id_x + 3 < width) ? (input[pixIdx + width + 3]) : 0;
    bCValues[6] = (id_y + 2 < height && id_x + 2 < width) ? (input[pixIdx + width * 2 + 2]) : 0;
    bCValues[7] = (id_y + 3 < height && id_x + 1 < width) ? (input[pixIdx + width * 3 + 1]) : 0;
    bCValues[8] = (id_y + 3 < height) ? (input[pixIdx + width * 3]) : 0;
    bCValues[9] = (id_y + 3 < height && id_x - 1 >= 0) ? (input[pixIdx + width * 3 - 1]) : 0;
    bCValues[10] = (id_y + 2 < height && id_x - 2 >= 0) ? (input[pixIdx + width * 2 - 2]) : 0;
    bCValues[11] = (id_y + 1 < height && id_x - 3 >= 0) ? (input[pixIdx + width - 3]) : 0;
    bCValues[12] = (id_x - 3 >= 0) ? (input[pixIdx - 3]) : 0;
    bCValues[13] = (id_y - 1 >= 0 && id_x - 3 >= 0) ? (input[pixIdx - width - 3]) : 0;
    bCValues[14] = (id_y - 2 >= 0 && id_x - 2 >= 0) ? (input[pixIdx + width * (-2) - 2]) : 0;
    bCValues[15] = (id_y - 3 >= 0 && id_x - 1 >= 0) ? (input[pixIdx + width * (-3) - 1]) : 0;

    unsigned char max = saturate_8u(input[pixIdx] + threshold);
    unsigned char min = saturate_8u(input[pixIdx] - threshold);
    unsigned int conditions = 0;

    float flag = 0;
    
    int a = 0,b = 0,c = 0,d = 0;
    a = ((RPPISLESSER(bCValues[1], min) == -1) ? -1 : ((RPPISGREATER(bCValues[1], max) == 1) ? 1 : 0));
    b = ((RPPISLESSER(bCValues[8], min) == -1) ? -1 : ((RPPISGREATER(bCValues[8], max) == 1) ? 1 : 0));
    c = ((RPPISLESSER(bCValues[4], min) == -1) ? -1 : ((RPPISGREATER(bCValues[4], max) == 1) ? 1 : 0));
    d = ((RPPISLESSER(bCValues[12], min) == -1) ? -1 : ((RPPISGREATER(bCValues[12], max) == 1) ? 1 : 0));    

    flag = a + b + c + d;
    if(flag >= 2 || flag <= -2)
    {
        conditions += (a == 0) ? 1 : 0;
        conditions += (b == 0) ? 1 : 0;
        conditions += (c == 0) ? 1 : 0;
        conditions += (d == 0) ? 1 : 0;
        if(conditions < 2)
        {
            output[pixIdx] = 0;
            return;
        }
        else
        {
            bCOutputMin[0] = RPPISLESSER(bCValues[0], min);
            bCOutputMin[1] = RPPISLESSER(bCValues[1], min);
            bCOutputMin[2] = RPPISLESSER(bCValues[2], min);
            bCOutputMin[3] = RPPISLESSER(bCValues[3], min);
            bCOutputMin[4] = RPPISLESSER(bCValues[4], min);
            bCOutputMin[5] = RPPISLESSER(bCValues[5], min);
            bCOutputMin[6] = RPPISLESSER(bCValues[6], min);
            bCOutputMin[7] = RPPISLESSER(bCValues[7], min);
            bCOutputMin[8] = RPPISLESSER(bCValues[8], min);
            bCOutputMin[9] = RPPISLESSER(bCValues[9], min);
            bCOutputMin[10] = RPPISLESSER(bCValues[10], min);
            bCOutputMin[11] = RPPISLESSER(bCValues[11], min);
            bCOutputMin[12] = RPPISLESSER(bCValues[12], min);
            bCOutputMin[13] = RPPISLESSER(bCValues[13], min);
            bCOutputMin[14] = RPPISLESSER(bCValues[14], min);
            bCOutputMin[15] = RPPISLESSER(bCValues[15], min);  

            bCOutputMax[0] = RPPISGREATER(bCValues[0], max);
            bCOutputMax[1] = RPPISGREATER(bCValues[1], max);
            bCOutputMax[2] = RPPISGREATER(bCValues[2], max);
            bCOutputMax[3] = RPPISGREATER(bCValues[3], max);
            bCOutputMax[4] = RPPISGREATER(bCValues[4], max);
            bCOutputMax[5] = RPPISGREATER(bCValues[5], max);
            bCOutputMax[6] = RPPISGREATER(bCValues[6], max);
            bCOutputMax[7] = RPPISGREATER(bCValues[7], max);
            bCOutputMax[8] = RPPISGREATER(bCValues[8], max);
            bCOutputMax[9] = RPPISGREATER(bCValues[9], max);
            bCOutputMax[10] = RPPISGREATER(bCValues[10], max);
            bCOutputMax[11] = RPPISGREATER(bCValues[11], max);
            bCOutputMax[12] = RPPISGREATER(bCValues[12], max);
            bCOutputMax[13] = RPPISGREATER(bCValues[13], max);
            bCOutputMax[14] = RPPISGREATER(bCValues[14], max);
            bCOutputMax[15] = RPPISGREATER(bCValues[15], max); 
        }
        
        int max = 0, min = 0;
        max = bCOutputMax[0] + bCOutputMax[1] + bCOutputMax[2] + bCOutputMax[3] + bCOutputMax[4] + bCOutputMax[5] + bCOutputMax[6] + bCOutputMax[7] + bCOutputMax[8] + bCOutputMax[9] + bCOutputMax[10] + bCOutputMax[11] + bCOutputMax[12] + bCOutputMax[13] + bCOutputMax[14] + bCOutputMax[15];
        min = bCOutputMin[0] + bCOutputMin[1] + bCOutputMin[2] + bCOutputMin[3] + bCOutputMin[4] + bCOutputMin[5] + bCOutputMin[6] + bCOutputMin[7] + bCOutputMin[8] + bCOutputMin[9] + bCOutputMin[10] + bCOutputMin[11] + bCOutputMin[12] + bCOutputMin[13] + bCOutputMin[14] + bCOutputMin[15]; 
        
        min = -min;

        if(min >= numOfPixels || max >= numOfPixels)
        {
            unsigned int count = 0;
            unsigned int maxLength = 0;
            if(min >= max)
            {
                for (int i = 0; i < 32; i++)
                {
                    if (bCOutputMin[(i % 16)] == -1)
                    {
                        count = 0;
                        if (i >= 16)
                        {
                            break;
                        } 
                    }
                    else
                    {
                        count++;
                        maxLength = RPPMAX2(maxLength, count);
                    }
                }   
            }
            else
            {
                for (int i = 0; i < 32; i++)
                {
                    if (bCOutputMax[(i % 16)] == 11)
                    {
                        count = 0;
                        if (i >= 16)
                        {
                            break;
                        } 
                    }
                    else
                    {
                        count++;
                        maxLength = RPPMAX2(maxLength, count);
                    }
                }  
            }
            if (maxLength >= numOfPixels)
            {
                output[pixIdx] = 255;
            }
            else
            {
                output[pixIdx] = 0;
            } 
        }
        else
        {
            output[pixIdx] = 0;
            return;
        }
    }
}

__kernel void fast_corner_detector_nms_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y * width + id_x;

    unsigned int bCValues[16];

    bCValues[0] = (id_y - 3 >= 0) ? (input[pixIdx + width * (-3)]) : 0;
    bCValues[1] = (id_y - 3 >= 0 && id_x + 1 < width) ? (input[pixIdx - width * 3 + 1]) : 0;
    bCValues[2] = (id_y - 2 >= 0 && id_x + 2 < width) ? (input[pixIdx - width * 2 + 2]) : 0;
    bCValues[3] = (id_y - 1 >= 0 && id_x + 3 < width) ? (input[pixIdx - width  + 3]) : 0;
    bCValues[4] = (id_x + 3 < width) ? (input[pixIdx + 3]) : 0;
    bCValues[5] = (id_y + 1 < height && id_x + 3 < width) ? (input[pixIdx + width + 3]) : 0;
    bCValues[6] = (id_y + 2 < height && id_x + 2 < width) ? (input[pixIdx + width * 2 + 2]) : 0;
    bCValues[7] = (id_y + 3 < height && id_x + 1 < width) ? (input[pixIdx + width * 3 + 1]) : 0;
    bCValues[8] = (id_y + 3 < height) ? (input[pixIdx + width * 3]) : 0;
    bCValues[9] = (id_y + 3 < height && id_x - 1 >= 0) ? (input[pixIdx + width * 3 - 1]) : 0;
    bCValues[10] = (id_y + 2 < height && id_x - 2 >= 0) ? (input[pixIdx + width * 2 - 2]) : 0;
    bCValues[11] = (id_y + 1 < height && id_x - 3 >= 0) ? (input[pixIdx + width - 3]) : 0;
    bCValues[12] = (id_x - 3 >= 0) ? (input[pixIdx - 3]) : 0;
    bCValues[13] = (id_y - 1 >= 0 && id_x - 3 >= 0) ? (input[pixIdx - width - 3]) : 0;
    bCValues[14] = (id_y - 2 >= 0 && id_x - 2 >= 0) ? (input[pixIdx + width * (-2) - 2]) : 0;
    bCValues[15] = (id_y - 3 >= 0 && id_x - 1 >= 0) ? (input[pixIdx + width * (-3) - 1]) : 0;



    unsigned int valueCircle = 0, valueNeighbour = 0;
    valueCircle = RPPABS((input[pixIdx] - bCValues[0]) + (input[pixIdx] - bCValues[1]) + (input[pixIdx] - bCValues[2]) + (input[pixIdx] - bCValues[3]) + (input[pixIdx] - bCValues[4]) + (input[pixIdx] - bCValues[5]) + (input[pixIdx] - bCValues[6]) + (input[pixIdx] - bCValues[7]) + (input[pixIdx] - bCValues[8]) + (input[pixIdx] - bCValues[9]) + (input[pixIdx] - bCValues[10]) + (input[pixIdx] - bCValues[11]) + (input[pixIdx] - bCValues[12]) + (input[pixIdx] - bCValues[13]) + (input[pixIdx] - bCValues[14]) + (input[pixIdx] - bCValues[15]));

    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            unsigned int index = pixIdx + (j * channel) + (i * width * channel);
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                valueNeighbour += RPPABS(input[pixIdx] - input[index]);
            }
            else
            {
                valueNeighbour += RPPABS(input[pixIdx]);
            }
        }
    }

    if(valueCircle > valueNeighbour)
    {
        if(input[pixIdx] == 255)
        {
            if(id_y - 3 >= 0)
                output[pixIdx + width * (-3)] = 255;
            if(id_y - 3 >= 0 && id_x + 1 < width)
                output[pixIdx - width * 3 + 1] = 255;
            if(id_y - 2 >= 0 && id_x + 2 < width) 
                output[pixIdx - width * 2 + 2] = 255;
            if(id_y - 1 >= 0 && id_x + 3 < width) 
                output[pixIdx - width  + 3] = 255;
            if(id_x + 3 < width) 
                output[pixIdx + 3] = 255;
            if(id_y + 1 < height && id_x + 3 < width) 
                output[pixIdx + width + 3] = 255;
            if(id_y + 2 < height && id_x + 2 < width) 
                output[pixIdx + width * 2 + 2] = 255;
            if(id_y + 3 < height && id_x + 1 < width) 
                output[pixIdx + width * 3 + 1] = 255;
            if(id_y + 3 < height) 
                output[pixIdx + width * 3] = 255;
            if(id_y + 3 < height && id_x - 1 >= 0) 
                output[pixIdx + width * 3 - 1] = 255;
            if(id_y + 2 < height && id_x - 2 >= 0) 
                output[pixIdx + width * 2 - 2] = 255;
            if(id_y + 1 < height && id_x - 3 >= 0) 
                output[pixIdx + width - 3] = 255;
            if(id_x - 3 >= 0) 
                output[pixIdx - 3] = 255;
            if(id_y - 1 >= 0 && id_x - 3 >= 0) 
                output[pixIdx - width - 3] = 255;
            if(id_y - 2 >= 0 && id_x - 2 >= 0) 
                output[pixIdx + width * (-2) - 2] = 255;
            if(id_y - 3 >= 0 && id_x - 1 >= 0) 
                output[pixIdx + width * (-3) - 1] = 255;
        }
    }
}

__kernel void fast_corner_detector_nms_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y * width + id_x;
    int opPixIdx = id_y * width * 3 + id_x * 3;

    unsigned int bCValues[16];

    bCValues[0] = (id_y - 3 >= 0) ? (input[pixIdx + width * (-3)]) : 0;
    bCValues[1] = (id_y - 3 >= 0 && id_x + 1 < width) ? (input[pixIdx - width * 3 + 1]) : 0;
    bCValues[2] = (id_y - 2 >= 0 && id_x + 2 < width) ? (input[pixIdx - width * 2 + 2]) : 0;
    bCValues[3] = (id_y - 1 >= 0 && id_x + 3 < width) ? (input[pixIdx - width  + 3]) : 0;
    bCValues[4] = (id_x + 3 < width) ? (input[pixIdx + 3]) : 0;
    bCValues[5] = (id_y + 1 < height && id_x + 3 < width) ? (input[pixIdx + width + 3]) : 0;
    bCValues[6] = (id_y + 2 < height && id_x + 2 < width) ? (input[pixIdx + width * 2 + 2]) : 0;
    bCValues[7] = (id_y + 3 < height && id_x + 1 < width) ? (input[pixIdx + width * 3 + 1]) : 0;
    bCValues[8] = (id_y + 3 < height) ? (input[pixIdx + width * 3]) : 0;
    bCValues[9] = (id_y + 3 < height && id_x - 1 >= 0) ? (input[pixIdx + width * 3 - 1]) : 0;
    bCValues[10] = (id_y + 2 < height && id_x - 2 >= 0) ? (input[pixIdx + width * 2 - 2]) : 0;
    bCValues[11] = (id_y + 1 < height && id_x - 3 >= 0) ? (input[pixIdx + width - 3]) : 0;
    bCValues[12] = (id_x - 3 >= 0) ? (input[pixIdx - 3]) : 0;
    bCValues[13] = (id_y - 1 >= 0 && id_x - 3 >= 0) ? (input[pixIdx - width - 3]) : 0;
    bCValues[14] = (id_y - 2 >= 0 && id_x - 2 >= 0) ? (input[pixIdx + width * (-2) - 2]) : 0;
    bCValues[15] = (id_y - 3 >= 0 && id_x - 1 >= 0) ? (input[pixIdx + width * (-3) - 1]) : 0;



    unsigned int valueCircle = 0, valueNeighbour = 0;
    valueCircle = RPPABS((input[pixIdx] - bCValues[0]) + (input[pixIdx] - bCValues[1]) + (input[pixIdx] - bCValues[2]) + (input[pixIdx] - bCValues[3]) + (input[pixIdx] - bCValues[4]) + (input[pixIdx] - bCValues[5]) + (input[pixIdx] - bCValues[6]) + (input[pixIdx] - bCValues[7]) + (input[pixIdx] - bCValues[8]) + (input[pixIdx] - bCValues[9]) + (input[pixIdx] - bCValues[10]) + (input[pixIdx] - bCValues[11]) + (input[pixIdx] - bCValues[12]) + (input[pixIdx] - bCValues[13]) + (input[pixIdx] - bCValues[14]) + (input[pixIdx] - bCValues[15]));

    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            unsigned int index = pixIdx + (j * channel) + (i * width * channel);
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                valueNeighbour += RPPABS(input[pixIdx] - input[index]);
            }
            else
            {
                valueNeighbour += RPPABS(input[pixIdx]);
            }
        }
    }

    if(valueCircle > valueNeighbour)
    {
        if(input[pixIdx] == 255)
        {
            if(id_y - 3 >= 0)
            {
                output[opPixIdx - width * 9] = 255;
                output[opPixIdx - width * 9 + 1] = 0;
                output[opPixIdx - width * 9 + 2] = 0;
            }
            if(id_y - 3 >= 0 && id_x + 1 < width)
            {
                output[opPixIdx - width * 9 + 3] = 255;
                output[opPixIdx - width * 9 + 3 + 1] = 0;
                output[opPixIdx - width * 9 + 3 + 2] = 0;
            }
            if(id_y - 2 >= 0 && id_x + 2 < width) 
            {
                output[opPixIdx - width * 6 + 6] = 255;
                output[opPixIdx - width * 6 + 6 + 1] = 0;
                output[opPixIdx - width * 6 + 6 + 2] = 0;
            }
            if(id_y - 1 >= 0 && id_x + 3 < width) 
            {
                output[opPixIdx - width * 3  + 9] = 255;
                output[opPixIdx - width * 3  + 9 + 1] = 0;
                output[opPixIdx - width * 3  + 9 + 2] = 0;
            }
            if(id_x + 3 < width) 
            {
                output[opPixIdx + 9] = 255;
                output[opPixIdx + 9 + 1] = 0;
                output[opPixIdx + 9 + 2] = 0;
            }
            if(id_y + 1 < height && id_x + 3 < width) 
            {
                output[opPixIdx + width * 3 + 9] = 255;
                output[opPixIdx + width * 3 + 9 + 1] = 0;
                output[opPixIdx + width * 3 + 9 + 2] = 0;
            }
            if(id_y + 2 < height && id_x + 2 < width) 
            {
                output[opPixIdx + width * 6 + 6] = 255;
                output[opPixIdx + width * 6 + 6 + 1] = 0;
                output[opPixIdx + width * 6 + 6 + 2] = 0;
            }
            if(id_y + 3 < height && id_x + 1 < width) 
            {
                output[opPixIdx + width * 9 + 3] = 255;
                output[opPixIdx + width * 9 + 3 + 1] = 0;
                output[opPixIdx + width * 9 + 3 + 2] = 0;
            }
            if(id_y + 3 < height) 
            {
                output[opPixIdx + width * 9] = 255;
                output[opPixIdx + width * 9 + 1] = 0;
                output[opPixIdx + width * 9 + 2] = 0;
            }
            if(id_y + 3 < height && id_x - 1 >= 0) 
            {
                output[opPixIdx + width * 9 - 3] = 255;
                output[opPixIdx + width * 9 - 3 + 1] = 0;
                output[opPixIdx + width * 9 - 3 + 2] = 0;
            }
            if(id_y + 2 < height && id_x - 2 >= 0) 
            {
                output[opPixIdx + width * 6 - 6] = 255;
                output[opPixIdx + width * 6 - 6 + 1] = 0;
                output[opPixIdx + width * 6 - 6 + 2] = 0;
            }
            if(id_y + 1 < height && id_x - 3 >= 0) 
            {
                output[opPixIdx + width * 3 - 9] = 255;
                output[opPixIdx + width * 3 - 9 + 1] = 0;
                output[opPixIdx + width * 3 - 9 + 2] = 0;
            }
            if(id_x - 3 >= 0) 
            {
                output[opPixIdx - 9] = 255;
                output[opPixIdx - 9 + 1] = 0;
                output[opPixIdx - 9 + 2] = 0;
            }
            if(id_y - 1 >= 0 && id_x - 3 >= 0) 
            {
                output[opPixIdx - width * 3 - 9] = 255;
                output[opPixIdx - width * 3 - 9 + 1] = 0;
                output[opPixIdx - width * 3 - 9 + 2] = 0;
            }
            if(id_y - 2 >= 0 && id_x - 2 >= 0) 
            {
                output[opPixIdx - width * 6 - 6] = 255;
                output[opPixIdx - width * 6 - 6 + 1] = 0;
                output[opPixIdx - width * 6 - 6 + 2] = 0;
            }
            if(id_y - 3 >= 0 && id_x - 1 >= 0) 
            {
                output[opPixIdx - width * 9 - 3] = 255;
                output[opPixIdx - width * 9 - 3 + 1] = 0;
                output[opPixIdx - width * 9 - 3 + 2] = 0;
            }      
        }
    }
}