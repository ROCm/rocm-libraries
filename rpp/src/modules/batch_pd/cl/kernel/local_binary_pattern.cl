#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

int power_function(int a, int b)
{
    int product = 1;
    for(int i = 0; i < b; i++)
        product *= product * a;
    return product;
}

__kernel void local_binary_pattern_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;


    int pixIdx = id_y * channel * width + id_x * channel + id_z;

    unsigned int pixel = 0;
    unsigned char neighborhood[9];

    if((id_x - 1) >= 0 && (id_y - 1) >= 0)
        neighborhood[0] = input [pixIdx - width * channel - channel];
    else
        neighborhood[0] = 0;

    if((id_y - 1) >= 0)
        neighborhood[1] = input [pixIdx - width * channel];
    else
        neighborhood[1] = 0;

    if((id_x + 1) <= width && (id_y - 1) >= 0)
        neighborhood[2] = input [pixIdx - width * channel + channel];
    else
        neighborhood[2] = 0;

    if((id_x + 1) <= width)
        neighborhood[3] = input [pixIdx + channel];
    else
        neighborhood[3] = 0;

    if((id_x + 1) <= width && (id_y + 1) <= height)
        neighborhood[4] = input [pixIdx + width * channel + channel];
    else
        neighborhood[4] = 0;

    if((id_y + 1) <= height)
        neighborhood[5] = input [pixIdx + width * channel];
    else
        neighborhood[5] = 0;

    if((id_x - 1) >= 0 && (id_y + 1) <= height)
        neighborhood[6] = input [pixIdx + width * channel -channel];
    else
        neighborhood[6] = 0;

    if((id_x - 1) >= 0)
        neighborhood[7] = input [pixIdx - channel];
    else
        neighborhood[7] = 0;

    neighborhood[8] = input[pixIdx];

    for(int i = 0 ; i < 8 ; i++)
    {
        if(neighborhood[i] - input[pixIdx] >= 0)
        {
            pixel += (unsigned char) power_function(2, i);
        }
    }
    output[pixIdx] = saturate_8u(pixel);

}

__kernel void local_binary_pattern_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;

    unsigned char pixel = 0;
    unsigned char neighborhood[9];

    if((id_x - 1) >= 0 && (id_y - 1) >= 0)
        neighborhood[0] = input [pixIdx - width - 1];
    else
        neighborhood[0] = 0;

    if((id_y - 1) >= 0)
        neighborhood[1] = input [pixIdx - width];
    else
        neighborhood[1] = 0;

    if((id_x + 1) <= width && (id_y - 1) >= 0)
        neighborhood[2] = input [pixIdx - width + 1];
    else
        neighborhood[2] = 0;

    if((id_x + 1) <= width)
        neighborhood[3] = input [pixIdx + 1];
    else
        neighborhood[3] = 0;

    if((id_x + 1) <= width && (id_y + 1) <= height)
        neighborhood[4] = input [pixIdx + width + 1];
    else
        neighborhood[4] = 0;

    if((id_y + 1) <= height)
        neighborhood[5] = input [pixIdx + width];
    else
        neighborhood[5] = 0;

    if((id_x - 1) >= 0 && (id_y + 1) <= height)
        neighborhood[6] = input [pixIdx + width -1];
    else
        neighborhood[6] = 0;

    if((id_x - 1) >= 0)
        neighborhood[7] = input [pixIdx - 1];
    else
        neighborhood[7] = 0;

    neighborhood[8] = input[pixIdx];

    for(int i = 0 ; i < 8 ; i++)
    {
        if(neighborhood[i] - input[pixIdx] >= 0)
        {
            pixel += power_function(2, i);
        }
    }
    output[pixIdx] = saturate_8u(pixel);
}

__kernel void local_binary_pattern_batch(  __global unsigned char* input,
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
    int pixel;
    if(id_x < width[id_z]-1 && id_y < height[id_z]-1 && id_x > 0 && id_y > 0)
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        unsigned char neighborhoodR[3][9];
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            for(int i = 0 ; i < channel ; i++)
            {
                if((id_x - 1) >= 0 && (id_y - 1) >= 0)
                    neighborhoodR[i][0] = input [pixIdx - (max_width[id_z] + 1) * plnpkdindex];
                else
                    neighborhoodR[i][0] = 0;

                if((id_y - 1) >= 0)
                    neighborhoodR[i][1] = input [pixIdx - (max_width[id_z]) * plnpkdindex];
                else
                    neighborhoodR[i][1] = 0;

                if((id_x + 1) < width[id_z] && (id_y - 1) >= 0)
                    neighborhoodR[i][2] = input [pixIdx - (max_width[id_z] - 1) * plnpkdindex];
                else
                    neighborhoodR[i][2] = 0;

                if((id_x + 1) < width[id_z])
                    neighborhoodR[i][3] = input [pixIdx + (1) * plnpkdindex];
                else
                    neighborhoodR[i][3] = 0;

                if((id_x + 1) < width[id_z] && (id_y + 1) < height[id_z])
                    neighborhoodR[i][4] = input [pixIdx + (max_width[id_z] + 1) * plnpkdindex];
                else
                    neighborhoodR[i][4] = 0;

                if((id_y + 1) < height[id_z])
                    neighborhoodR[i][5] = input [pixIdx + (max_width[id_z]) * plnpkdindex];
                else
                    neighborhoodR[i][5] = 0;

                if((id_x - 1) >= 0 && (id_y + 1) < height[id_z])
                    neighborhoodR[i][6] = input [pixIdx + (max_width[id_z] - 1) * plnpkdindex];
                else
                    neighborhoodR[i][6] = 0;

                if((id_x - 1) >= 0)
                    neighborhoodR[i][7] = input [pixIdx - (1) * plnpkdindex];
                else
                    neighborhoodR[i][7] = 0;

                neighborhoodR[i][8] = input[pixIdx];

                pixIdx += inc[id_z];
            }
            pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
            for(int j = 0 ; j < channel ; j++)
            {
                pixel = 0;
                for(int i = 0 ; i < 8 ; i++)
                {
                    if(neighborhoodR[j][i] - input[pixIdx] >= 0)
                    {
                        pixel += powr((float)2, (float)i);
                    }
                }
                output[pixIdx] = (unsigned char) saturate_8u(pixel);
                pixIdx += inc[id_z];
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