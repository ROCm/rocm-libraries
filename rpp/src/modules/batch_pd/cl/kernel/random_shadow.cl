__kernel void random_shadow(
    const __global unsigned char* input,
    __global  unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) 
        return;
     int pixIdx = (width * height * id_z) + (width * id_y) + id_x;
    output[pixIdx] = input[pixIdx];
}
__kernel void random_shadow_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int pixIdx = ((y1 - 1 + id_y) * srcwidth) + (x1 + id_x) + (id_z * srcheight * srcwidth);
    if(output[pixIdx] != input[pixIdx] / 2)
    {    
        output[pixIdx] = input[pixIdx] / 2;
    }
}

__kernel void random_shadow_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);

    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int width = x2 - x1;
    int pixIdx = ((y1 - 1 + id_y) * channel * srcwidth) + ((x1 + id_x) * channel) + (id_z);
    if(output[pixIdx] != input[pixIdx] / 2)
    {    
        output[pixIdx] = input[pixIdx] / 2;
    }

}