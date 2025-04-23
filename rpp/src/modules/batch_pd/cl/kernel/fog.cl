#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void fog_planar(  __global unsigned char* input,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float fogValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;
    int pixId= width * id_y  + id_x;
    int c=width*height;
    float check=input[pixId];
    if(channel>1)
    {
        check+=input[pixId+c]+input[pixId+c*2];
        check=check/3;
    }
    if(check >= (240) && fogValue!=0)
    {}
    else if(check>=(170))
    {
        float pixel = ((float) input[pixId])  * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
        input[pixId] = saturate_8u(pixel);
        if(channel>1)
        {
            pixel = ((float) input[pixId+c]) * (1.5 + fogValue) + (7*fogValue);
            input[pixId+c] = saturate_8u(pixel);
            pixel = ((float) input[pixId+c*2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
            input[pixId+c*2] = saturate_8u(pixel);
        }
    }

    else if(check<=(85))
    {
        float pixel = ((float) input[pixId]) * (1.5 + (fogValue*fogValue)) - (fogValue*4) + (130*fogValue);
        input[pixId] = saturate_8u(pixel);
        if(channel>1)
        {
            pixel = ((float) input[pixId+c]) * (1.5 + (fogValue*fogValue)) + (130*fogValue);
            input[pixId+c] = saturate_8u(pixel);
            pixel = ((float) input[pixId+c*2]) * (1.5 + (fogValue*fogValue)) + (fogValue*4) + 130*fogValue;
            input[pixId+c*2] = saturate_8u(pixel);
        }
    }
    else
    {
        float pixel = ((float) input[pixId]) * (1.5 + (fogValue * ( fogValue * 1.414))) - (fogValue*4) + 20 + (100*fogValue);
        input[pixId] = saturate_8u(pixel);
        if(channel>1)
        {
            pixel = ((float) input[pixId+c]) * (1.5 + (fogValue * ( fogValue * 1.414))) + 20 + (100*fogValue);
            input[pixId+c] = saturate_8u(pixel);
            pixel = ((float) input[pixId+c*2]) * (1.5 + (fogValue * ( fogValue * 1.414))) + (fogValue*4) + (100*fogValue);
            input[pixId+c*2] = saturate_8u(pixel);
        }
    }
}

__kernel void fog_pkd(  __global unsigned char* input,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float fogValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;
    int i= width * id_y * channel + id_x * channel;
    float check=input[i]+input[i+1]+input[i+2];
    if(check >= (240*3) && fogValue!=0)
    {}
    else if(check>=(170*3) && fogValue!=0)
    {
        float pixel = ((float) input[i]) * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + fogValue) + (7*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
        input[i+2] = saturate_8u(pixel);
    }
    else if(check<=(85*3) && fogValue!=0)
    {
        float pixel = ((float) input[i]) * (1.5 + (fogValue*fogValue)) - (fogValue*4) + (130*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + (fogValue*fogValue)) + (130*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + (fogValue*fogValue)) + (fogValue*4) + 130*fogValue;
        input[i+2] = saturate_8u(pixel);
    }
    else if(fogValue!=0)
    {
        float pixel = ((float) input[i]) * (1.5 + (fogValue * ( fogValue * 1.414))) - (fogValue*4) + 20 + (100*fogValue);
        input[i] = saturate_8u(pixel);
        pixel = ((float) input[i + 1]) * (1.5 + (fogValue * ( fogValue * 1.414))) + 20 + (100*fogValue);
        input[i+1] = saturate_8u(pixel);
        pixel = ((float) input[i + 2]) * (1.5 + (fogValue * ( fogValue * 1.414))) + (fogValue*4) + (100*fogValue);
        input[i+2] = saturate_8u(pixel);
    }

}

__kernel void fog_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float* fogValue,
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
    float tempFogValue = fogValue[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
    {   
        float check = input[pixIdx];
        if(channel == 3)
            check = (check + input[pixIdx + inc[id_z]] + input[pixIdx + inc[id_z] * 2]) / 3;
        if(check >= (240) && tempFogValue!=0)
        {
            output [pixIdx] = input [pixIdx];
            if(channel > 1)
            {
                output [pixIdx + inc[id_z]] = input [pixIdx + inc[id_z]];
                output [pixIdx + inc[id_z] * 2] = input [pixIdx + inc[id_z] * 2];
            }
        }
        else if(check>=(170) && tempFogValue!=0 )
        {
            float pixel = ((float) input[pixIdx])  * (1.5 + tempFogValue) - (tempFogValue*4) + (7*tempFogValue);
            output[pixIdx] = saturate_8u(pixel);
            if(channel>1)
            {
                pixel = ((float) input[pixIdx + inc[id_z]]) * (1.5 + tempFogValue) + (7*tempFogValue);
                output[pixIdx + inc[id_z]] = saturate_8u(pixel);
                pixel = ((float) input[pixIdx + inc[id_z] * 2]) * (1.5 + tempFogValue) + (tempFogValue*4) + (7*tempFogValue);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(pixel);
            }
        }

        else if(check<=(85) && tempFogValue!=0 )
        {
            float pixel = ((float) input[pixIdx]) * (1.5 + (tempFogValue*tempFogValue)) - (tempFogValue*4) + (130*tempFogValue);
            output[pixIdx] = saturate_8u(pixel);
            if(channel>1)
            {
                pixel = ((float) input[pixIdx + inc[id_z]]) * (1.5 + (tempFogValue*tempFogValue)) + (130*tempFogValue);
                output[pixIdx + inc[id_z]] = saturate_8u(pixel);
                pixel = ((float) input[pixIdx + inc[id_z] * 2]) * (1.5 + (tempFogValue*tempFogValue)) + (tempFogValue*4) + 130*tempFogValue;
                output[pixIdx + inc[id_z] * 2] = saturate_8u(pixel);
            }
        }
        else if(tempFogValue != 0)
        {
            float pixel = ((float) input[pixIdx]) * (1.5 + (tempFogValue * ( tempFogValue * 1.414))) - (tempFogValue*4) + 20 + (100*tempFogValue);
            output[pixIdx] = saturate_8u(pixel);
            if(channel>1)
            {
                pixel = ((float) input[pixIdx + inc[id_z]]) * (1.5 + (tempFogValue * ( tempFogValue * 1.414))) + 20 + (100*tempFogValue);
                output[pixIdx + inc[id_z]] = saturate_8u(pixel);
                pixel = ((float) input[pixIdx + inc[id_z] * 2]) * (1.5 + (tempFogValue * ( tempFogValue * 1.414))) + (tempFogValue*4) + (100*tempFogValue);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(pixel);
            }
        }        
    }
}