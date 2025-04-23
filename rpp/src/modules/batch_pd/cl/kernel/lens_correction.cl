#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void lenscorrection_pkd(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
        const float strength,
        const float zoom,
        const float halfWidth,
        const float halfHeight,
        const float correctionRadius,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel

)
{    
    int pix, pix_right, pix_right_down, pix_down, pixVal;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;
    float theta;
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;
    if (r == 0) 
        theta = 1.0;
    else
    theta = atan(r) / r;
    float new_idx = (halfWidth + theta * newX * zoom);
    float new_idy = (halfHeight + theta * newY * zoom);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;
    if ((x >= 0) && (y >= 0) && 
        (x < width - 2) && (y < height - 2))
    {
        pix = input[x * channel + y * width * channel + id_z];
        pix_right = input[(x +1) * channel + y * width * channel + id_z];
        pix_right_down = input[x * channel + (y+1) * width * channel + id_z];
        pix_down = input[(x+1) * channel + (y+1) * width * channel + id_z];

        pixVal = (int)(  pix*(1-x_diff)*(1-y_diff) +  pix_right*(x_diff)*(1-y_diff) +
                    pix_right_down*(y_diff)*(1-x_diff)   +  pix_down*(x_diff*y_diff)) ;
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
    else {
        output[dstpixIdx] = 0;
    }
}

__kernel void lenscorrection_pln(
	    const __global unsigned char* input,
	    __global  unsigned char* output,
        const float strength,
        const float zoom,
        const float halfWidth,
        const float halfHeight,
        const float correctionRadius,
	    const unsigned int height,
	    const unsigned int width,
	    const unsigned int channel
)
{    
    int pix, pix_right, pix_right_down, pix_down, pixVal;
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x + id_y * width + id_z  * channel;
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / (float)correctionRadius;

    float theta;
    if (r == 0) 
        theta = 1.0;
    else
        theta = atan(r) / r;
    float new_idx = (halfWidth + theta * newX * zoom);
    float new_idy = (halfHeight + theta * newY * zoom);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;
    if ((x >= 0) && (y >= 0) && 
        (x < width - 2) && (y < height - 2))
    {
        pix = input[x + y * width + id_z * channel ];
        pix_right = input[(x +1)+ y * width + id_z * channel ];
        pix_right_down = input[x + (y+1) * width + id_z * channel  ];
        pix_down = input[(x+1) + (y+1) * width + id_z * channel ];

        pixVal = (int)(  pix*(1-x_diff)*(1-y_diff) +  pix_right*(x_diff)*(1-y_diff) +
                    pix_right_down*(y_diff)*(1-x_diff)   +  pix_down*(x_diff*y_diff)) ;
        output[dstpixIdx] =  saturate_8u(pixVal);
    }
    else {
        output[dstpixIdx] = 0;
    }
   
}


__kernel void lens_correction_batch(    __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global float *strength,
                                    __global float *zoom,
                                    __global unsigned int *xroi_begin,
                                    __global unsigned int *xroi_end,
                                    __global unsigned int *yroi_begin,
                                    __global unsigned int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex){
    
    int pix, pix_right, pix_right_down, pix_down, pixVal;
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    int dstpixIdx = 0;
    dstpixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    float strength_img = strength[id_z];
    if (strength_img == 0)
        strength_img = 0.000001;
    float temp_val = width[id_z] * width[id_z] + height[id_z] * height[id_z];
    float correctionRadius = (float)sqrt(temp_val) / (float)strength_img;
    float halfWidth = width[id_z]/2.0;
    float halfHeight = height[id_z]/2.0;
    
    float newX = id_x - halfWidth;
    float newY = id_y - halfHeight;
    float r = (float)(sqrt(newX * newX + newY * newY)) / correctionRadius;

    float theta;
    if (r == 0) 
        theta = 1.0;
    else
        theta = atan(r) / r;
    float new_idx = (halfWidth + theta * newX * zoom[id_z]);
    float new_idy = (halfHeight + theta * newY * zoom[id_z]);
    int x = (int) new_idx;
    int y = (int) new_idy;
    float x_diff = new_idx - x;
    float y_diff = new_idy - y;
    //Should be done in FOR LOOP //correct it!!!!!
    
    if ((x >= xroi_begin[id_z]) && (y >= yroi_begin[id_z]) && (x < xroi_end[id_z]) && (y < yroi_end[id_z]))
    {
        for(indextmp=0; indextmp < channel; indextmp ++){
            pix = input[batch_index[id_z] + (x  + y * max_width[id_z] ) * plnpkdindex + indextmp*inc[id_z]];
            // pix_right = input[batch_index[id_z] + ((x+1)  + y * max_width[id_z] ) * plnpkdindex + indextmp*inc[id_z]];
            // pix_right_down = input[batch_index[id_z] + (x  + (y+1) * max_width[id_z] ) * plnpkdindex + indextmp*inc[id_z] ];
            // pix_down = input[batch_index[id_z] + ((x+1)  + (y+1) * max_width[id_z] ) * plnpkdindex + indextmp*inc[id_z]];
            pixVal = (int)(  pix*(1-x_diff)*(1-y_diff) +  pix_right*(x_diff)*(1-y_diff) +
                    pix_right_down*(y_diff)*(1-x_diff)   +  pix_down*(x_diff*y_diff)) ;
            output[dstpixIdx + indextmp*inc[id_z]] =  saturate_8u(pix);
        }
    }

    else {
        dstpixIdx = batch_index[id_z]   + (id_x  + id_y * max_width[id_z] ) * plnpkdindex;
        for(indextmp = 0; indextmp < channel; indextmp++){
            output[dstpixIdx] = 0;
            dstpixIdx += inc[id_z];
        }
    }
}