__kernel void fisheye_planar(
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
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x + id_y * width  + id_z  * channel;
    float normY = ((float)(2*id_y)/(float) (height))-1;
    float normX = ((float)(2*id_x)/(float) (width))-1;
    float dist = sqrt((normX*normX)+(normY*normY));
    if ((0.0 <= dist) && (dist <= 1.0)) {
        float newDist = sqrt(1.0-dist*dist);
        newDist = (dist + (1.0-newDist)) / 2.0;
        if (newDist <= 1.0) {
            float theta = atan2(normY,normX);
            float newX = newDist*cos(theta);
            float newY = newDist*sin(theta);
            int srcX = (int)(((newX+1)*width)/2.0);
            int srcY = (int)(((newY+1)*height)/2.0);
            int srcpixIdx = srcY * width + srcX + id_z * channel ;
            if (srcpixIdx >= 0 && srcpixIdx < width*height*channel) {
                output[dstpixIdx] = input[srcpixIdx];
            }
        } else {
            output[dstpixIdx] = 0;
        }
    }
}
__kernel void fisheye_packed(
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
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int dstpixIdx = id_x * channel + id_y * width * channel + id_z;
    output[dstpixIdx] = 0;
    float normY = ((float)(2*id_y)/(float) (height))-1;
    float normX = ((float)(2*id_x)/(float) (width))-1;
    float dist = sqrt((normX*normX)+(normY*normY));
    if ((0.0 <= dist) && (dist <= 1.0)) {
        float newDist = sqrt(1.0-dist*dist);
        newDist = (dist + (1.0-newDist)) / 2.0;
        if (newDist <= 1.0) {
            float theta = atan2(normY,normX);
            float newX = newDist*cos(theta);
            float newY = newDist*sin(theta);
            int srcX = (int)(((newX+1)*width)/2.0);
            int srcY = (int)(((newY+1)*height)/2.0);
            int srcpixIdx = srcY * width  * channel + srcX  * channel + id_z ;
            if (srcpixIdx >= 0 && srcpixIdx < width*height*channel) {
                output[dstpixIdx] = input[srcpixIdx];
            }
        }
    }
}


__kernel void fisheye_batch(    __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned int *xroi_begin,
                                    __global unsigned int *xroi_end,
                                    __global unsigned int *yroi_begin,
                                    __global unsigned int *yroi_end,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex)
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    int dstpixIdx = 0;
    dstpixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    float normY = ((float)(2*id_y)/(float) (height[id_z]))-1;
    float normX = ((float)(2*id_x)/(float) (width[id_z]))-1;
    float dist = sqrt((normX*normX)+(normY*normY));
    if ((0.0 <= dist) && (dist <= 1.0)) {
        float newDist = sqrt(1.0-dist*dist);
        newDist = (dist + (1.0-newDist)) / 2.0;
        if (newDist <= 1.0) {
            float theta = atan2(normY,normX);
            float newX = newDist*cos(theta);
            float newY = newDist*sin(theta);
            int srcX = (int)(((newX+1)*width[id_z])/2.0);
            int srcY = (int)(((newY+1)*height[id_z])/2.0);
            int srcpixIdx = batch_index[id_z] + (srcX  + srcY * max_width[id_z] ) * plnpkdindex ;
            // if (srcX < width[id_z] && srcY < height[id_z])
            if(srcY < yroi_end[id_z] && (srcY >=yroi_begin[id_z]) && srcX < xroi_end[id_z] && (srcX >=xroi_begin[id_z]))
            {
                if(srcpixIdx >= batch_index[id_z] && srcpixIdx <= batch_index[id_z+1] )
                {
                    for(indextmp = 0; indextmp < channel; indextmp++)
                    {
                            output[dstpixIdx] = input[srcpixIdx];
                            dstpixIdx += inc[id_z];
                            srcpixIdx += inc[id_z];
                    }
                }
            }
        }
    }
    
}