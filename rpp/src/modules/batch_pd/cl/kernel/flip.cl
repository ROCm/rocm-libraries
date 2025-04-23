__kernel void flip_horizontal_planar(
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

    int oPixIdx =   id_x + id_y * width + id_z * width * height;

    int nPixIdx =   id_x + (height-1 - id_y) * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_vertical_planar(
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

    int oPixIdx =   id_x + id_y * width + id_z * width * height;

    // TODO:Vertical flip has to be fixed

    int nPixIdx =   (width-1 - id_x) + id_y * width + id_z * width * height;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_bothaxis_planar(
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

    int oPixIdx =   id_x + id_y * width + id_z * width * height;

    // TODO:Vertical flip has to be fixed
    int nPixIdx =   (width-1 - id_x) + (height-1 - id_y) * width + id_z * width * height;


    output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_horizontal_packed(
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

    int oPixIdx =   id_x*channel +  id_y*width*channel + id_z ;
    /*             |size element | size Row |Channel | */

    int nPixIdx =   id_x*channel + (height-1 - id_y)*width*channel + id_z ;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_vertical_packed(
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

    int oPixIdx =   id_x*channel +  id_y*width*channel + id_z ;
    /*             |size element | size Row |Channel | */

    int nPixIdx =   (width-1 - id_x)*channel + id_y*width*channel + id_z ;

	output[nPixIdx] = input[oPixIdx];

}

__kernel void flip_bothaxis_packed(
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

    int oPixIdx =   id_x*channel +  id_y*width*channel + id_z ;
    /*             |size element | size Row |Channel | */

    int nPixIdx =   (width-1 - id_x)*channel + (height-1 - id_y)*width*channel + id_z ;

	output[nPixIdx] = input[oPixIdx];

}


__kernel void flip_batch(__global unsigned char* srcPtr,
                                    __global unsigned char* dstPtr,
                                    __global unsigned int *flipAxis,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    __global unsigned int *xroi_begin,
                                    __global unsigned int *xroi_end,
                                    __global unsigned int *yroi_begin,
                                    __global unsigned int *yroi_end,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd)
                                    ) 
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long src_pixIdx = 0, dst_pixIdx = 0; 

   if(id_y < yroi_end[id_z] && (id_y >=yroi_begin[id_z]) && id_x < xroi_end[id_z] && (id_x >=xroi_begin[id_z]))
    {
        if(flipAxis[id_z] == 0)
            src_pixIdx = batch_index[id_z] + (id_x + (height[id_z] -1 -id_y) * max_width[id_z]) * plnpkdindex;
        if(flipAxis[id_z] == 1)
            src_pixIdx = batch_index[id_z] + ((width[id_z] -1 -id_x) + (id_y) * max_width[id_z]) * plnpkdindex;
        if(flipAxis[id_z] == 2)
            src_pixIdx = batch_index[id_z] + ((width[id_z] -1 -id_x) + (height[id_z] -1 -id_y) * max_width[id_z]) * plnpkdindex;
            
        dst_pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex;  
        for(indextmp = 0; indextmp < channel; indextmp++){
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
            src_pixIdx += inc[id_z];
            dst_pixIdx += inc[id_z];
        }
    }

    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
        dst_pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex;
            for(indextmp = 0; indextmp < channel; indextmp++){  
                dstPtr[dst_pixIdx] = srcPtr[dst_pixIdx];
                dst_pixIdx += inc[id_z];
            }
    }
}