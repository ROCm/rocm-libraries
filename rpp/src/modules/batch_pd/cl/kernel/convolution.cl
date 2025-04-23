#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void naive_convolution_planar(
	const __global unsigned char* input,
	__global  unsigned char* output,
	__global  float* filter,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int hfFiltSz = filterSize/2;
    if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
            (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    float sum = 0.0;
    for (int ri = (-1 * hfFiltSz) , rf = 0;
            (ri <= hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0;
                (ci <= hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxF = rf + cf * filterSize ;
            const int idxI = pixIdx + ri + ci * width;
            sum += filter[idxF]*input[idxI];
        }
    }
    int res = (int)sum;
    output[pixIdx] = saturate_8u(res);

}

__kernel void naive_convolution_packed(
	const __global unsigned char* input,
	__global  unsigned char* output,
	__global  float* filter,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x * channel + id_y * width * channel + id_z ;

    int hfFiltSz = filterSize/2;
    if ( (id_x < hfFiltSz) || (id_y < hfFiltSz) ||
            (id_y >= (height-hfFiltSz)) || (id_x >= (width-hfFiltSz)) )
    {
        output[pixIdx] = input[pixIdx];
        return;
    }

    int res;

    float sum = 0.0;
    for (int ri = (-1 * hfFiltSz) , rf = 0;
            (ri <= hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0;
                (ci <= hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxF = rf + cf * filterSize ;
            const int idxI = pixIdx + ri * channel + ci * width *channel;
            sum += filter[idxF]*input[idxI];
        }
    }
    res = (int)sum;
    output[pixIdx] = saturate_8u(res);
}


