#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void naive_convolution_planar(
	const  unsigned char* input,
	  unsigned char* output,
	  float* filter,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize
)
{

    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
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

extern "C" __global__ void naive_convolution_packed(
	const  unsigned char* input,
	  unsigned char* output,
	  float* filter,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize
)
{

    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
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



// extern "C" __global__ void convolution_batch(   unsigned char* input,
//                                      unsigned char* output,
//                                       float* filter,
//                                      int *xroi_begin,
//                                      int *xroi_end,
//                                      int *yroi_begin,
//                                      int *yroi_end,
//                                      unsigned int *height,
//                                      unsigned int *width,
//                                      unsigned int *batch_index,
//                                     const unsigned int channel,
//                                     const unsigned int kernelSize,
//                                      unsigned int *inc, // use width * height for pln and 1 for pkd
//                                     const int plnpkdindex // use 1 pln 3 for pkd
//                                     )
// {
//     int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//     unsigned char pixel;
//     int indextmp = 0;
//     int pixIdx = 0;
//     int bound = (kernelSize - 1) / 2;
//     unsigned int index; 
//     float sum;

//     pixIdx = batch_index[id_z] + (id_x  + id_y * width[id_z] ) * plnpkdindex ;
//     if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
//     {   
//         for(indextmp = 0; indextmp < channel; indextmp++){
//             pixel = input[pixIdx];
//             sum = 0.0;
//             for (int ri = (-1 * bound) , rf = 0;
//                     (ri <= bound) && (rf < filterSize);
//                         ri++, rf++)
//                 {
//                 for (int ci = (-1 * bound) , cf = 0;
//                         (ci <= bound) && (cf < filterSize);
//                             ci++, cf++)
//                     {
//                     const int idxF = rf + cf * filterSize ;
//                     const int idxI = pixIdx + ri + ci * width;
//                     sum += filter[idxF]*input[idxI];
//                     }
//                 }   
//         output[pixIdx] = saturate_8u(int(sum));
//         output[pixIdx] = pixel;
//         pixIdx += inc[id_z];     
//         }
//     }
//     else if((id_x < width[id_z] ) && (id_y < height[id_z])){
//             for(indextmp = 0; indextmp < channel; indextmp++){
//                 output[pixIdx] = input[pixIdx];
//                 pixIdx += inc[id_z];
//             }
//     }
// }