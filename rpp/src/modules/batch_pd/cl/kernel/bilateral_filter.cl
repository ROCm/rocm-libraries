#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
double gaussian(double x, double sigmaI) {
    double a = 2.0;
    return exp(-(pow(x, a))/(2 * pow(sigmaI, a))) / (2 * M_PI * pow(sigmaI, a));
}

double distance(int x1, int y1, int x2, int y2){
    double d_x = x2-x1;
    double d_y = y2-y1;
    double a = 2.0;
    double dis = sqrt(pow(d_x,a)+pow(d_y,a));
    return dis;
}

__kernel void bilateral_filter_planar(
    const __global unsigned char* input,
    __global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize,
    const double sigmaI,
    const double sigmaS
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
        //output[pixIdx] = 0;
        return;
    }
    
    double sum = 0.0;
    double w_sum = 0.0;
    for (int ri = (-1 * hfFiltSz) , rf = 0;
            (ri <= hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0;
                (ci <= hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxI = pixIdx + ri + ci * width;
            double gi  = gaussian(input[idxI]-input[pixIdx],sigmaI);
            double dis = distance(id_y, id_x, id_y+ri, id_x+ci);
            double gs  = gaussian(dis,sigmaS);
            double w = gi * gs;
            sum += input[idxI] * w;
            w_sum += w;
        }
    }
    int res = sum/w_sum;
    output[pixIdx] = saturate_8u(res);

}

__kernel void bilateral_filter_packed(
    const __global unsigned char* input,
    __global  unsigned char* output,
    const unsigned int height,
    const unsigned int width,
    const unsigned int channel,
    const unsigned int filterSize,
    const double sigmaI,
    const double sigmaS
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

    double sum = 0.0;
    double w_sum = 0.0;
    for (int ri = (-1 * hfFiltSz) , rf = 0;
            (ri <= hfFiltSz) && (rf < filterSize);
                ri++, rf++)
    {
        for (int ci = (-1 * hfFiltSz) , cf = 0;
                (ci <= hfFiltSz) && (cf < filterSize);
                    ci++, cf++)
        {
            const int idxI = pixIdx + ri * channel + ci * width * channel;
            
            double gi  = gaussian(input[idxI]-input[pixIdx],sigmaI);
            double dis = distance(id_y, id_x, id_y + (ri * channel) , id_x + (ci * channel));
            double gs  = gaussian(dis,sigmaS);
            double w = gi * gs;
            sum += input[idxI] * w;
            w_sum += w;
        }
    }
    int res = sum/w_sum;
    output[pixIdx] = saturate_8u(res);

}


__kernel void bilateral_filter_batch(   __global unsigned char* input,
                                        __global unsigned char* output,
                                        __global unsigned int *kernelSize,
                                        __global double *sigmaS,
                                        __global double *sigmaI,
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
    int kernelSizeTemp = kernelSize[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int bound = (kernelSizeTemp - 1) / 2;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            double sum1 = 0.0;
            double w_sum1 = 0.0;  
            double sum2 = 0.0;
            double w_sum2 = 0.0;  
            double sum3 = 0.0;
            double w_sum3 = 0.0;  

            for(int i = -bound ; i <= bound ; i++)
            {
                for(int j = -bound ; j <= bound ; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        
                        double gi1  = gaussian(input[index] - input[pixIdx], sigmaI[id_z]);
                        double dis1 = distance(id_y, id_x, id_y + (i * plnpkdindex) , id_x + (j * plnpkdindex));
                        double gs1  = gaussian(dis1,sigmaS[id_z]);
                        double w1 = gi1 * gs1;
                        sum1 += input[index] * w1;
                        w_sum1 += w1;
                        
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            double gi2  = gaussian(input[index] - input[pixIdx], sigmaI[id_z]);
                            double dis2 = distance(id_y, id_x, id_y + (i * plnpkdindex) , id_x + (j * plnpkdindex));
                            double gs2  = gaussian(dis2,sigmaS[id_z]);
                            double w2 = gi2 * gs2;
                            sum2 += input[index] * w2;
                            w_sum2 += w2;
                            
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            double gi3  = gaussian(input[index] - input[pixIdx], sigmaI[id_z]);
                            double dis3 = distance(id_y, id_x, id_y + (i * plnpkdindex) , id_x + (j * plnpkdindex));
                            double gs3  = gaussian(dis3,sigmaS[id_z]);
                            double w3 = gi3 * gs3;
                            sum3 += input[index] * w3;
                            w_sum3 += w3;
                        }
                    }

                }
            }

            int res1 = sum1/w_sum1;
            int res2 = sum2/w_sum2;
            int res3 = sum3/w_sum3;

            output[pixIdx] = saturate_8u(res1);
            if(channel == 3)
            {
                output[pixIdx + inc[id_z]] = saturate_8u(res2);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(res3);
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