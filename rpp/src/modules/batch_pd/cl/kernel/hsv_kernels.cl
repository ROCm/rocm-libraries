//////////////////Conversion Functions ////////////////////////////_
__kernel void rgb2hsv_pln(__global unsigned char *input,
                         __global double *output,
                         const unsigned int height,
                         const unsigned int width)
{
    //Get our global thread ID
    int id = get_global_id(0);
    double r,g,b, min, max, delta;

    //Make sure we do not go out of bounds

    if (id < height * width ){
        r = input[id] / 255.0;
        g = input[id + height * width] / 255.0;
        b = input[id + 2* height * width]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        if (delta == 0) output[id] = 0;
        else {
            if (max == r)
                output[id] = 60 * ((g - b)/delta);
            else if (max == g)
                output[id] = 60 * ((b - r)/delta + 2);
            else
                output[id] = 60 * ((r - g)/delta + 4);
        }


        if ( output[id] < 0) output[id] = output[id] +360;
        if (max == 0) output[id +  height * width] = 0;
        else output[id + height * width] = delta / max;
        output[id + 2* height * width] = max;

    }

}

__kernel void rgb2hsv_pkd(  __global unsigned char *input,
                            __global double *output,
                              const unsigned int height,
                              const unsigned int width)
{
    //Get our global thread ID
    int id = get_global_id(0);
    double r,g,b, min, max, delta;

    //Make sure we do not go out of bounds
    id = id * 3;
    if (id < 3 *height * width ){
        r = input[id] / 255.0;
        g = input[id + 1] / 255.0;
        b = input[id + 2]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        if (delta == 0) output[id] = 0;
        else {
            if (max == r)
                output[id] = 60 * ((g - b)/delta);
            else if (max == g)
                output[id] = 60 * ((b - r)/delta + 2);
            else
                output[id] = 60 * ((r - g)/delta + 4);
        }


        if ( output[id] < 0)  output[id] = output[id] +360;
        if (max == 0) output[id +  1] = 0;
        else output[id + 1] = delta / max;
        output[id + 2] = max;

    }

}

__kernel void hsv2rgb_pln(   __global const double *input,
                         __global  unsigned char *output,
                         const unsigned int height,
                        const unsigned int width)
{
    //Get our global thread ID



    int pixIdx  = get_global_id(0);
    double     hh, p, q, t, ff;
    int        i;
    double     h,s,v;
    pixIdx = 3 * pixIdx;
    //Make sure we do not go out of bounds

    if (pixIdx < height*width ){

        h = input[pixIdx];
        s = input[pixIdx + height * width ] ;
        v = input[pixIdx + 2* height * width] ;

        if (s <= 0){
            output[pixIdx] = 0;
            output[pixIdx +  height * width] = 0;
            output[pixIdx + 2* height * width] = 0;
        }

        hh = h;
        if(h == 360.0) hh = 0.0;
        hh /= 60.0;
        i = (int)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));

    switch(i){
    case 0:
        output[pixIdx] = v * 255;
        output[pixIdx +  height * width] = t * 255  ;
        output[pixIdx + 2* height * width] = p * 255;
        break;
    case 1:
        output[pixIdx] = q * 255;
        output[pixIdx +  height * width] = v * 255  ;
        output[pixIdx + 2* height * width] = p * 255 ;
        break;
    case 2:
        output[pixIdx] = p * 255 ;
        output[pixIdx +  height * width] = v * 255;
        output[pixIdx + 2* height * width] = t * 255;
        break;

    case 3:
        output[pixIdx] = p * 255;
        output[pixIdx +  height * width] = q * 255;
        output[pixIdx + 2* height * width] = v * 255;
        break;

    case 4:
        output[pixIdx] = t * 255;
        output[pixIdx +  height * width] = p * 255 ;
        output[pixIdx + 2* height * width] = v * 255;
        break;
    case 5:
    default:
        output[pixIdx] = v * 255;
        output[pixIdx +  height * width] = p * 255 ;
        output[pixIdx + 2* height * width] = q * 255;
        break;
     }

    }

}

__kernel void hsv2rgb_pkd(__global const double *input,
                          __global  unsigned char *output,
                          const unsigned int height,
                          const unsigned int width)
{
    //Get our global thread ID


    int pixIdx = get_global_id(0);
    double     hh, p, q, t, ff;
    int        i;
    double     h,s,v;
    pixIdx = 3 * pixIdx;
    //Make sure we do not go out of bounds

    if (pixIdx < height*width*3 ){

        h = input[pixIdx];
        s = input[pixIdx + 1 ] ;
        v = input[pixIdx + 2] ;

        if (s <= 0){
            output[pixIdx] = 0;
            output[pixIdx + 1] = 0;
            output[pixIdx + 2] = 0;
        }

        hh = h;
        if(h == 360.0) { hh = 0.0; }
        hh /= 60.0;
        i = (int)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));

    switch(i)
    {
    case 0:
        output[pixIdx] = v * 255;
        output[pixIdx + 1] = t * 255 ;
        output[pixIdx + 2] = p * 255;
        break;
    case 1:
        output[pixIdx] = q * 255;
        output[pixIdx + 1] = v * 255  ;
        output[pixIdx + 2] = p * 255 ;
        break;
    case 2:
        output[pixIdx] = p * 255;
        output[pixIdx + 1] = v * 255;
        output[pixIdx + 2] = t * 255;
        break;

    case 3:
        output[pixIdx] = p * 255;
        output[pixIdx +  1] = q * 255;
        output[pixIdx + 2] = v * 255;
        break;

    case 4:
        output[pixIdx] = t * 255;
        output[pixIdx +  1] = p * 255 ;
        output[pixIdx + 2] = v * 255;
        break;
    case 5:
    default:
        output[pixIdx] = v * 255;
        output[pixIdx +  1] = p * 255 ;
        output[pixIdx + 2] = q * 255;
        break;
     }

    }

}


// Hue and Satutation Modification /////////////////////////////

__kernel void huergb_pln(   __global  unsigned char *input,
                            __global  unsigned char *output,
                            const  double hue,
                            const  double sat,
                            const unsigned int height,
                            const unsigned int width)
{
    //Get our global thread ID
    int id = get_global_id(0);
    double r,g,b, min, max, delta;
    double temp1, temp2, temp3;


    //Make sure we do not go out of bounds
    //id = id ;
    if (id < 3 *height * width ){
        r = input[id] / 255.0;
        g = input[id + height * width] / 255.0;
        b = input[id + 2 *height * width]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        if (delta == 0) output[id] = 0;
        else {
            if (max == r)
                temp1 = 60 * ((g - b)/delta);
            else if (max == g)
                temp1 = 60 * ((b - r)/delta + 2);
            else
                temp1 = 60 * ((r - g)/delta + 4);
        }

        temp1 += hue;
        if ( temp1 < 0)  temp1 = temp1 +360.0;
        else if (temp1 > 360) temp1 = temp1 - 360.0;
        
        if (max == 0) temp2 = 0;
        else temp2 = delta / max;
        temp2 += sat;
        temp3 = max;

    //barrier(CLK_GLOBAL_MEM_FENCE);
    double     hh, p, q, t, ff;
    int        i;
    double     h,s,v;

    int pixIdx = id;
    //Make sure we do not go out of bounds


        h = temp1;
        s = temp2 ;
        v = temp3 ;

        if (s <= 0){
            output[pixIdx] = 0;
            output[pixIdx + height * width] = 0;
            output[pixIdx + 2*height * width] = 0;
        }

        hh = h;
        if(h == 360.0) { hh = 0.0; }
        hh /= 60.0;
        i = (int)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));

    switch(i)
    {
    case 0:
        output[pixIdx] = v * 255;
        output[pixIdx + height * width] = t * 255 ;
        output[pixIdx + 2*height * width] = p * 255;
        break;
    case 1:
        output[pixIdx] = q * 255;
        output[pixIdx + height * width] = v * 255  ;
        output[pixIdx + 2*height * width] = p * 255 ;
        break;
    case 2:
        output[pixIdx] = p * 255;
        output[pixIdx + height * width] = v * 255;
        output[pixIdx + 2*height * width] = t * 255;
        break;

    case 3:
        output[pixIdx] = p * 255;
        output[pixIdx + height * width] = q * 255;
        output[pixIdx + 2* height * width] = v * 255;
        break;

    case 4:
        output[pixIdx] = t * 255;
        output[pixIdx +  height * width] = p * 255 ;
        output[pixIdx + 2*height * width] = v * 255;
        break;
    case 5:
    default:
        output[pixIdx] = v * 255;
        output[pixIdx +  height * width] = p * 255 ;
        output[pixIdx + 2*height * width] = q * 255;
        break;
     }

    }

}


__kernel void huergb_pkd(   __global  unsigned char *input,
                            __global  unsigned char *output,
                            const  double hue,
                            const  double sat,
                            const unsigned int height,
                            const unsigned int width)
{
    //Get our global thread ID
    int id = get_global_id(0);
    double r,g,b, min, max, delta;
    double temp1, temp2, temp3;

    //Make sure we do not go out of bounds
    id = id * 3;
    if (id < 3 *height * width ){
        r = input[id] / 255.0;
        g = input[id + 1] / 255.0;
        b = input[id + 2]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        if (delta == 0) output[id] = 0;
        else {
            if (max == r)
                temp1 = 60 * ((g - b)/delta);
            else if (max == g)
                temp1 = 60 * ((b - r)/delta + 2);
            else
                temp1 = 60 * ((r - g)/delta + 4);
        }

        temp1 += hue;
        if ( temp1 < 0)  temp1 = temp1 +360;
        else if (temp1 > 360) temp1 = temp1 - 360.0;

        if (max == 0) temp2 = 0;
        else temp2= delta / max;
        temp2 += sat;
        temp3 = max;

   // barrier(CLK_GLOBAL_MEM_FENCE);

    double     hh, p, q, t, ff;
    int        i;
    double     h,s,v;

    int pixIdx = id;
    //Make sure we do not go out of bounds


        h = temp1;
        s = temp2;
        v = temp3;

        if (s <= 0){
            output[pixIdx] = 0;
            output[pixIdx + 1] = 0;
            output[pixIdx + 2] = 0;
        }

        hh = h;
        if(h == 360.0) { hh = 0.0; }
        hh /= 60.0;
        i = (int)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));

    switch(i)
    {
    case 0:
        output[pixIdx] = v * 255;
        output[pixIdx + 1] = t * 255 ;
        output[pixIdx + 2] = p * 255;
        break;
    case 1:
        output[pixIdx] = q * 255;
        output[pixIdx + 1] = v * 255  ;
        output[pixIdx + 2] = p * 255 ;
        break;
    case 2:
        output[pixIdx] = p * 255;
        output[pixIdx + 1] = v * 255;
        output[pixIdx + 2] = t * 255;
        break;

    case 3:
        output[pixIdx] = p * 255;
        output[pixIdx +  1] = q * 255;
        output[pixIdx + 2] = v * 255;
        break;

    case 4:
        output[pixIdx] = t * 255;
        output[pixIdx +  1] = p * 255 ;
        output[pixIdx + 2] = v * 255;
        break;
    case 5:
    default:
        output[pixIdx] = v * 255;
        output[pixIdx +  1] = p * 255 ;
        output[pixIdx + 2] = q * 255;
        break;
     }

    }

}

__kernel void huehsv_pln(   __global  double *input,
                            __global  double *output,
                            const  double hue,
                            const  double sat,
                            const unsigned int height,
                            const unsigned int width)
{
    int id = get_global_id(0);
    output[id] = input[id] + hue;
    output[id + height * width] = input[id + height * width]+ sat;
    /*Boundary Conditions needs to be taken care of */
}

__kernel void huehsv_pkd(   __global  double *input,
                            __global  double *output,
                            const  double hue,
                            const  double sat,
                            const unsigned int height,
                            const unsigned int width)
{
    int id = get_global_id(0);
    id = id * 3;
    output[id] = input[id] + hue;
    output[id + 1] = input[id + 1] + sat;
}
////////////////////////////////////////////

__kernel void hue_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned float *hue,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    const unsigned int batch_size,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )

{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long r_pixIdx = 0, g_pixIdx = 0, b_pixIdx = 0;

    //Make sure we do not go out of bounds
    if((id_z < batch_size) && (id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])){
        r_pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        g_pixIdx = r_pixIdx + inc[id_z];
        b_pixIdx = g_pixIdx + inc[id_z];

        r = input[r_pixIdx] / 255.0;
        g = input[g_pixIdx] / 255.0;
        b = input[b_pixIdx]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        if (delta == 0) output[id] = 0;
        else {
            if (max == r)
                temp1 = 60 * ((g - b)/delta);
            else if (max == g)
                temp1 = 60 * ((b - r)/delta + 2);
            else
                temp1 = 60 * ((r - g)/delta + 4);
        }

        temp1 += hue;
        if ( temp1 < 0)  temp1 = temp1 +360;
        else if (temp1 > 360) temp1 = temp1 - 360.0;

        if (max == 0) temp2 = 0;
        else temp2= delta / max;
        temp3 = max;

   // barrier(CLK_GLOBAL_MEM_FENCE);

    double     hh, p, q, t, ff;
    int        i;
    double     h,s,v;


    //Make sure we do not go out of bounds


        h = temp1;
        s = temp2;
        v = temp3;

        if (s <= 0){
            output[r_pixIdx] = 0;
            output[g_pixIdx] = 0;
            output[b_pixIdx] = 0;
        }

        hh = h;
        if(h == 360.0) { hh = 0.0; }
        hh /= 60.0;
        i = (int)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));

    switch(i)
    {
    case 0:
        output[r_pixIdx] = v * 255;
        output[g_pixIdx] = t * 255 ;
        output[b_pixIdx] = p * 255;
        break;
    case 1:
        output[r_pixIdx] = q * 255;
        output[g_pixIdx] = v * 255  ;
        output[b_pixIdx] = p * 255 ;
        break;
    case 2:
        output[r_pixIdx] = p * 255;
        output[g_pixIdx] = v * 255;
        output[b_pixIdx] = t * 255;
        break;

    case 3:
        output[r_pixIdx] = p * 255;
        output[g_pixIdx] = q * 255;
        output[b_pixIdx] = v * 255;
        break;

    case 4:
        output[r_pixIdx] = t * 255;
        output[g_pixIdx] = p * 255 ;
        output[b_pixIdx] = v * 255;
        break;
    case 5:
    default:
        output[r_pixIdx] = v * 255;
        output[g_pixIdx] = p * 255 ;
        output[b_pixIdx] = q * 255;
        break;
     }

    }
}

__kernel void saturation_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned float *sat,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    const unsigned int batch_size,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )

{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long r_pixIdx = 0, g_pixIdx = 0, b_pixIdx = 0;

    //Make sure we do not go out of bounds
    if((id_z < batch_size) && (id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])){
        r_pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        g_pixIdx = r_pixIdx + inc[id_z];
        b_pixIdx = g_pixIdx + inc[id_z];

        r = input[r_pixIdx] / 255.0;
        g = input[g_pixIdx] / 255.0;
        b = input[b_pixIdx]/ 255.0;

        min = (r < g && r< b)? r : ((g < b)? g: b);
        max = (r > g && r > b)? r : ((g > b)? g: b);

        delta = max - min;

        if (delta == 0) output[id] = 0;
        else {
            if (max == r)
                temp1 = 60 * ((g - b)/delta);
            else if (max == g)
                temp1 = 60 * ((b - r)/delta + 2);
            else
                temp1 = 60 * ((r - g)/delta + 4);
        }

        if ( temp1 < 0)  temp1 = temp1 +360;
        else if (temp1 > 360) temp1 = temp1 - 360.0;

        if (max == 0) temp2 = 0;
        else temp2= delta / max;
        temp2 += sat;
        temp3 = max;

   // barrier(CLK_GLOBAL_MEM_FENCE);

    double     hh, p, q, t, ff;
    int        i;
    double     h,s,v;


    //Make sure we do not go out of bounds


        h = temp1;
        s = temp2;
        v = temp3;

        if (s <= 0){
            output[r_pixIdx] = 0;
            output[g_pixIdx] = 0;
            output[b_pixIdx] = 0;
        }

        hh = h;
        if(h == 360.0) { hh = 0.0; }
        hh /= 60.0;
        i = (int)hh;
        ff = hh - i;
        p = v * (1.0 - s);
        q = v * (1.0 - (s * ff));
        t = v * (1.0 - (s * (1.0 - ff)));

    switch(i)
    {
    case 0:
        output[r_pixIdx] = v * 255;
        output[g_pixIdx] = t * 255 ;
        output[b_pixIdx] = p * 255;
        break;
    case 1:
        output[r_pixIdx] = q * 255;
        output[g_pixIdx] = v * 255  ;
        output[b_pixIdx] = p * 255 ;
        break;
    case 2:
        output[r_pixIdx] = p * 255;
        output[g_pixIdx] = v * 255;
        output[b_pixIdx] = t * 255;
        break;

    case 3:
        output[r_pixIdx] = p * 255;
        output[g_pixIdx] = q * 255;
        output[b_pixIdx] = v * 255;
        break;

    case 4:
        output[r_pixIdx] = t * 255;
        output[g_pixIdx] = p * 255 ;
        output[b_pixIdx] = v * 255;
        break;
    case 5:
    default:
        output[r_pixIdx] = v * 255;
        output[g_pixIdx] = p * 255 ;
        output[b_pixIdx] = q * 255;
        break;
     }

    }
}