#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

__kernel void scale_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel,
                            const unsigned int exp_dest_height,
                            const unsigned int exp_dest_width
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/exp_dest_width ;
    float y_ratio = ((float)(source_height -1 ))/exp_dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel || id_x >= exp_dest_width || id_y >= exp_dest_height) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    A = srcPtr[x + y * source_width + id_z * source_height * source_width];
    B = srcPtr[x + 1  + y * source_width + id_z * source_height * source_width];
    C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+1) + (y+1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}

__kernel void scale_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel,
                            const unsigned int exp_dest_height,
                            const unsigned int exp_dest_width
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/exp_dest_width ;
    float y_ratio = ((float)(source_height -1 ))/exp_dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel || id_x >= exp_dest_width || id_y >= exp_dest_height) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[x * channel + y * source_width * channel + id_z];
    B = srcPtr[(x +1) * channel + y * source_width * channel + id_z];
    C = srcPtr[x * channel + (y+1) * source_width * channel + id_z];
    D = srcPtr[(x+1) * channel + (y+1) * source_width * channel + id_z];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                  C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
    dstPtr[pixId] =  saturate_8u(pixVal);

}

__kernel void scale_batch(    __global unsigned char* srcPtr,
                                    __global unsigned char* dstPtr,
                                    __global float* percentage,
                                    __global unsigned int *source_height,
                                    __global unsigned int *source_width,
                                    __global unsigned int *dest_height,
                                    __global unsigned int *dest_width,
                                    __global unsigned int *max_source_width,
                                    __global unsigned int *max_dest_width,
                                     __global int *xroi_begin,
                                    __global int *xroi_end,
                                     __global int *yroi_begin,
                                     __global int *yroi_end,
                                     __global unsigned long *source_batch_index,
                                    __global unsigned long *dest_batch_index,
                                      const unsigned int channel,
                                    __global unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                    __global unsigned int *dest_inc,
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) * 100 / (percentage[id_z] * dest_width[id_z]);
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) * 100 / (percentage[id_z] * dest_height[id_z]);
  float x_diff, y_diff, ya, yb;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * plnpkdindex;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * plnpkdindex +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * plnpkdindex +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * plnpkdindex +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * plnpkdindex +
                 indextmp * source_inc[id_z]];

      pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = saturate_8u(pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * plnpkdindex;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}