#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__kernel void
resize_pln(__global unsigned char *srcPtr, __global unsigned char *dstPtr,
           const unsigned int source_height, const unsigned int source_width,
           const unsigned int dest_height, const unsigned int dest_width,
           const unsigned int channel) {
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio = ((float)(source_width - 1)) / dest_width;
  float y_ratio = ((float)(source_height - 1)) / dest_height;
  float x_diff, y_diff, ya, yb;

  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  unsigned int pixId;
  pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
  A = srcPtr[x + y * source_width + id_z * source_height * source_width];
  B = srcPtr[x + 1 + y * source_width + id_z * source_height * source_width];
  C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
  D = srcPtr[(x + 1) + (y + 1) * source_width +
             id_z * source_height * source_width];

  pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                 C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));

  dstPtr[pixId] = saturate_8u(pixVal);
}

__kernel void
resize_pkd(__global unsigned char *srcPtr, __global unsigned char *dstPtr,
           const unsigned int source_height, const unsigned int source_width,
           const unsigned int dest_height, const unsigned int dest_width,
           const unsigned int channel) {
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio = ((float)(source_width - 1)) / dest_width;
  float y_ratio = ((float)(source_height - 1)) / dest_height;
  float x_diff, y_diff, ya, yb;

  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  unsigned int pixId;
  pixId = id_x * channel + id_y * dest_width * channel + id_z;

  A = srcPtr[x * channel + y * source_width * channel + id_z];
  B = srcPtr[(x + 1) * channel + y * source_width * channel + id_z];
  C = srcPtr[x * channel + (y + 1) * source_width * channel + id_z];
  D = srcPtr[(x + 1) * channel + (y + 1) * source_width * channel + id_z];

  pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                 C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
  dstPtr[pixId] = saturate_8u(pixVal);
}

__kernel void
resize_crop_pln(__global unsigned char *srcPtr, __global unsigned char *dstPtr,
                const unsigned int source_height,
                const unsigned int source_width, const unsigned int dest_height,
                const unsigned int dest_width, const unsigned int x1,
                const unsigned int y1, const unsigned int x2,
                const unsigned int y2, const unsigned int padding,
                const unsigned int type, const unsigned int channel) {
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio = ((float)(x2 - x1)) / dest_width;
  float y_ratio = ((float)(y2 - y1)) / dest_height;
  float x_diff, y_diff, ya, yb;
  A = B = C = D = 0;
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  unsigned int pixId;
  if (type == 0)
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
  else
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height +
            ((dest_width + padding * 2) * padding) + (id_y * padding * 2) +
            (padding);
  A = srcPtr[(x + x1) + (y + y1) * source_width +
             id_z * source_height * source_width];
  B = srcPtr[(x + x1 + 1) + (y + y1) * source_width +
             id_z * source_height * source_width];
  C = srcPtr[(x + x1) + (y + y1 + 1) * source_width +
             id_z * source_height * source_width];
  D = srcPtr[(x + x1 + 1) + (y + y1 + 1) * source_width +
             id_z * source_height * source_width];

  pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                 C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));

  dstPtr[pixId] = saturate_8u(pixVal);
}

__kernel void
resize_crop_pkd(__global unsigned char *srcPtr, __global unsigned char *dstPtr,
                const unsigned int source_height,
                const unsigned int source_width, const unsigned int dest_height,
                const unsigned int dest_width, const unsigned int x1,
                const unsigned int y1, const unsigned int x2,
                const unsigned int y2, const unsigned int padding,
                const unsigned int type, const unsigned int channel) {
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio = ((float)(x2 - x1)) / dest_width;
  float y_ratio = ((float)(y2 - y1)) / dest_height;
  float x_diff, y_diff, ya, yb;
  A = B = C = D = 0;
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  unsigned int pixId;
  if (type == 0)
    pixId = id_x * channel + id_y * dest_width * channel + id_z;
  else
    pixId = id_x * channel + id_y * dest_width * channel + id_z +
            ((dest_width + padding * 2) * channel * padding) +
            (id_y * padding * 2 * channel) + (padding * channel);
  A = srcPtr[(x + x1) * channel + (y + y1) * source_width * channel + id_z];
  B = srcPtr[(x + x1 + 1) * channel + (y + y1) * source_width * channel + id_z];
  C = srcPtr[(x + x1) * channel + (y + y1 + 1) * source_width * channel + id_z];
  D = srcPtr[(x + x1 + 1) * channel + (y + y1 + 1) * source_width * channel +
             id_z];

  pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                 C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));

  dstPtr[pixId] = saturate_8u(pixVal);
}

__kernel void resize_batch(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio = ((float)(source_width[id_z] - 1)) / dest_width[id_z];
  float y_ratio = ((float)(source_height[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * in_plnpkdind;
  for (indextmp = 0; indextmp < channel; indextmp++) {
    A = srcPtr[source_batch_index[id_z] +
               (x + y * max_source_width[id_z]) * in_plnpkdind +
               indextmp * source_inc[id_z]];
    B = srcPtr[source_batch_index[id_z] +
               ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
               indextmp * source_inc[id_z]];
    C = srcPtr[source_batch_index[id_z] +
               (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
               indextmp * source_inc[id_z]];
    D = srcPtr[source_batch_index[id_z] +
               ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
               indextmp * source_inc[id_z]];

    pixVal =
        (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
              C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
    dstPtr[dst_pixIdx] = saturate_8u(pixVal);
    dst_pixIdx += dest_inc[id_z];
  }
}

__kernel void resize_crop_batch(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc, const unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

__kernel void resize_crop_batch_int8(
    __global char *srcPtr, __global char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc, const unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal =
          (char)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}


__kernel void resize_crop_batch_fp16(
    __global half *srcPtr, __global half *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc, const unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float A, B, C, D, pixVal;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = (half)pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}
__kernel void resize_crop_batch_fp32(
    __global float *srcPtr, __global float *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float A, B, C, D, pixVal;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

__kernel void resize_crop_batch_u8_fp32(
    __global unsigned char *srcPtr, __global float *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float A, B, C, D, pixVal;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = pixVal / 255.0;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

__kernel void resize_crop_batch_u8_fp16(
    __global unsigned char *srcPtr, __global half *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float A, B, C, D, pixVal;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = (half)(pixVal/255.0);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

__kernel void resize_crop_batch_u8_int8(
    __global unsigned char *srcPtr, __global char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float A, B, C, D, pixVal;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = (char)(pixVal - 128);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = -128;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}


__kernel void resize_crop_mirror_batch(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global int *mirror, __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                     out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = saturate_8u(pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

__kernel void resize_crop_mirror_batch_int8(
    __global char *srcPtr, __global char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global int *mirror, __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                     out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal =
          (char)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = -128;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}
__kernel void resize_crop_mirror_batch_fp16(
    __global half *srcPtr, __global half *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global int *mirror, __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int x, y, index;
  float A, B, C, D, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                     out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = (half)pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}
__kernel void resize_crop_mirror_batch_fp32(
    __global float *srcPtr, __global float *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global int *mirror, __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int x, y, index;
  float A, B, C, D, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
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
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                    out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

__kernel void random_crop_letterbox_batch(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width, __global int *xroi_begin,
    __global int *xroi_end, __global int *yroi_begin, __global int *yroi_end,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc, unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  unsigned int minVal = ((dest_height[id_z] < dest_width[id_z]) ? dest_height[id_z] : dest_width[id_z]);
  padding = (5 * minVal / 100);

  if (id_x >= dest_width[id_z] - padding || id_y >= dest_height[id_z] - padding || id_x < padding || id_y < padding)
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
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
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}
