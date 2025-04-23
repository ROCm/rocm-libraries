#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void crop_and_patch_batch(
    __global unsigned char *srcPtr1, __global unsigned char *srcPtr2,
    __global unsigned char *dstPtr, __global unsigned int *source_height,
    __global unsigned int *source_width, __global unsigned int *dest_height,
    __global unsigned int *dest_width, __global unsigned int *x11,
    __global unsigned int *y11, __global unsigned int *x12,
    __global unsigned int *y12, __global unsigned int *x21,
    __global unsigned int *y21, __global unsigned int *x22,
    __global unsigned int *y22, __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}




__kernel void crop_and_patch_batch_fp16(
    __global half *srcPtr1, __global half *srcPtr2,
    __global half *dstPtr, __global unsigned int *source_height,
    __global unsigned int *source_width, __global unsigned int *dest_height,
    __global unsigned int *dest_width, __global unsigned int *x11,
    __global unsigned int *y11, __global unsigned int *x12,
    __global unsigned int *y12, __global unsigned int *x21,
    __global unsigned int *y21, __global unsigned int *x22,
    __global unsigned int *y22, __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  half A, B, C, D;
  int x, y;
  half pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (half)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}



__kernel void crop_and_patch_batch_fp32(
    __global float *srcPtr1, __global float *srcPtr2,
    __global float *dstPtr, __global unsigned int *source_height,
    __global unsigned int *source_width, __global unsigned int *dest_height,
    __global unsigned int *dest_width, __global unsigned int *x11,
    __global unsigned int *y11, __global unsigned int *x12,
    __global unsigned int *y12, __global unsigned int *x21,
    __global unsigned int *y21, __global unsigned int *x22,
    __global unsigned int *y22, __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  float A, B, C, D;
  int x, y;
  float pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (float)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}



__kernel void crop_and_patch_batch_int8(
    __global  char *srcPtr1, __global  char *srcPtr2,
    __global  char *dstPtr, __global unsigned int *source_height,
    __global unsigned int *source_width, __global unsigned int *dest_height,
    __global unsigned int *dest_width, __global unsigned int *x11,
    __global unsigned int *y11, __global unsigned int *x12,
    __global unsigned int *y12, __global unsigned int *x21,
    __global unsigned int *y21, __global unsigned int *x22,
    __global unsigned int *y22, __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  char A, B, C, D;
  int x, y;
   char pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (char)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}