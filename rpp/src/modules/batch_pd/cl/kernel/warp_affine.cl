#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void
warp_affine_pln(__global unsigned char *srcPtr, __global unsigned char *dstPtr,
                __global float *affine, const unsigned int source_height,
                const unsigned int source_width, const unsigned int dest_height,
                const unsigned int dest_width, const unsigned int channel) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  int xc = id_x - source_width / 2;
  int yc = id_y - source_height / 2;

  int k;
  int l;

  k = (int)((affine[0] * xc) + (affine[1] * yc)) + affine[2];
  l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];

  k = k + source_width / 2;
  l = l + source_height / 2;

  if (l < source_height && l >= 0 && k < source_width && k >= 0)
    dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] =
        srcPtr[(id_z * source_height * source_width) + (l * source_width) + k];
  else
    dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = 0;
}

__kernel void
warp_affine_pkd(__global unsigned char *srcPtr, __global unsigned char *dstPtr,
                __global float *affine, const unsigned int source_height,
                const unsigned int source_width, const unsigned int dest_height,
                const unsigned int dest_width, const unsigned int channel) {

  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  int xc = id_x - source_width / 2;
  int yc = id_y - source_height / 2;

  int k;
  int l;

  k = (int)((affine[0] * xc) + (affine[1] * yc)) + affine[2];
  l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];

  k = k + source_width / 2;
  l = l + source_height / 2;

  if (l < source_height && l >= 0 && k < source_width && k >= 0)
    dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] =
        srcPtr[id_z + (channel * l * source_width) + (channel * k)];
  else
    dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] = 0;
}

__kernel void warp_affine_batch(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global float *affine, __global unsigned int *source_height,
    __global unsigned int *source_width, __global unsigned int *dest_height,
    __global unsigned int *dest_width, __global unsigned int *xroi_begin,
    __global unsigned int *xroi_end, __global unsigned int *yroi_begin,
    __global unsigned int *yroi_end, __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);
  int affine_index = id_z * 6;
  int k =
      (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) +
      affine[affine_index + 2];
  int l =
      (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) +
      affine[affine_index + 5];
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
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

__kernel void warp_affine_batch_fp32(
    __global float *srcPtr, __global float *dstPtr, __global float *affine,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *xroi_begin, __global unsigned int *xroi_end,
    __global unsigned int *yroi_begin, __global unsigned int *yroi_end,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);
  int affine_index = id_z * 6;

  int k =
      (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) +
      affine[affine_index + 2];
  int l =
      (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) +
      affine[affine_index + 5];
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
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

__kernel void warp_affine_batch_fp16(
    __global half *srcPtr, __global half *dstPtr, __global float *affine,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *xroi_begin, __global unsigned int *xroi_end,
    __global unsigned int *yroi_begin, __global unsigned int *yroi_end,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);
  int affine_index = id_z * 6;

  int k =
      (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) +
      affine[affine_index + 2];
  int l =
      (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) +
      affine[affine_index + 5];
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
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

__kernel void warp_affine_batch_int8(
    __global char *srcPtr, __global char *dstPtr, __global float *affine,
    __global unsigned int *source_height, __global unsigned int *source_width,
    __global unsigned int *dest_height, __global unsigned int *dest_width,
    __global unsigned int *xroi_begin, __global unsigned int *xroi_end,
    __global unsigned int *yroi_begin, __global unsigned int *yroi_end,
    __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int *source_inc, __global unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);
  int affine_index = id_z * 6;

  int k =
      (int)((affine[affine_index + 0] * xc) + (affine[affine_index + 1] * yc)) +
      affine[affine_index + 2];
  int l =
      (int)((affine[affine_index + 3] * xc) + (affine[affine_index + 4] * yc)) +
      affine[affine_index + 5];
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = -128;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}
