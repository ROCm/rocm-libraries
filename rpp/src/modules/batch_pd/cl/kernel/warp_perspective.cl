__kernel void warp_perspective_pln(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global float *perspective, const unsigned int source_height,
    const unsigned int source_width, const unsigned int dest_height,
    const unsigned int dest_width, const unsigned int channel) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  int xc = id_x - source_width / 2;
  int yc = id_y - source_height / 2;

  int k;
  int l;
  int m;

  k = (int)((perspective[0] * xc) + (perspective[1] * yc)) + perspective[2];
  l = (int)((perspective[3] * xc) + (perspective[4] * yc)) + perspective[5];
  m = (int)((perspective[6] * xc) + (perspective[7] * yc)) + perspective[8];

  k = ((k / m) + source_width / 2);
  l = ((l / m) + source_height / 2);

  if (l < source_height && l >= 0 && k < source_width && k >= 0)
    dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] =
        srcPtr[(id_z * source_height * source_width) + (l * source_width) + k];
  else
    dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = 0;
}

__kernel void warp_perspective_pkd(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global float *perspective, const unsigned int source_height,
    const unsigned int source_width, const unsigned int dest_height,
    const unsigned int dest_width, const unsigned int channel) {

  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  int xc = id_x - source_width / 2;
  int yc = id_y - source_height / 2;

  int k;
  int l;
  int m;

  k = (int)((perspective[0] * xc) + (perspective[1] * yc)) + perspective[2];
  l = (int)((perspective[3] * xc) + (perspective[4] * yc)) + perspective[5];
  m = (int)((perspective[6] * xc) + (perspective[7] * yc)) + perspective[8];

  k = ((k / m) + source_width / 2);
  l = ((l / m) + source_height / 2);

  if (l < source_height && l >= 0 && k < source_width && k >= 0)
    dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] =
        srcPtr[id_z + (channel * l * source_width) + (channel * k)];
  else
    dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] = 0;
}

__kernel void warp_perspective_batch(
    __global unsigned char *srcPtr, __global unsigned char *dstPtr,
    __global float *perspective, __global unsigned int *source_height,
    __global unsigned int *source_width, __global unsigned int *dest_height,
    __global unsigned int *dest_width, __global unsigned int *xroi_begin,
    __global unsigned int *xroi_end, __global unsigned int *yroi_begin,
    __global unsigned int *yroi_end, __global unsigned int *max_source_width,
    __global unsigned int *max_dest_width,
    __global unsigned long *source_batch_index,
    __global unsigned long *dest_batch_index, const unsigned int channel,
    __global unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dest_inc,
    const int plnpkdindex // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);
  int perspective_index = id_z * 9;

  float k_float = (float)((perspective[perspective_index + 0] * xc) +
                (perspective[perspective_index + 1] * yc)) +
          perspective[perspective_index + 2];
  float l_float = (float)((perspective[perspective_index + 3] * xc) +
                (perspective[perspective_index + 4] * yc)) +
          perspective[perspective_index + 5];
  float m_float = (float)((perspective[perspective_index + 6] * xc) +
                (perspective[perspective_index + 7] * yc)) +
          perspective[perspective_index + 8];

  int k = (int) ((k_float / m_float) + (source_width[id_z] >> 1));
  int l = (int) ((l_float / m_float) + (source_height[id_z] >> 1));

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * plnpkdindex;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * plnpkdindex;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
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