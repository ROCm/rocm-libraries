#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

kernel void crop_mirror_normalize_batch(
    __global unsigned char *input,
    __global unsigned char *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc,
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const float local_mean = mean[id_z];
  const float local_std_dev = std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx = dst_batch_index[id_z] +
                             (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = input[src_pixIdx]; ;
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}

kernel void crop_mirror_normalize_batch_fp16(
    __global half *input,
    __global half *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const half local_mean = (half)mean[id_z];
  const half local_std_dev = (half)std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long  src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = (input[src_pixIdx] - local_mean) / local_std_dev;
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}

kernel void crop_mirror_normalize_batch_int8(
    __global char *input,
    __global char *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc,
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const float local_mean = mean[id_z];
  const float local_std_dev = std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) *
          out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = (char) ((input[src_pixIdx] - local_mean) / local_std_dev);
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = -128;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}

kernel void crop_mirror_normalize_batch_fp32(
    __global float *input,
    __global float *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const float local_mean = mean[id_z];
  const float local_std_dev = std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) *
          out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = (input[src_pixIdx] - local_mean) / local_std_dev;
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}

kernel void crop_mirror_normalize_batch_u8_fp16(
    __global unsigned char *input,
    __global half *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const float local_mean    = mean[id_z];
  const float local_std_dev = std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = (half)((input[src_pixIdx] - local_mean) / 255.0 * local_std_dev);
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0.0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}

kernel void crop_mirror_normalize_batch_u8_fp32(
    __global unsigned char *input,
    __global float *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const float local_mean    = mean[id_z];
  const float local_std_dev = std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = (float)((input[src_pixIdx] - local_mean) / 255.0 * local_std_dev);
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0.0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}

kernel void crop_mirror_normalize_batch_u8_int8(
    __global unsigned char *input,
    __global char *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global float *mean,
    __global float *std_dev, __global unsigned int *flip,
    __global unsigned int *max_src_width, __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  const float local_mean    = mean[id_z];
  const float local_std_dev = std_dev[id_z];
  const unsigned int local_flip = flip[id_z];
  unsigned long src_pixIdx;
  if (local_flip == 1) {
    src_pixIdx = src_batch_index[id_z] +
                 ((dst_width[id_z] - 1 - id_x + start_x[id_z]) +
                  (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  else{
     src_pixIdx = src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) * in_plnpkdind;
  }
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) * out_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = (char)((input[src_pixIdx] - 128 - local_mean ) /  local_std_dev);
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0.0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}



kernel void crop_batch(
    __global unsigned char *input, __global unsigned char *output,
    __global unsigned int *dst_height, __global unsigned int *dst_width,
    __global unsigned int *src_width, __global unsigned int *start_x,
    __global unsigned int *start_y, __global unsigned int *max_src_width,
    __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    // const unsigned int batch_size,
    __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
    __global unsigned int *dst_inc,
    const int in_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  unsigned long src_pixIdx =
      src_batch_index[id_z] +
      (id_x + start_x[id_z] + (id_y + start_y[id_z]) * max_src_width[id_z]) *
          in_plnpkdind;
  unsigned long dst_pixIdx =
      dst_batch_index[id_z] +
      (id_x + id_y * max_dst_width[id_z]) *
          in_plnpkdind;
  if ((id_x < dst_width[id_z]) && (id_y < dst_height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = input[src_pixIdx];
      src_pixIdx += src_inc[id_z];
      dst_pixIdx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pixIdx] = 0;
      dst_pixIdx += dst_inc[id_z];
    }
  }
}