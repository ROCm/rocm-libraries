#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

float gaussian(int x, int y, float std_dev) {
  float res, pi = 3.14;
  res = 1 / (2 * pi * std_dev * std_dev);
  float exp1, exp2;
  exp1 = -(x * x) / (2 * std_dev * std_dev);
  exp2 = -(y * y) / (2 * std_dev * std_dev);
  exp1 = exp1 + exp2;
  exp1 = exp(exp1);
  res *= exp1;
  return res;
}

__kernel void non_linear_blend_batch(
    __global unsigned char *input1, __global unsigned char *input2,
    __global unsigned char *output, __global float *std_dev,
    __global int *xroi_begin, __global int *xroi_end, __global int *yroi_begin,
    __global int *yroi_end, __global unsigned int *height,
    __global unsigned int *width, __global unsigned int *max_width,
    __global unsigned long *batch_index, const unsigned int channel,
    __global unsigned int *src_inc,
    __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  unsigned char valuergb1, valuergb2;
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    int x = (id_x - (width[id_z] >> 1));
    int y = (id_y - (height[id_z] >> 1));
    float gaussianvalue =
        gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
    for (indextmp = 0; indextmp < channel; indextmp++) {
      valuergb1 = input1[src_pix_idx];
      valuergb2 = input2[src_pix_idx];
      output[dst_pix_idx] = gaussianvalue * input1[src_pix_idx] +
                            (1 - gaussianvalue) * input2[src_pix_idx];
      src_pix_idx += src_inc[id_z];
      dst_pix_idx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dst_inc[id_z];
    }
  }
}

__kernel void non_linear_blend_batch_fp16(
    __global half *input1, __global half *input2, __global half *output,
    __global float *std_dev, __global int *xroi_begin, __global int *xroi_end,
    __global int *yroi_begin, __global int *yroi_end,
    __global unsigned int *height, __global unsigned int *width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    const unsigned int channel, __global unsigned int *src_inc,
    __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  unsigned char valuergb1, valuergb2;
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    int x = (id_x - (width[id_z] >> 1));
    int y = (id_y - (height[id_z] >> 1));
    float gaussianvalue =
        gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
    for (indextmp = 0; indextmp < channel; indextmp++) {
      valuergb1 = input1[src_pix_idx];
      valuergb2 = input2[src_pix_idx];
      output[dst_pix_idx] = gaussianvalue * input1[src_pix_idx] +
                            (1 - gaussianvalue) * input2[src_pix_idx];
      src_pix_idx += src_inc[id_z];
      dst_pix_idx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dst_inc[id_z];
    }
  }
}

__kernel void non_linear_blend_batch_fp32(
    __global float *input1, __global float *input2, __global float *output,
    __global float *std_dev, __global int *xroi_begin, __global int *xroi_end,
    __global int *yroi_begin, __global int *yroi_end,
    __global unsigned int *height, __global unsigned int *width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    const unsigned int channel, __global unsigned int *src_inc,
    __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  unsigned char valuergb1, valuergb2;
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    int x = (id_x - (width[id_z] >> 1));
    int y = (id_y - (height[id_z] >> 1));
    float gaussianvalue =
        gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
    for (indextmp = 0; indextmp < channel; indextmp++) {
      valuergb1 = input1[src_pix_idx];
      valuergb2 = input2[src_pix_idx];

      output[dst_pix_idx] = (half)gaussianvalue * input1[src_pix_idx] +
                            (1 - gaussianvalue) * input2[src_pix_idx];
      src_pix_idx += src_inc[id_z];
      dst_pix_idx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dst_inc[id_z];
    }
  }
}

__kernel void non_linear_blend_batch_int8(
    __global char *input1, __global char *input2, __global char *output,
    __global float *std_dev, __global int *xroi_begin, __global int *xroi_end,
    __global int *yroi_begin, __global int *yroi_end,
    __global unsigned int *height, __global unsigned int *width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    const unsigned int channel, __global unsigned int *src_inc,
    __global unsigned int *dst_inc, // use width * height for pln and 1 for pkd
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  unsigned char valuergb1, valuergb2;
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    int x = (id_x - (width[id_z] >> 1));
    int y = (id_y - (height[id_z] >> 1));
    float gaussianvalue =
        gaussian(x, y, std_dev[id_z]) / gaussian(0.0, 0.0, std_dev[id_z]);
    for (indextmp = 0; indextmp < channel; indextmp++) {
      valuergb1 = input1[src_pix_idx];
      valuergb2 = input2[src_pix_idx];
      output[dst_pix_idx] = gaussianvalue * input1[src_pix_idx] +
                            (1 - gaussianvalue) * input2[src_pix_idx];
      src_pix_idx += src_inc[id_z];
      dst_pix_idx += dst_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dst_inc[id_z];
    }
  }
}