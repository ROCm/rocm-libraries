#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void erase_batch(
    __global unsigned char *input, __global unsigned char *output,
    __global unsigned int *box_info, __global unsigned char *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  uchar3 pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i) * 3;
      pixel.x = colors[temp];
      pixel.y = colors[temp + 1];
      pixel.z = colors[temp + 2];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel.x;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.y;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.z;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_pln1_batch(
    __global unsigned char *input, __global unsigned char *output,
    __global unsigned int *box_info, __global unsigned char *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  uchar pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i);
      pixel = colors[temp];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_batch_int8(
    __global char *input, __global char *output,
    __global unsigned int *box_info, __global char *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  char3 pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i) * 3;
      pixel.x = colors[temp];
      pixel.y = colors[temp + 1];
      pixel.z = colors[temp + 2];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel.x;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.y;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.z;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_pln1_batch_int8(
    __global char *input, __global char *output,
    __global unsigned int *box_info, __global char *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  char pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i);
      pixel = colors[temp];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_batch_fp32(
    __global float *input, __global float *output,
    __global unsigned int *box_info, __global float *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float3 pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i) * 3;
      pixel.x = colors[temp];
      pixel.y = colors[temp + 1];
      pixel.z = colors[temp + 2];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel.x;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.y;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.z;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_pln1_batch_fp32(
    __global float *input, __global float *output,
    __global unsigned int *box_info, __global float *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i);
      pixel = colors[temp];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_batch_fp16(
    __global half *input, __global half *output,
    __global unsigned int *box_info, __global half *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  float3 pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i) * 3;
      pixel.x = (float) colors[temp];
      pixel.y = (float) colors[temp + 1];
      pixel.z = (float) colors[temp + 2];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = (half) pixel.x;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = (half) pixel.y;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = (half) pixel.z;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
    dst_pix_idx += dst_inc[id_z];
    src_pix_idx += src_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
  }
}

kernel void erase_pln1_batch_fp16(
    __global half *input, __global half *output,
    __global unsigned int *box_info, __global half *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *src_inc, __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  half pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i);
      pixel = colors[temp];
      break;
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  if (is_erase == true)
  {
    output[dst_pix_idx] = pixel;
  }
  else
  {
    output[dst_pix_idx] = input[src_pix_idx];
  }
}