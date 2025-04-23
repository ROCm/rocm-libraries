#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

uchar4 convert_one_pixel_to_rgb(float4 pixel) {
  float r, g, b;
  float h, s, v;

  h = pixel.x;
  s = pixel.y;
  v = pixel.z;

  float f = h / 60.0f;
  float hi = floor(f);
  f = f - hi;
  float p = v * (1 - s);
  float q = v * (1 - s * f);
  float t = v * (1 - s * (1 - f));

  if (hi == 0.0f || hi == 6.0f) {
    r = v;
    g = t;
    b = p;
  } else if (hi == 1.0f) {
    r = q;
    g = v;
    b = p;
  } else if (hi == 2.0f) {
    r = p;
    g = v;
    b = t;
  } else if (hi == 3.0f) {
    r = p;
    g = q;
    b = v;
  } else if (hi == 4.0f) {
    r = t;
    g = p;
    b = v;
  } else {
    r = v;
    g = p;
    b = q;
  }

  unsigned char red = (unsigned char)(255.0f * r);
  unsigned char green = (unsigned char)(255.0f * g);
  unsigned char blue = (unsigned char)(255.0f * b);
  unsigned char alpha = 0.0; //(unsigned char)(pixel.w);
  return (uchar4){red, green, blue, alpha};
}

float4 convert_one_pixel_to_hsv(uchar4 pixel) {
  float r, g, b, a;
  float h, s, v;

  r = pixel.x / 255.0f;
  g = pixel.y / 255.0f;
  b = pixel.z / 255.0f;
  a = pixel.w;

  float max = amd_max3(r, g, b);
  float min = amd_min3(r, g, b);
  float diff = max - min;

  v = max;

  if (v == 0.0f) { // black
    h = s = 0.0f;
  } else {
    s = diff / v;
    if (diff < 0.001f) { // grey
      h = 0.0f;
    } else { // color
      if (max == r) {
        h = 60.0f * (g - b) / diff;
        if (h < 0.0f) {
          h += 360.0f;
        }
      } else if (max == g) {
        h = 60.0f * (2 + (b - r) / diff);
      } else {
        h = 60.0f * (4 + (r - g) / diff);
      }
    }
  }

  return (float4){h, s, v, a};
}
__kernel void huergb_pkd(__global unsigned char *input,
                         __global unsigned char *output, const float hue,
                         const float sat, const unsigned int height,
                         const unsigned int width) {
  int id_x = get_global_id(0), id_y = get_global_id(1);
  float r, g, b, min, max, delta, h, s, v;
  if (id_x >= width || id_y >= height)
    return;

  int pixIdx = (id_y * width + id_x) * 3;
  uchar4 pixel;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + 1];
  pixel.z = input[pixIdx + 2];
  pixel.w = 0; // Transpency factor yet to come
  float4 hsv;
  hsv = convert_one_pixel_to_hsv(pixel);
  hsv.x += hue;
  if (hsv.x > 360.0) {
    hsv.x = hsv.x - 360.0;
  } else if (hsv.x < 0) {
    hsv.x = hsv.x + 360.0;
  }

  hsv.y *= sat;
  if (hsv.y > 1.0) {
    hsv.y = 1.0;
  } else if (hsv.y < 0.0) {
    hsv.y = 0.0;
  }

  pixel = convert_one_pixel_to_rgb(hsv);

  output[pixIdx] = pixel.x;
  output[pixIdx + 1] = pixel.y;
  output[pixIdx + 2] = pixel.z;
}

__kernel void huergb_pln(__global unsigned char *input,
                         __global unsigned char *output, const float hue,
                         const float sat, const unsigned int height,
                         const unsigned int width) {
  int id_x = get_global_id(0), id_y = get_global_id(1);
  float r, g, b, min, max, delta, h, s, v;
  if (id_x >= width || id_y >= height)
    return;

  int pixIdx = (id_y * width + id_x);
  uchar4 pixel;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + width * height];
  pixel.z = input[pixIdx + 2 * width * height];
  pixel.w = 0; // Transpency factor yet to come
  float4 hsv;
  hsv = convert_one_pixel_to_hsv(pixel);
  hsv.x += hue;
  if (hsv.x > 360.0) {
    hsv.x = hsv.x - 360.0;
  } else if (hsv.x < 0) {
    hsv.x = hsv.x + 360.0;
  }

  hsv.y *= sat;
  if (hsv.y > 1.0) {
    hsv.y = 1.0;
  } else if (hsv.y < 0.0) {
    hsv.y = 0.0;
  }

  pixel = convert_one_pixel_to_rgb(hsv);

  output[pixIdx] = pixel.x;
  output[pixIdx + width * height] = pixel.y;
  output[pixIdx + 2 * width * height] = pixel.z;
}

__kernel void hue_batch(
    __global unsigned char *input, __global unsigned char *output,
    __global float *hue, __global int *xroi_begin, __global int *xroi_end,
    __global int *yroi_begin, __global int *yroi_end,
    __global unsigned int *height, __global unsigned int *width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *inc, // use width * height for pln and 1 for pkd
    const int plnpkdindex       // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= width[id_z] || id_y >= height[id_z])
    return;
  uchar4 pixel;
  float4 hsv;

  int pixIdx =
      batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + inc[id_z]];
  pixel.z = input[pixIdx + 2 * inc[id_z]];
  pixel.w = 0.0;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
    hsv.x += hue[id_z];
    if (hsv.x > 360.0) {
      hsv.x = hsv.x - 360.0;
    } else if (hsv.x < 0) {
      hsv.x = hsv.x + 360.0;
    }
    pixel = convert_one_pixel_to_rgb(
        hsv); // Converting to RGB back with hue modification
    output[pixIdx] = pixel.x;
    output[pixIdx + inc[id_z]] = pixel.y;
    output[pixIdx + 2 * inc[id_z]] = pixel.z;
  } else {
    output[pixIdx] = pixel.x;
    output[pixIdx + inc[id_z]] = pixel.y;
    output[pixIdx + 2 * inc[id_z]] = pixel.z;
  }
}

__kernel void saturation_batch(
    __global unsigned char *input, __global unsigned char *output,
    __global float *sat, __global int *xroi_begin, __global int *xroi_end,
    __global int *yroi_begin, __global int *yroi_end,
    __global unsigned int *height, __global unsigned int *width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    __global unsigned int *inc, // use width * height for pln and 1 for pkd
    const int plnpkdindex       // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= width[id_z] || id_y >= height[id_z])
    return;
  uchar4 pixel;
  float4 hsv;

  int pixIdx =
      batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + inc[id_z]];
  pixel.z = input[pixIdx + 2 * inc[id_z]];
  pixel.w = 0.0;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    hsv = convert_one_pixel_to_hsv(pixel);
    hsv.y *= sat[id_z];
    if (hsv.y > 1.0) {
      hsv.y = 1.0;
    } else if (hsv.y < 0.0) {
      hsv.y = 0.0;
    }
    pixel = convert_one_pixel_to_rgb(
        hsv); // Converting to RGB back with saturation modification
    output[pixIdx] = pixel.x;
    output[pixIdx + inc[id_z]] = pixel.y;
    output[pixIdx + 2 * inc[id_z]] = pixel.z;
  } else {
    output[pixIdx] = pixel.x;
    output[pixIdx + inc[id_z]] = pixel.y;
    output[pixIdx + 2 * inc[id_z]] = pixel.z;
  }
}

__kernel void convert_single_rgb_hsv(__global unsigned char *input,
                                     __global float  *output,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int inc,
                                     const unsigned int plnpkdindex) {
  int id_x = get_global_id(0), id_y = get_global_id(1);
  if (id_x >= width || id_y >= height)
    return;

  int pixIdx = (id_y * width + id_x) * plnpkdindex;
  uchar4 pixel;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + inc];
  pixel.z = input[pixIdx + 2 * inc];
  pixel.w = 0; // Transpency factor yet to come
  float4 hsv;
  hsv = convert_one_pixel_to_hsv(pixel);

  output[pixIdx] = hsv.x;
  output[pixIdx + inc] = hsv.y;
  output[pixIdx + 2 * inc] = hsv.z;
}

__kernel void convert_single_hsv_rgb(__global  float *input,
                                     __global unsigned char *output,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int inc,
                                     const unsigned int plnpkdindex) {
  int id_x = get_global_id(0), id_y = get_global_id(1);
  if (id_x >= width || id_y >= height)
    return;

  int pixIdx = (id_y * width + id_x) * plnpkdindex;
  float4 pixel;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + inc];
  pixel.z = input[pixIdx + 2 * inc];
  pixel.w = 0; // Transpency factor yet to come
  uchar4 rgb;
  rgb = convert_one_pixel_to_rgb(pixel);

  output[pixIdx] = rgb.x;
  output[pixIdx + inc] = rgb.y;
  output[pixIdx + 2 * inc] = rgb.z;
}

__kernel void convert_batch_rgb_hsv(
    __global unsigned char *input, __global float *output,
    __global int *xroi_begin, __global int *xroi_end, __global int *yroi_begin,
    __global int *yroi_end, __global unsigned int *height,
    __global unsigned int *width, __global unsigned int *max_width,
    __global unsigned long *batch_index,
    __global unsigned int *inc, // use width * height for pln and 1 for pkd
    const int plnpkdindex       // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= width[id_z] || id_y >= height[id_z])
    return;
  uchar4 pixel;
  float4 hsv;

  int pixIdx =
      batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + inc[id_z]];
  pixel.z = input[pixIdx + 2 * inc[id_z]];
  pixel.w = 0.0;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
    output[pixIdx] = hsv.x;
    output[pixIdx + inc[id_z]] = hsv.y;
    output[pixIdx + 2 * inc[id_z]] = hsv.z;
  } else {
    output[pixIdx] = 0.0;
    output[pixIdx + inc[id_z]] = 0.0;
    output[pixIdx + 2 * inc[id_z]] = 0.0;
  }
}

__kernel void convert_batch_hsv_rgb(
    __global float *input, __global unsigned char *output,
    __global int *xroi_begin, __global int *xroi_end, __global int *yroi_begin,
    __global int *yroi_end, __global unsigned int *height,
    __global unsigned int *width, __global unsigned int *max_width,
    __global unsigned long *batch_index,
    __global unsigned int *inc, // use width * height for pln and 1 for pkd
    const int plnpkdindex       // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  if (id_x >= width[id_z] || id_y >= height[id_z])
    return;
  uchar4 rgb;
  float4 pixel;

  int pixIdx =
      batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
  pixel.x = input[pixIdx];
  pixel.y = input[pixIdx + inc[id_z]];
  pixel.z = input[pixIdx + 2 * inc[id_z]];
  pixel.w = 0.0;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    rgb = convert_one_pixel_to_rgb(pixel); // Converting to HSV
    output[pixIdx] = rgb.x;
    output[pixIdx + inc[id_z]] = rgb.y;
    output[pixIdx + 2 * inc[id_z]] = rgb.z;
  } else {
    output[pixIdx] = 0;
    output[pixIdx + inc[id_z]] = 0;
    output[pixIdx + 2 * inc[id_z]] = 0;
  }
}
