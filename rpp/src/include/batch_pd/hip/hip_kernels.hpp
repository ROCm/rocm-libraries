/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef HIP_KERNELS_H
#define HIP_KERNELS_H

#include <map>

// color_model_conversions
extern "C" __global__ void channel_extract_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int extractChannelNumber);
extern "C" __global__ void channel_extract_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int extractChannelNumber);
extern "C" __global__ void channel_extract_batch(unsigned char* input, unsigned char* output, unsigned int* channelNumber, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void channel_combine_pln(unsigned char* input1, unsigned char* input2, unsigned char* input3, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void channel_combine_pkd(unsigned char* input1, unsigned char* input2, unsigned char* input3, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void channel_combine_batch(unsigned char* input1, unsigned char* input2, unsigned char* input3, unsigned char* output, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void look_up_table_pkd(unsigned char* input, unsigned char* output, unsigned char* lutPtr, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void look_up_table_pln(unsigned char* input, unsigned char* output, unsigned char* lutPtr, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void look_up_table_batch(unsigned char* input, unsigned char* output, unsigned char* lutPtr, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void tensor_look_up_table(const unsigned int tensorDimension, unsigned char* input, unsigned char* output, const unsigned int a, const unsigned int b, const unsigned int c, unsigned char* lutPtr);
extern "C" __global__ void huergb_pkd(unsigned char *input, unsigned char *output, const  float hue, const  float sat, const unsigned int height, const unsigned int width);
extern "C" __global__ void huergb_pln(unsigned char *input, unsigned char *output, const  float hue, const  float sat, const unsigned int height, const unsigned int width);
extern "C" __global__ void hue_batch(unsigned char* input, unsigned char* output, float *hue, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void saturation_batch(unsigned char* input, unsigned char* output, float *sat, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void color_temperature_batch(unsigned char *input, unsigned char *output, int *value, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void temperature_packed(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const int modificationValue);
extern "C" __global__ void temperature_planar(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const int modificationValue);
extern "C" __global__ void vignette_batch(unsigned char *input, unsigned char *output, float *stdDev, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void vignette_pln(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const float stdDev);
extern "C" __global__ void vignette_pkd(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const float stdDev);
double gaussian(double x, double sigmaI);
float gaussian(int x, int y, float stdDev);
extern "C" __global__ void gaussian(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const float mean, const float sigma, const unsigned int channel);

// filter_operations
extern "C" __global__ void sobel_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int sobelType);
extern "C" __global__ void sobel_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int sobelType);
extern "C" __global__ void sobel_batch(unsigned char* input, unsigned char* output, unsigned int *sobelType, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void box_filter_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void median_filter_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void median_filter_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void median_filter_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void non_max_suppression_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void non_max_suppression_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void non_max_suppression_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void gaussian_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth);
extern "C" __global__ void gaussian_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth);
extern "C" __global__ void gaussian_filter_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, float *stdDev, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void gaussian(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const float mean, const float sigma, const unsigned int channel);
extern "C" __global__ void custom_convolution_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth);
extern "C" __global__ void custom_convolution_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth);
extern "C" __global__ void custom_convolution_batch(unsigned char* input, unsigned char* output, float *kernelValue, const unsigned int kHeight, const unsigned int kWidth, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void convolution_batch(unsigned char *input, unsigned char *output, float *filter, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *batch_index, const unsigned int channel, const unsigned int kerneSize, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void naive_convolution_packed(const unsigned char *input, unsigned char *output, float *filter, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int filterSize);
extern "C" __global__ void naive_convolution_planar(const unsigned char *input, unsigned char *output, float *filter, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int filterSize);
extern "C" __global__ void bilateral_filter_batch(unsigned char *input, unsigned char *output, unsigned int *kernelSize, double *sigmaS, double *sigmaI, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void bilateral_filter_packed(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int filterSize, const double sigmaI, const double sigmaS);
extern "C" __global__ void bilateral_filter_planar(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int filterSize, const double sigmaI, const double sigmaS);
double distance(int x1, int y1, int x2, int y2);
double gaussian(double x, double sigmaI);
float gaussian(int x, int y, float stdDev);

// geometry_transforms
extern "C" __global__ void lenscorrection_pln(const unsigned char *input, unsigned char *output, const float strength, const float zoom, const float halfWidth, const float halfHeight, const float correctionRadius, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void lenscorrection_pkd(const unsigned char *input, unsigned char *output, const float strength, const float zoom, const float halfWidth, const float halfHeight, const float correctionRadius, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void lens_correction_batch(unsigned char* input, unsigned char* output, unsigned int *height, unsigned int *width, unsigned int *max_width, float *strength, float *zoom, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex, const unsigned int batch_size);
extern "C" __global__ void fisheye_batch(unsigned char* input, unsigned char* output, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex, const unsigned int batch_size);
extern "C" __global__ void fisheye_packed(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void fisheye_planar(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_bothaxis_packed(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_vertical_packed(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_horizontal_packed(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_bothaxis_planar(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_vertical_planar(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_horizontal_planar(const unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void flip_batch(unsigned char* srcPtr, unsigned char* dstPtr, unsigned int *flipAxis, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, unsigned int *xroi_begin, unsigned int *xroi_end, unsigned int *yroi_begin, unsigned int *yroi_end, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void scale_batch(unsigned char* srcPtr, unsigned char* dstPtr, float* percentage, unsigned int *source_height, unsigned int *source_width, unsigned int *dest_height, unsigned int *dest_width, unsigned int *max_source_width, unsigned int *max_dest_width, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned long *source_batch_index, unsigned long *dest_batch_index, const unsigned int channel, unsigned int *source_inc, unsigned int *dest_inc, const int plnpkdindex);
extern "C" __global__ void scale_pkd(unsigned char* srcPtr, unsigned char* dstPtr, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel, const unsigned int exp_dest_height, const unsigned int exp_dest_width);
extern "C" __global__ void scale_pln(unsigned char* srcPtr, unsigned char* dstPtr, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel, const unsigned int exp_dest_height, const unsigned int exp_dest_width);
extern "C" __global__ void warp_perspective_pln(unsigned char* srcPtr, unsigned char* dstPtr, float* perspective, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void warp_perspective_pkd (unsigned char* srcPtr, unsigned char* dstPtr, float* perspective, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void warp_perspective_batch(unsigned char* srcPtr, unsigned char* dstPtr, float *perspective, unsigned int *source_height, unsigned int *source_width, unsigned int *dest_height, unsigned int *dest_width, unsigned int *max_source_width, unsigned int *max_dest_width, unsigned long *source_batch_index, unsigned long *dest_batch_index, const unsigned int channel, unsigned int *source_inc, unsigned int *dest_inc, const int plnpkdindex);
extern "C" __global__ void resize_crop_batch(unsigned char* srcPtr, unsigned char* dstPtr, unsigned int *source_height, unsigned int *source_width, unsigned int *dest_height, unsigned int *dest_width, unsigned int *max_source_width, unsigned int *max_dest_width, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned long *source_batch_index, unsigned long *dest_batch_index, const unsigned int channel, unsigned int *source_inc, unsigned int *dest_inc, const unsigned int padding, const unsigned int type, const int plnpkdindex);
extern "C" __global__ void resize_batch(unsigned char* srcPtr, unsigned char* dstPtr, unsigned int *source_height, unsigned int *source_width, unsigned int *dest_height, unsigned int *dest_width, unsigned int *max_source_width, unsigned int *max_dest_width, unsigned long *source_batch_index, unsigned long *dest_batch_index, const unsigned int channel, unsigned int *source_inc, unsigned int *dest_inc, const int plnpkdindex);
extern "C" __global__ void resize_crop_pkd(unsigned char *srcPtr, unsigned char *dstPtr, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int x1, const unsigned int y1, const unsigned int x2, const unsigned int y2, const unsigned int padding, const unsigned int type, const unsigned int channel);
extern "C" __global__ void resize_crop_pln(unsigned char *srcPtr, unsigned char *dstPtr, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int x1, const unsigned int y1, const unsigned int x2, const unsigned int y2, const unsigned int padding, const unsigned int type, const unsigned int channel);
extern "C" __global__ void resize_pkd(unsigned char *srcPtr, unsigned char *dstPtr, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void resize_pln(unsigned char *srcPtr, unsigned char *dstPtr, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void rotate_batch(unsigned char* srcPtr, unsigned char* dstPtr, float *angleDeg, unsigned int *source_height, unsigned int *source_width, unsigned int *dest_height, unsigned int *dest_width, unsigned int *xroi_begin, unsigned int *xroi_end, unsigned int *yroi_begin, unsigned int *yroi_end, unsigned int *max_source_width, unsigned int *max_dest_width, unsigned long *source_batch_index, unsigned long *dest_batch_index, const unsigned int channel, unsigned int *source_inc, unsigned int *dest_inc, const int plnpkdindex);
extern "C" __global__ void rotate_pkd(unsigned char *srcPtr, unsigned char *dstPtr, const float angleDeg, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void rotate_pln(unsigned char *srcPtr, unsigned char *dstPtr, const float angleDeg, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void warp_affine_pkd(unsigned char *srcPtr, unsigned char *dstPtr, float *affine, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);
extern "C" __global__ void warp_affine_pln(unsigned char *srcPtr, unsigned char *dstPtr, float *affine, const unsigned int source_height, const unsigned int source_width, const unsigned int dest_height, const unsigned int dest_width, const unsigned int channel);

// arithmetic_operations
extern "C" __global__ void absolute_difference_batch(unsigned char* input1, unsigned char* input2, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void absolute_difference(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void magnitude_batch(unsigned char* input1, unsigned char* input2, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void magnitude(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void multiply_batch(unsigned char* input1, unsigned char* input2, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void multiply(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void phase_batch(unsigned char* input1, unsigned char* input2, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void phase(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void tensor_add(const unsigned int tensorDimension, unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int a, const unsigned int b, const unsigned int c);
extern "C" __global__ void tensor_subtract(const unsigned int tensorDimension, unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int a, const unsigned int b, const unsigned int c);
extern "C" __global__ void tensor_multiply(const unsigned int tensorDimension, unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int a, const unsigned int b, const unsigned int c);
extern "C" __global__ void accumulate_squared_batch(unsigned char *input, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void accumulate_weighted_batch(unsigned char *input1, unsigned char *input2, float *alpha, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void accumulate_batch(unsigned char *input1, unsigned char *input2, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void accumulate_squared(unsigned char *input, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void accumulate_weighted(unsigned char *input1, unsigned char *input2, const double alpha, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void accumulate(unsigned char *input1, unsigned char *input2, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void add_batch(unsigned char *input1, unsigned char *input2, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void add(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void subtract_batch(unsigned char *input1, unsigned char *input2, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void subtract(unsigned char *a, unsigned char *b, unsigned char *c, const unsigned int height, const unsigned int width, const unsigned int channel);
unsigned char accumulate_weight_formula(unsigned char input_pixel1, unsigned char input_pixel2, float alpha);

// morphological_transforms
extern "C" __global__ void erode_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void erode_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void erode_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void dilate_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void dilate_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void dilate_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);

// statistical_operations
extern "C" __global__ void thresholding(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned char min, const unsigned char max);
extern "C" __global__ void thresholding_batch(unsigned char* input, unsigned char* output, unsigned char *min, unsigned char *max, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void min_hip(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void min_batch(unsigned char* input1, unsigned char* input2, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int  *inc, const int plnpkdindex);
extern "C" __global__ void max_hip(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void max_batch(unsigned char* input1, unsigned char* input2, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int  *inc, const int plnpkdindex);
extern "C" __global__ void min_loc(const unsigned char* input, unsigned char *partial_min, unsigned char *partial_min_location);
extern "C" __global__ void max_loc(const unsigned char* input, unsigned char *partial_max, unsigned char *partial_max_location);
extern "C" __global__ void integral_pkd_col(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void integral_pln_col(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void integral_pkd_row(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void integral_pln_row(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void integral_up_pln(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel, const int loop, const int diag);
extern "C" __global__ void integral_low_pln(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel, const int loop, const int diag);
extern "C" __global__ void integral_up_pkd(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel, const int loop, const int diag);
extern "C" __global__ void integral_low_pkd(unsigned char* input, unsigned int* output, const unsigned int height, const unsigned int width, const unsigned int channel, const int loop, const int diag);
extern "C" __global__ void sum(const unsigned char* input, long *partialSums);
extern "C" __global__ void mean_stddev(const unsigned char* input, float *partial_mean_sum, const float mean);
__device__ unsigned int get_pkd_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);
unsigned int get_pkd_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);
__device__ unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);
unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);

// computer_vision
extern "C" __global__ void local_binary_pattern_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void local_binary_pattern_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void local_binary_pattern_batch(unsigned char* input, unsigned char* output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void gaussian_image_pyramid_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth);
extern "C" __global__ void gaussian_image_pyramid_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth);
extern "C" __global__ void gaussian_image_pyramid_batch(unsigned char* input, unsigned char* output, unsigned int *kernelSize, float *stdDev, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void gaussian_image_pyramid_pkd_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth, const unsigned long batchIndex);
extern "C" __global__ void gaussian_image_pyramid_pln_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth, const unsigned long batchIndex);
extern "C" __global__ void laplacian_image_pyramid_pkd_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth, const unsigned long batchIndex);
extern "C" __global__ void laplacian_image_pyramid_pln_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, float* kernal, const unsigned int kernalheight, const unsigned int kernalwidth, const unsigned long batchIndex);
extern "C" __global__ void ced_pln3_to_pln1_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned long batchIndex);
extern "C" __global__ void ced_pkd3_to_pln1(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void ced_pln1_to_pkd3(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void ced_pkd3_to_pln1_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned long batchIndex);
extern "C" __global__ void ced_non_max_suppression(unsigned char* input, unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned char min, const unsigned char max);
extern "C" __global__ void ced_non_max_suppression_batch(unsigned char* input, unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned char min, const unsigned char max);
extern "C" __global__ void canny_edge_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned char min, const unsigned char max, const unsigned long batchIndex, const unsigned int originalChannel);
extern "C" __global__ void ced_pln1_to_pln3_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned long batchIndex);
extern "C" __global__ void ced_pln1_to_pkd3_batch(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned long batchIndex);
extern "C" __global__ void convert_bit_depth_u8s8(unsigned char* input, char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void convert_bit_depth_u8u16(unsigned char* input, unsigned short* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void convert_bit_depth_u8s16(unsigned char* input, short* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void convert_bit_depth_batch_u8s8(unsigned char* input, char* output, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void convert_bit_depth_batch_u8u16(unsigned char* input, unsigned short* output, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void convert_bit_depth_batch_u8s16(unsigned char* input, short* output, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void harris_corner_detector_strength(unsigned char* sobelX, unsigned char* sobelY, float* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize, const float kValue, const float threshold);
extern "C" __global__ void harris_corner_detector_nonmax_supression(float* input, float* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void harris_corner_detector_pln(unsigned char* input, float* inputFloat, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void harris_corner_detector_pkd(unsigned char* input, float* inputFloat, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void fast_corner_detector(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned char threshold, const unsigned int numOfPixels);
extern "C" __global__ void fast_corner_detector_nms_pln(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void fast_corner_detector_nms_pkd(unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void reconstruction_laplacian_image_pyramid_pkd(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int height2, const unsigned int width2, const unsigned int channel);
extern "C" __global__ void reconstruction_laplacian_image_pyramid_pln(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int height2, const unsigned int width2, const unsigned int channel);
extern "C" __global__ void tensor_matrix_multiply(unsigned char* input1, unsigned char* input2, unsigned char* output, const unsigned int r1, const unsigned int c1, const unsigned int r2, const unsigned int c2);
extern "C" __global__ void tensor_convert_bit_depth_u8s8(const unsigned int tensorDimension, unsigned char* input, char* output, const unsigned int a, const unsigned int b, const unsigned int c);
extern "C" __global__ void tensor_convert_bit_depth_u8u16(const unsigned int tensorDimension, unsigned char* input, unsigned short* output, const unsigned int a, const unsigned int b, const unsigned int c);
extern "C" __global__ void tensor_convert_bit_depth_u8s16(const unsigned int tensorDimension, unsigned char* input, short* output, const unsigned int a, const unsigned int b, const unsigned int c);

// logical_operations
extern "C" __global__ void bitwise_AND_batch(unsigned char *input1, unsigned char *input2, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void bitwise_AND(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void bitwise_NOT_batch(unsigned char *input, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void bitwise_NOT(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void exclusive_OR_batch(unsigned char *input1, unsigned char *input2, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void exclusive_OR(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void inclusive_OR_batch(unsigned char *input1, unsigned char *input2, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void inclusive_OR(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);

// fused_functions
extern "C" __global__ void color_twist_batch(unsigned char *input, unsigned char *output, float *alpha, float *beta, float *hue, float *sat, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, unsigned int *inc, unsigned int *dst_inc, const int in_plnpkdind, const int out_plnpkdind);
extern "C" __global__ void resize_crop_mirror_batch(unsigned char *srcPtr, unsigned char *dstPtr, unsigned int *source_height, unsigned int *source_width, unsigned int *dest_height, unsigned int *dest_width, unsigned int *max_source_width, unsigned int *max_dest_width, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, int *mirror, unsigned long *source_batch_index, unsigned long *dest_batch_index, const unsigned int channel, unsigned int *source_inc, unsigned int *dest_inc, const int in_plnpkdind, const int out_plnpkdind);
extern "C" __global__ void crop_batch(unsigned char *input, unsigned char *output, unsigned int *dst_height, unsigned int *dst_width, unsigned int *src_width, unsigned int *start_x, unsigned int *start_y, unsigned int *max_src_width, unsigned int *max_dst_width, unsigned long *src_batch_index, unsigned long *dst_batch_index, const unsigned int channel, unsigned int *src_inc, unsigned int *dst_inc, const int in_plnpkdind, const int out_plnpkdind);
extern "C" __global__ void crop_mirror_normalize_batch(unsigned char *input, unsigned char *output, unsigned int *dst_height, unsigned int *dst_width, unsigned int *src_width, unsigned int *start_x, unsigned int *start_y, float *mean, float *std_dev, unsigned int *flip, unsigned int *max_src_width, unsigned int *max_dst_width, unsigned long *src_batch_index, unsigned long *dst_batch_index, const unsigned int channel, unsigned int *src_inc, unsigned int *dst_inc, const int in_plnpkdind, const int out_plnpkdind);

// image_augmentations
extern "C" __global__ void brightness(unsigned char *input, unsigned char *output, const float alpha, const int beta, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void brightness_batch(unsigned char* input, unsigned char* output, float *alpha, float *beta, unsigned int *xroi_begin, unsigned int *xroi_end, unsigned int *yroi_begin, unsigned int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
unsigned char gamma_correct(unsigned char input, float gamma);
extern "C" __global__ void gamma_correction(unsigned char *input, unsigned char *output, const float gamma, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void gamma_correction_batch(unsigned char *input, unsigned char *output, float *gamma, unsigned int *xroi_begin, unsigned int *xroi_end, unsigned int *yroi_begin, unsigned int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
unsigned char blend_formula(unsigned char input_pixel1, unsigned char input_pixel2, float alpha);
extern "C" __global__ void blend(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const float alpha, const unsigned int channel);
extern "C" __global__ void blend_batch(unsigned char *input1, unsigned char *input2, unsigned char *output, float *alpha, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void contrast_stretch(unsigned char *input, unsigned char *output, const unsigned int min, const unsigned int max, const unsigned int new_min, const unsigned int new_max, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void contrast_batch(unsigned char *input, unsigned char *output, unsigned int *min, unsigned int *max, unsigned int *new_min, unsigned int *new_max, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
unsigned char exposure_value(unsigned char input, float value);
extern "C" __global__ void exposure(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const float value);
extern "C" __global__ void exposure_batch(unsigned char *input, unsigned char *output, float *value, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void random_shadow_packed(unsigned char* input, unsigned char* output, const unsigned int srcheight, const unsigned int srcwidth, const unsigned int channel, const unsigned int x1, const unsigned int y1, const unsigned int x2, const unsigned int y2);
extern "C" __global__ void random_shadow_planar(unsigned char* input, unsigned char* output, const unsigned int srcheight, const unsigned int srcwidth, const unsigned int channel, const unsigned int x1, const unsigned int y1, const unsigned int x2, const unsigned int y2);
extern "C" __global__ void random_shadow(const unsigned char* input, unsigned char* output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void fog_batch(unsigned char *input, unsigned char *output, float *fogValue, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void fog_pkd(unsigned char *input, const unsigned int height, const unsigned int width, const unsigned int channel, const float fogValue);
extern "C" __global__ void fog_planar(unsigned char *input, const unsigned int height, const unsigned int width, const unsigned int channel, const float fogValue);
extern "C" __global__ void jitter_batch(unsigned char *input, unsigned char *output, unsigned int *kernelSize, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void jitter_pln(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
extern "C" __global__ void jitter_pkd(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int kernelSize);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
extern "C" __global__ void snp_pln(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int pixelDistance);
extern "C" __global__ void snp_pkd(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int pixelDistance);
extern "C" __global__ void noise_batch(unsigned char* input, unsigned char* output, float *noiseProbability, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
double gaussian(double x, double sigmaI);
float gaussian(int x, int y, float stdDev);
extern "C" __global__ void gaussian(unsigned char *input1, unsigned char *input2, unsigned char *output, const unsigned int height, const unsigned int width, const float mean, const float sigma, const unsigned int channel);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
extern "C" __global__ void pixelate_batch(unsigned char *input, unsigned char *output, int *xroi_begin, int *xroi_end, int *yroi_begin, int *yroi_end, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void pixelate_pln(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void pixelate_pkd(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__ void rain_batch(unsigned char *input, unsigned char *output, float *rainPercentage, unsigned int *rainWidth, unsigned int *rainHeight, float *transparency, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void rain_pln(unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int pixelDistance, const unsigned int rainWidth, const unsigned int rainHeight, const float transparency);
extern "C" __global__ void rain_pkd(unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int pixelDistance, const unsigned int rainWidth, const unsigned int rainHeight, const float transparency);
extern "C" __global__ void rain(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
extern "C" __global__ void snow_pln(unsigned char *output,const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int pixelDistance);
extern "C" __global__ void snow_pkd(unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel, const unsigned int pixelDistance);
extern "C" __global__ void snow(unsigned char *input, unsigned char *output, const unsigned int height, const unsigned int width, const unsigned int channel);
extern "C" __global__  void snow_batch(unsigned char* input, unsigned char* output, const float *snowPercentage, const unsigned int *height, const unsigned int *width, const unsigned int *max_width, const unsigned long *batch_index, const unsigned int channel, const unsigned int *inc, const int plnpkdindex);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
inline unsigned int xorshift(int pixid);
extern "C" __global__ void histogram_equalize_pkd(unsigned char *input, unsigned char *output, unsigned int *cum_histogram, const unsigned int width, const unsigned int height, const unsigned int channel);
extern "C" __global__ void histogram_equalize_pln(unsigned char *input, unsigned char *output, unsigned int *cum_histogram, const unsigned int width, const unsigned int height, const unsigned int channel);
extern "C" __global__ void histogram_equalize_batch(unsigned char* input, unsigned char* output, unsigned int *cum_histogram, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int channel, const unsigned int batch_size, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void histogram_sum_partial(unsigned int *histogramPartial, unsigned int *histogram, const unsigned int num_groups);
extern "C" __global__ void histogram_sum_partial_batch(unsigned int *histogramPartial, unsigned int *histogram, const unsigned int batch_size, const unsigned int num_groups, const unsigned int channel);
extern "C" __global__ void partial_histogram_batch(unsigned char* input, unsigned int *histogramPartial, unsigned int *height, unsigned int *width, unsigned int *max_width, unsigned long *batch_index, const unsigned int num_groups, const unsigned int channel, const unsigned int batch_size, unsigned int *inc, const int plnpkdindex);
extern "C" __global__ void partial_histogram_semibatch(unsigned char* input, unsigned int *histogramPartial, const unsigned int height, const unsigned int width, const unsigned int max_width, const unsigned long batch_index, const unsigned  int hist_index, const unsigned int channel, const unsigned int inc, const int plnpkdindex);
extern "C" __global__ void partial_histogram_pkd(unsigned char *input, unsigned int *histogramPartial, const unsigned int width, const unsigned int height, const unsigned int channel);
extern "C" __global__ void partial_histogram_pln(unsigned char *input, unsigned int *histogramPartial, const unsigned int width, const unsigned int height, const unsigned int channel);
__device__ unsigned int get_pkd_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);
unsigned int get_pkd_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);
__device__ unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);
unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel);

// other
extern "C" __global__ void scan(int *input, int *output);
extern "C" __global__ void scan_1c(int *input, int *output, __local int *b, __local int *c);
extern "C" __global__ void scan_batch(int *input, int *output, const unsigned int batch_size, __local  int *b, __local  int *c);

const std::map<std::string, const void*>& funMap1()
{
    static const std::map<std::string, const void*> data{
        {"absolute_difference", reinterpret_cast<const void*>(absolute_difference)},
        {"absolute_difference_batch", reinterpret_cast<const void*>(absolute_difference_batch)},
        {"accumulate_weighted", reinterpret_cast<const void*>(accumulate_weighted)},
        {"accumulate_weighted_batch", reinterpret_cast<const void*>(accumulate_weighted_batch)},
        {"accumulate", reinterpret_cast<const void*>(accumulate)},
        {"accumulate_batch", reinterpret_cast<const void*>(accumulate_batch)},
        {"add", reinterpret_cast<const void*>(add)},
        {"add_batch", reinterpret_cast<const void*>(add_batch)},
        {"subtract", reinterpret_cast<const void*>(subtract)},
        {"subtract_batch", reinterpret_cast<const void*>(subtract_batch)},
        {"magnitude", reinterpret_cast<const void*>(magnitude)},
        {"magnitude_batch", reinterpret_cast<const void*>(magnitude_batch)},
        {"multiply", reinterpret_cast<const void*>(multiply)},
        {"multiply_batch", reinterpret_cast<const void*>(multiply_batch)},
        {"phase", reinterpret_cast<const void*>(phase)},
        {"phase_batch", reinterpret_cast<const void*>(phase_batch)},
        {"accumulate_squared", reinterpret_cast<const void*>(accumulate_squared)},
        {"accumulate_squared_batch", reinterpret_cast<const void*>(accumulate_squared_batch)},
        {"color_twist_batch", reinterpret_cast<const void*>(color_twist_batch)},
        {"color_temperature_batch", reinterpret_cast<const void*>(color_temperature_batch)},
        {"temperature_packed", reinterpret_cast<const void*>(temperature_packed)},
        {"temperature_planar", reinterpret_cast<const void*>(temperature_planar)},
        {"vignette_batch", reinterpret_cast<const void*>(vignette_batch)},
        {"vignette_pln", reinterpret_cast<const void*>(vignette_pln)},
        {"vignette_pkd", reinterpret_cast<const void*>(vignette_pkd)},
        {"snp_pln", reinterpret_cast<const void*>(snp_pln)},
        {"snp_pkd", reinterpret_cast<const void*>(snp_pkd)},
        {"noise_batch", reinterpret_cast<const void*>(noise_batch)},
        {"channel_extract_pln", reinterpret_cast<const void*>(channel_extract_pln)},
        {"channel_extract_pkd", reinterpret_cast<const void*>(channel_extract_pkd)},
        {"channel_extract_batch", reinterpret_cast<const void*>(channel_extract_batch)},
        {"channel_combine_pln", reinterpret_cast<const void*>(channel_combine_pln)},
        {"channel_combine_pkd", reinterpret_cast<const void*>(channel_combine_pkd)},
        {"channel_combine_batch", reinterpret_cast<const void*>(channel_combine_batch)},
        {"erode_batch", reinterpret_cast<const void*>(erode_batch)},
        {"erode_pln", reinterpret_cast<const void*>(erode_pln)},
        {"erode_pkd", reinterpret_cast<const void*>(erode_pkd)},
        {"dilate_batch", reinterpret_cast<const void*>(dilate_batch)},
        {"dilate_pln", reinterpret_cast<const void*>(dilate_pln)},
        {"dilate_pkd", reinterpret_cast<const void*>(dilate_pkd)},
        {"local_binary_pattern_pkd", reinterpret_cast<const void*>(local_binary_pattern_pkd)},
        {"local_binary_pattern_pln", reinterpret_cast<const void*>(local_binary_pattern_pln)},
        {"local_binary_pattern_batch", reinterpret_cast<const void*>(local_binary_pattern_batch)},
        {"sobel_pkd", reinterpret_cast<const void*>(sobel_pkd)},
        {"sobel_pln", reinterpret_cast<const void*>(sobel_pln)},
        {"sobel_batch", reinterpret_cast<const void*>(sobel_batch)},
        {"custom_convolution_pkd", reinterpret_cast<const void*>(custom_convolution_pkd)},
        {"custom_convolution_pln", reinterpret_cast<const void*>(custom_convolution_pln)},
        {"custom_convolution_batch", reinterpret_cast<const void*>(custom_convolution_batch)},
        {"box_filter_batch", reinterpret_cast<const void*>(box_filter_batch)},
        {"median_filter_pkd", reinterpret_cast<const void*>(median_filter_pkd)},
        {"median_filter_pln", reinterpret_cast<const void*>(median_filter_pln)},
        {"median_filter_batch", reinterpret_cast<const void*>(median_filter_batch)},
        {"non_max_suppression_pkd", reinterpret_cast<const void*>(non_max_suppression_pkd)},
        {"non_max_suppression_pln", reinterpret_cast<const void*>(non_max_suppression_pln)},
        {"non_max_suppression_batch", reinterpret_cast<const void*>(non_max_suppression_batch)},
        {"bilateral_filter_batch", reinterpret_cast<const void*>(bilateral_filter_batch)},
        {"bilateral_filter_packed", reinterpret_cast<const void*>(bilateral_filter_packed)},
        {"bilateral_filter_planar", reinterpret_cast<const void*>(bilateral_filter_planar)},
        {"gaussian_pkd", reinterpret_cast<const void*>(gaussian_pkd)},
        {"gaussian_pln", reinterpret_cast<const void*>(gaussian_pln)},
        {"gaussian_filter_batch", reinterpret_cast<const void*>(gaussian_filter_batch)},
        {"bitwise_AND", reinterpret_cast<const void*>(bitwise_AND)},
        {"bitwise_AND_batch", reinterpret_cast<const void*>(bitwise_AND_batch)},
        {"bitwise_NOT", reinterpret_cast<const void*>(bitwise_NOT)},
        {"bitwise_NOT_batch", reinterpret_cast<const void*>(bitwise_NOT_batch)},
        {"exclusive_OR", reinterpret_cast<const void*>(exclusive_OR)},
        {"exclusive_OR_batch", reinterpret_cast<const void*>(exclusive_OR_batch)},
        {"inclusive_OR", reinterpret_cast<const void*>(inclusive_OR)},
        {"inclusive_OR_batch", reinterpret_cast<const void*>(inclusive_OR_batch)},
        {"brightness", reinterpret_cast<const void*>(brightness)},
        {"brightness_batch",  reinterpret_cast<const void*>(brightness_batch)},
        {"gamma_correction_batch", reinterpret_cast<const void*>(gamma_correction_batch)},
        {"gamma_correction", reinterpret_cast<const void*>(gamma_correction)},
        {"contrast_batch", reinterpret_cast<const void*>(contrast_batch)},
        {"contrast_stretch", reinterpret_cast<const void*>(contrast_stretch)},
        {"exposure_batch", reinterpret_cast<const void*>(exposure_batch)},
        {"exposure", reinterpret_cast<const void*>(exposure)},
        {"jitter_batch", reinterpret_cast<const void*>(jitter_batch)},
        {"jitter_pln", reinterpret_cast<const void*>(jitter_pln)},
        {"jitter_pkd", reinterpret_cast<const void*>(jitter_pkd)},
        {"blend", reinterpret_cast<const void*>(blend)},
        {"blend_batch", reinterpret_cast<const void*>(blend_batch)},
        {"rain_batch", reinterpret_cast<const void*>(rain_batch)},
        {"rain_pkd", reinterpret_cast<const void*>(rain_pkd)},
        {"rain_pln", reinterpret_cast<const void*>(rain_pln)},
        {"fog_batch", reinterpret_cast<const void*>(fog_batch)},
        {"fog_pkd", reinterpret_cast<const void*>(fog_pkd)},
        {"fog_planar", reinterpret_cast<const void*>(fog_planar)},
        {"snow_batch", reinterpret_cast<const void*>(snow_batch)},
        {"snow_pln", reinterpret_cast<const void*>(snow_pln)},
        {"snow_pkd", reinterpret_cast<const void*>(snow_pkd)},
        {"lenscorrection_pln", reinterpret_cast<const void*>(lenscorrection_pln)},
        {"lenscorrection_pkd", reinterpret_cast<const void*>(lenscorrection_pkd)},
        {"lens_correction_batch", reinterpret_cast<const void*>(lens_correction_batch)},
        {"fisheye_packed", reinterpret_cast<const void*>(fisheye_packed)},
        {"fisheye_planar", reinterpret_cast<const void*>(fisheye_planar)},
        {"fisheye_batch", reinterpret_cast<const void*>(fisheye_batch)},
        {"pixelate_pln", reinterpret_cast<const void*>(pixelate_pln)},
        {"pixelate_pkd", reinterpret_cast<const void*>(pixelate_pkd)},
        {"pixelate_batch", reinterpret_cast<const void*>(pixelate_batch)},
        {"thresholding", reinterpret_cast<const void*>(thresholding)},
        {"thresholding_batch", reinterpret_cast<const void*>(thresholding_batch)},
        {"min_hip", reinterpret_cast<const void*>(min_hip)},
        {"min_batch", reinterpret_cast<const void*>(min_batch)},
        {"max_hip", reinterpret_cast<const void*>(max_hip)},
        {"max_batch", reinterpret_cast<const void*>(max_batch)},
        {"random_shadow_packed", reinterpret_cast<const void*>(random_shadow_packed)},
        {"random_shadow_planar", reinterpret_cast<const void*>(random_shadow_planar)},
        {"random_shadow", reinterpret_cast<const void*>(random_shadow)},
        {"histogram_equalize_batch", reinterpret_cast<const void*>(histogram_equalize_batch)},
        {"histogram_equalize_pkd", reinterpret_cast<const void*>(histogram_equalize_pkd)},
        {"histogram_equalize_pln", reinterpret_cast<const void*>(histogram_equalize_pln)},
        {"histogram_sum_partial_batch", reinterpret_cast<const void*>(histogram_sum_partial_batch)},
        {"histogram_sum_partial", reinterpret_cast<const void*>(histogram_sum_partial)},
        {"partial_histogram_batch", reinterpret_cast<const void*>(partial_histogram_batch)},
        {"partial_histogram_semibatch", reinterpret_cast<const void*>(partial_histogram_semibatch)},
        {"partial_histogram_pkd", reinterpret_cast<const void*>(partial_histogram_pkd)},
        {"partial_histogram_pln", reinterpret_cast<const void*>(partial_histogram_pln)},
        {"min_loc", reinterpret_cast<const void*>(min_loc)},
        {"max_loc", reinterpret_cast<const void*>(max_loc)},
        {"integral_pkd_col", reinterpret_cast<const void*>(integral_pkd_col)},
        {"integral_pln_col", reinterpret_cast<const void*>(integral_pln_col)},
        {"integral_pkd_row", reinterpret_cast<const void*>(integral_pkd_row)},
        {"integral_pln_row", reinterpret_cast<const void*>(integral_pln_row)},
        {"integral_up_pln", reinterpret_cast<const void*>(integral_up_pln)},
        {"integral_low_pln", reinterpret_cast<const void*>(integral_low_pln)},
        {"integral_up_pkd", reinterpret_cast<const void*>(integral_up_pkd)},
        {"integral_low_pkd", reinterpret_cast<const void*>(integral_low_pkd)},
        {"sum", reinterpret_cast<const void*>(sum)},
        {"mean_stddev", reinterpret_cast<const void*>(mean_stddev)},
        {"flip_horizontal_planar", reinterpret_cast<const void*>(flip_horizontal_planar)},
        {"flip_vertical_planar", reinterpret_cast<const void*>(flip_vertical_planar)},
        {"flip_bothaxis_planar", reinterpret_cast<const void*>(flip_bothaxis_planar)},
        {"flip_horizontal_packed", reinterpret_cast<const void*>(flip_horizontal_packed)},
        {"flip_vertical_packed", reinterpret_cast<const void*>(flip_vertical_packed)},
        {"flip_bothaxis_packed", reinterpret_cast<const void*>(flip_bothaxis_packed)},
        {"flip_batch", reinterpret_cast<const void*>(flip_batch)},
        {"resize_pln", reinterpret_cast<const void*>(resize_pln)},
        {"resize_pkd", reinterpret_cast<const void*>(resize_pkd)},
        {"resize_batch", reinterpret_cast<const void*>(resize_batch)},
        {"crop_batch", reinterpret_cast<const void*>(crop_batch)},
        {"crop_mirror_normalize_batch", reinterpret_cast<const void*>(crop_mirror_normalize_batch)},
        {"resize_crop_batch", reinterpret_cast<const void*>(resize_crop_batch)},
        {"resize_crop_pln", reinterpret_cast<const void*>(resize_crop_pln)},
        {"resize_crop_pkd", reinterpret_cast<const void*>(resize_crop_pkd)},
        {"huergb_pln", reinterpret_cast<const void*>(huergb_pln)},
        {"huergb_pkd", reinterpret_cast<const void*>(huergb_pkd)},
        {"hue_batch", reinterpret_cast<const void*>(hue_batch)},
        {"saturation_batch", reinterpret_cast<const void*>(saturation_batch)},
        {"rotate_pln", reinterpret_cast<const void*>(rotate_pln)},
        {"rotate_pkd", reinterpret_cast<const void*>(rotate_pkd)},
        {"rotate_batch", reinterpret_cast<const void*>(rotate_batch)},
        {"warp_affine_pln", reinterpret_cast<const void*>(warp_affine_pln)},
        {"warp_affine_pkd", reinterpret_cast<const void*>(warp_affine_pkd)},
        {"scale_pln", reinterpret_cast<const void*>(scale_pln)},
        {"scale_pkd", reinterpret_cast<const void*>(scale_pkd)},
        {"scale_batch", reinterpret_cast<const void*>(scale_batch)},
        {"warp_perspective_batch", reinterpret_cast<const void*>(warp_perspective_batch)},
        {"warp_perspective_pkd", reinterpret_cast<const void*>(warp_perspective_pkd)},
        {"warp_perspective_pln", reinterpret_cast<const void*>(warp_perspective_pln)},
        {"look_up_table_pkd", reinterpret_cast<const void*>(look_up_table_pkd)},
        {"look_up_table_pln", reinterpret_cast<const void*>(look_up_table_pln)},
        {"look_up_table_batch", reinterpret_cast<const void*>(look_up_table_batch)},
        {"naive_convolution_planar", reinterpret_cast<const void*>(naive_convolution_planar)},
        {"naive_convolution_packed", reinterpret_cast<const void*>(naive_convolution_packed)},
        {"gaussian_image_pyramid_pkd", reinterpret_cast<const void*>(gaussian_image_pyramid_pkd)},
        {"gaussian_image_pyramid_pln", reinterpret_cast<const void*>(gaussian_image_pyramid_pln)},
        {"gaussian_image_pyramid_batch", reinterpret_cast<const void*>(gaussian_image_pyramid_batch)},
        {"gaussian_image_pyramid_pkd_batch", reinterpret_cast<const void*>(gaussian_image_pyramid_pkd_batch)},
        {"gaussian_image_pyramid_pln_batch", reinterpret_cast<const void*>(gaussian_image_pyramid_pln_batch)},
        {"laplacian_image_pyramid_pkd_batch", reinterpret_cast<const void*>(laplacian_image_pyramid_pkd_batch)},
        {"laplacian_image_pyramid_pln_batch", reinterpret_cast<const void*>(laplacian_image_pyramid_pln_batch)},
        {"ced_non_max_suppression_batch", reinterpret_cast<const void*>(ced_non_max_suppression_batch)},
        {"ced_non_max_suppression", reinterpret_cast<const void*>(ced_non_max_suppression)},
        {"canny_edge_batch", reinterpret_cast<const void*>(canny_edge_batch)},
        {"ced_pln1_to_pln3_batch", reinterpret_cast<const void*>(ced_pln1_to_pln3_batch)},
        {"ced_pln1_to_pkd3_batch", reinterpret_cast<const void*>(ced_pln1_to_pkd3_batch)},
        {"convert_bit_depth_u8s8", reinterpret_cast<const void*>(convert_bit_depth_u8s8)},
        {"convert_bit_depth_u8u16", reinterpret_cast<const void*>(convert_bit_depth_u8u16)},
        {"convert_bit_depth_u8s16", reinterpret_cast<const void*>(convert_bit_depth_u8s16)},
        {"convert_bit_depth_batch_u8s8", reinterpret_cast<const void*>(convert_bit_depth_batch_u8s8)},
        {"convert_bit_depth_batch_u8u16", reinterpret_cast<const void*>(convert_bit_depth_batch_u8u16)},
        {"convert_bit_depth_batch_u8s16", reinterpret_cast<const void*>(convert_bit_depth_batch_u8s16)},
        {"harris_corner_detector_strength", reinterpret_cast<const void*>(harris_corner_detector_strength)},
        {"harris_corner_detector_nonmax_supression", reinterpret_cast<const void*>(harris_corner_detector_nonmax_supression)},
        {"harris_corner_detector_pln", reinterpret_cast<const void*>(harris_corner_detector_pln)},
        {"harris_corner_detector_pkd", reinterpret_cast<const void*>(harris_corner_detector_pkd)},
        {"fast_corner_detector", reinterpret_cast<const void*>(fast_corner_detector)},
        {"fast_corner_detector_nms_pln", reinterpret_cast<const void*>(fast_corner_detector_nms_pln)},
        {"fast_corner_detector_nms_pkd", reinterpret_cast<const void*>(fast_corner_detector_nms_pkd)},
        {"reconstruction_laplacian_image_pyramid_pkd", reinterpret_cast<const void*>(reconstruction_laplacian_image_pyramid_pkd)},
        {"reconstruction_laplacian_image_pyramid_pln", reinterpret_cast<const void*>(reconstruction_laplacian_image_pyramid_pln)},
        {"tensor_look_up_table", reinterpret_cast<const void*>(tensor_look_up_table)},
        {"tensor_convert_bit_depth_u8s16", reinterpret_cast<const void*>(tensor_convert_bit_depth_u8s16)},
        {"tensor_convert_bit_depth_u8u16", reinterpret_cast<const void*>(tensor_convert_bit_depth_u8u16)},
        {"tensor_convert_bit_depth_u8s8", reinterpret_cast<const void*>(tensor_convert_bit_depth_u8s8)},
        {"tensor_matrix_multiply", reinterpret_cast<const void*>(tensor_matrix_multiply)},
        {"tensor_multiply", reinterpret_cast<const void*>(tensor_multiply)},
        {"tensor_subtract", reinterpret_cast<const void*>(tensor_subtract)},
        {"tensor_add", reinterpret_cast<const void*>(tensor_add)},
        {"scan", reinterpret_cast<const void*>(scan)},
    };
    return data;
}

#endif // HIP_KERNELS_H