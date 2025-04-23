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

#ifndef HIP_DECLARATIONS_H
#define HIP_DECLARATIONS_H

#include "rpp.h"
#include "handle.hpp"
#include "rpp_hip_load_store.hpp"

// ===== Utils

void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width);
void get_kernel_name(std::string &kernel_name, const RPPTensorFunctionMetaData &tensor_info);

//===== Internal HIP functions

/******************** advanced_augmentations ********************/

RppStatus
non_linear_blend_hip_batch_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
non_linear_blend_hip_batch_tensor_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
non_linear_blend_hip_batch_tensor_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
non_linear_blend_hip_batch_tensor_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
water_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
water_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
water_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
water_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
erase_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32u* anchor_box_info, Rpp8u* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
erase_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, Rpp32u* anchor_box_info, Rpp16f* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
erase_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, Rpp32u* anchor_box_info, Rpp32f* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
erase_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp32u* anchor_box_info, Rpp8s* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_cast_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_cast_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_cast_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_cast_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
lut_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp8u *lut,rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
lut_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp8s *lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_and_patch_hip_batch_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_and_patch_hip_batch_tensor_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_and_patch_hip_batch_tensor_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_and_patch_hip_batch_tensor_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
glitch_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
glitch_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
glitch_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
glitch_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);

/******************** image_augmentations ********************/

RppStatus
brightness_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f alpha, Rpp32s beta, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
brightness_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
gamma_correction_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float gamma, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
gamma_correction_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
contrast_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u newMin, Rpp32u newMax, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
contrast_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
pixelate_hip(Rpp8u* srcPtr, RppiSize srcSize,Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel,rpp::Handle& handle);
RppStatus
pixelate_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
exposure_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f exposureValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
exposure_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
jitter_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, unsigned int kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
jitter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
noise_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f noiseProbability, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
noise_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
blend_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, float alpha, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
blend_hip_batch(Rpp8u* srcPtr1, Rpp8u* srcPtr2,Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
random_crop_letterbox_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
rain_hip(Rpp8u* srcPtr, RppiSize srcSize,Rpp8u* dstPtr, Rpp32f rainPercentage, Rpp32u rainWidth, Rpp32u rainHeight, Rpp32f transparency, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
rain_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
snow_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float snowCoefficient, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
snow_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
fog_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *temp, Rpp32f fogValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
fog_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
random_shadow_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
random_shadow_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
histogram_balance_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
histogram_balance_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** fused_functions ********************/

RppStatus
color_twist_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, float alpha, float beta, float hue_shift, float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
color_twist_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
color_twist_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_twist_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_twist_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
color_twist_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_mirror_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_mirror_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_mirror_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_mirror_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize, Rpp32u crop_pox_x, Rpp32u crop_pos_y,
                            Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
crop_mirror_normalize_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
crop_mirror_normalize_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);

/******************** arithmetic_operations ********************/

RppStatus
absolute_difference_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
absolute_difference_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
accumulate_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
accumulate_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
accumulate_weighted_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp64f alpha, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
accumulate_weighted_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
add_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
add_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
subtract_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
subtract_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
magnitude_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
magnitude_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
phase_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
phase_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
multiply_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
multiply_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
accumulate_squared_hip(Rpp8u* srcPtr, RppiSize srcSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
accumulate_squared_hip_batch(Rpp8u* srcPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** logical_operations ********************/

RppStatus
bitwise_AND_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
bitwise_AND_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
bitwise_NOT_hip(Rpp8u* srcPtr1, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
bitwise_NOT_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
exclusive_OR_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
exclusive_OR_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
inclusive_OR_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
inclusive_OR_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** color_model_conversions ********************/

RppStatus
color_temperature_hip(Rpp8u* srcPtr1, RppiSize srcSize, Rpp8u* dstPtr, int adjustmentValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
color_temperature_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
vignette_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, float stdDev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
vignette_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
hueRGB_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float hue_factor, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
hueRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
saturationRGB_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float saturation_factor, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
saturationRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
color_convert_hip_batch_u8_fp32(Rpp8u* srcPtr, Rpp32f* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
color_convert_hip_batch_fp32_u8(Rpp32f* srcPtr, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
channel_extract_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u extractChannelNumber, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
channel_extract_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
channel_combine_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* srcPtr3, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
channel_combine_hip_batch(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* srcPtr3, Rpp8u* dstPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
look_up_table_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,Rpp8u* lutPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
look_up_table_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lutPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** filter_functions ********************/

RppStatus
sobel_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u sobelType, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
sobel_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
box_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
box_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
median_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
median_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
non_max_suppression_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
non_max_suppression_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
bilateral_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, unsigned int filterSize, double sigmaI, double sigmaS, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
bilateral_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
gaussian_filter_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
gaussian_filter_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle&handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
custom_convolution_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32f *kernel, RppiSize KernelSize, rpp::Handle& handle,RppiChnFormat chnFormat, unsigned int channel);

/******************** morphological_transforms ********************/

RppStatus
dilate_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
dilate_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
erode_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
erode_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** geometry_transforms ********************/

RppStatus
flip_hip(Rpp8u *srcPtr, RppiSize srcSize, Rpp8u *dstPtr, uint flipAxis, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
flip_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
resize_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
resize_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
resize_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u x2, Rpp32u y1, Rpp32u y2, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
resize_crop_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
resize_crop_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
resize_crop_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
rotate_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize, float angleDeg, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
rotate_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
rotate_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
rotate_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
rotate_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
rotate_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info);
RppStatus
warp_affine_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize, float *affine, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
warp_affine_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle, Rpp32f *affine, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
warp_affine_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info);
RppStatus
warp_affine_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info);
RppStatus
warp_affine_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info);
RppStatus
warp_affine_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info);
RppStatus
warp_perspective_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize,float *perspective, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
warp_perspective_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr,rpp::Handle& handle, Rpp32f *perspective,RppiChnFormat chnFormat, unsigned int channel);
RppStatus
scale_hip(Rpp8u * srcPtr, RppiSize srcSize, Rpp8u * dstPtr, RppiSize dstSize, Rpp32f percentage, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
scale_hip_batch(Rpp8u * srcPtr, Rpp8u * dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
fisheye_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
fisheye_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
lens_correction_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float strength,float zoom, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
lens_correction_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle&handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** statistical_operations ********************/

RppStatus
thresholding_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp8u min, Rpp8u max, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
thresholding_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
min_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
min_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
max_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
max_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
min_max_loc_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32u* minLoc, Rpp32u* maxLoc, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
min_max_loc_hip_batch(Rpp8u* srcPtr, Rpp8u *min, Rpp8u *max, Rpp32u *minLoc, Rpp32u *maxLoc, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
integral_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp32u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
integral_hip_batch(Rpp8u* srcPtr, Rpp32u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
mean_stddev_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
mean_stddev_hip_batch(Rpp8u* srcPtr, Rpp32f *mean, Rpp32f *stddev, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** computer_vision ********************/

RppStatus
data_object_copy_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
data_object_copy_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
local_binary_pattern_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
local_binary_pattern_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
gaussian_image_pyramid_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
gaussian_image_pyramid_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
control_flow_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
control_flow_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, Rpp32u type, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
laplacian_image_pyramid_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
laplacian_image_pyramid_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
canny_edge_detector_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp8u minThreshold, Rpp8u maxThreshold, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
canny_edge_detector_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
harris_corner_detector_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u gaussianKernelSize, Rpp32f stdDev, Rpp32u kernelSize, Rpp32f kValue, Rpp32f threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
harris_corner_detector_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
tensor_transpose_hip_u8(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32u* in_dims, Rpp32u *perm, rpp::Handle& handle);
RppStatus
tensor_transpose_hip_fp16(Rpp16f* srcPtr, Rpp16f* dstPtr, Rpp32u* in_dims, Rpp32u *perm, rpp::Handle& handle);
RppStatus
tensor_transpose_hip_fp32(Rpp32f* srcPtr, Rpp32f* dstPtr, Rpp32u* in_dims, Rpp32u *perm, rpp::Handle& handle);
RppStatus
tensor_transpose_hip_i8(Rpp8s* srcPtr, Rpp8s* dstPtr, Rpp32u* in_dims, Rpp32u *perm, rpp::Handle& handle);
RppStatus
fast_corner_detector_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u numOfPixels, Rpp8u threshold, Rpp32u nonmaxKernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
fast_corner_detector_hip_batch( Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
reconstruction_laplacian_image_pyramid_hip(Rpp8u* srcPtr1, RppiSize srcSize1, Rpp8u* srcPtr2, RppiSize srcSize2, Rpp8u* dstPtr,
                                            Rpp32f stdDev, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
reconstruction_laplacian_image_pyramid_hip_batch( Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
RppStatus
remap_hip(Rpp8u *srcPtr, RppiSize srcSize,Rpp8u *dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
RppStatus
remap_hip_batch(Rpp8u *srcPtr, Rpp8u* dstPtr, Rpp32u* rowRemapTable, Rpp32u* colRemapTable, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
template <typename T, typename U>
RppStatus
convert_bit_depth_hip(T* srcPtr, RppiSize srcSize, U* dstPtr, Rpp32u type, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle);
template <typename T, typename U>
RppStatus
convert_bit_depth_hip_batch(T* srcPtr, U* dstPtr, Rpp32u type,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

/******************** tensor_functions ********************/

RppStatus
tensor_add_hip(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle);
RppStatus
tensor_subtract_hip(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle);
RppStatus
tensor_multiply_hip(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle);
RppStatus
tensor_matrix_multiply_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp32u *tensorDimensionValues1, Rpp32u *tensorDimensionValues2, Rpp8u* dstPtr, rpp::Handle &handle);
template <typename T, typename U>
RppStatus
tensor_convert_bit_depth_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, T* srcPtr, U* dstPtr, Rpp32u type, rpp::Handle& handle);
RppStatus
tensor_look_up_table_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lutPtr, rpp::Handle& handle);

#endif // HIP_DECLARATIONS_H