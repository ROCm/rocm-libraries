/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_AUGMENTATION_H
#define MIVISIONX_ROCAL_API_AUGMENTATION_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Augmentations.
 *
 * \defgroup group_rocal_augmentations API: AMD rocAL - Augmentation API
 * \brief The AMD rocAL augmentation functions.
 */

/*!
 * \brief Rearranges the order of the frames in the sequences with respect to new_order. new_order can have values in the range [0, sequence_length). Frames can be repeated or dropped in the new_order.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] p_context context for the pipeline.
 * \param [in] p_input Input Rocal Tensor
 * \param [in] new_order represents the new order of the frames in the sequence
 * \param [in] is_output True: the output image is needed by user and will be copied to output buffers using the data transfer API calls. False: the output image is just an intermediate image, user is not interested in using it directly. This option allows certain optimizations to be achieved.
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSequenceRearrange(RocalContext p_context, RocalTensor p_input,
                                                             std::vector<unsigned int> &new_order,
                                                             bool is_output);

/*! \brief Resize images.
 * \note Accepts U8 and RGB24 input.
 * \ingroup group_rocal_augmentations
 * \note: Accepts U8 and RGB24 input.
 * \param [in] context context for the pipeline.
 * \param [in] input Input Rocal Tensor
 * \param [in] dest_width output width
 * \param [in] dest_height ouput Height
 * \param [in] is_output True: the output image is needed by user and will be copied to output buffers using the data transfer API calls. False: the output image is just an intermediate image, user is not interested in using it directly. This option allows certain optimizations to be achieved.
 * \param [in] scaling_mode The resize scaling_mode to resize the image.
 * \param [in] max_size Limits the size of the resized image.
 * \param [in] resize_shorter The length of the shorter dimension of the image.
 * \param [in] resize_longer The length of the larger dimension of the image.
 * \param [in] interpolation_type The type of interpolation to be used for resize.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResize(RocalContext context, RocalTensor input,
                                                  unsigned dest_width, unsigned dest_height,
                                                  bool is_output,
                                                  RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                  std::vector<unsigned> max_size = {},
                                                  unsigned resize_shorter = 0,
                                                  unsigned resize_longer = 0,
                                                  RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                  RocalTensorLayout output_layout = ROCAL_NONE,
                                                  RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Fused function which performs resize, normalize and flip on images.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal Tensor
 * \param [in] dest_width output width
 * \param [in] dest_height output height
 * \param [in] mean The channel mean values
 * \param [in] std_dev The channel standard deviation values
 * \param [in] is_output True: the output image is needed by user and will be copied to output buffers using the data transfer API calls. False: the output image is just an intermediate image, user is not interested in using it directly. This option allows certain optimizations to be achieved.
 * \param [in] scaling_mode The resize scaling_mode to resize the image.
 * \param [in] max_size Limits the size of the resized image.
 * \param [in] resize_shorter The length of the shorter dimension of the image.
 * \param [in] resize_longer The length of the larger dimension of the image.
 * \param [in] interpolation_type The type of interpolation to be used for resize.
 * \param [in] mirror Parameter to enable horizontal flip for output image.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, RocalTensor p_input, unsigned dest_width,
                                                                 unsigned dest_height, std::vector<float> &mean, std::vector<float> &std_dev,
                                                                 bool is_output,
                                                                 RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                                 std::vector<unsigned> max_size = {}, unsigned resize_shorter = 0,
                                                                 unsigned resize_longer = 0,
                                                                 RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                                 RocalIntParam mirror = NULL,
                                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Fused function which perrforms crop and resize on images.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] context Rocal context
 * \param [in] input Input Rocal Tensor
 * \param [in] dest_width output width
 * \param [in] dest_height output height
 * \param [in] is_output True: the output image is needed by user and will be copied to output buffers using the data transfer API calls. False: the output image is just an intermediate image, user is not interested in using it directly. This option allows certain optimizations to be achieved.
 * \param [in] area Target area for the crop
 * \param [in] aspect_ratio specifies the aspect ratio of the cropped region
 * \param [in] x_center_drift Horizontal shift of the crop center from its original position in the input image
 * \param [in] y_center_drift Vertical shift of the crop center from its original position in the input image
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropResize(RocalContext context, RocalTensor input,
                                                      unsigned dest_width, unsigned dest_height,
                                                      bool is_output,
                                                      RocalFloatParam area = NULL,
                                                      RocalFloatParam aspect_ratio = NULL,
                                                      RocalFloatParam x_center_drift = NULL,
                                                      RocalFloatParam y_center_drift = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Fused function which perrforms crop and resize on images with fixed crop coordinates.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] dest_width output width
 * \param [in] dest_height output height
 * \param [in] is_output True: the output image is needed by user and will be copied to output buffers using the data transfer API calls. False: the output image is just an intermediate image, user is not interested in using it directly. This option allows certain optimizations to be achieved.
 * \param [in] area Target area for the crop
 * \param [in] aspect_ratio specifies the aspect ratio of the cropped region
 * \param [in] x_center_drift Horizontal shift of the crop center from its original position in the input image
 * \param [in] y_center_drift Vertical shift of the crop center from its original position in the input image
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropResizeFixed(RocalContext context, RocalTensor input,
                                                           unsigned dest_width, unsigned dest_height,
                                                           bool is_output,
                                                           float area, float aspect_ratio,
                                                           float x_center_drift, float y_center_drift,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Resizes images based on the ROI region passed by the user.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] dest_width output width
 * \param [in] dest_height output height
 * \param [in] roi_h height of the ROI region
 * \param [in] roi_w width of the ROI region
 * \param [in] roi_pos_x specifies a specific horizontal position for the ROI region
 * \param [in] roi_pos_y specifies a specific vertical position for the ROI region
 * \param [in] interpolation_type The type of interpolation to be used for resize.
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalROIResize(RocalContext context, RocalTensor input,
                                                           unsigned dest_width, unsigned dest_height,
                                                           bool is_output,
                                                           unsigned roi_h,
                                                           unsigned roi_w,
                                                           float roi_pos_x = 0.0f,
                                                           float roi_pos_y = 0.0f,
                                                           RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Rotates images.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] context Rocal context
 * \param [in] input Input Rocal Tensor
 * \param [in] is_output True: the output tensor is needed by user and will be copied to output buffers using the data transfer API calls. False: the output tensor is just an intermediate tensor, user is not interested in using it directly. This option allows certain optimizations to be achieved.
 * \param [in] angle Rocal parameter defining the rotation angle value in degrees.
 * \param [in] dest_width output width
 * \param [in] dest_height output height
 * \param [in] interpolation_type The type of interpolation to be used for rotate.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRotate(RocalContext context, RocalTensor input, bool is_output,
                                                  RocalFloatParam angle = NULL, unsigned dest_width = 0,
                                                  unsigned dest_height = 0,
                                                  RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                  RocalTensorLayout output_layout = ROCAL_NONE,
                                                  RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Rotates images with fixed angle value.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal Tensor
 * \param [in] dest_width output width
 * \param [in] dest_height output height
 * \param [in] is_output Is the output tensor part of the graph output
 * \param [in] angle The rotation angle value in degrees.
 * \param [in] interpolation_type The type of interpolation to be used for rotate.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRotateFixed(RocalContext context, RocalTensor input, float angle,
                                                       bool is_output, unsigned dest_width = 0, unsigned dest_height = 0,
                                                       RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                       RocalTensorLayout output_layout = ROCAL_NONE,
                                                       RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts brightness of the image.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] alpha controls contrast of the image
 * \param [in] beta controls brightness of the image
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input, bool is_output,
                                                      RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts brightness of the image with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] alpha controls contrast of the image
 * \param [in] beta controls brightness of the image
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBrightnessFixed(RocalContext context, RocalTensor input,
                                                           float alpha, float beta,
                                                           bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies gamma correction on image.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] gamma gamma value for the image.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalGamma(RocalContext context, RocalTensor input,
                                                 bool is_output,
                                                 RocalFloatParam gamma = NULL,
                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies gamma correction on image with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] gamma gamma value for the image.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalGammaFixed(RocalContext context, RocalTensor input,
                                                      float gamma,
                                                      bool is_output,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts contrast of the image.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] contrast_factor parameter representing the contrast factor for the contrast operation
 * \param [in] contrast_center parameter representing the contrast center for the contrast operation
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalContrast(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam contrast_factor = NULL, RocalFloatParam contrast_center = NULL,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts contrast of the image with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] contrast_factor  parameter representing the contrast factor for the contrast operation
 * \param [in] contrast_center  parameter representing the contrast center for the contrast operation
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalContrastFixed(RocalContext context, RocalTensor input,
                                                         float contrast_factor, float contrast_center,
                                                         bool is_output,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Flip images horizontally and/or vertically based on inputs.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] horizonal_flag  determines whether the input tensor should be flipped horizontally
 * \param [in] vertical_flag  determines whether the input tensor should be flipped vertically
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFlip(RocalContext context, RocalTensor input, bool is_output,
                                                RocalIntParam horizonal_flag = NULL, RocalIntParam vertical_flag = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Flip images horizontally and/or vertically with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] horizonal_flag  determines whether the input tensor should be flipped horizontally
 * \param [in] vertical_flag  determines whether the input tensor should be flipped vertically
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFlipFixed(RocalContext context, RocalTensor input,
                                                     int horizonal_flag, int vertical_flag, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies blur effect to images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] kernel_size size ofthr kernel used for blurring
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlur(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalIntParam kernel_size = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies blur effect to images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] kernel_size size of the kernel used for blurring
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlurFixed(RocalContext context, RocalTensor input,
                                                     int kernel_size, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input1 Input1 Rocal tensor
 * \param [in] input2 Input2 Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] ratio Rocal parameter defining the blending ratio, should be between 0.0 and 1.0
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlend(RocalContext context, RocalTensor input1, RocalTensor input2,
                                                 bool is_output,
                                                 RocalFloatParam ratio = NULL,
                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Blends two input images given the fixed ratio: output = input1*ratio + input2*(1-ratio)
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input1 Input1 Rocal tensor
 * \param [in] input2 Input2 Rocal tensor
 * \param [in] ratio Float value defining the blending ratio, should be between 0.0 and 1.0.
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalBlendFixed(RocalContext context, RocalTensor input1, RocalTensor input2,
                                                      float ratio, bool is_output,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies affine transformation to images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] x0 float parameter representing the coefficient of affine tensor matrix
 * \param [in] x1 float parameter representing the coefficient of affine tensor matrix
 * \param [in] y0 float parameter representing the coefficient of affine tensor matrix
 * \param [in] y1 float parameter representing the coefficient of affine tensor matrix
 * \param [in] o0 float parameter representing the coefficient of affine tensor matrix
 * \param [in] o1 float parameter representing the coefficient of affine tensor matrix
 * \param [in] dest_height output height
 * \param [in] dest_width output width
 * \param [in] interpolation_type The type of interpolation to be used for warp affine.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalWarpAffine(RocalContext context, RocalTensor input, bool is_output,
                                                      unsigned dest_height = 0, unsigned dest_width = 0,
                                                      RocalFloatParam x0 = NULL, RocalFloatParam x1 = NULL,
                                                      RocalFloatParam y0 = NULL, RocalFloatParam y1 = NULL,
                                                      RocalFloatParam o0 = NULL, RocalFloatParam o1 = NULL,
                                                      RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies affine transformation to images with fixed affine matrix.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] x0 float parameter representing the coefficient of affine tensor matrix
 * \param [in] x1 float parameter representing the coefficient of affine tensor matrix
 * \param [in] y0 float parameter representing the coefficient of affine tensor matrix
 * \param [in] y1 float parameter representing the coefficient of affine tensor matrix
 * \param [in] o0 float parameter representing the coefficient of affine tensor matrix
 * \param [in] o1 float parameter representing the coefficient of affine tensor matrix
 * \param [in] dest_height output height
 * \param [in] dest_width output width
 * \param [in] interpolation_type The type of interpolation to be used for warp affine.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalWarpAffineFixed(RocalContext context, RocalTensor input, float x0, float x1,
                                                           float y0, float y1, float o0, float o1, bool is_output,
                                                           unsigned int dest_height = 0, unsigned int dest_width = 0,
                                                           RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies fish eye effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFishEye(RocalContext context, RocalTensor input, bool is_output,
                                                   RocalTensorLayout output_layout = ROCAL_NONE,
                                                   RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies vignette effect on images.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] sdev standard deviation for the vignette effect
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVignette(RocalContext context, RocalTensor input,
                                                    bool is_output, RocalFloatParam sdev = NULL,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies vignette effect on images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \note Accepts U8 and RGB24 input.
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] sdev standard deviation for the vignette effect
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVignetteFixed(RocalContext context, RocalTensor input,
                                                         float sdev, bool is_output,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies jitter effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] kernel_size kernel size used for the jitter effect
 * \param [in] seed seed value for the random number generator
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJitter(RocalContext context, RocalTensor input,
                                                  bool is_output,
                                                  RocalIntParam kernel_size = NULL,
                                                  int seed = 0,
                                                  RocalTensorLayout output_layout = ROCAL_NONE,
                                                  RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies jitter effect on images with fixed kernel size.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] kernel_size kernel size used for the jitter effect
 * \param [in] seed seed value for the random number generator
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJitterFixed(RocalContext context, RocalTensor input,
                                                       int kernel_size, bool is_output, int seed = 0,
                                                       RocalTensorLayout output_layout = ROCAL_NONE,
                                                       RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies salt and pepper noise effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] noise_prob probability of applying the Salt and Pepper noise.
 * \param [in] salt_prob probability of applying salt noise
 * \param [in] salt_val specifies the value of the salt noise
 * \param [in] pepper_val specifies the value of the pepper noise
 * \param [in] seed seed value for the random number generator
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnPNoise(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam noise_prob = NULL, RocalFloatParam salt_prob = NULL,
                                                    RocalFloatParam salt_val = NULL, RocalFloatParam pepper_val = NULL,
                                                    int seed = 0,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies salt and pepper noise on images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] noise_prob probability of applying the Salt and Pepper noise.
 * \param [in] salt_prob probability of applying salt noise
 * \param [in] salt_val specifies the value of the salt noise
 * \param [in] pepper_val specifies the value of the pepper noise
 * \param [in] seed seed value for the random number generator
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnPNoiseFixed(RocalContext context, RocalTensor input,
                                                         float noise_prob, float salt_prob,
                                                         float salt_val, float pepper_val,
                                                         bool is_output, int seed = 0,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies snow effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] snow Float param representing the intensity of snow effect
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnow(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam snow = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies snow effect on images with fixed parameter.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] snow Float param representing the intensity of snow effect
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSnowFixed(RocalContext context, RocalTensor input,
                                                     float snow, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies rain effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] rain_value parameter represents the intensity of rain effect
 * \param [in] rain_width parameter represents the width of the rain effect
 * \param [in] rain_height parameter represents the width of the rain effect
 * \param [in] rain_transparency parameter represents the transperancy of the rain effect
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRain(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam rain_value = NULL,
                                                RocalIntParam rain_width = NULL,
                                                RocalIntParam rain_height = NULL,
                                                RocalFloatParam rain_transparency = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies rain effect on images with fixed parameter.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] rain_value parameter represents the intensity of rain effect
 * \param [in] rain_width parameter represents the width of the rain effect
 * \param [in] rain_height parameter represents the width of the rain effect
 * \param [in] rain_transparency parameter represents the transperancy of the rain effect
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRainFixed(RocalContext context, RocalTensor input,
                                                     float rain_value,
                                                     int rain_width,
                                                     int rain_height,
                                                     float rain_transparency,
                                                     bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the color temperature in images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] adjustment color temperature adjustment value
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTemp(RocalContext context, RocalTensor input,
                                                     bool is_output,
                                                     RocalIntParam adjustment = NULL,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the color temperature in images with fixed value.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] adjustment color temperature adjustment value
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTempFixed(RocalContext context, RocalTensor input,
                                                          int adjustment, bool is_output,
                                                          RocalTensorLayout output_layout = ROCAL_NONE,
                                                          RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies fog effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] fog_value parameter representing the intensity of fog effect
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFog(RocalContext context, RocalTensor input,
                                               bool is_output,
                                               RocalFloatParam fog_value = NULL,
                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies fog effect on images with fixed parameter.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] fog_value parameter representing the intensity of fog effect
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFogFixed(RocalContext context, RocalTensor input,
                                                    float fog_value, bool is_output,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies lens correction effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] strength parameter representing the strength of the lens correction.
 * \param [in] zoom parameter representing the zoom factor of the lens correction.
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalLensCorrection(RocalContext context, RocalTensor input, bool is_output,
                                                          RocalFloatParam strength = NULL,
                                                          RocalFloatParam zoom = NULL,
                                                          RocalTensorLayout output_layout = ROCAL_NONE,
                                                          RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies lens correction effect on images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] strength parameter representing the strength of the lens correction.
 * \param [in] zoom parameter representing the zoom factor of the lens correction.
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalLensCorrectionFixed(RocalContext context, RocalTensor input,
                                                               float strength, float zoom, bool is_output,
                                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies pixelate effect on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalPixelate(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the exposure in images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] exposure_factor exposure adjustment factor
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalExposure(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam exposure_factor = NULL,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the exposure in images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] exposure_factor exposure adjustment factor
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalExposureFixed(RocalContext context, RocalTensor input,
                                                         float exposure_factor, bool is_output,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the hue in images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] hue hue adjustment value in degrees
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalHue(RocalContext context, RocalTensor input,
                                               bool is_output,
                                               RocalFloatParam hue = NULL,
                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the hue in images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] hue hue adjustment value in degrees
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalHueFixed(RocalContext context, RocalTensor input,
                                                    float hue,
                                                    bool is_output,
                                                    RocalTensorLayout output_layout = ROCAL_NONE,
                                                    RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the saturation in images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] saturation saturation adjustment value
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSaturation(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam saturation = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the saturation in images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] saturation saturation adjustment value
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSaturationFixed(RocalContext context, RocalTensor input,
                                                           float saturation, bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Copies input tensor to output tensor.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCopy(RocalContext context, RocalTensor input, bool is_output);

/*! \brief Performs no operation.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalNop(RocalContext context, RocalTensor input, bool is_output);

/*! \brief Adjusts the brightness, hue and saturation of the images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] alpha parameter that controls the brightness of an image
 * \param [in] beta parameter that helps in tuning the color balance of an image
 * \param [in] hue parameter that adjusts the hue of an image
 * \param [in] sat parameter that controls the intensity of colors
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTwist(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalFloatParam beta = NULL,
                                                      RocalFloatParam hue = NULL,
                                                      RocalFloatParam sat = NULL,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Adjusts the brightness, hue and saturation of the images with fixed parameters.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] alpha parameter that controls the brightness of an image
 * \param [in] beta parameter that helps in tuning the color balance of an image
 * \param [in] hue parameter that adjusts the hue of an image
 * \param [in] sat parameter that controls the intensity of colors
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalColorTwistFixed(RocalContext context, RocalTensor input,
                                                           float alpha,
                                                           float beta,
                                                           float hue,
                                                           float sat,
                                                           bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Fused function which performs crop, normalize and flip on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] crop_height crop width of the tensor
 * \param [in] crop_width crop height of the tensor
 * \param [in] start_x x-coordinate, start of the input tensor to be cropped
 * \param [in] start_y y-coordinate, start of the input tensor to be cropped
 * \param [in] mean mean value (specified for each channel) for tensor normalization
 * \param [in] std_dev standard deviation value (specified for each channel) for tensor normalization
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] mirror controls horizontal flip of the tensor
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext context, RocalTensor input,
                                                               unsigned crop_height,
                                                               unsigned crop_width,
                                                               float start_x,
                                                               float start_y,
                                                               std::vector<float> &mean,
                                                               std::vector<float> &std_dev,
                                                               bool is_output,
                                                               RocalIntParam mirror = NULL,
                                                               RocalTensorLayout output_layout = ROCAL_NONE,
                                                               RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Crops images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] crop_height crop width of the tensor
 * \param [in] crop_width crop height of the tensor
 * \param [in] crop_depth crop depth of the tensor
 * \param [in] crop_pox_x x-coordinate, start of the input tensor to be cropped
 * \param [in] crop_pos_y y-coordinate, start of the input tensor to be cropped
 * \param [in] crop_pos_z z-coordinate, start of the input tensor to be cropped
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCrop(RocalContext context, RocalTensor input, bool is_output,
                                                RocalFloatParam crop_width = NULL,
                                                RocalFloatParam crop_height = NULL,
                                                RocalFloatParam crop_depth = NULL,
                                                RocalFloatParam crop_pox_x = NULL,
                                                RocalFloatParam crop_pos_y = NULL,
                                                RocalFloatParam crop_pos_z = NULL,
                                                RocalTensorLayout output_layout = ROCAL_NONE,
                                                RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Crops images with fixed coordinates.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] crop_height crop width of the tensor
 * \param [in] crop_width crop height of the tensor
 * \param [in] crop_depth crop depth of the tensor
 * \param [in] crop_pox_x x-coordinate, start of the input tensor to be cropped
 * \param [in] crop_pos_y y-coordinate, start of the input tensor to be cropped
 * \param [in] crop_pos_z z-coordinate, start of the input tensor to be cropped
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropFixed(RocalContext context, RocalTensor input,
                                                     unsigned crop_width,
                                                     unsigned crop_height,
                                                     unsigned crop_depth,
                                                     bool is_output,
                                                     float crop_pox_x,
                                                     float crop_pos_y,
                                                     float crop_pos_z,
                                                     RocalTensorLayout output_layout = ROCAL_NONE,
                                                     RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Crops images at the center with fixed coordinates.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] crop_height crop width of the tensor
 * \param [in] crop_width crop height of the tensor
 * \param [in] crop_depth crop depth of the tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalCropCenterFixed(RocalContext context, RocalTensor input,
                                                           unsigned crop_width,
                                                           unsigned crop_height,
                                                           unsigned crop_depth,
                                                           bool is_output,
                                                           RocalTensorLayout output_layout = ROCAL_NONE,
                                                           RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Fused function which performs resize, crop and flip on images with fixed crop.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] dest_height output height
 * \param [in] dest_width output width
 * \param [in] crop_h crop width of the tensor
 * \param [in] crop_w crop height of the tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] mirror controls horizontal flip of the tensor
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResizeCropMirrorFixed(RocalContext context, RocalTensor input,
                                                                 unsigned dest_width, unsigned dest_height,
                                                                 bool is_output,
                                                                 unsigned crop_h,
                                                                 unsigned crop_w,
                                                                 RocalIntParam mirror,
                                                                 RocalTensorLayout output_layout = ROCAL_NONE,
                                                                 RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Fused function which performs resize, crop and flip on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] dest_height output height
 * \param [in] dest_width output width
 * \param [in] crop_height crop width of the tensor
 * \param [in] crop_width crop height of the tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] mirror controls horizontal flip of the tensor
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResizeCropMirror(RocalContext context, RocalTensor input,
                                                            unsigned dest_width, unsigned dest_height,
                                                            bool is_output, RocalFloatParam crop_height = NULL,
                                                            RocalFloatParam crop_width = NULL, RocalIntParam mirror = NULL,
                                                            RocalTensorLayout output_layout = ROCAL_NONE,
                                                            RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Crops images randomly.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] num_of_attempts maximum number of attempts the function will make to find a valid crop
 * \param [in] crop_area_factor specifies the proportion of the input image to be included in the cropped region
 * \param [in] crop_aspect_ratio specifies the aspect ratio of the cropped region
 * \param [in] crop_pos_x specifies a specific horizontal position for the crop
 * \param [in] crop_pos_y specifies a specific vertical position for the crop
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRandomCrop(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam crop_area_factor = NULL,
                                                      RocalFloatParam crop_aspect_ratio = NULL,
                                                      RocalFloatParam crop_pos_x = NULL,
                                                      RocalFloatParam crop_pos_y = NULL,
                                                      int num_of_attempts = 20,
                                                      RocalTensorLayout output_layout = ROCAL_NONE,
                                                      RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Crops images randomly used for SSD training.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] threshold the threshold parameter for crop operation
 * \param [in] crop_area_factor specifies the proportion of the input image to be included in the cropped region
 * \param [in] crop_aspect_ratio specifies the aspect ratio of the cropped region
 * \param [in] crop_pos_x specifies a specific horizontal position for the crop
 * \param [in] crop_pos_y specifies a specific vertical position for the crop
 * \param [in] num_of_attempts he maximum number of attempts the function will make to find a valid crop
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSSDRandomCrop(RocalContext context, RocalTensor input,
                                                         bool is_output,
                                                         RocalFloatParam threshold = NULL,
                                                         RocalFloatParam crop_area_factor = NULL,
                                                         RocalFloatParam crop_aspect_ratio = NULL,
                                                         RocalFloatParam crop_pos_x = NULL,
                                                         RocalFloatParam crop_pos_y = NULL,
                                                         int num_of_attempts = 20,
                                                         RocalTensorLayout output_layout = ROCAL_NONE,
                                                         RocalTensorOutputType output_datatype = ROCAL_UINT8);

/*! \brief Applies preemphasis filter to the input data.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output Sets to True if the output tensor is part of the graph output
 * \param [in] preemph_coeff Preemphasis coefficient
 * \param [in] preemph_border_type Border value policy. Possible values are "zero", "clamp", "reflect".
 * \param [in] output_datatype The data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalPreEmphasisFilter(RocalContext context,
                                                             RocalTensor input,
                                                             bool is_output,
                                                             RocalFloatParam preemph_coeff = NULL,
                                                             RocalAudioBorderType preemph_border_type = RocalAudioBorderType::ROCAL_CLAMP,
                                                             RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief Produces a spectrogram from a 1D audio signal.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] window_fn values of the window function
 * \param [in] center_windows boolean value to specify whether extracted windows should be padded so that the window function is centered at multiples of window_step
 * \param [in] reflect_padding Indicates the padding policy when sampling outside the bounds of the audio data
 * \param [in] spectrogram_layout output spectrogram layout
 * \param [in] power Exponent of the magnitude of the spectrum
 * \param [in] nfft Size of the Fast Fourier transform (FFT)
 * \param [in] window_length Window size in the number of samples
 * \param [in] window_step Step between the Short-time Fourier transform (STFT) windows in number of samples
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSpectrogram(RocalContext context,
                                                       RocalTensor input,
                                                       bool is_output,
                                                       std::vector<float> &window_fn,
                                                       bool center_windows,
                                                       bool reflect_padding,
                                                       int power,
                                                       int nfft,
                                                       int window_length = 512,
                                                       int window_step = 256,
                                                       RocalTensorLayout output_layout = ROCAL_NFT,
                                                       RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief A
 * \ingroup group_rocal_augmentations
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param[in] cutoff_db minimum or cut-off ratio in dB
 * \param[in] multiplier factor by which the logarithm is multiplied
 * \param[in] reference_magnitude Reference magnitude which if not provided uses maximum value of input as reference
 * \param [in] rocal_tensor_output_type the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalToDecibels(RocalContext p_context,
                                                      RocalTensor p_input,
                                                      bool is_output,
                                                      float cutoff_db,
                                                      float multiplier,
                                                      float reference_magnitude,
                                                      RocalTensorOutputType rocal_tensor_output_type);

/*! \brief Applies resample augmentation to input tensors
 * \ingroup group_rocal_augmentations
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal tensor
 * \param [in] p_output_resample_rate the output resample rate for a batch of audio samples
 * \param [in] is_output Is the output tensor part of the graph output
 * \param [in] sample_hint sample_hint value is the value required to allocate the max memory for output tensor wrt resample_rate and the samples
 * \param [in] quality The resampling is achieved by applying a sinc filter with Hann window with an extent controlled by the quality argument
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalResample(RocalContext p_context,
                                                    RocalTensor p_input,
                                                    RocalTensor p_output_resample_rate,
                                                    bool is_output,
                                                    float sample_hint,
                                                    float quality = 50.0,
                                                    RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief Creates and returns rocALTensor generated from an uniform distribution
 * \ingroup group_rocal_augmentations
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal tensor
 * \param [in] is_output Is the output tensor part of the graph output
 * \param [in] range The range for generating uniform distribution
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalUniformDistribution(RocalContext p_context,
                                                               RocalTensor p_input,
                                                               bool is_output,
                                                               std::vector<float> &range);

/*! \brief Creates and returns rocALTensor generated from an normal distribution
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal tensor
 * \param [in] is_output Is the output tensor part of the graph output
 * \param [in] mean The mean value for generating the normal distribution
 * \param [in] stddev The stddev value for generating the normal distribution
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalNormalDistribution(RocalContext p_context,
                                                              RocalTensor p_input,
                                                              bool is_output,
                                                              float mean = 0.0,
                                                              float stddev = 0.0);

/*! \brief Multiples a tensor and a scalar and returns the output
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal tensor
 * \param [in] is_output Is the output tensor part of the graph output
 * \param [in] scalar The scalar value to be multiplied with the input tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalTensorMulScalar(RocalContext p_context,
                                                           RocalTensor p_input,
                                                           bool is_output,
                                                           float scalar = 0.0,
                                                           RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief Adds two tensors and returns the output.
 * \param [in] p_context Rocal context
 * \param [in] p_input1 Input Rocal tensor1
 * \param [in] p_input2 Input Rocal tensor2
 * \param [in] is_output Is the output tensor part of the graph output
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalTensorAddTensor(RocalContext p_context,
                                                           RocalTensor p_input1,
                                                           RocalTensor p_input2,
                                                           bool is_output,
                                                           RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief Performs silence detection in the input audio tensor
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] cutoff_db threshold(dB) below which the signal is considered silent
 * \param [in] reference_power reference power that is used to convert the signal to dB
 * \param [in] reset_interval number of samples after which the moving mean average is recalculated to avoid loss of precision
 * \param [in] window_length size of the sliding window used to calculate of the short-term power of the signal
 * \return RocalNSROutput
 */
extern "C" RocalNSROutput ROCAL_API_CALL rocalNonSilentRegionDetection(RocalContext context,
                                                                       RocalTensor input,
                                                                       bool is_output,
                                                                       float cutoff_db,
                                                                       float reference_power,
                                                                       int reset_interval,
                                                                       int window_length);

/*! \brief Extracts the sub-tensor from a given input tensor
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] anchor anchor values used for specifying the starting indices of slice
 * \param [in] shape shape values used for specifying the length of slice
 * \param [in] fill_values fill values based on out of Bound policy
 * \param [in] policy
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSlice(RocalContext context,
                                                 RocalTensor input,
                                                 bool is_output,
                                                 RocalTensor anchor,
                                                 RocalTensor shape,
                                                 std::vector<float> fill_values,
                                                 RocalOutOfBoundsPolicy policy = RocalOutOfBoundsPolicy::ROCAL_ERROR,
                                                 RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief Performs mean-stddev normalization on images.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] axes axes list for tensor normalization
 * \param [in] mean mean value (specified for each channel) for tensor normalization
 * \param [in] std_dev standard deviation value (specified for each channel) for tensor normalization
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] scale scale value (specified for each channel) for tensor normalization
 * \param [in] shift shift value (specified for each channel) for tensor normalization
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalNormalize(RocalContext context, RocalTensor input,
                                                     std::vector<unsigned> &axes,
                                                     std::vector<float> &mean,
                                                     std::vector<float> &std_dev,
                                                     bool is_output,
                                                     float scale = 1.0, float shift = 0.0,
                                                     RocalTensorOutputType output_datatype = ROCAL_FP32);

/*! \brief Applies mel-filter bank augmentation on the given input tensor
 * \ingroup group_rocal_augmentations
 * \param [in] p_context Rocal context
 * \param [in] p_input Input Rocal tensor
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] freq_high maximum frequency
 * \param [in] freq_low minimum frequency
 * \param [in] mel_formula formula used to convert frequencies from hertz to mel and from mel to hertz
 * \param [in] nfilter number of mel filters
 * \param [in] normalize boolean variable that determine whether to normalize weights / not
 * \param [in] sample_rate sampling rate of the audio data
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */

extern "C" RocalTensor ROCAL_API_CALL rocalMelFilterBank(RocalContext p_context,
                                                         RocalTensor p_input,
                                                         bool is_output,
                                                         float freq_high,
                                                         float freq_low,
                                                         RocalMelScaleFormula mel_formula,
                                                         int nfilter,
                                                         bool normalize,
                                                         float sample_rate,
                                                         RocalTensorOutputType output_datatype);

/*! \brief Transposes the tensors by reordering the dimensions based on the perm parameter.
 * \ingroup group_rocal_augmentations
 * \param [in] context Rocal context
 * \param [in] input Input Rocal tensor
 * \param [in] perm Permutation of the dimensions of the input
 * \param [in] is_output is the output tensor part of the graph output
 * \param [in] output_layout the layout of the output tensor
 * \param [in] output_datatype the data type of the output tensor
 * \return RocalTensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalTranspose(RocalContext context, RocalTensor input, std::vector<unsigned> perm, bool is_output,
                                                     RocalTensorLayout output_layout = ROCAL_NONE);

#endif  // MIVISIONX_ROCAL_API_AUGMENTATION_H
