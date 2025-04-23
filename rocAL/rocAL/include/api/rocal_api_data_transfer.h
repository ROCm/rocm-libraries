/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MIVISIONX_ROCAL_API_DATA_TRANSFER_H
#define MIVISIONX_ROCAL_API_DATA_TRANSFER_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Data Transfer
 *
 * \defgroup group_rocal_data_transfer API: AMD rocAL - Data Transfer API
 * \brief The AMD rocAL data transfer functions.
 */

/*!
 * \brief copies data to output buffer
 * \ingroup group_rocal_data_transfer
 * \param [in] context Rocal context
 * \param [in] out_ptr pointer to output buffer
 * \param [in] out_size size of output buffer
 * \return Rocal status indicating success or failure
 */
extern "C" RocalStatus ROCAL_API_CALL rocalCopyToOutput(RocalContext context, unsigned char *out_ptr, size_t out_size);

/*!
 * \brief converts data to a tensor
 * \ingroup group_rocal_data_transfer
 * \param [in] rocal_context Rocal context
 * \param [in] out_ptr pointer to output buffer
 * \param [in] tensor_format the layout of the tensor data
 * \param [in] tensor_output_type the output type of the tensor data
 * \param [in] multiplier0 the multiplier for channel 0
 * \param [in] multiplier1 the multiplier for channel 1
 * \param [in] multiplier2 the multiplier for channel 2
 * \param [in] offset0 the offset for channel 0
 * \param [in] offset1 the offset for channel 1
 * \param [in] offset2 the offset for channel 2
 * \param [in] reverse_channels flag to reverse the channel orders
 * \param [in] output_mem_type the memory type of output tensor buffer
 * \return Rocal status indicating success or failure
 */
extern "C" RocalStatus ROCAL_API_CALL rocalToTensor(RocalContext rocal_context, void *out_ptr,
                                                    RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type,
                                                    float multiplier0, float multiplier1, float multiplier2, float offset0,
                                                    float offset1, float offset2,
                                                    bool reverse_channels, RocalOutputMemType output_mem_type, int max_roi_height = 0, int max_roi_width = 0);

/*!
 * \brief Sets the output images in the RocalContext
 * \ingroup group_rocal_data_transfer
 * \param [in] p_context Rocal context
 * \param [in] num_of_outputs number of output images
 * \param [in] output_images output images
 */
extern "C" void ROCAL_API_CALL rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalTensor> &output_images);

/*!
 * \brief gives the list of output tensors from rocal context
 * \ingroup group_rocal_data_transfer
 * \param [in] p_context Rocal Context
 * \return A RocalTensorList containing the list of output tensors
 */
extern "C" RocalTensorList ROCAL_API_CALL rocalGetOutputTensors(RocalContext p_context);

/*!
 * \brief Creates ExternalSourceFeedInput for data transfer
 * \ingroup group_rocal_data_transfer
 * \param rocal_context Rocal context
 * \param input_images Strings pointing to the location on the disk
 * \param labels Labels whose values is passed by the user using an external source
 * \param input_buffer Compressed or uncompressed input buffer
 * \param roi_width The roi width of the images
 * \param roi_height The roi height of the images
 * \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param channels The number of channels for the image
 * \param mode Determines the mode of the source passed from the user - file_names / uncompressed data / compressed data
 * \param layout Determines the layout of the images - NCHW / NHWC
 * \return Reference to the output tensor
 */
extern "C" RocalStatus ROCAL_API_CALL rocalExternalSourceFeedInput(RocalContext p_context, const std::vector<std::string>& input_images_names,
                                                                   bool is_labels, const std::vector<unsigned char *>& input_buffer,
                                                                   const std::vector<ROIxywh>& roi_xywh,
                                                                   unsigned int max_width, unsigned int max_height, unsigned int channels,
                                                                   RocalExternalSourceMode mode, RocalTensorLayout layout, bool eos);

#endif  // MIVISIONX_ROCAL_API_DATA_TRANSFER_H
