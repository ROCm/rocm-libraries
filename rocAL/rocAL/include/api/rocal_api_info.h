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

#ifndef MIVISIONX_ROCAL_API_INFO_H
#define MIVISIONX_ROCAL_API_INFO_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Info
 *
 * \defgroup group_rocal_info API: AMD rocAL - Info API
 * \brief The AMD rocAL Informational functions.
 */

/*!
 * \brief Retrieves the width of the output.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext
 * \return The width of the output.
 */
extern "C" int ROCAL_API_CALL rocalGetOutputWidth(RocalContext rocal_context);

/*!
 * \brief Retrieves the height of the output.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext
 * \return The height of the output.
 */
extern "C" int ROCAL_API_CALL rocalGetOutputHeight(RocalContext rocal_context);

/*!
 * \brief Retrieves the color format of the output.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext.
 * \return The color format of the output.
 */
extern "C" int ROCAL_API_CALL rocalGetOutputColorFormat(RocalContext rocal_context);

/*!
 * \brief Retrieves the number of remaining images.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext.
 * \return The number of remaining images yet to be processed.
 */

extern "C" size_t ROCAL_API_CALL rocalGetRemainingImages(RocalContext rocal_context);

/*!
 * \brief Retrieves the width of the image.
 * \ingroup group_rocal_info
 * \param [in] image The RocalTensor data.
 * \return The width of the image.
 */
extern "C" size_t ROCAL_API_CALL rocalGetImageWidth(RocalTensor image);

/*!
 * \brief Retrieves the height of the image.
 * \ingroup group_rocal_info
 * \param [in] image The RocalTensor data.
 * \return The height of the image.
 */
extern "C" size_t ROCAL_API_CALL rocalGetImageHeight(RocalTensor image);

/*!
 * \brief Retrieves the number of planes (channels) in the image.
 * \ingroup group_rocal_info
 * \param [in] image The RocalTensor data.
 * \return The number of planes (channels) in the image.
 */
extern "C" size_t ROCAL_API_CALL rocalGetImagePlanes(RocalTensor image);

/*!
 * \brief Checks if the RocalContext is empty.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext
 * \return return if RocalContext is empty or not.
 */
extern "C" size_t ROCAL_API_CALL rocalIsEmpty(RocalContext rocal_context);

/*!
 * \brief Retrieves the number of augmentation branches.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext
 * \return Number of augmentation graph branches. Defined by number of calls to the augmentation API's with the is_output flag set to true.
 */
extern "C" size_t ROCAL_API_CALL rocalGetAugmentationBranchCount(RocalContext rocal_context);

/*!
 * \brief Retrieves the status.
  * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext from which to retrieve the status.
 * \return The status of tha last API call
 */
extern "C" RocalStatus ROCAL_API_CALL rocalGetStatus(RocalContext rocal_context);

/*!
 * \brief Retrieves the error message.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext
 * \return A pointer to the error message string.
 */
extern "C" const char* ROCAL_API_CALL rocalGetErrorMessage(RocalContext rocal_context);

/*!
 * \brief Retrieves timing information.
 * \ingroup group_rocal_info
 * \param [in] rocal_context The RocalContext
 * \return The timing info associated with recent execution.
 */
extern "C" TimingInfo ROCAL_API_CALL rocalGetTimingInfo(RocalContext rocal_context);

/*!
 * \brief Retrieves the information about the size of the last batch.
 * \ingroup group_rocal_info
 * \param rocal_context
 * \return The number of samples that were padded in the last batch in adherence with last_batch_policy and pad_last_batch_repeated.
 */
extern "C" size_t ROCAL_API_CALL rocalGetLastBatchPaddedSize(RocalContext rocal_context);

#endif  // MIVISIONX_ROCAL_API_INFO_H
