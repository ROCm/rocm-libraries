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

#ifndef MIVISIONX_ROCAL_API_TENSOR_H
#define MIVISIONX_ROCAL_API_TENSOR_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Tensor
 *
 * \defgroup group_rocal_tensor API: AMD rocAL - Tensor API
 * \brief The AMD rocAL Tensor.
 */

/*!
 * \brief class representing rocal tensor
 */
class rocalTensor {
   public:
    virtual ~rocalTensor() = default;
    virtual void* buffer() = 0;
    virtual unsigned copy_data(void* user_buffer, RocalOutputMemType external_mem_type = ROCAL_MEMCPY_HOST) = 0;
    virtual unsigned copy_data(void* user_buffer, uint x_offset, uint y_offset, uint max_cols, uint max_rows) = 0; // Copy only the ROI to the user_buffer [The padded region is not copied]
    virtual unsigned num_of_dims() = 0;
    virtual unsigned batch_size() = 0;
    virtual std::vector<size_t> dims() = 0;
    virtual std::vector<size_t> strides() = 0;
    virtual RocalTensorLayout layout() = 0;
    virtual RocalTensorBackend backend() = 0;
    virtual RocalTensorOutputType data_type() = 0;
    virtual RocalOutputMemType mem_type() = 0;
    virtual size_t data_size() = 0;
    virtual RocalROICordsType roi_type() = 0;
    virtual size_t get_roi_dims_size() = 0;
    virtual void copy_roi(void* roi_buffer) = 0;
    virtual std::vector<size_t> shape() = 0;
    virtual void set_dims(std::vector<size_t> dims) = 0;
    virtual void set_mem_handle(void* buffer) = 0;
    virtual void set_tensor_layout(RocalTensorLayout layout) = 0;
    virtual uint64_t data_type_size() = 0;
};

/*!
 * \brief class representing rocal tensor list
 */
class rocalTensorList {
   public:
    virtual ~rocalTensorList() = default;
    virtual uint64_t size() = 0;
    virtual rocalTensor* at(size_t index) = 0;
    // isDenseTensor
};

/*!
 * \brief class representing a list of rocal tensor list
 */
class rocalListOfTensorList {
   public:
    virtual ~rocalListOfTensorList() = default;
    virtual uint64_t size() = 0;
    virtual rocalTensorList* at(size_t index) = 0;
};

/*! 
 * \brief  RocalNSROutput contains the anchor and shape tensor for NonSilentRegionDetection
 */
struct RocalNSROutput {
    rocalTensor* anchor;
    rocalTensor* shape;
};

typedef rocalTensor* RocalTensor;
typedef rocalTensorList* RocalTensorList;
typedef rocalListOfTensorList* RocalMetaData;

#endif  // MIVISIONX_ROCAL_API_TENSOR_H
