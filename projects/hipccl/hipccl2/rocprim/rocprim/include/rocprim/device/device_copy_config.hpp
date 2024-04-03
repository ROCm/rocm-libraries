// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_COPY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_COPY_CONFIG_HPP_

#include "config_types.hpp"
#include "detail/device_config_helper.hpp"
#include "device_memcpy_config.hpp"

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief
///
/// \tparam NonBlevBlockSize - number of threads per block for thread- and warp-level copy.
/// \tparam NonBlevBuffersPerThreaed - number of buffers processed per thread.
/// \tparam TlevBytesPerThread - number of bytes per thread for thread-level copy.
/// \tparam BlevBlockSize - number of thread per block for block-level copy.
/// \tparam BlevBytesPerThread - number of bytes per thread for block-level copy.
/// \tparam WlevSizeThreshold - minimum size to use warp-level copy instead of thread-level.
/// \tparam BlevSizeThreshold - minimum size to use block-level copy instead of warp-level.
template<unsigned int NonBlevBlockSize         = 256,
         unsigned int NonBlevBuffersPerThreaed = 2,
         unsigned int TlevBytesPerThread       = 8,
         unsigned int BlevBlockSize            = 128,
         unsigned int BlevBytesPerThread       = 32,
         unsigned int WlevSizeThreshold        = 128,
         unsigned int BlevSizeThreshold        = 1024>
struct batch_copy_config
    : public batch_memcpy_config<NonBlevBlockSize,
                                 NonBlevBuffersPerThreaed,
                                 TlevBytesPerThread,
                                 BlevBlockSize,
                                 BlevBytesPerThread,
                                 WlevSizeThreshold,
                                 BlevSizeThreshold>
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
#endif
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif
