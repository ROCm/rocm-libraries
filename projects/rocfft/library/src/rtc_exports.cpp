// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocfft_exception.h"
#include "rtc_cache.h"

rocfft_status rocfft_cache_serialize(void** buffer, size_t* buffer_len_bytes)
try
{
    if(!buffer || !buffer_len_bytes)
        return rocfft_status_invalid_arg_value;

    if(!RTCCache::single)
        return rocfft_status_failure;

    return RTCCache::single->serialize(buffer, buffer_len_bytes);
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_cache_buffer_free(void* buffer)
try
{
    if(!RTCCache::single)
        return rocfft_status_failure;
    RTCCache::single->serialize_free(buffer);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_cache_deserialize(const void* buffer, size_t buffer_len_bytes)
try
{
    if(!buffer || !buffer_len_bytes)
        return rocfft_status_invalid_arg_value;

    if(!RTCCache::single)
        return rocfft_status_failure;

    return RTCCache::single->deserialize(buffer, buffer_len_bytes);
}
catch(...)
{
    return rocfft_handle_exception();
}
