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

#pragma once

#include "device_code.h"

const static std::string data_transfer_program_name = "utility";

const static std::vector<std::string> data_transfer_kernel_names = {"copyInt8ToNHWC", "copyInt8ToNCHW"};

const static std::string data_transfer_source =
    "__kernel void copyInt8ToNHWC(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned w, unsigned h, unsigned c, float multiplier0, float multiplier1, float multiplier2, float offset0, float offset1, float offset2, unsigned reverse_channels) {"
    "   if(c > 3 || c < 1) return;"
    "	int i = get_global_id(0);"
    "   unsigned channel_size = h*w;"
    "   unsigned size = channel_size*c;"
    "   if(i >= size) return;"
    "   unsigned channel_idx = i % c;"
    "   unsigned pixel_idx = i % channel_size;"
    "   float out_val = 0;"
    "   float multiplier[3] = {multiplier0, multiplier1, multiplier2};"
    "   float offset[3] = {offset0, offset1, offset2};"
    "   if(reverse_channels) {"
    "       out_val = multiplier[c-channel_idx-1]*((float)(in[c*pixel_idx+c-channel_idx-1]))+offset[c-channel_idx-1]; "
    "   } else {"
    "       out_val = multiplier[channel_idx]*((float)(in[c*pixel_idx+channel_idx]))+offset[channel_idx]; "
    "   }"
    "   out [out_offset + c*pixel_idx + channel_idx] = out_val;}"
    ""
    ""
    "__kernel void copyInt8ToNCHW(__global const unsigned char* in, __global float* out, unsigned out_offset, unsigned w, unsigned h, unsigned c, float multiplier0, float multiplier1, float multiplier2, float offset0, float offset1, float offset2, unsigned reverse_channels) {"
    "   if(c > 3 || c < 1) return;"
    "	int i = get_global_id(0);"
    "   unsigned channel_size = h*w;"
    "   unsigned size = channel_size*c;"
    "   if(i >= size) return; "
    "   unsigned channel_idx = i % c;"
    "   unsigned pixel_idx = i % channel_size;"
    "   float out_val = 0;"
    "   float multiplier[3] = {multiplier0, multiplier1, multiplier2};"
    "   float offset[3] = {offset0, offset1, offset2};"
    "   if(reverse_channels) {"
    "       out_val = multiplier[c-channel_idx-1]*((float)(in[c*pixel_idx+c-channel_idx-1]))+offset[c-channel_idx-1]; "
    "   } else {"
    "       out_val = multiplier[channel_idx]*((float)(in[c*pixel_idx+channel_idx]))+offset[channel_idx]; "
    "   }"
    "   out [out_offset + channel_idx*channel_size + pixel_idx] = out_val;}";

class OCLUtility : public DeviceCode {
   public:
    OCLUtility() : DeviceCode(data_transfer_source, data_transfer_program_name, data_transfer_kernel_names) {}
    // TODO : delete other implicit constructors
};
