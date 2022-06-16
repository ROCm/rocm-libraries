# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(read_config_autotune_settings file list_across_names list_across output_pattern_suffix)
  if(file STREQUAL "benchmark_device_adjacent_difference")
    set(list_across_names "DataType;Left;InPlace;BlockSize;ItemsPerThread" PARENT_SCOPE)
    set(list_across "int int64_t uint8_t rocprim::half float double custom_type<float,float>;\
true false;true false;64 128;1 2 4 8 16" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@Left@_@InPlace@_@BlockSize@_@ItemsPerThread@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_merge_sort")
    set(list_across_names "KeyType_ValueType;MergeBlockSizeExponent;SortBlockSizeExponent" PARENT_SCOPE)
    # first list is keys, second list is key,value pairs
    set(list_across "\
int int64_t int8_t uint8_t rocprim::half short \
\
int,float int64_t,double int8_t,int8_t uint8_t,uint8_t rocprim::half,rocprim::half short,short int,custom_type<float,float> \
int64_t,custom_type<double,double> custom_type<double,double>,custom_type<double,double> custom_type<int,int>,custom_type<double,double> \
custom_type<int,int>,custom_type<char,double> custom_type<int,int>,custom_type<int64_t,double>;\
6 7 8 9 10;7 8 9 10" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType_ValueType@_@MergeBlockSizeExponent@_@SortBlockSizeExponent@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_radix_sort")
    set(list_across_names "KeyType_ValueType;LongRadixBits_ShortRadixBits;ItemsPerThread2" PARENT_SCOPE)
    # first list is keys, second list is key,value pairs
    set(list_across "\
int int64_t int8_t uint8_t rocprim::half short \
\
int,float int,double int,float2 int,custom_type<float,float> int,double2 int,custom_type<double,double> \
int64_t,float int64_t,double int64_t,float2 int64_t,custom_type<float,float> int64_t,double2 int64_t,custom_type<double,double> \
int8_t,int8_t uint8_t,uint8_t rocprim::half,rocprim::half;\
4,3 5,4 6,4 7,6 8,7;1 2 4 8" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType_ValueType@_@LongRadixBits_ShortRadixBits@_@ItemsPerThread2@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_radix_sort_single")
    set(list_across_names "KeyType_ValueType;BlockSize" PARENT_SCOPE)
    # first list is keys, second list is key,value pairs
    set(list_across "\
int int64_t int8_t uint8_t rocprim::half short \
\
int,float int,double int,float2 int,custom_type<float,float> int,double2 int,custom_type<double,double> \
int64_t,float int64_t,double int64_t,float2 int64_t,custom_type<float,float> int64_t,double2 int64_t,custom_type<double,double> \
int8_t,int8_t uint8_t,uint8_t rocprim::half,rocprim::half;\
64 128 256 512 1024" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType_ValueType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_reduce")
    set(list_across_names "DataType;BlockSize;ItemsPerThread" PARENT_SCOPE)
    set(list_across "int float double int8_t int64_t rocprim::half;64 128 256;1 2 4 8 16" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@BlockSize@_@ItemsPerThread@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_scan")
    set(list_across_names "ByKey;Excl;DataType" PARENT_SCOPE)
    set(list_across "true false;true false;\
int float double int64_t custom_type<double,double> int8_t rocprim::half" PARENT_SCOPE)
    set(output_pattern_suffix "@ByKey@_@Excl@_@DataType@" PARENT_SCOPE)
  endif()
endfunction()