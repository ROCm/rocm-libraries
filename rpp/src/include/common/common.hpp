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

#ifndef GUARD_RPP_COMMON_HPP_
#define GUARD_RPP_COMMON_HPP_

#include "rpp.h"
#include "manage_ptr.hpp"

#ifdef HIP_COMPILE
using Data_t        = void*;
using ConstData_t   = const void*;
using ManageDataPtr = RPP_MANAGE_PTR(void, hipFree);
inline Data_t DataCast(void* p) { return p; }
inline ConstData_t DataCast(const void* p) { return p; }
#elif defined(OCL_COMPILE)
using Data_t        = cl_mem;
using ConstData_t   = Data_t;    // Const doesnt apply to cl_mem
using ManageDataPtr = RPP_MANAGE_PTR(cl_mem, clReleaseMemObject);
inline Data_t DataCast(void* p) { return reinterpret_cast<Data_t>(p); }
inline ConstData_t DataCast(const void* p) { return reinterpret_cast<ConstData_t>(const_cast<void*>(p)); }
#endif

#endif    // GUARD_RPP_COMMON_HPP_
