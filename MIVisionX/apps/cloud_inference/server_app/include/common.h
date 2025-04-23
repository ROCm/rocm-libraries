/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef COMMON_H
#define COMMON_H

// Maximum number of GPUs supported
#define MAX_NUM_GPU    8

// Module configuration
#define MODULE_CONFIG  "annmodule.txt"

// Module library name
#ifdef __APPLE__
#define MODULE_LIBNAME "libannmodule.dylib"
#else
#define MODULE_LIBNAME "libannmodule.so"
#endif

// Useful macros
#define ERRCHK(call) if(call) return -1

// Useful functions
void info(const char * format, ...);
void warning(const char * format, ...);
void fatal(const char * format, ...);
int error(const char * format, ...);

#endif
