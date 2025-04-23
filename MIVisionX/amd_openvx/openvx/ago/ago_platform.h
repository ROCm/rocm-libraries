/* 
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 
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


#ifndef __ago_platform_h__
#define __ago_platform_h__

// OpenCL: enabled unless disabled explicitly by setting ENABLE_OPENCL=0
#ifndef ENABLE_OPENCL
#define ENABLE_OPENCL  1
#endif

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <fenv.h>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <functional>
#include <chrono>
#include <thread>

#if _WIN32
#include <Windows.h>
#include <cmath>
#include <intrin.h>
#else
#include <dlfcn.h>
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#if __APPLE__
#include <cstdlib>
#include <cmath>
#endif
#include <strings.h>
#define _strnicmp strncasecmp
#define _stricmp  strcasecmp
#endif
using namespace std;

#if ENABLE_OPENCL
#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#if ENABLE_HIP
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#define HIPRTC_GET_TYPE_NAME
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip/hiprtc.h"
#endif

// platform specific shared library file extension
#if _WIN32
#define SHARED_LIBRARY_PREFIX    ""
#define SHARED_LIBRARY_EXTENSION ".dll"
#elif __APPLE__
#define SHARED_LIBRARY_PREFIX    "lib"
#define SHARED_LIBRARY_EXTENSION ".dylib"
#else
#define SHARED_LIBRARY_PREFIX    "lib"
#define SHARED_LIBRARY_EXTENSION ".so"
#endif

// platform specific alignment attributes
#if _WIN32
#define DECL_ALIGN(n) __declspec(align(n))
#define ATTR_ALIGN(n)
#else
#define DECL_ALIGN(n)
#define ATTR_ALIGN(n) __attribute__((aligned(n)))
#endif

// macro to port VisualStudio m128i fields of __m128i to g++
#if _WIN32
#define M128I(m128i_register) m128i_register
#else
#define M128I(m128i_register) (*((_m128i_union*)&m128i_register))
typedef union {
	char               m128i_i8[16];
	short              m128i_i16[8];
	int                m128i_i32[4];
	long long          m128i_i64[2];
	unsigned char      m128i_u8[16];
	unsigned short     m128i_u16[8];
	unsigned int       m128i_u32[4];
	unsigned long long m128i_u64[2];
} _m128i_union;
#endif

// platform independent data types
typedef struct _ago_module    * ago_module;

// platform independent functions
bool       agoIsCpuHardwareSupported();
uint32_t   agoControlFpSetRoundEven();
void       agoControlFpReset(uint32_t state);
int64_t    agoGetClockCounter();
int64_t    agoGetClockFrequency();
bool       agoGetEnvironmentVariable(const char * name, char * value, size_t valueSize); // returns true if success
bool       agoSetEnvironmentVariable(const char * name, const char * value); // returns true if success
bool       agoUnsetEnvironmentVariable(const char * name); // returns true if success
ago_module agoOpenModule(const char * libFileName);
void *     agoGetFunctionAddress(ago_module module, const char * functionName);
void       agoCloseModule(ago_module module);

#if !_WIN32
typedef void * CRITICAL_SECTION;
typedef void * HANDLE;
typedef unsigned long DWORD;
typedef void * LPVOID;
typedef int BOOL;
typedef long LONG, * LPLONG;
typedef void (*LPTHREAD_START_ROUTINE)(LPVOID lpThreadParameter);
extern void EnterCriticalSection(CRITICAL_SECTION* cs);
extern void LeaveCriticalSection(CRITICAL_SECTION* cs);
extern void InitializeCriticalSection(CRITICAL_SECTION* cs);
extern void DeleteCriticalSection(CRITICAL_SECTION* cs);
extern void CloseHandle(HANDLE h);
extern HANDLE CreateSemaphore(void *, LONG, LONG, void *);
extern HANDLE CreateThread(void *, size_t dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, LPVOID lpParameter, DWORD dwCreationFlags, void *);
extern DWORD WaitForSingleObject(HANDLE hHandle, DWORD dwMilliseconds);
extern BOOL ReleaseSemaphore(HANDLE hSemaphore, LONG lReleaseCount, LPLONG lpPreviousCount);
#define WINAPI
#define INFINITE 0xFFFFFFFF
#define WAIT_OBJECT_0 0
#endif

#endif