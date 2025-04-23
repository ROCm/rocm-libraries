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

#ifndef __profiler_h__
#define __profiler_h__


// PROFILER_MODE:
//   0 - no profiling
//   1 - default profiling
#define PROFILER_MODE 0

#if PROFILER_MODE
#ifndef _WIN32
#include <inttypes.h>
#define __stdcall
#define __int64 int64_t
#endif

#define PROFILER_DEFINE_EVENT_ENUM(g,e) ePROFILER_EVENT_ENUM_ ## g ## e,
enum ProfilerEventEnum {
    PROFILER_DEFINE_EVENT_ENUM(inference_server_app, workMasterInputQ)
    PROFILER_DEFINE_EVENT_ENUM(inference_server_app, workDeviceInputCopyBatch)
    PROFILER_DEFINE_EVENT_ENUM(inference_server_app, workDeviceInputCopyJpegDecode)
    PROFILER_DEFINE_EVENT_ENUM(inference_server_app, workDeviceProcess)
    PROFILER_DEFINE_EVENT_ENUM(inference_server_app, workDeviceOutputCopy)
    PROFILER_DEFINE_EVENT_ENUM(inference_server_app, workRGBtoTensor)
	PROFILER_NUM_EVENTS
};
void __stdcall PROFILER_INITIALIZE();
void __stdcall PROFILER_SHUTDOWN();
#else
#define PROFILER_INITIALIZE()
#define PROFILER_SHUTDOWN()
#endif

#if PROFILER_MODE
void __stdcall _PROFILER_START(ProfilerEventEnum e);
void __stdcall _PROFILER_STOP(ProfilerEventEnum e);
void __stdcall _PROFILER_DATA(ProfilerEventEnum e, __int64 value);
#define PROFILER_START(g,e)  _PROFILER_START(ePROFILER_EVENT_ENUM_ ## g ## e);
#define PROFILER_STOP(g,e)   _PROFILER_STOP(ePROFILER_EVENT_ENUM_ ## g ## e);
#define PROFILER_DATA(g,e,v) _PROFILER_DATA(ePROFILER_EVENT_ENUM_ ## g ## e, (__int64)v);
#define PROFILER_DATA2(g,e,v0,v1) _PROFILER_DATA(ePROFILER_EVENT_ENUM_ ## g ## e, (__int64)(v0)|((__int64)(v1)<<32));
#define PROFILER_START_INDEX(g,e,i)  _PROFILER_START((ProfilerEventEnum)(ePROFILER_EVENT_ENUM_ ## g ## e + (i)));
#define PROFILER_STOP_INDEX(g,e,i)   _PROFILER_STOP((ProfilerEventEnum)(ePROFILER_EVENT_ENUM_ ## g ## e + (i)));
#define PROFILER_DATA_INDEX(g,e,i,v) _PROFILER_DATA((ProfilerEventEnum)(ePROFILER_EVENT_ENUM_ ## g ## e + (i)), (__int64)v);
#define PROFILER_DATA2_INDEX(g,e,i,v0,v1) _PROFILER_DATA((ProfilerEventEnum)(ePROFILER_EVENT_ENUM_ ## g ## e + (i)), (__int64)(v0)|((__int64)(v1)<<32));
#else
#define PROFILER_START(g,e)
#define PROFILER_STOP(g,e)
#define PROFILER_DATA(g,e,v)
#define PROFILER_DATA2(g,e,v0,v1)
#define PROFILER_START_INDEX(g,e,i)
#define PROFILER_STOP_INDEX(g,e,i)
#define PROFILER_DATA_INDEX(g,e,i,v)
#define PROFILER_DATA2_INDEX(g,e,i,v0,v1)
#endif

#endif
