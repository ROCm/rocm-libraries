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


#ifndef __VX_UTILS_H__
#define __VX_UTILS_H__

// OpenCL: enabled unless disabled explicitly by setting ENABLE_OPENCL=0
#ifndef ENABLE_OPENCL
#define ENABLE_OPENCL  1
#endif
// OpenCV: enabled unless disabled explicitly by setting ENABLE_OPENCV=0
#ifndef ENABLE_OPENCV
#define ENABLE_OPENCV 1
#endif

#include <VX/vx.h>
#include <VX/vx_compatibility.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>

#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <algorithm>

#if _WIN32
#include <Windows.h>
#include <wincrypt.h>
#else
#include <chrono>
#include <unistd.h>
#if HAVE_OpenSSL
#include <openssl/hmac.h>
#include <openssl/md5.h>
#endif

#include <strings.h>
#define _strnicmp strncasecmp
#define _stricmp  strcasecmp
#endif

#include "vx_ext_amd.h"
#define _USE_MATH_DEFINES
#include <math.h>

#if ENABLE_OPENCL
#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#if ENABLE_HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#endif

#if ENABLE_OPENCV
#include "opencv2/opencv.hpp"
using namespace cv;
#if USE_OPENCV_4
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#endif
#endif

// Scalar to String SIZE
#define MAX_SCALAR_TO_STRING_SIZE 256

using namespace std;

///////////////////////////////////////////////////////////////////////////
// macros
///////////////////////////////////////////////////////////////////////////

// error check/report macros
#define ReportError(...) { printf(__VA_ARGS__); throw -1; }
#define ERROR_CHECK(call) { vx_status status = call; if(status) ReportError("ERROR: " #call "=> %d (%s) [" __FILE__ "#%d]\n", status, ovxEnum2Name(status), __LINE__); }
#define ERROR_CHECK_AND_WARN(call,warncode) { vx_status status = call; if(status == warncode) printf("WARNING: " #call "=> %d (%s) [" __FILE__ "#%d]\n", status, ovxEnum2Name(status), __LINE__); else if(status) ReportError("ERROR: " #call "=> %d (%s) [" __FILE__ "#%d]\n", status, ovxEnum2Name(status), __LINE__); }
#define NULLPTR_CHECK(call) if((call) == nullptr) ReportError("ERROR: " #call "=> nullptr [" __FILE__ "#%d]\n", __LINE__)

///////////////////////////////////////////////////////////////////////////
// platform independent functions
///////////////////////////////////////////////////////////////////////////

// get clock counter
inline int64_t utilGetClockCounter()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	return v.QuadPart;
#else
	return chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

// get clock frequency
inline int64_t utilGetClockFrequency()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceFrequency(&v);
	return v.QuadPart;
#else
	return chrono::high_resolution_clock::period::den / chrono::high_resolution_clock::period::num;
#endif
}


///////////////////////////////////////////////////////////////////////////
// class CHasher for checksum computation
///////////////////////////////////////////////////////////////////////////
class CHasher {
public:
	CHasher();
	~CHasher();

	void Initialize();
	void Process(vx_uint8 * data_ptr, vx_size count);
	const char * GetCheckSum();
	void Shutdown();

private:
#if _WIN32
	HCRYPTPROV m_cryptProv;
	HCRYPTHASH m_cryptHash;
#elif HAVE_OpenSSL
	MD5_CTX m_handle;
#endif
	vx_uint8 m_hash[16];
	char m_checkSum[33];

};

///////////////////////////////////////////////////////////////////////////
// utility functions
///////////////////////////////////////////////////////////////////////////

// Utility functions to replace ~ with a user specified directory
//   SetRootDir -- set the root directory (default ".")
//   RootDirUpdated -- replace ~ in filePath with root directory
void SetRootDir(const char * rootDir);
const char * RootDirUpdated(const char * filePath);

// enumeration constants
//  ovxEnum2Name -- the returns a global pointer, so returned string has to be saved by caller immediately
//  ovxEnum2String -- return enum name or hex value as a string
//  ovxName2Enum -- returns enum corresponding to name or hex value in the input string
const char * ovxEnum2Name(vx_enum e);
void ovxEnum2String(vx_enum e, char str[]);
vx_enum ovxName2Enum(const char * name);

// compute checksum of rectangular region specified within an image
void ComputeChecksum(char checkSumString[64], vx_image image, vx_rectangle_t * rectRegion);
// compare rectangular region specified within an image and return number of pixels mismatching
size_t CompareImage(vx_image image, vx_rectangle_t * rectRegion, vx_uint8 * refImage, float errLimitMin, float errLimitMax, int frameNumber, const char * fileNameRef);
// get image width in bytes from image
vx_size CalculateImageWidthInBytes(vx_image image);
// read image
int ReadImage(vx_image image, vx_rectangle_t * rectFull, FILE * fp);
// write image
int WriteImage(vx_image image, vx_rectangle_t * rectFull, FILE * fp);
// write image compressed
int WriteImageCompressed(vx_image image, vx_rectangle_t * rectFull, const char * fileName);

// read & write scalar value to/from a string
int ReadScalarToString(vx_scalar scalar, char str[]);
int WriteScalarFromString(vx_scalar scalar, const char str[]);
int GetScalarValueFromString(vx_enum type, const char str[], vx_uint64 * value);
int PutScalarValueToString(vx_enum type, const void * value, char str[]);
int GetScalarValueForStructTypes(vx_enum type, const char str[], void * value);

// useful utility functions:
//   stristr -- case insensitive version of strstr
const char * stristr(const char * str1, const char * str2);
vector<string> &split(const string &s, char delim, vector<string> &elems);
int convert_image_format(string format);

#endif

