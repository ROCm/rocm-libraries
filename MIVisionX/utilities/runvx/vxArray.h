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


#ifndef __VX_ARRAY_H__
#define __VX_ARRAY_H__

#include "vxParameter.h"
#include "vxParamHelper.h"
#include "vxUtils.h"

class CVxParamArray : public CVxParameter
{
public:
	CVxParamArray();
	virtual ~CVxParamArray();
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);
	virtual int Finalize();
	virtual int ReadFrame(int frameNumber);
	virtual int WriteFrame(int frameNumber);
	virtual int CompareFrame(int frameNumber);
	virtual int Shutdown();

protected:
	// read file into m_bufForRead: returns numItems
	size_t ReadFileIntoBuffer(FILE * fp, bool readFileIsBinary);
	// compare routines: return true if mismatch is detected, other returns false
	bool CompareFrameBitwiseExact(size_t numItems, size_t numItemsRef, vx_uint8 * bufItems, int frameNumber, const char * fileName);
	bool CompareFrameKeypoints(size_t numItems, size_t numItemsRef, vx_uint8 * bufItems, int frameNumber, const char * fileName);
	bool CompareFrameCoord2d(size_t numItems, size_t numItemsRef, vx_uint8 * bufItems, int frameNumber, const char * fileName);

private:
	// vx configuration
	vx_enum m_format;
	vx_size m_capacity;
	vx_size m_itemSize;
	// I/O configuration
	bool m_readFileIsBinary;
	bool m_writeFileIsBinary;
	bool m_compareFileIsBinary;
	int m_compareCountMatches;
	int m_compareCountMismatches;
	std::string m_fileNameCompareLog;
	vx_int32 m_errX;
	vx_int32 m_errY;
	vx_float32 m_errStrength;
	vx_float32 m_errMismatchPercent;
	// vx object
	vx_array m_array;
	vx_uint8 * m_bufForRead;
};


#endif /* __VX_ARRAY_H__ */