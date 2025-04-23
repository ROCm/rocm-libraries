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


#ifndef __VX_DISTRIBUTION_H__
#define __VX_DISTRIBUTION_H__

#include "vxParameter.h"
#include "vxParamHelper.h"
#include "vxUtils.h"

// CVxParamDistribution: wrapper for vx_distribution object
class CVxParamDistribution : public CVxParameter
{
public:
	// constructor and destructor
	CVxParamDistribution();
	virtual ~CVxParamDistribution();
	virtual int Shutdown();

	// Initialize: create OpenVX object and further uses InitializeIO to input/output initialization
	//   desc: object description as specified on command-line or in script
	//   returns 0 on SUCCESS, else error code
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);

	// InitializeIO: performs I/O initialization using the OpenVX object already created
	//   ref: OpenVX object already created
	//   io_params: I/O description as specified on command-line or in script
	//   returns 0 on SUCCESS, else error code
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);

	// Finalize: for final initialization after vxVerifyGraph
	//   meant for querying object parameters which are not available until vxVerifyGraph
	virtual int Finalize();

	// get OpenVX object type (e.g., VX_TYPE_IMAGE, VX_TYPE_SCALAR, ...)
	// TBD: change getObjectType to GetObjectType

	// frame-level read, write, and compare
	//   returns 0 on SUCCESS, else error code
	//   ReadFrame() returns +ve value to indicate data unavailability
	virtual int ReadFrame(int frameNumber);
	virtual int WriteFrame(int frameNumber);
	virtual int CompareFrame(int frameNumber);

protected:
	// read file into m_bufForRead: returns 0 if successful, 1 on EOF
	int ReadFileIntoBuffer(FILE * fp, vx_uint32 * buf);

private:
	// vx_distribution configuration
	vx_size m_numBins;
	vx_int32 m_offset;
	vx_uint32 m_range;
	// I/O configuration
	bool m_readFileIsBinary;
	bool m_writeFileIsBinary;
	bool m_compareFileIsBinary;
	int m_compareCountMatches;
	int m_compareCountMismatches;
	// vx object
	vx_distribution m_distribution;
	vx_uint32 * m_bufForCompare;
};

#endif /* __VX_DISTRIBUTION_H__ */