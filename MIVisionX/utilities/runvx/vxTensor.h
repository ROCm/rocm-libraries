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


#ifndef __VX_TENSOR_H__
#define __VX_TENSOR_H__

#include "vxParameter.h"
#include "vxParamHelper.h"
#include "vxUtils.h"

#define MAX_TENSOR_DIMENSIONS     4

class CVxParamTensor : public CVxParameter
{
public:
	CVxParamTensor();
	virtual ~CVxParamTensor();
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);
	virtual int Finalize();
	virtual int SyncFrame(int frameNumber);
	virtual int ReadFrame(int frameNumber);
	virtual int WriteFrame(int frameNumber);
	virtual int CompareFrame(int frameNumber);
	virtual int Shutdown();

private:
	// vx configuration
	vx_size m_num_of_dims;
	vx_size m_dims[MAX_TENSOR_DIMENSIONS];
	vx_enum m_data_type;
	vx_uint32 m_fixed_point_pos;
	// I/O configuration
	bool m_readFileIsBinary;
	bool m_writeFileIsBinary;
	bool m_compareFileIsBinary;
	int m_compareCountMatches;
	int m_compareCountMismatches;
	float m_maxErrorLimit;
	float m_avgErrorLimit;
	// vx object
	vx_tensor m_tensor;
	vx_uint8 * m_data;
	vx_size m_size;
	vx_size m_stride[MAX_TENSOR_DIMENSIONS];
	vx_size m_start[MAX_TENSOR_DIMENSIONS];
	vx_size m_end[MAX_TENSOR_DIMENSIONS];
	vx_enum m_memory_type;
	vx_size m_num_handles;
	vx_size m_active_handle;
	void * m_memory_handle[MAX_BUFFER_HANDLES];
};

#endif /* __VX_TENSOR_H__ */
