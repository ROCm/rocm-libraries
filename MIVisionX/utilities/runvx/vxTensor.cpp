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

#define _CRT_SECURE_NO_WARNINGS
#include "vxTensor.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamTensor
//
CVxParamTensor::CVxParamTensor()
{
	// vx configuration
	m_vxObjType = VX_TYPE_TENSOR;
	m_num_of_dims = 0;
	for (vx_size i = 0; i < MAX_TENSOR_DIMENSIONS; i++)
		m_dims[i] = 1;
	m_data_type = VX_TYPE_INT16;
	m_fixed_point_pos = 0;
	m_num_handles = 0;
	m_memory_type = VX_MEMORY_TYPE_NONE;
	memset(m_memory_handle, 0, sizeof(m_memory_handle));
	// I/O configuration
	m_maxErrorLimit = 0;
	m_avgErrorLimit = 0;
	m_readFileIsBinary = false;
	m_writeFileIsBinary = false;
	m_compareFileIsBinary = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	// vx object
	m_tensor = nullptr;
	m_vxObjRef = nullptr;
	m_data = nullptr;
	m_size = 0;
}

CVxParamTensor::~CVxParamTensor()
{
	Shutdown();
}

int CVxParamTensor::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: tensor COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	if (m_tensor) {
		vxReleaseTensor(&m_tensor);
		m_tensor = nullptr;
	}
	if (m_data) {
		delete[] m_data;
		m_data = nullptr;
	}
	if (m_memory_type == VX_MEMORY_TYPE_HOST) {
		for (vx_size active_handle = 0; active_handle < m_num_handles; active_handle++) {
			if (m_memory_handle[active_handle])
				free(m_memory_handle[active_handle]);
			m_memory_handle[active_handle] = nullptr;
		}
	}
#if ENABLE_OPENCL
	else if (m_memory_type == VX_MEMORY_TYPE_OPENCL) {
		for (vx_size active_handle = 0; active_handle < m_num_handles; active_handle++) {
			if (m_memory_handle[active_handle]) {
				int err = clReleaseMemObject((cl_mem)m_memory_handle[active_handle]);
				if (err)
					ReportError("ERROR: clReleaseMemObject(*) failed (%d)\n", err);
			}
			m_memory_handle[active_handle] = nullptr;
		}
	}
#elif ENABLE_HIP
    else if (m_memory_type == VX_MEMORY_TYPE_HIP) {
		for (vx_size active_handle = 0; active_handle < m_num_handles; active_handle++) {
			if (m_memory_handle[active_handle]) {
				if (m_memory_handle[active_handle]) {
				    hipFree(m_memory_handle[active_handle]);
				}
				m_memory_handle[active_handle] = nullptr;
			}
		}
	}
#endif
    return 0;
}

int CVxParamTensor::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	const char * ioParams = desc;
	if (!_strnicmp(desc, "tensor:", 7) || !_strnicmp(desc, "virtual-tensor:", 15)) {
		bool isVirtual = false;
		if (!_strnicmp(desc, "virtual-tensor:", 15)) {
			isVirtual = true;
			desc += 8;
		}
		char objType[64], data_type[64];
		ioParams = ScanParameters(desc, "tensor:<num-of-dims>,{dims},<data-type>,<fixed-point-pos>", "s:D,L,s,d", objType, &m_num_of_dims, &m_num_of_dims, m_dims, data_type, &m_fixed_point_pos);
		m_data_type = ovxName2Enum(data_type);
		if (isVirtual) {
			m_tensor = vxCreateVirtualTensor(graph, m_num_of_dims, m_dims, m_data_type, m_fixed_point_pos);
		}
		else {
			m_tensor = vxCreateTensor(context, m_num_of_dims, m_dims, m_data_type, m_fixed_point_pos);
		}
	}
	else if (!_strnicmp(desc, "tensor-from-roi:", 16)) {
		char objType[64], masterName[64];
		ioParams = ScanParameters(desc, "tensor-from-view:<tensor>,<view>", "s:s,D,L,L", objType, masterName, &m_num_of_dims, &m_num_of_dims, m_start, &m_num_of_dims, m_end);
		auto itMaster = m_paramMap->find(masterName);
		if (itMaster == m_paramMap->end())
			ReportError("ERROR: tensor [%s] doesn't exist for %s\n", masterName, desc);
		vx_tensor masterTensor = (vx_tensor)itMaster->second->GetVxObject();
		m_tensor = vxCreateTensorFromView(masterTensor, m_num_of_dims, m_start, m_end);
	}
	else if (!_strnicmp(desc, "tensor-from-handle:", 19)) {
		char objType[64], data_type[64], memory_type_str[64];
		ioParams = ScanParameters(desc, "tensor-from-handle:<num-of-dims>,{dims},<data-type>,<fixed-point-pos>,{strides},<num-handles>,<memory-type>",
			"s:D,L,s,d,L,D,s", objType, &m_num_of_dims, &m_num_of_dims, m_dims, data_type, &m_fixed_point_pos, &m_num_of_dims, m_stride, &m_num_handles, memory_type_str);
		if(m_num_handles > MAX_BUFFER_HANDLES)
			ReportError("ERROR: num-handles is out of range: " VX_FMT_SIZE " (must be less than %d)\n", m_num_handles, MAX_BUFFER_HANDLES);
		m_data_type = ovxName2Enum(data_type);
		vx_uint64 memory_type = 0;
		if (GetScalarValueFromString(VX_TYPE_ENUM, memory_type_str, &memory_type) < 0)
			ReportError("ERROR: invalid memory type enum: %s\n", memory_type_str);
		m_memory_type = (vx_enum)memory_type;
		memset(m_memory_handle, 0, sizeof(m_memory_handle));
		if (m_memory_type == VX_MEMORY_TYPE_HOST) {
			// allocate all handles on host
			for (vx_size active_handle = 0; active_handle < m_num_handles; active_handle++) {
				vx_size size = m_dims[m_num_of_dims-1] * m_stride[m_num_of_dims-1];
				m_memory_handle[active_handle] = malloc(size);
				if (!m_memory_handle[active_handle])
					ReportError("ERROR: malloc(%d) failed\n", (int)size);
			}
		}
#if ENABLE_OPENCL
		else if (m_memory_type == VX_MEMORY_TYPE_OPENCL) {
			// allocate all handles on opencl
			cl_context opencl_context = nullptr;
			vx_status status = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &opencl_context, sizeof(opencl_context));
			if (status)
				ReportError("ERROR: vxQueryContext(*,VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,...) failed (%d)\n", status);
			for (vx_size active_handle = 0; active_handle < m_num_handles; active_handle++) {
				vx_size size = m_dims[m_num_of_dims-1] * m_stride[m_num_of_dims-1];
				cl_int err = CL_SUCCESS;
				m_memory_handle[active_handle] = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, size, NULL, &err);
				if (!m_memory_handle[active_handle] || err)
					ReportError("ERROR: clCreateBuffer(*,CL_MEM_READ_WRITE,%d,NULL,*) failed (%d)\n", (int)size, err);
			}
		}
#elif ENABLE_HIP
        else if (m_memory_type == VX_MEMORY_TYPE_HIP) {
            int hip_device = -1;
            vx_status status = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE, &hip_device, sizeof(hip_device));
            if (status || hip_device<0 )
                ReportError("ERROR: vxQueryContext(*,VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE,...) failed (%d, %d)\n", status, hip_device);
			for (vx_size active_handle = 0; active_handle < m_num_handles; active_handle++) {
				vx_size size = m_dims[m_num_of_dims-1] * m_stride[m_num_of_dims-1];
				hipError_t err = hipMalloc(&m_memory_handle[active_handle], size);
				if (!m_memory_handle[active_handle] || err != hipSuccess)
					ReportError("ERROR: VxTensor hipMalloc(%d) failed (%d)\n", (int)size, err);
			}
		}
#endif
		else ReportError("ERROR: invalid memory-type enum: %s\n", memory_type_str);
		m_active_handle = 0;
		m_tensor = vxCreateTensorFromHandle(context, m_num_of_dims, m_dims, m_data_type, m_fixed_point_pos, m_stride, m_memory_handle[m_active_handle], m_memory_type);
	}
	else ReportError("ERROR: unsupported tensor type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_tensor);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: tensor creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_tensor) vxReleaseTensor(&m_tensor);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_tensor;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamTensor::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_tensor = (vx_tensor)m_vxObjRef;
	ERROR_CHECK(vxQueryTensor(m_tensor, VX_TENSOR_NUMBER_OF_DIMS, &m_num_of_dims, sizeof(m_num_of_dims)));
	ERROR_CHECK(vxQueryTensor(m_tensor, VX_TENSOR_DIMS, &m_dims, sizeof(m_dims[0])*m_num_of_dims));
	ERROR_CHECK(vxQueryTensor(m_tensor, VX_TENSOR_DATA_TYPE, &m_data_type, sizeof(m_data_type)));
	ERROR_CHECK(vxQueryTensor(m_tensor, VX_TENSOR_FIXED_POINT_POSITION, &m_fixed_point_pos, sizeof(vx_uint8)));
	if(m_data_type == VX_TYPE_UINT8 || m_data_type == VX_TYPE_INT8)
		m_size = 1;
	else if(m_data_type == VX_TYPE_UINT16 || m_data_type == VX_TYPE_INT16 || m_data_type == VX_TYPE_FLOAT16)
		m_size = 2;
	else if(m_data_type == VX_TYPE_INT64)
		m_size = 8;
	else
	    m_size = 4;
	for (vx_uint32 i = 0; i < m_num_of_dims; i++) {
		m_stride[i] = m_size;
		m_size *= m_dims[i];
	}
	m_data = new vx_uint8[m_size];
	if (!m_data) ReportError("ERROR: memory allocation failed for tensor: %u\n", (vx_uint32)m_size);

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read"))
		{ // read request syntax: read,<fileName>[,ascii|binary]
			m_fileNameRead.assign(RootDirUpdated(fileName));
			m_fileNameForReadHasIndex = (m_fileNameRead.find("%") != m_fileNameRead.npos) ? true : false;
			m_readFileIsBinary = true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",binary", ",s", option);
				if (!_stricmp(option, "binary")) {
					m_readFileIsBinary = true;
				}
				else ReportError("ERROR: invalid tensor read option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "init"))
		{ // init request syntax: init,<fileName>
			if(!_strnicmp(fileName, "@fill~f32~", 10)) {
				float value = (float)atof(&fileName[10]);
				float * buf = (float *)m_data;
				for(size_t i = 0; i < m_size/4; i++)
					buf[i] = value;
			}
			else if(!_strnicmp(fileName, "@fill~i32~", 10)) {
				vx_int32 value = atoi(&fileName[10]);
				vx_int32 * buf = (vx_int32 *)m_data;
				for(size_t i = 0; i < m_size/4; i++)
					buf[i] = value;
			}
			else if(!_strnicmp(fileName, "@fill~i16~", 10)) {
				vx_int16 value = (vx_int16)atoi(&fileName[10]);
				vx_int16 * buf = (vx_int16 *)m_data;
				for(size_t i = 0; i < m_size/2; i++)
					buf[i] = value;
			}
			else if(!_strnicmp(fileName, "@fill~u8~", 9)) {
				int value = atoi(&fileName[9]);
				memset(m_data, value, m_size);
			}
			else {
				int count = 1;
				const char * tensorFileName = fileName;
				if(!_strnicmp(tensorFileName, "@repeat~", 8)) {
					tensorFileName += 8;
					for(count = 0; *tensorFileName >= '0' && *tensorFileName <= '9'; tensorFileName++) {
						count = count * 10 + *tensorFileName - '0';
					}
					if(*tensorFileName++ != '~' || count < 1)
						ReportError("ERROR: invalid init @repeat~<n>~fileName syntax -- %s\n", fileName);
					if((m_size % count) != 0)
						ReportError("ERROR: file size is not multiple of tensor size -- %s\n", fileName);
				}
				if(!_stricmp(fileName + strlen(fileName) - 4, ".dat")) {
					ReportError("ERROR: read from .dat files not supported: %s\n", fileName);
				}
				FILE * fp = fopen(RootDirUpdated(tensorFileName), "rb");
				if (!fp) {
					ReportError("ERROR: Unable to open: %s\n", tensorFileName);
				}
				vx_size size = m_size / count;
				if (fread(m_data, 1, size, fp) != size)
					ReportError("ERROR: not enough data (%d bytes) in %s\n", (vx_uint32)size, tensorFileName);
				for(int i = 1; i < count; i++) {
					memcpy(m_data + i * size, m_data, size);
				}
				vx_status status = vxCopyTensorPatch(m_tensor, m_num_of_dims, nullptr, nullptr, m_stride, m_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
				fclose(fp);
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxCopyTensorPatch: write failed (%d)\n", status);
			}
		}
		else if (!_stricmp(ioType, "write"))
		{ // write request syntax: write,<fileName>[,ascii|binary]
			m_fileNameWrite.assign(RootDirUpdated(fileName));
			m_writeFileIsBinary = true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",binary", ",s", option);
				if (!_stricmp(option, "binary")) {
					m_writeFileIsBinary = true;
				}
				else ReportError("ERROR: invalid tensor write option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // write request syntax: compare,<fileName>[,ascii|binary]
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			m_compareFileIsBinary = true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",binary|maxerr=<value>|avgerr=<value>", ",s", option);
				if (!_stricmp(option, "binary")) {
					m_compareFileIsBinary = true;
				}
				else if (!_strnicmp(option, "maxerr=", 7)) {
					m_maxErrorLimit = (float)atof(&option[7]);
					m_avgErrorLimit = 1e20f;
				}
				else if (!_strnicmp(option, "avgerr=", 7)) {
					m_avgErrorLimit = (float)atof(&option[7]);
					m_maxErrorLimit = 1e20f;
				}
				else ReportError("ERROR: invalid tensor compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "directive") && (!_stricmp(fileName, "VX_DIRECTIVE_AMD_COPY_TO_OPENCL") || !_stricmp(fileName, "sync-cl-write"))) {
			m_useSyncOpenCLWriteDirective = true;
		}
		else ReportError("ERROR: invalid tensor operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamTensor::Finalize()
{
	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_tensor, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamTensor::SyncFrame(int frameNumber)
{
	if (m_num_handles > 1) {
		// swap handles if requested for tensor created from handle
		vx_size prev_handle = m_active_handle;
		void * prev_ptr = nullptr;
		m_active_handle = (m_active_handle + 1) % m_num_handles;
		vx_status status = vxSwapTensorHandle(m_tensor, m_memory_handle[m_active_handle], &prev_ptr);
		if (status)
			ReportError("ERROR: vxSwapTensorHandle(%s,*,*) failed (%d)\n", m_vxObjName, status);
		if (prev_ptr != m_memory_handle[prev_handle])
			ReportError("ERROR: vxSwapTensorHandle(%s,*,*) didn't return correct prev_ptr at [" VX_FMT_SIZE "]\n", m_vxObjName, prev_handle);
	}
	return 0;
}

int CVxParamTensor::ReadFrame(int frameNumber)
{
	// check if there is no user request to read
	if (m_fileNameRead.length() < 1) return 0;

	// for single frame reads, there is no need to read the array again
	// as it is already read into the object
	if (!m_fileNameForReadHasIndex && frameNumber != m_captureFrameStart) {
		return 0;
	}

	// reading data from input file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameRead.c_str(), frameNumber);
	if(!_stricmp(fileName + strlen(fileName) - 4, ".dat")) {
		ReportError("ERROR: read from .dat files not supported: %s\n", fileName);
	}
	FILE * fp = fopen(fileName, m_readFileIsBinary ? "rb" : "r");
	if (!fp) {
		if (frameNumber == m_captureFrameStart) {
			ReportError("ERROR: Unable to open: %s\n", fileName);
		}
		else {
			return 1; // end of sequence detected for multiframe sequences
		}
	}
	if (fread(m_data, 1, m_size, fp) != m_size)
		ReportError("ERROR: not enough data (%d bytes) in %s\n", (vx_uint32)m_size, fileName);
	vx_status status = vxCopyTensorPatch(m_tensor, m_num_of_dims, nullptr, nullptr, m_stride, m_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
	fclose(fp);
	if (status != VX_SUCCESS)
		ReportError("ERROR: vxCopyTensorPatch: write failed (%d)\n", status);

	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_tensor, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamTensor::WriteFrame(int frameNumber)
{
	// check if there is no user request to write
	if (m_fileNameWrite.length() < 1) return 0;
	// read data from tensor
	vx_status status = vxCopyTensorPatch(m_tensor, m_num_of_dims, nullptr, nullptr, m_stride, m_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	if (status != VX_SUCCESS)
		ReportError("ERROR: vxCopyTensorPatch: read failed (%d)\n", status);
	// write data to output file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_writeFileIsBinary ? "wb" : "w");
	if (!fp) ReportError("ERROR: Unable to create: %s\n", fileName);
	if(!_stricmp(fileName + strlen(fileName) - 4, ".dat")) {
		// write NNEF Tensor File Header
		struct HeaderPart1 {
			vx_uint8  magic[2];
			vx_uint8  version_major;
			vx_uint8  version_minor;
			vx_uint32 offset_to_data;
			vx_uint32 num_dims;
		} h1 = {
			{ 0x4e, 0xef }, 1, 0,
			12 + 4 * (vx_uint32)m_num_of_dims + 4,
			(vx_uint32)m_num_of_dims
		};
		vx_uint32 h2[4] = { 0 };
		for(size_t i = 0; i < m_num_of_dims; i++)
			h2[i] = (vx_uint32)m_dims[m_num_of_dims-1-i];
		if(m_num_of_dims == 1) {
			h2[0] = 1;
			h2[1] = (vx_uint32)m_dims[0];
			h1.num_dims += 1;
			h1.offset_to_data += 4;
		}
		else if(m_num_of_dims == 3) {
			h2[3] = h2[2];
			h2[2] = h2[1];
			h2[1] = h2[0];
			h2[0] = 1;
			h1.num_dims += 1;
			h1.offset_to_data += 4;
		}
		struct HeaderPart3 {
			vx_uint8  data_type;
			vx_uint8  bit_width;
			vx_uint16 len_of_quant_string;
		} h3 = {
			0, 32, 0
		};
		if(m_data_type == VX_TYPE_FLOAT32) h3.data_type = 0, h3.bit_width = 32;
		else if(m_data_type == VX_TYPE_FLOAT16) h3.data_type = 0, h3.bit_width = 16;
		else if(m_data_type == VX_TYPE_INT8) h3.data_type = 2, h3.bit_width = 8;
		else if(m_data_type == VX_TYPE_UINT8) h3.data_type = 3, h3.bit_width = 8;
		else if(m_data_type == VX_TYPE_INT16) h3.data_type = 2, h3.bit_width = 16;
		else if(m_data_type == VX_TYPE_UINT16) h3.data_type = 3, h3.bit_width = 16;
		else if(m_data_type == VX_TYPE_INT32) h3.data_type = 2, h3.bit_width = 32;
		else if(m_data_type == VX_TYPE_UINT32) h3.data_type = 3, h3.bit_width = 32;
		else if(m_data_type == VX_TYPE_INT64) h3.data_type = 2, h3.bit_width = 64;
		fwrite(&h1, 1, sizeof(h1), fp);
		fwrite(&h2, 1, h1.num_dims * sizeof(vx_uint32), fp);
		fwrite(&h3, 1, sizeof(h3), fp);
	}
	fwrite(m_data, 1, m_size, fp);
	fclose(fp);

	return 0;
}

int CVxParamTensor::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	if(!_stricmp(fileName + strlen(fileName) - 4, ".dat")) {
		ReportError("ERROR: read from .dat files not supported: %s\n", fileName);
	}
	FILE * fp = fopen(fileName, m_compareFileIsBinary ? "rb" : "r");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	if (fread(m_data, 1, m_size, fp) != m_size)
		ReportError("ERROR: not enough data (%d bytes) in %s\n", (vx_uint32)m_size, fileName);
	fclose(fp);

	// compare
	vx_map_id map_id;
	vx_size stride[MAX_TENSOR_DIMENSIONS];
	vx_uint8 * ptr;
	vx_status status = vxMapTensorPatch(m_tensor, m_num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	if (status != VX_SUCCESS)
		ReportError("ERROR: vxMapTensorPatch: read failed (%d)\n", status);

	bool mismatchDetected = false;
	if (m_data_type == VX_TYPE_INT16) {
		vx_int32 maxError = 0;
		vx_int64 sumError = 0;
		for (vx_size d3 = 0; d3 < m_dims[3]; d3++) {
			for (vx_size d2 = 0; d2 < m_dims[2]; d2++) {
				for (vx_size d1 = 0; d1 < m_dims[1]; d1++) {
					vx_size roffset = m_stride[3] * d3 + m_stride[2] * d2 + m_stride[1] * d1;
					vx_size doffset = stride[3] * d3 + stride[2] * d2 + stride[1] * d1;
					const vx_int16 * buf1 = (const vx_int16 *)(((vx_uint8 *)ptr) + doffset);
					const vx_int16 * buf2 = (const vx_int16 *)(m_data + roffset);
					for (vx_size d0 = 0; d0 < m_dims[0]; d0++) {
						vx_int32 v1 = buf1[d0];
						vx_int32 v2 = buf2[d0];
						vx_int32 d = v1 - v2;
						d = (d < 0) ? -d : d;
						maxError = (d > maxError) ? d : maxError;
						sumError += (long) d * d;
					}
				}
			}
		}
		vx_size count = m_dims[0] * m_dims[1] * m_dims[2] * m_dims[3];
		float avgError = (float)sumError / (float)count;
		mismatchDetected = true;
		if (((float)maxError <= m_maxErrorLimit) && ((float)avgError <= m_avgErrorLimit))
		    mismatchDetected = false;
		if (mismatchDetected)
			printf("ERROR: tensor COMPARE MISMATCHED [max-err: %d] [avg-err: %.6f] for %s with frame#%d of %s\n", maxError, avgError, GetVxObjectName(), frameNumber, fileName);
		else if (m_verbose)
			printf("OK: tensor COMPARE MATCHED [max-err: %d] [avg-err: %.6f] for %s with frame#%d of %s\n", maxError, avgError, GetVxObjectName(), frameNumber, fileName);
	}
	else if (m_data_type == VX_TYPE_FLOAT32) {
		vx_float32 maxError = 0;
		vx_float64 sumError = 0;
		for (vx_size d3 = 0; d3 < m_dims[3]; d3++) {
			for (vx_size d2 = 0; d2 < m_dims[2]; d2++) {
				for (vx_size d1 = 0; d1 < m_dims[1]; d1++) {
					vx_size roffset = m_stride[3] * d3 + m_stride[2] * d2 + m_stride[1] * d1;
					vx_size doffset = stride[3] * d3 + stride[2] * d2 + stride[1] * d1;
					const vx_float32 * buf1 = (const vx_float32 *)(((vx_uint8 *)ptr) + doffset);
					const vx_float32 * buf2 = (const vx_float32 *)(m_data + roffset);
					for (vx_size d0 = 0; d0 < m_dims[0]; d0++) {
						vx_float32 v1 = buf1[d0];
						vx_float32 v2 = buf2[d0];
						vx_float32 d = v1 - v2;
						d = (d < 0) ? -d : d;
						maxError = (d > maxError) ? d : maxError;
						sumError += (double) d * d;
					}
				}
			}
		}
		vx_size count = m_dims[0] * m_dims[1] * m_dims[2] * m_dims[3];
		float avgError = (float)sumError / (float)count;
		mismatchDetected = true;
		if ((maxError <= m_maxErrorLimit) && (avgError <= m_avgErrorLimit))
		    mismatchDetected = false;
		if (mismatchDetected)
			printf("ERROR: tensor COMPARE MISMATCHED [max-err: %.6f] [avg-err: %.6f] for %s with frame#%d of %s\n", maxError, avgError, GetVxObjectName(), frameNumber, fileName);
		else if (m_verbose)
			printf("OK: tensor COMPARE MATCHED [max-err: %.6f] [avg-err: %.6f] for %s with frame#%d of %s\n", maxError, avgError, GetVxObjectName(), frameNumber, fileName);
	}
	else if (m_data_type == VX_TYPE_FLOAT16) {
		vx_float32 maxError = 0;
		vx_float64 sumError = 0;
		for (vx_size d3 = 0; d3 < m_dims[3]; d3++) {
			for (vx_size d2 = 0; d2 < m_dims[2]; d2++) {
				for (vx_size d1 = 0; d1 < m_dims[1]; d1++) {
					vx_size roffset = m_stride[3] * d3 + m_stride[2] * d2 + m_stride[1] * d1;
					vx_size doffset = stride[3] * d3 + stride[2] * d2 + stride[1] * d1;
					const vx_uint16 * buf1 = (const vx_uint16 *)(((vx_uint8 *)ptr) + doffset);
					const vx_uint16 * buf2 = (const vx_uint16 *)(m_data + roffset);
					for (vx_size d0 = 0; d0 < m_dims[0]; d0++) {
						vx_uint16 h1 = buf1[d0];
						vx_uint16 h2 = buf2[d0];
						vx_uint32 d1 = ((h1 & 0x8000) << 16) | (((h1 & 0x7c00) + 0x1c000) << 13) | ((h1 & 0x03ff) << 13);
						vx_uint32 d2 = ((h2 & 0x8000) << 16) | (((h2 & 0x7c00) + 0x1c000) << 13) | ((h2 & 0x03ff) << 13);
						vx_float32 v1 = *(float *)&d1;
						vx_float32 v2 = *(float *)&d2;
						vx_float32 d = v1 - v2;
						d = (d < 0) ? -d : d;
						maxError = (d > maxError) ? d : maxError;
						sumError += (double) d * d;
					}
				}
			}
		}
		vx_size count = m_dims[0] * m_dims[1] * m_dims[2] * m_dims[3];
		float avgError = (float)sumError / (float)count;
		mismatchDetected = true;
		if ((maxError <= m_maxErrorLimit) && (avgError <= m_avgErrorLimit))
		    mismatchDetected = false;
		if (mismatchDetected)
			printf("ERROR: tensor COMPARE MISMATCHED [max-err: %.6f] [avg-err: %.6f] for %s with frame#%d of %s\n", maxError, avgError, GetVxObjectName(), frameNumber, fileName);
		else if (m_verbose)
			printf("OK: tensor COMPARE MATCHED [max-err: %.6f] [avg-err: %.6f] for %s with frame#%d of %s\n", maxError, avgError, GetVxObjectName(), frameNumber, fileName);
	}
	else {
		for (vx_size d3 = 0; d3 < m_dims[3]; d3++) {
			for (vx_size d2 = 0; d2 < m_dims[2]; d2++) {
				for (vx_size d1 = 0; d1 < m_dims[1]; d1++) {
					vx_size roffset = m_stride[3] * d3 + m_stride[2] * d2 + m_stride[1] * d1;
					vx_size doffset = stride[3] * d3 + stride[2] * d2 + stride[1] * d1;
					if (memcpy(((vx_uint8 *)ptr) + doffset, m_data + roffset, stride[0] * m_dims[0])) {
						mismatchDetected = true;
						break;
					}
				}
				if (mismatchDetected)
					break;
			}
			if (mismatchDetected)
				break;
		}
		if (mismatchDetected)
			printf("ERROR: tensor COMPARE MISMATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
		else if (m_verbose) 
			printf("OK: tensor COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}

	status = vxUnmapTensorPatch(m_tensor, map_id);
	if (status != VX_SUCCESS)
		ReportError("ERROR: vxUnmapTensorPatch: read failed (%d)\n", status);

	// report error if mismatched
	if (mismatchDetected) {
		m_compareCountMismatches++;
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
	}

	return 0;
}
