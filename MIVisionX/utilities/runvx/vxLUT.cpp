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
#include "vxLUT.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamLUT
//
CVxParamLUT::CVxParamLUT()
{
	// vx configuration
	m_vxObjType = VX_TYPE_LUT;
	m_data_type = VX_TYPE_UINT8;
	m_count = 0;
	// I/O configuration
	m_readFileIsBinary = false;
	m_writeFileIsBinary = false;
	m_compareFileIsBinary = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	// vx object
	m_lut = nullptr;
	m_vxObjRef = nullptr;
}

CVxParamLUT::~CVxParamLUT()
{
	Shutdown();
}

int CVxParamLUT::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: lut COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	if (m_lut) {
		vxReleaseLUT(&m_lut);
		m_lut = nullptr;
	}
	return 0;
}

int CVxParamLUT::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64], data_type[64];
	const char * ioParams = ScanParameters(desc, "lut:<data-type>,<count>", "s:s,D", objType, data_type, &m_count);
	if (!_stricmp(objType, "lut")) {
		m_data_type = ovxName2Enum(data_type);
		m_lut = vxCreateLUT(context, m_data_type, m_count);
	}
	else ReportError("ERROR: unsupported lut type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_lut);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: lut creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_lut) vxReleaseLUT(&m_lut);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_lut;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamLUT::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_lut = (vx_lut)m_vxObjRef;
	ERROR_CHECK(vxQueryLUT(m_lut, VX_LUT_ATTRIBUTE_TYPE, &m_data_type, sizeof(m_data_type)));
	ERROR_CHECK(vxQueryLUT(m_lut, VX_LUT_ATTRIBUTE_COUNT, &m_count, sizeof(m_count)));

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read"))
		{ // read request syntax: read,<fileName>[,ascii|binary]
			m_fileNameRead.assign(RootDirUpdated(fileName));
			m_fileNameForReadHasIndex = (m_fileNameRead.find("%") != m_fileNameRead.npos) ? true : false;
			m_readFileIsBinary = (m_fileNameRead.find(".txt") != m_fileNameRead.npos) ? false : true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",ascii|binary", ",s", option);
				if (!_stricmp(option, "ascii")) {
					m_readFileIsBinary = false;
				}
				else if (!_stricmp(option, "binary")) {
					m_readFileIsBinary = true;
				}
				else ReportError("ERROR: invalid lut read option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "write"))
		{ // write request syntax: write,<fileName>[,ascii|binary]
			m_fileNameWrite.assign(RootDirUpdated(fileName));
			m_writeFileIsBinary = (m_fileNameWrite.find(".txt") != m_fileNameWrite.npos) ? false : true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",ascii|binary", ",s", option);
				if (!_stricmp(option, "ascii")) {
					m_writeFileIsBinary = false;
				}
				else if (!_stricmp(option, "binary")) {
					m_writeFileIsBinary = true;
				}
				else ReportError("ERROR: invalid lut write option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // write request syntax: compare,<fileName>[,ascii|binary]
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			m_compareFileIsBinary = (m_fileNameCompare.find(".txt") != m_fileNameCompare.npos) ? false : true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",ascii|binary", ",s", option);
				if (!_stricmp(option, "ascii")) {
					m_compareFileIsBinary = false;
				}
				else if (!_stricmp(option, "binary")) {
					m_compareFileIsBinary = true;
				}
				else ReportError("ERROR: invalid lut compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "view")) {
			m_displayName.assign(fileName);
			m_paramList.push_back(this);
		}
		else if (!_stricmp(ioType, "directive") && (!_stricmp(fileName, "VX_DIRECTIVE_AMD_COPY_TO_OPENCL") || !_stricmp(fileName, "sync-cl-write"))) {
			m_useSyncOpenCLWriteDirective = true;
		}
		else ReportError("ERROR: invalid lut operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamLUT::Finalize()
{
	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_lut, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamLUT::ReadFrame(int frameNumber)
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
	FILE * fp = fopen(fileName, m_readFileIsBinary ? "rb" : "r");
	if (!fp) {
		if (frameNumber == m_captureFrameStart) {
			ReportError("ERROR: Unable to open: %s\n", fileName);
		}
		else {
			return 1; // end of sequence detected for multiframe sequences
		}
	}
	vx_size size; ERROR_CHECK(vxQueryLUT(m_lut, VX_LUT_ATTRIBUTE_SIZE, &size, sizeof(size)));
	vx_uint8 * data = nullptr; ERROR_CHECK(vxAccessLUT(m_lut, (void **)&data, VX_WRITE_ONLY));
	int status = 0;
	if (m_readFileIsBinary) {
		if (fread(data, 1, size, fp) != size)
			status = -1;
	}
	else {
		vx_size itemsize = size / m_count;
		for (vx_uint32 x = 0; x < m_count; x++) {
			vx_uint32 value;
			if (fscanf(fp, "%i", &value) != 1) {
				status = -1;
				break;
			}
			memcpy(&data[x * itemsize], &value, itemsize);
		}
	}
	ERROR_CHECK(vxCommitLUT(m_lut, data));
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on lut input file: %s\n", fileName);

	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_lut, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return status;
}

int CVxParamLUT::WriteFrame(int frameNumber)
{
	// check if there is no user request to write
	if (m_fileNameWrite.length() < 1) return 0;
	// write data to output file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_writeFileIsBinary ? "wb" : "w");
	if (!fp) ReportError("ERROR: Unable to create: %s\n", fileName);
	vx_size size; ERROR_CHECK(vxQueryLUT(m_lut, VX_LUT_ATTRIBUTE_SIZE, &size, sizeof(size)));
	vx_uint8 * data = nullptr; ERROR_CHECK(vxAccessLUT(m_lut, (void **)&data, VX_READ_ONLY));
	if (m_writeFileIsBinary) {
		fwrite(data, 1, size, fp);
	}
	else {
		vx_size itemsize = size / m_count;
		for (vx_uint32 x = 0; x < m_count; x++) {
			char value[MAX_SCALAR_TO_STRING_SIZE]; 
			PutScalarValueToString(m_data_type, &data[x * itemsize], value);
			fprintf(fp, "%s\n", value);
		}
	}
	ERROR_CHECK(vxCommitLUT(m_lut, data));
	fclose(fp);

	return 0;
}

int CVxParamLUT::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_compareFileIsBinary ? "rb" : "r");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	vx_size size; ERROR_CHECK(vxQueryLUT(m_lut, VX_LUT_ATTRIBUTE_SIZE, &size, sizeof(size)));
	vx_size itemsize = size / m_count;
	vx_uint8 * data = nullptr; ERROR_CHECK(vxAccessLUT(m_lut, (void **)&data, VX_WRITE_ONLY));
	int status = 0;
	bool mismatchDetected = false;
	for (vx_uint32 x = 0; x < m_count; x++) {
		vx_uint32 value;
		if (m_compareFileIsBinary) {
			if (fread(&value, itemsize, 1, fp) != 1) {
				status = -1;
				break;
			}
		}
		else {
			if (fscanf(fp, "%i", &value) != 1) {
				status = -1;
				break;
			}
		}
		if (memcmp(&data[x * itemsize], &value, itemsize) != 0) {
			mismatchDetected = true;
			break;
		}
	}
	ERROR_CHECK(vxCommitLUT(m_lut, data));
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on lut reference file: %s\n", fileName);

	if (mismatchDetected) {
		m_compareCountMismatches++;
		printf("ERROR: lut COMPARE MISMATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
		if (m_verbose) printf("OK: lut COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}

	return 0;
}
