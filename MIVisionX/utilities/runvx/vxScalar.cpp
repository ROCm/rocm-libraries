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
#include "vxScalar.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamScalar
//
CVxParamScalar::CVxParamScalar()
{
	// vx configuration
	m_vxObjType = VX_TYPE_SCALAR;
	m_format = VX_TYPE_INVALID;
	// I/O configuration
	m_compareRangeInRef = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	// vx object
	m_scalar = nullptr;
}

CVxParamScalar::~CVxParamScalar()
{
	Shutdown();
}

int CVxParamScalar::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: scalar COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	GuiTrackBarShutdown((vx_reference)m_scalar);
	if (m_scalar){
		vxReleaseScalar(&m_scalar);
		m_scalar = nullptr;
	}
	return 0;
}

int CVxParamScalar::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64], format[64], value[256];
	const char * ioParams = ScanParameters(desc, "scalar:<type>,<value>", "s:s,S", objType, format, value);
	if (!_stricmp(objType, "scalar")) {
		m_format = ovxName2Enum(format);
		if (m_format == VX_TYPE_STRING_AMD) {
			m_scalar = vxCreateScalar(context, m_format, value);
		}
		else if (m_format == (VX_TYPE_NN_CONVOLUTION_PARAMS) || m_format == (VX_TYPE_NN_DECONVOLUTION_PARAMS) ||
                 m_format == (VX_TYPE_NN_ROI_POOL_PARAMS) || m_format == (VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS))
        {
			if (m_format == VX_TYPE_NN_CONVOLUTION_PARAMS) {
				vx_nn_convolution_params_t v;
				if (!GetScalarValueForStructTypes(m_format, value, &v)) {
					m_scalar= vxCreateScalar(context, m_format, &v);
				}
			}
			else if (m_format == VX_TYPE_NN_DECONVOLUTION_PARAMS) {
				vx_nn_deconvolution_params_t v;
				if (!GetScalarValueForStructTypes(m_format, value, &v)) {
					m_scalar = vxCreateScalar(context, m_format, &v);
				}
			}
			else if (m_format == VX_TYPE_NN_ROI_POOL_PARAMS) {
				vx_nn_roi_pool_params_t v;
				if (!GetScalarValueForStructTypes(m_format, value, &v)) {
					m_scalar = vxCreateScalar(context, m_format, &v);
				}
			}
			else if (m_format == VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS) {
				vx_tensor_matrix_multiply_params_t v;
				if (!GetScalarValueForStructTypes(m_format, value, &v)) {
					m_scalar = vxCreateScalar(context, m_format, &v);
				}
			}
			else ReportError("ERROR: unsupported scalar value: %s [%s:0x%08x]\n", value, format, m_format);
		}
		else {
			vx_uint64 v = 0;
			if (!GetScalarValueFromString(m_format, value, &v)) {
				m_scalar = vxCreateScalar(context, m_format, &v);
			}
			else ReportError("ERROR: unsupported scalar value: %s [%s:0x%08x]\n", value, format, m_format);
		}
	}
	else ReportError("ERROR: unsupported scalar type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_scalar);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: scalar creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_scalar) vxReleaseScalar(&m_scalar);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_scalar;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamScalar::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_scalar = (vx_scalar)m_vxObjRef;
	ERROR_CHECK(vxQueryScalar(m_scalar, VX_SCALAR_ATTRIBUTE_TYPE, &m_format, sizeof(m_format)));

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read"))
		{ // read request syntax: read,<fileName>
			// close files if already open
			if (m_fpRead) {
				fclose(m_fpRead);
				m_fpRead = nullptr;
			}
			m_fileNameRead.assign(RootDirUpdated(fileName));
			if (*io_params == ',') {
				ReportError("ERROR: invalid scalar read option: %s\n", io_params);
			}
		}
		else if (!_stricmp(ioType, "write"))
		{ // write request syntax: write,<fileName>
			if (m_fpWrite) {
				fclose(m_fpWrite);
				m_fpWrite = nullptr;
			}
			m_fileNameWrite.assign(RootDirUpdated(fileName));
			if (*io_params == ',') {
				ReportError("ERROR: invalid scalar read option: %s\n", io_params);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare syntax: compare,fileName[,range]
			if (m_fpCompare) {
				fclose(m_fpCompare);
				m_fpCompare = nullptr;
			}
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",range", ",s", option);
				if (!_stricmp(option, "range")) {
					m_compareRangeInRef = true;
				}
				else ReportError("ERROR: invalid scalar compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "view")) {
			m_displayName.assign(fileName);
			m_paramList.push_back(this);
		}
		else if (!_stricmp(ioType, "ui") && (m_format == VX_TYPE_FLOAT32)) {
			int id = 0;
			float valueMin = 0.0f, valueMax = 1.0f, valueInc = 0.1f;
			ScanParameters(fileName, "{<id>;<min>;<max>;<inc>}", "{d;f;f;f}", &id, &valueMin, &valueMax, &valueInc);
			GuiTrackBarInitializeScalar((vx_reference)m_scalar, id-1, valueMin, valueMax, valueInc);
		}
		else ReportError("ERROR: invalid scalar operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamScalar::Finalize()
{
	// get attributes
	ERROR_CHECK(vxQueryScalar(m_scalar, VX_SCALAR_ATTRIBUTE_TYPE, &m_format, sizeof(m_format)));

	return 0;
}

int CVxParamScalar::ReadFrame(int frameNumber)
{
	// check if there is no user request to read
	if (m_fileNameRead.length() < 1) return 0;

	// make sure to open the input file
	if (!m_fpRead) {
		const char * fileName = m_fileNameRead.c_str();
		if (!(m_fpRead = fopen(fileName, "r")))
			ReportError("ERROR: unable to open: %s\n", fileName);
	}

	// read the next word and set the scalar value
	char str[256];
	if (fscanf(m_fpRead, "%s", str) != 1) {
		// end of file reached
		return 1;
	}
	return WriteScalarFromString(m_scalar, str);
}

int CVxParamScalar::WriteFrame(int frameNumber)
{
	// check if there is no user request to write
	if (m_fileNameWrite.length() < 1) return 0;

	// make sure to create the output file
	if (!m_fpWrite) {
		const char * fileName = m_fileNameWrite.c_str();
		if (!(m_fpWrite = fopen(fileName, "w")))
			ReportError("ERROR: unable to create: %s\n", fileName);
	}

	// write scalar value
	char str[MAX_SCALAR_TO_STRING_SIZE];
	if (ReadScalarToString(m_scalar, str) < 0)
		return -1;
	fprintf(m_fpWrite, "%s\n", str);

	return 0;
}

int CVxParamScalar::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// make sure to open the input file
	if (!m_fpCompare) {
		const char * fileName = m_fileNameCompare.c_str();
		if (!(m_fpCompare = fopen(fileName, "r")))
			ReportError("ERROR: unable to open: %s\n", fileName);
	}

	// read the next item for compare
	char strMin[256], strMax[256];
	vx_uint64 valueRefMin = 0, valueRefMax = 0;
	if (!m_compareRangeInRef) {
		// read one value and set it as min as well as max
		if (fscanf(m_fpCompare, "%s", strMin) != 1)
			ReportError("ERROR: compare: missing data item for %s\n", GetVxObjectName());
		if (GetScalarValueFromString(m_format, strMin, &valueRefMin) < 0)
			ReportError("ERROR: compare: invalid data item for %s: %s\n", GetVxObjectName(), strMin);
		valueRefMax = valueRefMin;
		strcpy(strMax, strMin);
	}
	else {
		// read min and max values for range compare
		if (fscanf(m_fpCompare, "%s%s", strMin, strMax) != 2)
			ReportError("ERROR: compare: missing data item for %s\n", GetVxObjectName());
		if (GetScalarValueFromString(m_format, strMin, &valueRefMin) < 0)
			ReportError("ERROR: compare: invalid data item for %s: %s\n", GetVxObjectName(), strMin);
		if (GetScalarValueFromString(m_format, strMax, &valueRefMax) < 0)
			ReportError("ERROR: compare: invalid data item for %s: %s\n", GetVxObjectName(), strMax);
	}
	// compare the value to be within the range
	vx_uint64 value = 0;
	ERROR_CHECK(vxReadScalarValue(m_scalar, &value));
	bool mismatchDetected = true;
	if (((m_format == VX_TYPE_FLOAT32) && (*(vx_float32 *)&value >= *(vx_float32 *)&valueRefMin) && (*(vx_float32 *)&value <= *(vx_float32 *)&valueRefMax))
		|| ((m_format == VX_TYPE_FLOAT64) && (*(vx_float64 *)&value >= *(vx_float64 *)&valueRefMin) && (*(vx_float64 *)&value <= *(vx_float64 *)&valueRefMax))
		|| ((m_format == VX_TYPE_DF_IMAGE) && (*(vx_df_image *)&value >= *(vx_df_image *)&valueRefMin) && (*(vx_df_image *)&value <= *(vx_df_image *)&valueRefMax))
		|| ((m_format == VX_TYPE_SIZE) && (*(vx_size *)&value >= *(vx_size *)&valueRefMin) && (*(vx_size *)&value <= *(vx_size *)&valueRefMax))
		|| ((m_format == VX_TYPE_ENUM) && (*(vx_enum *)&value >= *(vx_enum *)&valueRefMin) && (*(vx_enum *)&value <= *(vx_enum *)&valueRefMax))
		|| ((m_format == VX_TYPE_BOOL) && (*(vx_bool *)&value >= *(vx_bool *)&valueRefMin) && (*(vx_bool *)&value <= *(vx_bool *)&valueRefMax))
		|| ((m_format == VX_TYPE_UINT64) && (*(vx_uint64 *)&value >= *(vx_uint64 *)&valueRefMin) && (*(vx_uint64 *)&value <= *(vx_uint64 *)&valueRefMax))
		|| ((m_format == VX_TYPE_UINT32) && (*(vx_uint32 *)&value >= *(vx_uint32 *)&valueRefMin) && (*(vx_uint32 *)&value <= *(vx_uint32 *)&valueRefMax))
		|| ((m_format == VX_TYPE_UINT16) && (*(vx_uint16 *)&value >= *(vx_uint16 *)&valueRefMin) && (*(vx_uint16 *)&value <= *(vx_uint16 *)&valueRefMax))
		|| ((m_format == VX_TYPE_UINT8) && (*(vx_uint8 *)&value >= *(vx_uint8 *)&valueRefMin) && (*(vx_uint8 *)&value <= *(vx_uint8 *)&valueRefMax))
		|| ((m_format == VX_TYPE_INT64) && (*(vx_int64 *)&value >= *(vx_int64 *)&valueRefMin) && (*(vx_int64 *)&value <= *(vx_int64 *)&valueRefMax))
		|| ((m_format == VX_TYPE_INT32) && (*(vx_int32 *)&value >= *(vx_int32 *)&valueRefMin) && (*(vx_int32 *)&value <= *(vx_int32 *)&valueRefMax))
		|| ((m_format == VX_TYPE_INT16) && (*(vx_int16 *)&value >= *(vx_int16 *)&valueRefMin) && (*(vx_int16 *)&value <= *(vx_int16 *)&valueRefMax))
		|| ((m_format == VX_TYPE_INT8) && (*(vx_int8 *)&value >= *(vx_int8 *)&valueRefMin) && (*(vx_int8 *)&value <= *(vx_int8 *)&valueRefMax))
		|| ((m_format == VX_TYPE_CHAR) && (*(vx_char *)&value >= *(vx_char *)&valueRefMin) && (*(vx_char *)&value <= *(vx_char *)&valueRefMax)))
	{
		mismatchDetected = false;
	}

	char str[MAX_SCALAR_TO_STRING_SIZE];
	ReadScalarToString(m_scalar, str);
	if (mismatchDetected) {
		m_compareCountMismatches++;
		printf("ERROR: scalar COMPARE MISMATCHED for %s with frame#%d: %s in [%s .. %s]\n", GetVxObjectName(), frameNumber, str, strMin, strMax);
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
		if (m_verbose) printf("OK: scalar COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, str);
	}

	return 0;
}
