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
#include "vxThreshold.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamThreshold
//
CVxParamThreshold::CVxParamThreshold()
{
	// vx configuration
	m_vxObjType = VX_TYPE_THRESHOLD;
	m_thresh_type = VX_THRESHOLD_TYPE_BINARY;
	m_input_format = VX_DF_IMAGE_U8;
	m_output_format = VX_DF_IMAGE_U8;
	// vx object
	m_threshold = nullptr;
}

CVxParamThreshold::~CVxParamThreshold()
{
	Shutdown();
}

int CVxParamThreshold::Shutdown(void)
{
	if (m_threshold) {
		vxReleaseThreshold(&m_threshold);
		m_threshold = nullptr;
	}
	return 0;
}

int CVxParamThreshold::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64], thresh_type[64];
	const char * ioParams = ScanParameters(desc, "threshold:<thresh-type>,<input-image-format>,<output-image-format>", "s:s,c,c", objType, thresh_type, &m_input_format, &m_output_format);
	if (!_stricmp(objType, "threshold")) {
		m_thresh_type = ovxName2Enum(thresh_type);
		m_threshold = vxCreateThresholdForImage(context, m_thresh_type, m_input_format, m_output_format);
	}
	else ReportError("ERROR: unsupported threshold type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_threshold);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: threshold creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_threshold) vxReleaseThreshold(&m_threshold);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_threshold;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamThreshold::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_threshold = (vx_threshold)m_vxObjRef;
	ERROR_CHECK(vxQueryThreshold(m_threshold, VX_THRESHOLD_ATTRIBUTE_TYPE, &m_thresh_type, sizeof(m_thresh_type)));

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read"))
		{ // read request syntax: read,<fileName>
			if (m_fpRead) {
				fclose(m_fpRead);
				m_fpRead = nullptr;
			}
			m_fileNameRead.assign(RootDirUpdated(fileName));
			if (*io_params == ',') {
				ReportError("ERROR: invalid threshold read option: %s\n", io_params);
			}
		}
		else if (!_stricmp(ioType, "init")) {
			if (m_thresh_type == VX_THRESHOLD_TYPE_RANGE) {
				vx_int32 lower = 0, upper = 0;
				if (fileName[0] == '{') {
					ScanParameters(fileName, "{<threshold-lower>;<threshold-upper>}", "{d;d}", &lower, &upper);
				}
				else {
					ScanParameters(fileName, "<threshold-lower>", "d", &lower);
					io_params = ScanParameters(io_params, ",<threshold-upper>", ",d", &upper);
				}
				ERROR_CHECK(vxSetThresholdAttribute(m_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(vx_int32)));
				ERROR_CHECK(vxSetThresholdAttribute(m_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(vx_int32)));
			}
			else {
				vx_int32 value = 0;
				ScanParameters(fileName, "<threshold-value>", "d", &value);
				ERROR_CHECK(vxSetThresholdAttribute(m_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &value, sizeof(vx_int32)));
			}
		}
		else ReportError("ERROR: invalid threshold operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamThreshold::Finalize()
{
	return 0;
}

int CVxParamThreshold::ReadFrame(int frameNumber)
{
	// check if there is no user request to read
	if (m_fileNameRead.length() < 1) return 0;

	// make sure to open the input file
	if (!m_fpRead) {
		const char * fileName = m_fileNameRead.c_str();
		if (!(m_fpRead = fopen(fileName, "r")))
			ReportError("ERROR: unable to open: %s\n", fileName);
	}

	// read the next word(s) and set the threshold
	if (m_thresh_type == VX_THRESHOLD_TYPE_RANGE) {
		vx_int32 lower = 0, upper = 0;
		if (fscanf(m_fpRead, "%i%i", &lower, &upper) != 2)
			return 1; // end of file reached
		ERROR_CHECK(vxSetThresholdAttribute(m_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(vx_int32)));
		ERROR_CHECK(vxSetThresholdAttribute(m_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(vx_int32)));
	}
	else {
		vx_int32 value = 0;
		if (fscanf(m_fpRead, "%i", &value) != 1)
			return 1; // end of file reached
		ERROR_CHECK(vxSetThresholdAttribute(m_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &value, sizeof(vx_int32)));
	}

	return 0;
}

int CVxParamThreshold::WriteFrame(int frameNumber)
{
	return 0;
}

int CVxParamThreshold::CompareFrame(int frameNumber)
{
	return 0;
}
