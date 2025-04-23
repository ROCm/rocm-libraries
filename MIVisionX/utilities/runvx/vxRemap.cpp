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
#include "vxRemap.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamRemap
//
CVxParamRemap::CVxParamRemap()
{
	// vx configuration
	m_vxObjType = VX_TYPE_REMAP;
	m_srcWidth = 0;
	m_srcHeight = 0;
	m_dstWidth = 0;
	m_dstHeight = 0;
	// I/O configuration
	m_readFileIsBinary = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	m_xyErr[0] = m_xyErr[1] = 0.0f;
	// vx object
	m_remap = nullptr;
}

CVxParamRemap::~CVxParamRemap()
{
	Shutdown();
}

int CVxParamRemap::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: remap COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	if (m_remap) {
		vxReleaseRemap(&m_remap);
		m_remap = nullptr;
	}
	return 0;
}

int CVxParamRemap::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	//   syntax: remap:<srcWidth>,<srcHeight>,<dstWidth>,<dstHeight>[:<io-params>]
	char objType[64];
	const char * ioParams = ScanParameters(desc, "remap:<srcWidth>,<srcHeight>,<dstWidth>,<dstHeight>", "s:d,d,d,d", objType, &m_srcWidth, &m_srcHeight, &m_dstWidth, &m_dstHeight);
	if (!_stricmp(objType, "remap")) {
		m_remap = vxCreateRemap(context, m_srcWidth, m_srcHeight, m_dstWidth, m_dstHeight);
	}
	else ReportError("ERROR: invalid remap type: %s\n", objType);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_remap);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: pyramid creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_remap) vxReleaseRemap(&m_remap);
		throw - 1;
	}

	// io initialize
	return InitializeIO(context, graph, (vx_reference)m_remap, ioParams);
}

int CVxParamRemap::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_remap = (vx_remap)m_vxObjRef;
	ERROR_CHECK(vxQueryRemap(m_remap, VX_REMAP_ATTRIBUTE_SOURCE_WIDTH, &m_srcWidth, sizeof(m_srcWidth)));
	ERROR_CHECK(vxQueryRemap(m_remap, VX_REMAP_ATTRIBUTE_SOURCE_HEIGHT, &m_srcHeight, sizeof(m_srcHeight)));
	ERROR_CHECK(vxQueryRemap(m_remap, VX_REMAP_ATTRIBUTE_DESTINATION_WIDTH, &m_dstWidth, sizeof(m_dstWidth)));
	ERROR_CHECK(vxQueryRemap(m_remap, VX_REMAP_ATTRIBUTE_DESTINATION_HEIGHT, &m_dstHeight, sizeof(m_dstHeight)));

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read")) {
			m_fileNameRead.assign(RootDirUpdated(fileName));
			m_usingMultiFrameCapture = (m_fileNameRead.find("%") != std::string::npos) ? true : false;
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
				else ReportError("ERROR: invalid remap read option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "write")) {
			m_fileNameWrite.assign(RootDirUpdated(fileName));
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare request syntax: compare,<fileName>[,err{<x>;<y>}]
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",err{<x>;<y>}", ",s", option);
				if (!_strnicmp(option, "err{", 4)) {
					ScanParameters(&option[3], "{<errX>;<errY>}", "{f;f}", &m_xyErr[0], &m_xyErr[1]);
				}
				else ReportError("ERROR: invalid remap compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "directive") && (!_stricmp(fileName, "VX_DIRECTIVE_AMD_COPY_TO_OPENCL") || !_stricmp(fileName, "sync-cl-write"))) {
			m_useSyncOpenCLWriteDirective = true;
		}
		else if (!_stricmp(ioType, "init")) {
			const char * patternName = fileName;
			if (!_stricmp(patternName, "same")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (vx_float32)x, sy = (vx_float32)y;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else if (!_stricmp(patternName, "rotate-90")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (vx_float32)m_dstHeight - 1 - y, sy = (vx_float32)x;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else if (!_stricmp(patternName, "rotate-180")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (vx_float32)m_dstWidth - 1 - x, sy = (vx_float32)m_dstHeight - 1 - y;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else if (!_stricmp(patternName, "rotate-270")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (vx_float32)y, sy = (vx_float32)m_dstWidth - 1 - x;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else if (!_stricmp(patternName, "scale")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (x + 0.5f) * (vx_float32)m_srcWidth / (vx_float32)m_dstWidth - 0.5f;
						vx_float32 sy = (y + 0.5f) * (vx_float32)m_srcHeight / (vx_float32)m_dstHeight - 0.5f;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else if (!_stricmp(patternName, "hflip")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (vx_float32)m_dstWidth - 1 - x, sy = (vx_float32)y;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else if (!_stricmp(patternName, "vflip")) {
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 sx = (vx_float32)x, sy = (vx_float32)m_dstHeight - 1 - y;
						vx_status status = vxSetRemapPoint(m_remap, x, y, sx, sy);
						if (status) {
							printf("ERROR: vxSetRemapPoint(*,%d,%d,%g,%g) failed, status = %d\n", x, y, sx, sy, status);
							return -1;
						}
					}
				}
			}
			else {
				// initialize from binary file
				FILE * fp = fopen(fileName, "rb");
				if (!fp) ReportError("ERROR: CVxParamRemap::InitializeIO: unable to open: %s\n", fileName);
				for (vx_uint32 y = 0; y < m_dstHeight; y++){
					for (vx_uint32 x = 0; x < m_dstWidth; x++){
						vx_float32 src_xy[2];
						if (fread(src_xy, sizeof(src_xy), 1, fp) != 1)
							ReportError("ERROR: detected EOF at (%d,%d) on remap input file: %s\n", x, y, fileName);
						ERROR_CHECK(vxSetRemapPoint(m_remap, x, y, src_xy[0], src_xy[1]));
					}
				}
				fclose(fp);
			}
		}
		else {
			printf("ERROR: invalid remap I/O operation: %s\n", ioType);
			return -1;
		}
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}
	return 0;
}

int CVxParamRemap::Finalize()
{
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_remap, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}
	return 0;
}

int CVxParamRemap::ReadFrame(int frameNumber)
{
	if (m_fileNameRead.length() < 1) return 0;

	if (!m_usingMultiFrameCapture && frameNumber != m_captureFrameStart) {
		// for single frame reads, there is no need to read the array again
		// as it is already read into the object
		return 0;
	}

	// read from user specified file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameRead.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_readFileIsBinary ? "rb" : "r");
	if (!fp) ReportError("ERROR: unable to open: %s\n", fileName);
	for (vx_uint32 y = 0; y < m_dstHeight; y++){
		for (vx_uint32 x = 0; x < m_dstWidth; x++){
			vx_float32 src_xy[2];
			if (m_readFileIsBinary) {
				if (fread(src_xy, sizeof(src_xy), 1, fp) != 1)
					ReportError("ERROR: detected EOF at (%d,%d) on remap input file: %s\n", x, y, fileName);
			}
			else {
				if (fscanf(fp, "%g%g", &src_xy[0], &src_xy[1]) != 2)
					ReportError("ERROR: detected EOF at (%d,%d) on remap input file: %s (ASCII)\n", x, y, fileName);
			}
			ERROR_CHECK(vxSetRemapPoint(m_remap, x, y, src_xy[0], src_xy[1]));
		}
	}
	fclose(fp);

	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_remap, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamRemap::WriteFrame(int frameNumber)
{
	if (m_fileNameWrite.length() < 1) return 0;

	// write output into user specified file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, "wb");
	if (!fp) ReportError("ERROR: unable to create: %s\n", fileName);
	for (vx_uint32 y = 0; y < m_dstHeight; y++){
		for (vx_uint32 x = 0; x < m_dstWidth; x++){
			vx_float32 src_xy[2];
			ERROR_CHECK(vxGetRemapPoint(m_remap, x, y, &src_xy[0], &src_xy[1]));
			fwrite(src_xy, sizeof(src_xy), 1, fp);
		}
	}
	fclose(fp);

	return 0;
}

int CVxParamRemap::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	FILE * fp = fopen(fileName, "rb");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	bool mismatchDetected = false;
	int status = 0;
	for (vx_uint32 y = 0; y < m_dstHeight; y++){
		for (vx_uint32 x = 0; x < m_dstWidth; x++){
			vx_float32 xy[2];
			ERROR_CHECK(vxGetRemapPoint(m_remap, x, y, &xy[0], &xy[1]));
			vx_float32 xyRef[2];
			if (fread(xyRef, sizeof(xyRef), 1, fp) != 1) {
				status = -1;
				break;
			}
			if (fabsf(xy[0] - xyRef[0]) > m_xyErr[0] || fabsf(xy[1] - xyRef[1]) > m_xyErr[1]) {
				mismatchDetected = true;
				break;
			}
		}
		if (status || mismatchDetected)
			break;
	}
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on remap comapre reference file: %s\n", fileName);

	if (mismatchDetected) {
		m_compareCountMismatches++;
		printf("ERROR: remap COMPARE MISMATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
		if (m_verbose) printf("OK: remap COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}

	return 0;
}
