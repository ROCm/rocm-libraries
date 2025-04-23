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
#include "vxConvolution.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamConvolution
//
CVxParamConvolution::CVxParamConvolution()
{
	// vx configuration
	m_vxObjType = VX_TYPE_CONVOLUTION;
	m_columns = 0;
	m_rows = 0;
	m_scale = 0;
	// I/O configuration
	m_readFileIsBinary = false;
	m_writeFileIsBinary = false;
	m_compareFileIsBinary = false;
	m_readFileWithScale = false;
	m_writeFileWithScale = false;
	m_compareFileWithScale = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	m_bufForAccess = nullptr;
	// vx object
	m_convolution = nullptr;
}

CVxParamConvolution::~CVxParamConvolution()
{
	Shutdown();
}

int CVxParamConvolution::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: convolution COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	if (m_convolution) {
		vxReleaseConvolution(&m_convolution);
		m_convolution = nullptr;
	}
	if (m_bufForAccess) {
		delete[] m_bufForAccess;
		m_bufForAccess = nullptr;
	}
	return 0;
}

int CVxParamConvolution::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64];
	const char * ioParams = ScanParameters(desc, "convolution:<columns>,<rows>", "s:D,D", objType, &m_columns, &m_rows);
	if (!_stricmp(objType, "convolution")) {
		m_convolution = vxCreateConvolution(context, m_columns, m_rows);
	}
	else ReportError("ERROR: unsupported convolution type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_convolution);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: convolution creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_convolution) vxReleaseConvolution(&m_convolution);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_convolution;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamConvolution::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_convolution = (vx_convolution)m_vxObjRef;
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_COLUMNS, &m_columns, sizeof(m_columns)));
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_ROWS, &m_rows, sizeof(m_rows)));
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read"))
		{ // read request syntax: read,<fileName>[,ascii|binary|scale]
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
				else if (!_stricmp(option, "scale")) {
					m_readFileWithScale = true;
				}
				else ReportError("ERROR: invalid convolution read option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "write"))
		{ // write request syntax: write,<fileName>[,ascii|binary|scale]
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
				else if (!_stricmp(option, "scale")) {
					m_writeFileWithScale = true;
				}
				else ReportError("ERROR: invalid convolution write option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare request syntax: compare,<fileName>[,ascii|binary|scale]
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
				else if (!_stricmp(option, "scale")) {
					m_compareFileWithScale = true;
				}
				else ReportError("ERROR: invalid convolution compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "scale"))
		{ // write request syntax: scale,<scale>
			ScanParameters(fileName, "<scale>", "d", &m_scale);
			ERROR_CHECK(vxSetConvolutionAttribute(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
		}
		else if (!_stricmp(ioType, "init"))
		{ // write request syntax: init,{<value1>;<value2>;...<valueN>}|scale{<scale>}
			if (!_strnicmp(fileName, "scale{", 6)) {
				ScanParameters(&fileName[5], "{<scale>}", "{d}", &m_scale);
				ERROR_CHECK(vxSetConvolutionAttribute(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
			}
			else {
				NULLPTR_CHECK(m_bufForAccess = new vx_int16[m_columns * m_rows]);
				vx_size index = 0; char fmt[3] = "{d";
				for (const char * s = fileName; *s && index < (m_columns * m_rows); fmt[0] = ';', index++) {
					vx_uint32 value;
					s = ScanParameters(s, "<value>", fmt, &value);
					m_bufForAccess[index] = value;
				}
				if (index < (m_columns * m_rows)) ReportError("ERROR: convolution init have too few values: %s\n", fileName);
				ERROR_CHECK(vxWriteConvolutionCoefficients(m_convolution, m_bufForAccess));
			}
		}
		else if (!_stricmp(ioType, "directive") && !_stricmp(fileName, "readonly")) {
			ERROR_CHECK(vxDirective((vx_reference)m_convolution, VX_DIRECTIVE_AMD_READ_ONLY));
		}
		else ReportError("ERROR: invalid convolution operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamConvolution::Finalize()
{
	// get object attributes
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_COLUMNS, &m_columns, sizeof(m_columns)));
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_ROWS, &m_rows, sizeof(m_rows)));
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
	return 0;
}

int CVxParamConvolution::ReadFrame(int frameNumber)
{
	// check if there is no user request to read
	if (m_fileNameRead.length() < 1) return 0;

	// make sure buffer has been allocated
	if (!m_bufForAccess) NULLPTR_CHECK(m_bufForAccess = new vx_int16[m_columns * m_rows]);

	// for single frame reads, there is no need to read it again
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
	int status = 0;
	if (m_readFileIsBinary) {
		if (m_readFileWithScale) {
			if (fread(&m_scale, sizeof(vx_uint32), 1, fp) == 1) {
				ERROR_CHECK(vxSetConvolutionAttribute(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
			}
			else status = -1;
		}
		if (fread(m_bufForAccess, sizeof(vx_int16), m_columns * m_rows, fp) != (m_columns * m_rows))
			status = -1;
	}
	else {
		if (m_readFileWithScale) {
			if (fscanf(fp, "%i", &m_scale) == 1) {
				ERROR_CHECK(vxSetConvolutionAttribute(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
			}
			else status = -1;
		}
		for (vx_size index = 0; index < (m_columns * m_rows); index++) {
			vx_uint32 value;
			if (fscanf(fp, "%i", &value) != 1) {
				status = -1;
				break;
			}
			m_bufForAccess[index] = (vx_int16)value;
		}
	}
	ERROR_CHECK(vxWriteConvolutionCoefficients(m_convolution, m_bufForAccess));
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on convolution input file: %s\n", fileName);

	return status;
}

int CVxParamConvolution::WriteFrame(int frameNumber)
{
	// check if there is no user request to write
	if (m_fileNameWrite.length() < 1) return 0;

	// make sure buffer has been allocated and read the convolution data
	if (!m_bufForAccess) NULLPTR_CHECK(m_bufForAccess = new vx_int16[m_columns * m_rows]);
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
	ERROR_CHECK(vxReadConvolutionCoefficients(m_convolution, m_bufForAccess));

	// write data to output file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_writeFileIsBinary ? "wb" : "w");
	if (!fp) ReportError("ERROR: Unable to create: %s\n", fileName);
	if (m_writeFileIsBinary) {
		if (m_writeFileWithScale) {
			fwrite(&m_scale, sizeof(vx_uint32), 1, fp);
		}
		fwrite(m_bufForAccess, sizeof(vx_int16), m_columns * m_rows, fp);
	}
	else {
		if (m_writeFileWithScale) {
			fprintf(fp, "%d\n", m_scale);
		}
		for (vx_size row = 0; row < m_rows; row++) {
			fprintf(fp, "\n");
			for (vx_size col = 0; col < m_columns; col++) {
				fprintf(fp, " %6d", m_bufForAccess[row * m_columns + col]);
			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	return 0;
}

int CVxParamConvolution::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// make sure buffer has been allocated and read the convolution data
	if (!m_bufForAccess) NULLPTR_CHECK(m_bufForAccess = new vx_int16[m_columns * m_rows]);
	ERROR_CHECK(vxQueryConvolution(m_convolution, VX_CONVOLUTION_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));
	ERROR_CHECK(vxReadConvolutionCoefficients(m_convolution, m_bufForAccess));

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_compareFileIsBinary ? "rb" : "r");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	bool mismatchDetected = false;
	int status = 0;
	if (m_compareFileWithScale) {
		vx_uint32 scaleRef;
		if (m_compareFileIsBinary) {
			if (fread(&scaleRef, sizeof(vx_uint32), 1, fp) != 1)
				status = -1;
		}
		else {
			if (fscanf(fp, "%i", &scaleRef) != 1)
				status = -1;
		}
		if (m_scale != scaleRef)
			mismatchDetected = true;
	}
	for (vx_size index = 0; index < (m_columns * m_rows); index++) {
		vx_int16 coeffValue = 0;
		if (m_compareFileIsBinary) {
			if (fread(&coeffValue, sizeof(coeffValue), 1, fp) != 1) {
				status = -1;
				break;
			}
		}
		else {
			vx_int32 value;
			if (fscanf(fp, "%i", &value) != 1) {
				status = -1;
				break;
			}
			coeffValue = (vx_int16)value;
		}
		if (m_bufForAccess[index] != coeffValue) {
			mismatchDetected = true;
			break;
		}
	}
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on convolution comapre reference file: %s\n", fileName);

	if (mismatchDetected) {
		m_compareCountMismatches++;
		printf("ERROR: convolution COMPARE MISMATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
		if (m_verbose) printf("OK: convolution COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}

	return 0;
}
