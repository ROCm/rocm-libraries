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
#include "vxMatrix.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamMatrix
//
CVxParamMatrix::CVxParamMatrix()
{
	// vx configuration
	m_vxObjType = VX_TYPE_MATRIX;
	m_data_type = VX_TYPE_INT32;
	m_columns = 0;
	m_rows = 0;
	m_size = 0;
	// I/O configuration
	m_readFileIsBinary = false;
	m_writeFileIsBinary = false;
	m_compareFileIsBinary = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	m_errTolerance = 0.0f;
	m_bufForAccess = nullptr;
	// vx object
	m_matrix = nullptr;
}

CVxParamMatrix::~CVxParamMatrix()
{
	Shutdown();
}

int CVxParamMatrix::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: matrix COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	GuiTrackBarShutdown((vx_reference)m_matrix);
	if (m_matrix) {
		vxReleaseMatrix(&m_matrix);
		m_matrix = nullptr;
	}
	if (m_bufForAccess) {
		delete[] m_bufForAccess;
		m_bufForAccess = nullptr;
	}
	return 0;
}

int CVxParamMatrix::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64], data_type[64];
	const char * ioParams = ScanParameters(desc, "matrix:<data-type>,<columns>,<rows>", "s:s,D,D", objType, data_type, &m_columns, &m_rows);
	if (!_stricmp(objType, "matrix")) {
		m_data_type = ovxName2Enum(data_type);
		m_matrix = vxCreateMatrix(context, m_data_type, m_columns, m_rows);
	}
	else ReportError("ERROR: unsupported matrix type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_matrix);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: matrix creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_matrix) vxReleaseMatrix(&m_matrix);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_matrix;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamMatrix::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_matrix = (vx_matrix)m_vxObjRef;
	ERROR_CHECK(vxQueryMatrix(m_matrix, VX_MATRIX_ATTRIBUTE_TYPE, &m_data_type, sizeof(m_data_type)));
	ERROR_CHECK(vxQueryMatrix(m_matrix, VX_MATRIX_ATTRIBUTE_COLUMNS, &m_columns, sizeof(m_columns)));
	ERROR_CHECK(vxQueryMatrix(m_matrix, VX_MATRIX_ATTRIBUTE_ROWS, &m_rows, sizeof(m_rows)));
	ERROR_CHECK(vxQueryMatrix(m_matrix, VX_MATRIX_ATTRIBUTE_SIZE, &m_size, sizeof(m_size)));

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
				else ReportError("ERROR: invalid matrix read option: %s\n", option);
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
				else ReportError("ERROR: invalid matrix write option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare request syntax: compare,<fileName>[,ascii|binary][,err{<tolerance>}]
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			m_compareFileIsBinary = (m_fileNameCompare.find(".txt") != m_fileNameCompare.npos) ? false : true;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",ascii|binary|err{<tolerance>}", ",s", option);
				if (!_stricmp(option, "ascii")) {
					m_compareFileIsBinary = false;
				}
				else if (!_stricmp(option, "binary")) {
					m_compareFileIsBinary = true;
				}
				else if (!_strnicmp(option, "err{", 4)) {
					ScanParameters(&option[3], "{<tolerance>}", "{f}", &m_errTolerance);
				}
				else ReportError("ERROR: invalid matrix compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "init"))
		{ // write request syntax: init,{<value1>;<value2>;...<valueN>}
			NULLPTR_CHECK(m_bufForAccess = new vx_uint8[m_size]);
			if (fileName[0] == '{') {
				vx_size index = 0; char fmt[3] = { '{', (m_data_type == VX_TYPE_FLOAT32) ? 'f' : 'd', 0 };
				for (const char * s = fileName; *s && index < (m_columns * m_rows); fmt[0] = ';', index++) {
					if (m_data_type == VX_TYPE_INT32 || m_data_type == VX_TYPE_UINT8) {
						vx_uint32 value;
						s = ScanParameters(s, "<value>", fmt, &value);
						if (m_data_type == VX_TYPE_UINT8) ((vx_uint8 *)m_bufForAccess)[index] = (vx_uint8)value;
						else ((vx_int32 *)m_bufForAccess)[index] = value;
					}
					else if (m_data_type == VX_TYPE_FLOAT32) {
						s = ScanParameters(s, "<value>", fmt, &((vx_float32 *)m_bufForAccess)[index]);
					}
					else ReportError("ERROR: matrix init option not support for data_type of %s\n", GetVxObjectName());
				}
				if (index < (m_columns * m_rows)) ReportError("ERROR: matrix init have too few values: %s\n", fileName);
				ERROR_CHECK(vxWriteMatrix(m_matrix, m_bufForAccess));
			}
			else {
				std::string fileNameRead = m_fileNameRead;
				bool fileNameForReadHasIndex = m_fileNameForReadHasIndex;
				bool readFileIsBinary = m_readFileIsBinary;
				m_fileNameRead.assign(RootDirUpdated(fileName));
				m_fileNameForReadHasIndex = (m_fileNameRead.find("%") != m_fileNameRead.npos) ? true : false;
				m_readFileIsBinary = (m_fileNameRead.find(".txt") != m_fileNameRead.npos) ? false : true;
				if (ReadFrame(0)) {
					return -1;
				}
				m_fileNameRead = fileNameRead;
				m_fileNameForReadHasIndex = fileNameForReadHasIndex;
				m_readFileIsBinary = readFileIsBinary;
			}
		}
		else if (!_stricmp(ioType, "directive") && !_stricmp(fileName, "readonly")) {
			ERROR_CHECK(vxDirective((vx_reference)m_matrix, VX_DIRECTIVE_AMD_READ_ONLY));
		}
		else if (!_stricmp(ioType, "directive") && (!_stricmp(fileName, "VX_DIRECTIVE_AMD_COPY_TO_OPENCL") || !_stricmp(fileName, "sync-cl-write"))) {
			m_useSyncOpenCLWriteDirective = true;
		}
		else if (!_stricmp(ioType, "ui") && !_strnicmp(fileName, "f", 1) && m_data_type == VX_TYPE_FLOAT32 && m_columns == 3 && m_rows == 3) {
			int id = 0;
			float valueR = 200.0f, valueInc = 0.5f;
			if (sscanf(&fileName[1], "%d,%g,%g", &id, &valueR, &valueInc) != 3) {
				printf("ERROR: invalid matrix UI configuration '%s'\n", fileName);
				return -1;
			}
			id--;
			GuiTrackBarInitializeMatrix((vx_reference)m_matrix, id, valueR, valueInc);
			GuiTrackBarProcessKey(0); // just initialize the matrix
		}
		else ReportError("ERROR: invalid matrix operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamMatrix::Finalize()
{
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_matrix, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}
	return 0;
}

int CVxParamMatrix::ReadFrame(int frameNumber)
{
	// check if there is no user request to read
	if (m_fileNameRead.length() < 1) return 0;

	// make sure buffer has been allocated
	if (!m_bufForAccess) NULLPTR_CHECK(m_bufForAccess = new vx_uint8[m_size]);

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
		if (fread(m_bufForAccess, 1, m_size, fp) != m_size)
			status = -1;
	}
	else {
		for (vx_size index = 0; index < (m_columns * m_rows); index++) {
			if (m_data_type == VX_TYPE_INT32 || m_data_type == VX_TYPE_UINT8) {
				vx_uint32 value;
				if (fscanf(fp, "%i", &value) != 1) {
					status = -1;
					break;
				}
				if (m_data_type == VX_TYPE_UINT8) ((vx_uint8 *)m_bufForAccess)[index] = (vx_uint8)value;
				else ((vx_int32 *)m_bufForAccess)[index] = value;
			}
			else if (m_data_type == VX_TYPE_FLOAT32) {
				if (fscanf(fp, "%g", &((vx_float32 *)m_bufForAccess)[index]) != 1) {
					status = -1;
					break;
				}
			}
			else ReportError("ERROR: matrix ascii read option not support for data_type of %s\n", GetVxObjectName());
		}
	}
	ERROR_CHECK(vxWriteMatrix(m_matrix, m_bufForAccess));
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on matrix input file: %s\n", fileName);

	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_matrix, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return status;
}

int CVxParamMatrix::WriteFrame(int frameNumber)
{
	// check if there is no user request to write
	if (m_fileNameWrite.length() < 1) return 0;

	// make sure buffer has been allocated and read the matrix data
	if (!m_bufForAccess) NULLPTR_CHECK(m_bufForAccess = new vx_uint8[m_size]);
	ERROR_CHECK(vxReadMatrix(m_matrix, m_bufForAccess));

	// write data to output file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_writeFileIsBinary ? "wb" : "w");
	if (!fp) ReportError("ERROR: Unable to create: %s\n", fileName);
	if (m_writeFileIsBinary) {
		fwrite(m_bufForAccess, 1, m_size, fp);
	}
	else {
		for (vx_size index = 0; index < m_columns * m_rows; index++) {
			if (m_data_type == VX_TYPE_INT32) fprintf(fp, "%d ", ((vx_int32 *)m_bufForAccess)[index]);
			else if (m_data_type == VX_TYPE_FLOAT32) fprintf(fp, "%g ", ((vx_float32 *)m_bufForAccess)[index]);
			else if (m_data_type == VX_TYPE_UINT8) fprintf(fp, "%d ", ((vx_uint8 *)m_bufForAccess)[index]);
			else ReportError("ERROR: matrix ascii write option not support for data_type of %s\n", GetVxObjectName());
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	return 0;
}

int CVxParamMatrix::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// make sure buffer has been allocated and read the matrix data
	if (!m_bufForAccess) NULLPTR_CHECK(m_bufForAccess = new vx_uint8[m_size]);
	ERROR_CHECK(vxReadMatrix(m_matrix, m_bufForAccess));

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_compareFileIsBinary ? "rb" : "r");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	bool mismatchDetected = false;
	int status = 0, errTolerance = (int)m_errTolerance;
	vx_size itemsize = m_size / (m_columns * m_rows);
	for (vx_size index = 0; index < (m_columns * m_rows); index++) {
		union {
			vx_int32 i32; 
			vx_float32 f32; 
			vx_uint8 u8;
		} item;
		if (m_compareFileIsBinary) {
			if (fread(&item, itemsize, 1, fp) != 1) {
				status = -1;
				break;
			}
		}
		else {
			if (m_data_type == VX_TYPE_INT32 || m_data_type == VX_TYPE_UINT8) {
				if (fscanf(fp, "%i", &item.i32) != 1) {
					status = -1;
					break;
				}
			}
			else if (m_data_type == VX_TYPE_FLOAT32) {
				if (fscanf(fp, "%g", &item.f32) != 1) {
					status = -1;
					break;
				}
			}
			else ReportError("ERROR: matrix ascii compare option not support for data_type of %s\n", GetVxObjectName());
		}
		if (m_data_type == VX_TYPE_INT32) {
			if (abs(item.i32 - ((vx_int32 *)m_bufForAccess)[index]) > errTolerance)
				mismatchDetected = true;
		}
		else if (m_data_type == VX_TYPE_FLOAT32) {
			if (fabsf(item.f32 - ((vx_float32 *)m_bufForAccess)[index]) > m_errTolerance)
				mismatchDetected = true;
		}
		else if (m_data_type == VX_TYPE_UINT8) {
			if (abs((int)item.u8 - (int)((vx_uint8 *)m_bufForAccess)[index]) > errTolerance)
				mismatchDetected = true;
		}
		if (mismatchDetected)
			break;
	}
	fclose(fp);
	if (status < 0)
		ReportError("ERROR: detected EOF on matrix comapre reference file: %s\n", fileName);

	if (mismatchDetected) {
		m_compareCountMismatches++;
		printf("ERROR: matrix COMPARE MISMATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
		if (m_verbose) printf("OK: matrix COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}

	return 0;
}
