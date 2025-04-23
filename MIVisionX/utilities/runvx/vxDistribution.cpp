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
#include "vxDistribution.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamDistribution
//
CVxParamDistribution::CVxParamDistribution()
{
	// vx configuration
	m_vxObjType = VX_TYPE_DISTRIBUTION;
	m_numBins = 0;
	m_offset = 0;
	m_range = 0;
	// I/O configuration
	m_readFileIsBinary = false;
	m_writeFileIsBinary = false;
	m_compareFileIsBinary = false;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	// vx object
	m_distribution = nullptr;
	m_bufForCompare = nullptr;
}

CVxParamDistribution::~CVxParamDistribution()
{
	Shutdown();
}

int CVxParamDistribution::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: distribution COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	if (m_distribution){
		vxReleaseDistribution(&m_distribution);
		m_distribution = nullptr;
	}
	if (m_bufForCompare) {
		delete[] m_bufForCompare;
		m_bufForCompare = nullptr;
	}
	return 0;
}

int CVxParamDistribution::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64];
	const char * ioParams = ScanParameters(desc, "distribution:<numBins>,<offset>,<range>", "s:D,d,d", objType, &m_numBins, &m_offset, &m_range);
	if (!_stricmp(objType, "distribution")) {
		m_distribution = vxCreateDistribution(context, m_numBins, m_offset, m_range);
	}
	else ReportError("ERROR: unsupported distribution type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_distribution);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: distribution creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_distribution) vxReleaseDistribution(&m_distribution);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_distribution;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamDistribution::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_distribution = (vx_distribution)m_vxObjRef;
	ERROR_CHECK(vxQueryDistribution(m_distribution, VX_DISTRIBUTION_ATTRIBUTE_BINS, &m_numBins, sizeof(m_numBins)));
	ERROR_CHECK(vxQueryDistribution(m_distribution, VX_DISTRIBUTION_ATTRIBUTE_OFFSET, &m_offset, sizeof(m_offset)));
	ERROR_CHECK(vxQueryDistribution(m_distribution, VX_DISTRIBUTION_ATTRIBUTE_RANGE, &m_range, sizeof(m_range)));

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
				else ReportError("ERROR: invalid distribution read option: %s\n", option);
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
				else ReportError("ERROR: invalid distribution write option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare syntax: compare,fileName[,ascii|binary]
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
				else ReportError("ERROR: invalid distribution compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "view")) {
			m_displayName.assign(fileName);
			m_paramList.push_back(this);
		}
		else ReportError("ERROR: invalid distribution operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamDistribution::Finalize()
{
	return 0;
}

// read file into m_bufForRead: returns 0 if successful, 1 on EOF
int CVxParamDistribution::ReadFileIntoBuffer(FILE * fp, vx_uint32 * buf)
{
	// read file into m_bufForRead
	int status = 0;
	if (m_readFileIsBinary)
	{ // read in BINARY mode
		vx_size count = fread(buf, sizeof(vx_uint32), m_numBins, fp);
		if (count != m_numBins)
			status = 1;
	}
	else
	{ // read in ASCII mode
		for (size_t i = 0; i < m_numBins; i++){
			if (fscanf(fp, "%i", &buf[i]) != 1) {
				status = 1;
				break;
			}
		}
	}
	return status;
}

int CVxParamDistribution::ReadFrame(int frameNumber)
{
	// check if there is no user request to read
	if (m_fileNameRead.length() < 1) return 0;

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
	vx_uint32 * data = nullptr;
	ERROR_CHECK(vxAccessDistribution(m_distribution, (void **)&data, VX_WRITE_ONLY));
	int status = ReadFileIntoBuffer(fp, data);
	ERROR_CHECK(vxCommitDistribution(m_distribution, data));
	fclose(fp);

	return status;
}

int CVxParamDistribution::WriteFrame(int frameNumber)
{
	// check if there is no user request to write
	if (m_fileNameWrite.length() < 1) return 0;

	// reading data from input file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_writeFileIsBinary ? "wb" : "w");
	if (!fp) ReportError("ERROR: Unable to create: %s\n", fileName);
	vx_uint32 * data = nullptr;
	ERROR_CHECK(vxAccessDistribution(m_distribution, (void **)&data, VX_READ_ONLY));
	if (m_writeFileIsBinary)
	{ // write in BINARY mode
		fwrite(data, sizeof(data[0]), m_numBins, fp);
	}
	else
	{ // write in ASCII mode
		for (size_t i = 0; i < m_numBins; i++)
			fprintf(fp, "%8d\n", data[i]);
	}
	ERROR_CHECK(vxCommitDistribution(m_distribution, data));
	fclose(fp);

	return 0;
}

int CVxParamDistribution::CompareFrame(int frameNumber)
{
	// check if there is no user request to compare
	if (m_fileNameCompare.length() < 1) return 0;

	// make sure m_bufForRead is allocated
	if (!m_bufForCompare) NULLPTR_CHECK(m_bufForCompare = new vx_uint32[m_numBins]);

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_compareFileIsBinary ? "rb" : "r");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	int status = ReadFileIntoBuffer(fp, m_bufForCompare);
	fclose(fp);
	if (status) ReportError("ERROR: distribution compare reference doesn't have enough data: %s\n", fileName);

	// compare and report error if mismatched
	vx_uint32 * bufRef = nullptr;
	ERROR_CHECK(vxAccessDistribution(m_distribution, (void **)&bufRef, VX_READ_ONLY));
	status = memcmp(bufRef, m_bufForCompare, m_numBins * sizeof(vx_uint32)) ? -1 : 0;
	ERROR_CHECK(vxCommitDistribution(m_distribution, bufRef));
	if (status) {
		m_compareCountMismatches++;
		printf("ERROR: distribution COMPARE MISMATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
		if (!m_discardCompareErrors) return -1;
	}
	else {
		m_compareCountMatches++;
		if (m_verbose) printf("OK: distribution COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}
	return 0;
}
