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
#include "vxArray.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamArray
//

CVxParamArray::CVxParamArray()
{
	m_vxObjType = VX_TYPE_ARRAY;
	m_format = VX_TYPE_KEYPOINT;
	m_capacity = 0;
	m_itemSize = 0;
	m_readFileIsBinary = false;
	m_writeFileIsBinary = false;
	m_compareFileIsBinary = false;
	m_array = nullptr;
	m_bufForRead = nullptr;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;
	m_errX = 0;
	m_errY = 0;
	m_errStrength = 1e-10f;
	m_errMismatchPercent = 0.0f;
}

CVxParamArray::~CVxParamArray()
{
	Shutdown();
}

int CVxParamArray::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64];
	const char * ioParams = ScanParameters(desc, "array|virtual-array:", "s:", objType);
	if (!_stricmp(objType, "array") || !_stricmp(objType, "virtual-array") ||
		!_stricmp(objType, "array-virtual"))
	{
		// syntax: [virtual-]array:<format>,<capacity>[:<io-params>]
		char itemType[64];
		ioParams = ScanParameters(ioParams, "<format>,<capacity>", "s,D", &itemType, &m_capacity);
		bool found_userStruct = false;
		for (auto it = m_userStructMap->begin(); it != m_userStructMap->end(); ++it){
			if (strcmp(itemType, it->first.c_str()) == 0){
				found_userStruct = true;
				m_format = it->second;
			}
		}
		if (found_userStruct == false){
			m_format = ovxName2Enum(itemType);
			if (m_format == 0) {
				ReportError("ERROR: invalid array item type specified: %s\n", itemType);
			}
		}
		// create array object
		if (!_stricmp(objType, "virtual-array") || !_stricmp(objType, "array-virtual")) {
			m_array = vxCreateVirtualArray(graph, m_format, m_capacity);
			m_isVirtualObject = true;
		}
		else {
			m_array = vxCreateArray(context, m_format, m_capacity);
		}
	}
	else ReportError("ERROR: unsupported array type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_array);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: array creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_array) vxReleaseArray(&m_array);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_array;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamArray::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_array = (vx_array)m_vxObjRef;
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &m_format, sizeof(m_format)));
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_CAPACITY, &m_capacity, sizeof(m_capacity)));
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &m_itemSize, sizeof(m_itemSize)));

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
				else ReportError("ERROR: invalid array read option: %s\n", option);
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
				else ReportError("ERROR: invalid array write option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare syntax: compare,fileName[,ascii|binary][,err{<x>;<y>[;<strength>][;<%mismatch>]}][,log{<fileName>}]
			m_fileNameCompareLog = "";
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			m_compareFileIsBinary = (m_fileNameCompare.find(".txt") != m_fileNameCompare.npos) ? false : true;
			while (*io_params == ',') {
				char option[256];
				io_params = ScanParameters(io_params, ",ascii|binary|err{<x>;<y>[;<strength>][;<%mismatch>]}|log{<fileName>}", ",S", option);
				if (!_stricmp(option, "ascii")) {
					m_compareFileIsBinary = false;
				}
				else if (!_stricmp(option, "binary")) {
					m_compareFileIsBinary = true;
				}
				else if (!_strnicmp(option, "err{", 4)) {
					if (m_format == VX_TYPE_KEYPOINT) {
						const char * p = ScanParameters(&option[3], "{<x>;<y>;<strength>[;<%mismatch>]}", "{d;d;f", &m_errX, &m_errY, &m_errStrength);
						if (*p == ';') {
							ScanParameters(p, ";<%mismatch>}", ";f}", &m_errMismatchPercent);
						}
					}
					else if (m_format == VX_TYPE_COORDINATES2D) {
						const char * p = ScanParameters(&option[3], "{<x>;<y>[;<%mismatch>]}", "{d;d", &m_errX, &m_errY);
						if (*p == ';') {
							ScanParameters(p, ";<%mismatch>}", ";f}", &m_errMismatchPercent);
						}
					}
					else ReportError("ERROR: array compare option not supported for this array: %s\n", option);
				}
				else if (!_strnicmp(option, "log{", 4)) {
					option[strlen(option) - 1] = 0;
					m_fileNameCompareLog.assign(RootDirUpdated(&option[4]));
				}
				else ReportError("ERROR: invalid array compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "view")) {
			m_displayName.assign(fileName);
			m_paramList.push_back(this);
		}
		else if (!_stricmp(ioType, "directive") && (!_stricmp(fileName, "VX_DIRECTIVE_AMD_COPY_TO_OPENCL") || !_stricmp(fileName, "sync-cl-write"))) {
			m_useSyncOpenCLWriteDirective = true;
		}
		else if (!_stricmp(ioType, "init")) {
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
				else ReportError("ERROR: invalid array init option: %s\n", option);
			}
			if (ReadFrame(0) < 0)
				ReportError("ERROR: reading from input file for array init\n");
		}
		else ReportError("ERROR: invalid array operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamArray::Finalize()
{
	// get attributes
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &m_itemSize, sizeof(m_itemSize)));
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_CAPACITY, &m_capacity, sizeof(m_capacity)));

	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_array, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamArray::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: array COMPARE MATCHED for %d frame(s) of %s\n", m_compareCountMatches, GetVxObjectName());
	}
	if (m_array) {
		vxReleaseArray(&m_array);
		m_array = nullptr;
	}
	if (m_bufForRead) {
		delete[] m_bufForRead;
		m_bufForRead = nullptr;
	}

	return 0;
}

// read file into m_bufForRead: returns numItems
size_t CVxParamArray::ReadFileIntoBuffer(FILE * fp, bool readFileIsBinary)
{
	// make sure m_bufForRead is allocated
	if (!m_bufForRead) NULLPTR_CHECK(m_bufForRead = new vx_uint8[m_capacity * m_itemSize]);

	// read file into m_bufForRead
	size_t numItems = 0;
	if (readFileIsBinary)
	{ // data in file is in BINARY format
		numItems = fread(m_bufForRead, m_itemSize, m_capacity, fp);
	}
	else
	{ // data in file is in ASCII format
		if (m_format == VX_TYPE_KEYPOINT) {
			// input syntax of each item: <x> <y> <strength> <scale> <orientation> <tracking_status> <error>
			vx_keypoint_t * item = (vx_keypoint_t *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (7 != fscanf(fp, "%d%d%g%g%g%d%g", &item->x, &item->y, &item->strength, &item->scale, &item->orientation, &item->tracking_status, &item->error))
					break;
			}
		}
		else if (m_format == VX_TYPE_RECTANGLE) {
			// input syntax of each item: <start_x> <start_y> <end_x> <end_y>
			vx_rectangle_t * item = (vx_rectangle_t *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (4 != fscanf(fp, "%d%d%d%d", &item->start_x, &item->start_y, &item->end_x, &item->end_y))
					break;
			}
		}
		else if (m_format == VX_TYPE_COORDINATES2D) {
			// input syntax of each item: <x> <y>
			vx_coordinates2d_t * item = (vx_coordinates2d_t *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (2 != fscanf(fp, "%d%d", &item->x, &item->y))
					break;
			}
		}
		else if (m_format == VX_TYPE_COORDINATES3D) {
			// input syntax of each item: <x> <y> <z>
			vx_coordinates3d_t * item = (vx_coordinates3d_t *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (3 != fscanf(fp, "%d%d%d", &item->x, &item->y, &item->y))
					break;
			}
		}
		else if (m_format == VX_TYPE_INT32 || m_format == VX_TYPE_UINT32 || m_format == VX_TYPE_BOOL) {
			// input syntax of each item: <x> <y> <z>
			vx_uint32 * item = (vx_uint32 *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (1 != fscanf(fp, "%i", item))
					break;
			}
		}
		else if (m_format == VX_TYPE_FLOAT32) {
			// input syntax of each item: <x> <y> <z>
			vx_float32 * item = (vx_float32 *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (1 != fscanf(fp, "%g", item))
					break;
			}
		}
		else if (m_format == VX_TYPE_FLOAT64) {
			// input syntax of each item: <x> <y> <z>
			vx_float64 * item = (vx_float64 *)m_bufForRead;
			for (numItems = 0; numItems < m_capacity; numItems++, item++) {
				if (1 != fscanf(fp, "%lg", item))
					break;
			}
		}
		else {
			// read input as hex value of each byte
			vx_size numBytes = 0;
			while (numBytes < (m_itemSize * m_capacity)) {
				int value;
				if (1 != fscanf(fp, "%x", &value))
					break;
				m_bufForRead[numBytes++] = (vx_uint8)value;
			}
			numItems = numBytes / m_itemSize;
		}
	}

	return numItems;
}

int CVxParamArray::ReadFrame(int frameNumber)
{
	// check if user specified input file to read from
	if (m_fileNameRead.length() < 1) return 0;

	// for single frame reads, there is no need to read the array again
	// as it is already read into the object
	if (!m_fileNameForReadHasIndex && frameNumber != m_captureFrameStart) {
		return 0;
	}

	// reading data from input file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameRead.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_readFileIsBinary ? "rb" : "r");
	if(!fp) {
		if (frameNumber >= (int)m_captureFrameStart) {
			// end of sequence detected for multiframe sequences
			return 1;
		}
		else ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	size_t numItems = ReadFileIntoBuffer(fp, m_readFileIsBinary);
	fclose(fp);

	// set array size to numItems and write the data into array object
	ERROR_CHECK(vxTruncateArray(m_array, 0));
	if (numItems > 0) {
		ERROR_CHECK(vxAddArrayItems(m_array, numItems, m_bufForRead, m_itemSize));
	}

	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_array, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamArray::WriteFrame(int frameNumber)
{
	// check if user specified file to write
	if (m_fileNameWrite.length() < 1) return 0;

	// create the output file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_writeFileIsBinary ? "wb" : "w");
	if(!fp) ReportError("ERROR: Unable to create: %s\n", fileName);

	// get numItems and write if any items exist
	vx_size numItems;
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numItems, sizeof(numItems)));
	if (numItems > 0) {
		vx_uint8 * base = nullptr;
		vx_size stride;
		ERROR_CHECK(vxAccessArrayRange(m_array, 0, numItems, &stride, (void **)&base, VX_READ_ONLY));
		if (m_writeFileIsBinary)
		{ // write in binary
			for (size_t i = 0; i < numItems; i++) {
				vx_uint8 * item = vxFormatArrayPointer(base, i, stride);
				fwrite(item, 1, m_itemSize, fp);
			}
		}
		else
		{ // write in ASCII mode
			if (m_format == VX_TYPE_KEYPOINT) {
				for (size_t i = 0; i < numItems; i++) {
					vx_keypoint_t * item = (vx_keypoint_t *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%4d %4d %20.12e %20.12e %20.12e %d %20.12e\n", item->x, item->y, item->strength, item->scale, item->orientation, item->tracking_status, item->error);
				}
			}
			else if (m_format == VX_TYPE_COORDINATES2D) {
				for (size_t i = 0; i < numItems; i++) {
					vx_coordinates2d_t * item = (vx_coordinates2d_t *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%4d %4d\n", item->x, item->y);
				}
			}
			else if (m_format == VX_TYPE_COORDINATES3D) {
				for (size_t i = 0; i < numItems; i++) {
					vx_coordinates3d_t * item = (vx_coordinates3d_t *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%4d %4d %4d\n", item->x, item->y, item->z);
				}
			}
			else if (m_format == VX_TYPE_RECTANGLE) {
				for (size_t i = 0; i < numItems; i++) {
					vx_rectangle_t * item = (vx_rectangle_t *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%4d %4d %4d %4d\n", item->start_x, item->start_y, item->end_x, item->end_y);
				}
			}
			else if (m_format == VX_TYPE_INT32 || m_format == VX_TYPE_BOOL) {
				for (size_t i = 0; i < numItems; i++) {
					vx_int32 * item = (vx_int32 *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%d\n", *item);
				}
			}
			else if (m_format == VX_TYPE_UINT32) {
				for (size_t i = 0; i < numItems; i++) {
					vx_uint32 * item = (vx_uint32 *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%u\n", *item);
				}
			}
			else if (m_format == VX_TYPE_FLOAT32) {
				for (size_t i = 0; i < numItems; i++) {
					vx_float32 * item = (vx_float32 *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%.12g\n", *item);
				}
			}
			else if (m_format == VX_TYPE_FLOAT64) {
				for (size_t i = 0; i < numItems; i++) {
					vx_float64 * item = (vx_float64 *)vxFormatArrayPointer(base, i, stride);
					fprintf(fp, "%.12lg\n", *item);
				}
			}
			else {
				// write output as hex values
				for (size_t i = 0; i < numItems; i++) {
					vx_uint8 * item = vxFormatArrayPointer(base, i, stride);
					for (size_t j = 0; j < m_itemSize; j++)
						fprintf(fp, " %02X", item[j]);
					fprintf(fp, "\n");
				}
			}
		}
		ERROR_CHECK(vxCommitArrayRange(m_array, 0, numItems, base));
	}
	fclose(fp);

	return 0;
}

int CVxParamArray::CompareFrame(int frameNumber)
{
	// check if user specified file to write
	if (m_fileNameCompare.length() < 1) return 0;

	// clear items from m_arrayListForView
	m_arrayListForView.clear();

	// reading data from reference file
	char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), frameNumber);
	FILE * fp = fopen(fileName, m_compareFileIsBinary ? "rb" : "r");
	if (!fp) {
		ReportError("ERROR: Unable to open: %s\n", fileName);
	}
	size_t numItemsRef = ReadFileIntoBuffer(fp, m_compareFileIsBinary);
	fclose(fp);

	// get numItems of the array
	vx_size numItems;
	ERROR_CHECK(vxQueryArray(m_array, VX_ARRAY_ATTRIBUTE_NUMITEMS, &numItems, sizeof(numItems)));

	// compare array items
	bool mismatchDetected = false;
	if (m_format == VX_TYPE_KEYPOINT && numItems > 0)
	{ // keypoint compare with user specified tolerance limits
		mismatchDetected = CompareFrameKeypoints(numItems, numItemsRef, m_bufForRead, frameNumber, fileName);
	}
	else if (m_format == VX_TYPE_COORDINATES2D && numItems > 0)
	{ // coordinates2d compare with user specified tolerance limits
		mismatchDetected = CompareFrameCoord2d(numItems, numItemsRef, m_bufForRead, frameNumber, fileName);
	}
	else
	{ // fallback to bitwise exact compare
		mismatchDetected = CompareFrameBitwiseExact(numItems, numItemsRef, m_bufForRead, frameNumber, fileName);
	}

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

bool CVxParamArray::CompareFrameBitwiseExact(size_t numItems, size_t numItemsRef, vx_uint8 * bufItems, int frameNumber, const char * fileName)
{
	// bitwise exact compare
	size_t numItemsMin = min(numItems, numItemsRef);
	size_t numMismatches = 0;
	if (numItemsMin > 0) {
		void * ptr = nullptr;
		vx_size stride = 0;
		ERROR_CHECK(vxAccessArrayRange(m_array, 0, numItems, &stride, &ptr, VX_READ_ONLY));
		for (size_t i = 0; i < numItems; i++) {
			vx_uint8 * item = vxFormatArrayPointer(ptr, i, stride);
			if (memcmp(item, bufItems + i * m_itemSize, m_itemSize) != 0) {
				numMismatches++;
			}
		}
		ERROR_CHECK(vxCommitArrayRange(m_array, 0, numItems, ptr));
	}
	numMismatches += max(numItems, numItemsRef) - numItemsMin;
	bool mismatchDetected = false;
	if (numMismatches > 0) {
		printf("ERROR: array COMPARE MISMATCHED %d/%d for %s with frame#%d of %s\n", (int)numMismatches, (int)numItems, GetVxObjectName(), frameNumber, fileName);
		mismatchDetected = true;
	}
	else {
		if (m_verbose) printf("OK: array COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, fileName);
	}
	return mismatchDetected;
}

bool CVxParamArray::CompareFrameKeypoints(size_t numItems, size_t numItemsRef, vx_uint8 * bufItems, int frameNumber, const char * fileName)
{
	FILE * fpLog = NULL;
	if (m_fileNameCompareLog.length() > 0) {
		char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompareLog.c_str(), frameNumber);
		fpLog = fopen(fileName, "w");
		if (!fpLog) ReportError("ERROR: Unable to create: %s\n", fileName);
		printf("OK: creating array compare output log for %s in %s\n", GetVxObjectName(), fileName);
	}

	enum { // color indices of each list for viewing
		colorIndex_match_XYexact_S    = 0,
		colorIndex_match_XYexact_notS = 1,
		colorIndex_match_XYS = 2,
		colorIndex_missing_in_ref = 3,
		colorIndex_missing_in_cur = 4,
	};
	// number of keypoint counts
	size_t count_match_XYexact_S = 0;
	size_t count_match_XYexact_notS = 0;
	size_t count_match_XYS = 0;
	size_t count_missing_in_ref = 0;
	size_t count_missing_in_cur = 0;
	size_t count_non_trackable_in_ref = 0;
	size_t count_non_trackable_in_cur = 0;

	// reset array list for viewing
	ResetArrayListForView();

	// get reference and actual keypoint buffers
	vx_keypoint_t * kpRefBase = (vx_keypoint_t *)m_bufForRead, * kpActualBase = nullptr;
	vx_size stride;
	ERROR_CHECK(vxAccessArrayRange(m_array, 0, numItems, &stride, (void **)&kpActualBase, VX_READ_ONLY));

	// try matching reference keypoints with actual
	for (size_t j = 0; j < numItemsRef; j++) {
		vx_keypoint_t * kpRef = &kpRefBase[j];
		if (!kpRef->tracking_status) {
			count_non_trackable_in_ref++;
		}
		else {
			bool matched = false;
			for (size_t i = 0; i < numItems; i++) {
				vx_keypoint_t * kpCur = &vxArrayItem(vx_keypoint_t, kpActualBase, i, stride);
				if (kpCur->tracking_status) {
					if ((kpCur->x == kpRef->x) && (kpCur->y == kpRef->y)) {
						if (fabsf(kpCur->strength - kpRef->strength) <= m_errStrength) {
							AddToArrayListForView(colorIndex_match_XYexact_S, kpCur->x, kpCur->y, kpCur->strength);
							if (fpLog) fprintf(fpLog, "MATCH-XY-EXACT-S          -- %5d %5d %20.12e (ref:%06d) %5d %5d %20.12e (cur:%06d)\n", kpRef->x, kpRef->y, kpRef->strength, (int)j, kpCur->x, kpCur->y, kpCur->strength, (int)i);
							count_match_XYexact_S++;
						}
						else {
							AddToArrayListForView(colorIndex_match_XYexact_notS, kpCur->x, kpCur->y, kpCur->strength);
							if (fpLog) fprintf(fpLog, "MATCH-XY-EXACT-S-MISMATCH -- %5d %5d %20.12e (ref:%06d) %5d %5d %20.12e (cur:%06d)\n", kpRef->x, kpRef->y, kpRef->strength, (int)j, kpCur->x, kpCur->y, kpCur->strength, (int)i);
							count_match_XYexact_notS++;
						}
						matched = true;
					}
					else if ((abs(kpCur->x - kpRef->x) <= m_errX) && (abs(kpCur->y - kpRef->y) <= m_errY) &&
						(fabsf(kpCur->strength - kpRef->strength) <= m_errStrength))
					{
						AddToArrayListForView(colorIndex_match_XYS, kpCur->x, kpCur->y, kpCur->strength);
						if (fpLog) fprintf(fpLog, "MATCH-XYS                     -- %5d %5d %20.12e (ref:%06d) %5d %5d %20.12e (cur:%06d)\n", kpRef->x, kpRef->y, kpRef->strength, (int)j, kpCur->x, kpCur->y, kpCur->strength, (int)i);
						count_match_XYS++;
						matched = true;
					}
					if (matched)
						break;
				}
			}
			if (!matched) {
				AddToArrayListForView(colorIndex_missing_in_cur, kpRef->x, kpRef->y, kpRef->strength);
				if (fpLog) fprintf(fpLog, "MISMATCH-WITH-CUR         -- %5d %5d %20.12e (ref:%06d)\n", kpRef->x, kpRef->y, kpRef->strength, (int)j);
				count_missing_in_cur++;
			}
		}
	}

	// try matching actual keypoints with reference
	for (size_t i = 0; i < numItems; i++) {
		vx_keypoint_t * kpCur = &vxArrayItem(vx_keypoint_t, kpActualBase, i, stride);
		if (!kpCur->tracking_status) {
			count_non_trackable_in_cur++;
		}
		else {
			bool matched = false;
			for (size_t j = 0; j < numItemsRef; j++) {
				vx_keypoint_t * kpRef = &kpRefBase[j];
				if (kpRef->tracking_status) {
					if ((abs(kpCur->x - kpRef->x) <= m_errX) && (abs(kpCur->y - kpRef->y) <= m_errY) &&
						(fabsf(kpCur->strength - kpRef->strength) <= m_errStrength))
					{
						matched = true;
					}
					if (matched)
						break;
				}
			}
			if (!matched) {
				AddToArrayListForView(colorIndex_missing_in_ref, kpCur->x, kpCur->y, kpCur->strength);
				if (fpLog) fprintf(fpLog, "MISMATCH-WITH-REF         --                                               %5d %5d %20.12e (cur:%06d)\n", kpCur->x, kpCur->y, kpCur->strength, (int)i);
				count_missing_in_ref++;
			}
		}
	}

	ERROR_CHECK(vxCommitArrayRange(m_array, 0, numItems, kpActualBase));

	// check for overall mismatch criteria
	size_t totalMatched = count_match_XYexact_S + count_match_XYS;
	size_t totalMismatchesOrMissing = max(count_match_XYexact_notS + count_missing_in_ref, count_missing_in_cur);
	size_t total = totalMatched + totalMismatchesOrMissing;
	float percentMismatches = (total > 0) ? (100.0f * (float)totalMismatchesOrMissing / (float)total) : 0.0f;
	bool mismatched = false;
	if (percentMismatches > m_errMismatchPercent) {
		mismatched = true;
		char line[512];
		snprintf(line, sizeof(line), "ERROR: array COMPARE MISMATCHED [matched %d; mismatched/missing %d (%.3f%%)] [untracked %d/%d(ref) vs %d/%d] for %s with frame#%d of %s\n",
			(int)totalMatched, (int)totalMismatchesOrMissing, percentMismatches,
			(int)count_non_trackable_in_ref, (int)numItemsRef, (int)count_non_trackable_in_cur, (int)numItems,
			GetVxObjectName(), frameNumber, fileName);
		printf("%s", line);
		if (fpLog) fprintf(fpLog, "%s", line);
	}
	else {
		char line[512];
		snprintf(line, sizeof(line), "OK: array COMPARE MATCHED %.3f%% [untracked %d/%d(ref) vs %d/%d] for %s with frame#%d of %s\n", 100.0f - percentMismatches, (int)count_non_trackable_in_ref, (int)numItemsRef, (int)count_non_trackable_in_cur, (int)numItems, GetVxObjectName(), frameNumber, fileName);
		if (m_verbose) printf("%s", line);
		if (fpLog) fprintf(fpLog, "%s", line);
	}

	if (fpLog) fclose(fpLog);
	return mismatched;
}

bool CVxParamArray::CompareFrameCoord2d(size_t numItems, size_t numItemsRef, vx_uint8 * bufItems, int frameNumber, const char * fileName)
{
	FILE * fpLog = NULL;
	if (m_fileNameCompareLog.length() > 0) {
		char fileName[MAX_FILE_NAME_LENGTH]; snprintf(fileName, sizeof(fileName), m_fileNameCompareLog.c_str(), frameNumber);
		fpLog = fopen(fileName, "w");
		if (!fpLog) ReportError("ERROR: Unable to create: %s\n", fileName);
		printf("OK: creating array compare output log for %s in %s\n", GetVxObjectName(), fileName);
	}

	enum { // color indices of each list for viewing
		colorIndex_match_XYexact = 0,
		colorIndex_match_XY = 1,
		colorIndex_missing_in_ref = 2,
		colorIndex_missing_in_cur = 3,
	};
	// number of keypoint counts
	size_t count_match_XYexact = 0;
	size_t count_match_XY = 0;
	size_t count_missing_in_ref = 0;
	size_t count_missing_in_cur = 0;

	// reset array list for viewing
	ResetArrayListForView();

	// get reference and actual keypoint buffers
	vx_coordinates2d_t * kpRefBase = (vx_coordinates2d_t *)m_bufForRead, *kpActualBase = nullptr;
	vx_size stride;
	if (numItems > 0) {
		ERROR_CHECK(vxAccessArrayRange(m_array, 0, numItems, &stride, (void **)&kpActualBase, VX_READ_ONLY));
	}

	// try matching reference keypoints with actual
	for (size_t j = 0; j < numItemsRef; j++) {
		vx_coordinates2d_t * kpRef = &kpRefBase[j];
		bool matched = false;
		for (size_t i = 0; i < numItems; i++) {
			vx_coordinates2d_t * kpCur = &vxArrayItem(vx_coordinates2d_t, kpActualBase, i, stride);
			if ((kpCur->x == kpRef->x) && (kpCur->y == kpRef->y)) {
				AddToArrayListForView(colorIndex_match_XYexact, kpCur->x, kpCur->y, 0.0f);
				if (fpLog) fprintf(fpLog, "MATCH-XY-EXACT       -- %5d %5d (ref:%06d) %5d %5d (cur:%06d)\n", kpRef->x, kpRef->y, (int)j, kpCur->x, kpCur->y, (int)i);
				count_match_XYexact++;
				matched = true;
			}
			else if ((abs((vx_int32)kpCur->x - (vx_int32)kpRef->x) <= m_errX) && (abs((vx_int32)kpCur->y - (vx_int32)kpRef->y) <= m_errY)) {
				AddToArrayListForView(colorIndex_match_XY, kpCur->x, kpCur->y, 0.0f);
				if (fpLog) fprintf(fpLog, "MATCH-XY             -- %5d %5d (ref:%06d) %5d %5d (cur:%06d)\n", kpRef->x, kpRef->y, (int)j, kpCur->x, kpCur->y, (int)i);
				count_match_XY++;
				matched = true;
			}
			if (matched)
				break;
		}
		if (!matched) {
			AddToArrayListForView(colorIndex_missing_in_cur, kpRef->x, kpRef->y, 0.0f);
			if (fpLog) fprintf(fpLog, "MISMATCH-WITH-CUR    -- %5d %5d (ref:%06d)\n", kpRef->x, kpRef->y, (int)j);
			count_missing_in_cur++;
		}
	}

	// try matching actual keypoints with reference
	for (size_t i = 0; i < numItems; i++) {
		vx_coordinates2d_t * kpCur = &vxArrayItem(vx_coordinates2d_t, kpActualBase, i, stride);
		bool matched = false;
		for (size_t j = 0; j < numItemsRef; j++) {
			vx_coordinates2d_t * kpRef = &kpRefBase[j];
			if ((abs((vx_int32)kpCur->x - (vx_int32)kpRef->x) <= m_errX) && (abs((vx_int32)kpCur->y - (vx_int32)kpRef->y) <= m_errY)) {
				matched = true;
				break;
			}
		}
		if (!matched) {
			AddToArrayListForView(colorIndex_missing_in_ref, kpCur->x, kpCur->y, 0.0f);
			if (fpLog) fprintf(fpLog, "MISMATCH-WITH-REF    --                          %5d %5d (cur:%06d)\n", kpCur->x, kpCur->y, (int)i);
			count_missing_in_ref++;
		}
	}

	if (numItems > 0) {
		ERROR_CHECK(vxCommitArrayRange(m_array, 0, numItems, kpActualBase));
	}

	// check for overall mismatch criteria
	size_t totalMatched = count_match_XYexact + count_match_XY;
	size_t totalMismatchesOrMissing = max(count_missing_in_ref, count_missing_in_cur);
	size_t total = totalMatched + totalMismatchesOrMissing;
	float percentMismatches = (total > 0) ? (100.0f * (float)totalMismatchesOrMissing / (float)total) : 0.0f;
	bool mismatched = false;
	if (percentMismatches > m_errMismatchPercent) {
		mismatched = true;
		printf("ERROR: array COMPARE MISMATCHED [matched %d; mismatched/missing %d (%.3f%%)] for %s with frame#%d of %s\n", (int)totalMatched, (int)totalMismatchesOrMissing, percentMismatches, GetVxObjectName(), frameNumber, fileName);
		if (fpLog) fprintf(fpLog, "ERROR: array COMPARE MISMATCHED [matched %d; mismatched/missing %d (%.3f%%)] for %s with frame#%d of %s\n", (int)totalMatched, (int)totalMismatchesOrMissing, percentMismatches, GetVxObjectName(), frameNumber, fileName);
	}
	else {
		if (m_verbose) printf("OK: array COMPARE MATCHED %.3f%% for %s with frame#%d of %s\n", 100.0f - percentMismatches, GetVxObjectName(), frameNumber, fileName);
		if (fpLog) fprintf(fpLog, "OK: array COMPARE MATCHED %.3f%% for %s with frame#%d of %s\n", 100.0f - percentMismatches, GetVxObjectName(), frameNumber, fileName);
	}

	if (fpLog) fclose(fpLog);
	return mismatched;
}
