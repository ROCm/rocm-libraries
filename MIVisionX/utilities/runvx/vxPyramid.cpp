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
#include "vxPyramid.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamPyramid
//
CVxParamPyramid::CVxParamPyramid()
{
	// vx configuration
	m_vxObjType = VX_TYPE_PYRAMID;
	m_format = VX_DF_IMAGE_VIRT;
	m_width = 0;
	m_height = 0;
	m_numLevels = 0;
	m_scale = 0;
	// vx object
	m_pyramid = nullptr;
	// I/O configuration
	m_comparePixelErrorMin = 0;
	m_comparePixelErrorMax = 0;
	m_bufForCompare = nullptr;
	m_imageFrameSize = nullptr;
	m_rectFullLevel = nullptr;
	m_rectCompareLevel = nullptr;
	m_fpReadImage = nullptr;
	m_fpWriteImage = nullptr;
	m_fpCompareImage = nullptr;
	m_useCheckSumForCompare = false;
	m_generateCheckSumForCompare = false;
}

CVxParamPyramid::~CVxParamPyramid()
{
	Shutdown();
}

int CVxParamPyramid::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: pyramid %s MATCHED for %d frame(s) of %s\n", m_useCheckSumForCompare ? "CHECKSUM" : "COMPARE", m_compareCountMatches, GetVxObjectName());
	}
	if (m_pyramid) {
		vxReleasePyramid(&m_pyramid);
		m_pyramid = nullptr;
	}
	if (m_bufForCompare) {
		delete[] m_bufForCompare;
		m_bufForCompare = nullptr;
	}
	if (m_imageFrameSize) {
		delete[] m_imageFrameSize;
		m_imageFrameSize = nullptr;
	}
	if (m_rectFullLevel) {
		delete[] m_rectFullLevel;
		m_rectFullLevel = nullptr;
	}
	if (m_rectCompareLevel) {
		delete[] m_rectCompareLevel;
		m_rectCompareLevel = nullptr;
	}
	if (m_fpReadImage) {
		for (vx_size level = 0; level < m_numLevels; level++)
			if (m_fpReadImage[level]) fclose(m_fpReadImage[level]);
		m_fpReadImage = nullptr;
	}
	if (m_fpWriteImage) {
		for (vx_size level = 0; level < m_numLevels; level++)
			if (m_fpWriteImage[level]) fclose(m_fpWriteImage[level]);
		m_fpWriteImage = nullptr;
	}
	if (m_fpCompareImage) {
		for (vx_size level = 0; level < m_numLevels; level++)
			if (m_fpCompareImage[level]) fclose(m_fpCompareImage[level]);
		m_fpCompareImage = nullptr;
	}
	return 0;
}

int CVxParamPyramid::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters: syntax: [virtual-]pyramid:<numLevels>,half|orb|<scale-factor>,<width>,<height>,<format>[:<io-params>]
	char objType[64], scaleFactor[64];
	const char * ioParams = ScanParameters(desc, "pyramid|virtual-pyramid:<numLevels>,half|orb|<scale-factor>,<width>,<height>,<format>", "s:D,s,d,d,c", objType, &m_numLevels, scaleFactor, &m_width, &m_height, &m_format);
	if (!_strnicmp(scaleFactor, "half", 4)) m_scale = VX_SCALE_PYRAMID_HALF;
	else if (!_strnicmp(scaleFactor, "orb", 3)) m_scale = VX_SCALE_PYRAMID_ORB;
	else m_scale = (float)atof(scaleFactor);

	// create pyarmid object
	m_pyramid = nullptr;
	if (!_stricmp(objType, "pyramid")) {
		m_pyramid = vxCreatePyramid(context, m_numLevels, m_scale, m_width, m_height, m_format);
	}
	else if (!_stricmp(objType, "virtual-pyramid") || !_stricmp(objType, "pyramid-virtual")) {
		m_pyramid = vxCreateVirtualPyramid(graph, m_numLevels, m_scale, m_width, m_height, m_format);
		m_isVirtualObject = true;
	}
	else ReportError("ERROR: invalid pyramid type: %s\n", objType);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_pyramid);
	if (ovxStatus != VX_SUCCESS){
		printf("ERROR: pyramid creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_pyramid) vxReleasePyramid(&m_pyramid);
		throw - 1;
	}

	// io initialize
	return InitializeIO(context, graph, (vx_reference)m_pyramid, ioParams);
}

int CVxParamPyramid::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_pyramid = (vx_pyramid)m_vxObjRef;
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_FORMAT, &m_format, sizeof(m_format)));
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_WIDTH, &m_width, sizeof(m_width)));
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_HEIGHT, &m_height, sizeof(m_height)));
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_LEVELS, &m_numLevels, sizeof(m_numLevels)));
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_SCALE, &m_scale, sizeof(m_scale)));

	// process I/O parameters
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		if (!_stricmp(ioType, "read"))
		{ // read request syntax: read,<fileName>
			m_fileNameRead.assign(RootDirUpdated(fileName));
			m_fileNameForReadHasIndex = (m_fileNameRead.find("%") != m_fileNameRead.npos) ? true : false;
			if (!m_fileNameForReadHasIndex) ReportError("ERROR: invalid pyramid input fileName (expects %%d format for each level): %s\n", ioType);
			// mark multi-frame capture enabled
			m_usingMultiFrameCapture = true;
		}
		else if (!_stricmp(ioType, "write"))
		{ // write request syntax: write,<fileName>
			bool needDisplay = false;
			m_fileNameWrite.assign(RootDirUpdated(fileName));
			m_fileNameForWriteHasIndex = (m_fileNameWrite.find("%") != m_fileNameWrite.npos) ? true : false;
			if (!m_fileNameForWriteHasIndex) ReportError("ERROR: invalid pyramid output fileName (expects %%d format for each level): %s\n", ioType);
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare syntax: compare,fileName[,rect{<start-x>;<start-y>;<end-x>;<end-y>}][,err{<min>;<max>}][,checksum|checksum-save-instead-of-test]
			// save the reference image fileName
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			m_fileNameForCompareHasIndex = (m_fileNameCompare.find("%") != m_fileNameCompare.npos) ? true : false;
			if (!m_fileNameForCompareHasIndex) ReportError("ERROR: invalid pyramid compare fileName (expects %%d format for each level): %s\n", ioType);
			// initialize pixel error range for exact match
			m_comparePixelErrorMin = 0;
			m_comparePixelErrorMax = 0;
			// set the compare region
			m_rectCompare.start_x = 0;
			m_rectCompare.start_y = 0;
			m_rectCompare.end_x = m_width;
			m_rectCompare.end_y = m_height;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",rect{<start-x>;<start-y>;<end-x>;<end-y>}|err{<min>;<max>}|checksum|checksum-save-instead-of-test", ",s", option);
				if (!_strnicmp(option, "rect", 4)) {
					ScanParameters(option + 4, "{<start-x>;<start-y>;<end-x>;<end-y>}", "{d;d;d;d}", &m_rectCompare.start_x, &m_rectCompare.start_y, &m_rectCompare.end_x, &m_rectCompare.end_y);
				}
				else if (!_strnicmp(option, "err", 3)) {
					ScanParameters(option + 3, "{<min>;<max>}", "{f;f}", &m_comparePixelErrorMin, &m_comparePixelErrorMax);
					if (m_useCheckSumForCompare) ReportError("ERROR: can't support error range with checksum\n");
				}
				else if (!_stricmp(option, "checksum")) {
					m_useCheckSumForCompare = true;
					if (m_comparePixelErrorMin != m_comparePixelErrorMax) ReportError("ERROR: can't support error range with checksum\n");
				}
				else if (!_stricmp(option, "checksum-save-instead-of-test")) {
					m_generateCheckSumForCompare = true;
				}
				else ReportError("ERROR: invalid compare option: %s\n", option);
			}
		}
		else ReportError("ERROR: invalid pyramid operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamPyramid::Finalize()
{
	// get object attributes
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_FORMAT, &m_format, sizeof(m_format)));
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_WIDTH, &m_width, sizeof(m_width)));
	ERROR_CHECK(vxQueryPyramid(m_pyramid, VX_PYRAMID_ATTRIBUTE_HEIGHT, &m_height, sizeof(m_height)));

	// initialize other parameters
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;

	// compute frame size in bytes
	m_pyramidFrameSize = 0;
	if (!m_imageFrameSize) m_imageFrameSize = new size_t[m_numLevels];
	if (!m_rectFullLevel) m_rectFullLevel = new vx_rectangle_t[m_numLevels];
	if (!m_rectCompareLevel) m_rectCompareLevel = new vx_rectangle_t[m_numLevels];
	for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
		// get image at current level
		vx_image image = vxGetPyramidLevel(m_pyramid, level);
		// get attributes and initialize pyramid level rectangles
		vx_uint32 width, height; size_t num_planes;
		ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &num_planes, sizeof(num_planes)));
		m_rectFullLevel[level].start_x = 0;
		m_rectFullLevel[level].start_y = 0;
		m_rectFullLevel[level].end_x = width;
		m_rectFullLevel[level].end_y = height;
		m_rectCompareLevel[level].start_x = (vx_uint32)ceil((double)m_rectCompare.start_x * width / m_width);
		m_rectCompareLevel[level].start_y = (vx_uint32)ceil((double)m_rectCompare.start_y * height / m_height);
		m_rectCompareLevel[level].end_x = (vx_uint32)floor((double)m_rectCompare.end_x * width / m_width);
		m_rectCompareLevel[level].end_y = (vx_uint32)floor((double)m_rectCompare.end_y * height / m_height);
		// compute image level frame size
		m_imageFrameSize[level] = 0;
		for (vx_uint32 plane = 0; plane < (vx_uint32)num_planes; plane++) {
			vx_imagepatch_addressing_t addr = { 0 };
			vx_uint8 * dst = NULL;
			if (vxAccessImagePatch(image, &m_rectFullLevel[level], plane, &addr, (void **)&dst, VX_READ_ONLY) == VX_SUCCESS) {
				vx_size width = (addr.dim_x * addr.scale_x) / VX_SCALE_UNITY;
				vx_size height = (addr.dim_y * addr.scale_y) / VX_SCALE_UNITY;
				vx_size width_in_bytes = (m_format == VX_DF_IMAGE_U1_AMD) ? ((width + 7) >> 3) : (width * addr.stride_x);
				m_imageFrameSize[level] += width_in_bytes * height;
				ERROR_CHECK(vxCommitImagePatch(image, &m_rectFullLevel[level], plane, &addr, (void *)dst));
			}
		}
		ERROR_CHECK(vxReleaseImage(&image));
		// update pyramid level frame size
		m_pyramidFrameSize += m_imageFrameSize[level];
	}

	// close files if already open
	for (vx_size level = 0; level < m_numLevels; level++) {
		if (m_fpReadImage && m_fpReadImage[level]) fclose(m_fpReadImage[level]);
		if (m_fpWriteImage && m_fpWriteImage[level]) fclose(m_fpWriteImage[level]);
		if (m_fpCompareImage && m_fpCompareImage[level]) fclose(m_fpCompareImage[level]);
	}

	// open files for read/write/compare
	if (m_fileNameRead.length() > 0 && !m_fpReadImage) {
		m_fpReadImage = new FILE *[m_numLevels]();
		for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
			// get width and height of current level
			vx_uint32 width = 0, height = 0;
			vx_image image = vxGetPyramidLevel(m_pyramid, level);
			ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
			ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
			ERROR_CHECK(vxReleaseImage(&image));
			// generate fileName with level, width, height in formatting and open the file
			char fileName[256]; snprintf(fileName, sizeof(fileName), m_fileNameRead.c_str(), level, width, height);
			m_fpReadImage[level] = fopen(fileName, "rb");
			if (!m_fpReadImage[level]) ReportError("ERROR: Unable to open: %s\n", fileName);
		}
	}
	if (m_fileNameWrite.length() > 0 && !m_fpWriteImage) {
		m_fpWriteImage = new FILE *[m_numLevels]();
		for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
			// get width and height of current level
			vx_uint32 width = 0, height = 0;
			vx_image image = vxGetPyramidLevel(m_pyramid, level);
			ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
			ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
			ERROR_CHECK(vxReleaseImage(&image));
			// generate fileName with level, width, height in formatting and open the file
			char fileName[256]; snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), level, width, height);
			m_fpWriteImage[level] = fopen(fileName, "wb");
			if (!m_fpWriteImage[level]) ReportError("ERROR: Unable to create: %s\n", fileName);
		}
	}
	if (m_fileNameCompare.length() > 0 && !m_fpCompareImage) {
		m_fpCompareImage = new FILE *[m_numLevels]();
		for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
			// get width and height of current level
			vx_uint32 width = 0, height = 0;
			vx_image image = vxGetPyramidLevel(m_pyramid, level);
			ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
			ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
			ERROR_CHECK(vxReleaseImage(&image));
			// generate fileName with level, width, height in formatting and open the file
			char fileName[256]; snprintf(fileName, sizeof(fileName), m_fileNameCompare.c_str(), level, width, height);
			m_fpCompareImage[level] = fopen(fileName, m_generateCheckSumForCompare ? "w" : (m_useCheckSumForCompare ? "r" : "rb"));
			if (!m_fpCompareImage[level]) ReportError("ERROR: Unable to %s: %s\n", m_generateCheckSumForCompare ? "create" : "open", fileName);
		}
		// allocate buffer for comparision
		if (!m_useCheckSumForCompare && !m_generateCheckSumForCompare) {
			NULLPTR_CHECK(m_bufForCompare = new vx_uint8[m_imageFrameSize[0]]);
		}
	}

	return 0;
}

int CVxParamPyramid::ReadFrame(int frameNumber)
{
	if (!m_fpReadImage) return VX_SUCCESS;

	for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
		// get image for current level and read image
		vx_image image = vxGetPyramidLevel(m_pyramid, level);
		int status = ReadImage(image, &m_rectFullLevel[level], m_fpReadImage[level]);
		vxReleaseImage(&image);
		if (status) return status;
	}

	return 0;
}

int CVxParamPyramid::WriteFrame(int frameNumber)
{
	if (!m_fpWriteImage) return VX_SUCCESS;

	for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
		// get image for current level and write image
		vx_image image = vxGetPyramidLevel(m_pyramid, level);
		int status = WriteImage(image, &m_rectFullLevel[level], m_fpWriteImage[level]);
		vxReleaseImage(&image);
		if (status) return status;
	}

	return 0;
}

int CVxParamPyramid::CompareFrame(int frameNumber)
{
	if (!m_fpCompareImage) return VX_SUCCESS;

	for (vx_uint32 level = 0; level < (vx_uint32)m_numLevels; level++) {
		// get image and fp for current level
		vx_image image = vxGetPyramidLevel(m_pyramid, level);
		FILE * fp = m_fpCompareImage[level];

		if (m_generateCheckSumForCompare)
		{ // generate checksum //////////////////////////////////////////
			char checkSumString[64];
			ComputeChecksum(checkSumString, image, &m_rectCompareLevel[level]);
			fprintf(fp, "%s\n", checkSumString);
		}
		else if (m_useCheckSumForCompare)
		{ // compare checksum //////////////////////////////////////////
			char checkSumStringRef[64] = { 0 };
			if (fscanf(fp, "%s", checkSumStringRef) != 1) {
				printf("ERROR: pyramid level#%d checksum missing for %s with frame#%d\n", level, GetVxObjectName(), frameNumber);
				throw - 1;
			}
			char checkSumString[64];
			ComputeChecksum(checkSumString, image, &m_rectCompareLevel[level]);
			if (!strcmp(checkSumString, checkSumStringRef)) {
				m_compareCountMatches++;
				if (m_verbose) printf("OK: pyramid level#%d CHECKSUM MATCHED for %s with frame#%d\n", level, GetVxObjectName(), frameNumber);
			}
			else {
				m_compareCountMismatches++;
				printf("ERROR: pyramid level#%d CHECKSUM MISMATCHED for %s with frame#%d [%s instead of %s]\n", level, GetVxObjectName(), frameNumber, checkSumString, checkSumStringRef);
				if (!m_discardCompareErrors) return -1;
			}
		}
		else
		{ // compare raw frames //////////////////////////////////////////
			// read data from frame
			size_t bytesRead = fread(m_bufForCompare, 1, m_imageFrameSize[level], fp);
			if (m_imageFrameSize[level] != bytesRead) {
				// no more data to compare
				ReportError("ERROR: pyramid level#%d data missing for %s in frame#%d\n", level, GetVxObjectName(), frameNumber);
			}
			// compare image to reference from file
			size_t errorPixelCountTotal = CompareImage(image, &m_rectCompareLevel[level], m_bufForCompare, m_comparePixelErrorMin, m_comparePixelErrorMax, frameNumber, nullptr);
			if (!errorPixelCountTotal) {
				m_compareCountMatches++;
				if (m_verbose) printf("OK: pyramid level#%d COMPARE MATCHED for %s with frame#%d\n", level, GetVxObjectName(), frameNumber);
			}
			else {
				m_compareCountMismatches++;
				if (!m_discardCompareErrors) return -1;
			}
		}
		vxReleaseImage(&image);
	}

	return 0;
}
