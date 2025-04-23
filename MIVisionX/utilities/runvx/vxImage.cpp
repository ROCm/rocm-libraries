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
#include "vxImage.h"

///////////////////////////////////////////////////////////////////////
// class CVxParamImage
//
CVxParamImage::CVxParamImage()
{
	// vx configuration
	m_vxObjType = VX_TYPE_IMAGE;
	m_format = VX_DF_IMAGE_U8;
	m_width = 0;
	m_height = 0;
	m_planes = 0;
	
	// I/O configuration
	m_frameSize = 0;
	m_bufForCompare = nullptr;
	m_displayName = "";
	m_repeatFrames = 0;
	m_countFrames = 0;
	m_useCheckSumForCompare = false;
	m_generateCheckSumForCompare = false;
	m_usingDisplay = false;
	m_usingWriter = false;
	m_countInitializeIO = 0;
	m_memory_type = VX_MEMORY_TYPE_NONE;
	m_active_handle = 0;
	memset(m_addr, 0, sizeof(m_addr));
	memset(m_memory_handle, 0, sizeof(m_memory_handle));
	m_swap_handles = false;

#if ENABLE_OPENCV
	m_cvCapDev = NULL;
	m_cvCapMat = NULL;
	m_cvDispMat = NULL;
	m_cvImage = NULL;
	m_cvWriter = NULL;
	m_cvReadEofOccured = false;
#endif
	m_cameraName[0] = 0;
	m_comparePixelErrorMin = 0;
	m_comparePixelErrorMax = 0;
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;

	// vx object
	m_image = nullptr;
	m_vxObjRef = nullptr;
	m_disableWaitForKeyPress = false;

	// reset video capture
	m_gotCaptureVideoSize = false;
	m_doNotResizeCapturedImages = false;
	m_captureWidth = 0;
	m_captureHeight = 0;
	m_colorIndexDefault = 0;
	m_radiusDefault = 2.0;
}

CVxParamImage::~CVxParamImage()
{
	Shutdown();
}

int CVxParamImage::Shutdown(void)
{
	if (m_compareCountMatches > 0 && m_compareCountMismatches == 0) {
		printf("OK: image %s MATCHED for %d frame(s) of %s\n", m_useCheckSumForCompare ? "CHECKSUM" : "COMPARE", m_compareCountMatches, GetVxObjectName());
	}
	if (m_image) {
		vxReleaseImage(&m_image);
		m_image = nullptr;
	}
	if (m_bufForCompare) {
		delete[] m_bufForCompare;
		m_bufForCompare = nullptr;
	}

	if (m_memory_type == VX_MEMORY_TYPE_HOST) {
		for (int active_handle = 0; active_handle < 2; active_handle++) {
			for (vx_size plane = 0; plane < m_planes; plane++) {
				if (m_memory_handle[active_handle][plane])
					free(m_memory_handle[active_handle][plane]);
				m_memory_handle[active_handle][plane] = nullptr;
			}
		}
	}
#if ENABLE_OPENCL
	else if (m_memory_type == VX_MEMORY_TYPE_OPENCL) {
		for (int active_handle = 0; active_handle < 2; active_handle++) {
			for (vx_size plane = 0; plane < m_planes; plane++) {
				if (m_memory_handle[active_handle][plane]) {
					int err = clReleaseMemObject((cl_mem)m_memory_handle[active_handle][plane]);
					if (err)
						ReportError("ERROR: clReleaseMemObject(*) failed (%d)\n", err);
				}
				m_memory_handle[active_handle][plane] = nullptr;
			}
		}
	}
#elif ENABLE_HIP
    else if (m_memory_type == VX_MEMORY_TYPE_HIP) {
		for (int active_handle = 0; active_handle < 2; active_handle++) {
			for (vx_size plane = 0; plane < m_planes; plane++) {
				if (m_memory_handle[active_handle][plane]) {
				    hipFree(m_memory_handle[active_handle][plane]);
				}
				m_memory_handle[active_handle][plane] = nullptr;
			}
		}
	}
#endif

	m_memory_type = VX_MEMORY_TYPE_NONE;

#if ENABLE_OPENCV
	bool changed_numCvUse = false;
	if (m_cvDispMat) {
		if (m_usingDisplay) {
			g_numCvUse--;
			changed_numCvUse = true;
		}
		delete (Mat *)m_cvDispMat;
		m_cvDispMat = NULL;
	}
	if (m_cvCapMat) {
		g_numCvUse--;
		changed_numCvUse = true;
		delete (Mat *)m_cvCapMat;
		m_cvCapMat = NULL;
	}
	if (m_cvCapDev) {
		delete (VideoCapture *)m_cvCapDev;
		m_cvCapDev = NULL;
	}
	if (m_cvImage) {
		g_numCvUse--;
		changed_numCvUse = true;
		delete (Mat *)m_cvImage;
		m_cvImage = NULL;
	}
	if (m_cvWriter) {
		delete (VideoWriter *)m_cvWriter;
		m_cvWriter = NULL;
	}
	if (changed_numCvUse && g_numCvUse == 0) {
		if (!m_disableWaitForKeyPress) {
			printf("Abort: Press any key to exit...\n"); fflush(stdout);
			waitKey(0);
		}
	}
#endif

	return 0;
}

void CVxParamImage::DisableWaitForKeyPress()
{
	m_disableWaitForKeyPress = true;
}

int CVxParamImage::Initialize(vx_context context, vx_graph graph, const char * desc)
{
	// get object parameters and create object
	char objType[64];
	const char * ioParams = ScanParameters(desc, "image|virtual-image|uniform-image|image-from-roi|image-from-handle|image-from-channel:", "s:", objType);
	if (!_stricmp(objType, "image") || !_stricmp(objType, "virtual-image") || !_stricmp(objType, "uniform-image") ||
		!_stricmp(objType, "image-virtual") || !_stricmp(objType, "image-uniform"))
	{
		// syntax: [virtual-|uniform-]image:<width>,<height>,<format>[,<range>][,<space>][:<io-params>]
		ioParams = ScanParameters(ioParams, "<width>,<height>,<format>", "d,d,c", &m_width, &m_height, &m_format);
		if (!_stricmp(objType, "uniform-image") || !_stricmp(objType, "image-uniform")) {
			memset(&m_uniformValue, 0, sizeof(m_uniformValue));
			if (*ioParams == ',') {
				ioParams++;
				if (*ioParams == '{') {
					// scan get 8-bit values for RGB/RGBX/YUV formats as {R;G;B}/{R;G;B;X}/{Y;U;V}
					const char * p = ioParams;
					for (int index = 0; index < 4 && (*p == '{' || *p == ';');) {
						int value = 0;
						p = ScanParameters(&p[1], "<byte>", "d", &value);
						m_uniformValue.reserved[index++] = (vx_uint8)value;
					}
					if (*p == '}') p++;
					ioParams = p;
				}
				else ioParams = ScanParameters(ioParams, "<uniform-pixel-value>", "D", m_uniformValue.reserved);
			}
			m_image = vxCreateUniformImage(context, m_width, m_height, m_format, &m_uniformValue);
		}
		else if (!_stricmp(objType, "image-virtual") || !_stricmp(objType, "virtual-image")) {
			m_image = vxCreateVirtualImage(graph, m_width, m_height, m_format);
			m_isVirtualObject = true;
		}
		else {
			m_image = vxCreateImage(context, m_width, m_height, m_format);
		}
		if (vxGetStatus((vx_reference)m_image) == VX_SUCCESS) {
			// process optional parameters: channel range and color space
			while (*ioParams == ',') {
				char enumName[64];
				ioParams = ScanParameters(ioParams, ",<enum>", ",s", enumName);
				vx_enum enumValue = ovxName2Enum(enumName);
				if (enumValue == VX_CHANNEL_RANGE_FULL || enumValue == VX_CHANNEL_RANGE_RESTRICTED) {
					ERROR_CHECK(vxSetImageAttribute(m_image, VX_IMAGE_ATTRIBUTE_RANGE, &enumValue, sizeof(enumValue)));
				}
				else if (enumValue == VX_COLOR_SPACE_BT601_525 || enumValue == VX_COLOR_SPACE_BT601_625 || enumValue == VX_COLOR_SPACE_BT709) {
					ERROR_CHECK(vxSetImageAttribute(m_image, VX_IMAGE_ATTRIBUTE_SPACE, &enumValue, sizeof(enumValue)));
				}
				else {
					ReportError("ERROR: invalid enum specified: %s\n", enumName);
				}
			}
		}
	}
	else if (!_stricmp(objType, "image-from-roi") || !_stricmp(objType, "image-roi")) {
		// syntax: image-from-roi:<master-image>,rect{<start-x>;<start-y>;<end-x>;<end-y>}[:<io-params>]
		char roi[64];
		ioParams = ScanParameters(ioParams, "<master-image>,rect{<start-x>;<start-y>;<end-x>;<end-y>}", "s,s", m_roiMasterName, roi);
		if (_strnicmp(roi, "rect{", 5) != 0) 
			ReportError("ERROR: invalid image-from-roi syntax: %s\n", desc);
		ScanParameters(&roi[4], "{<start-x>;<start-y>;<end-x>;<end-y>}", "{d;d;d;d}", &m_roiRegion.start_x, &m_roiRegion.start_y, &m_roiRegion.end_x, &m_roiRegion.end_y);
		auto it = m_paramMap->find(m_roiMasterName);
		if (it == m_paramMap->end())
			ReportError("ERROR: image [%s] doesn't exist for %s\n", m_roiMasterName, desc);
		vx_image masterImage = (vx_image)it->second->GetVxObject();
		m_image = vxCreateImageFromROI(masterImage, &m_roiRegion);
	}
	else if (!_stricmp(objType, "image-from-channel")) {
		// syntax: image-from-channel:<master-image>,<channel>[:<io-params>]
		char roi[64];
		ioParams = ScanParameters(ioParams, "<master-image>,<channel>", "s,s", m_roiMasterName, roi);
		vx_uint64 channel = 0;
		if (GetScalarValueFromString(VX_TYPE_ENUM, roi, &channel) < 0)
			ReportError("ERROR: invalid channel enum: %s\n", roi);
		auto it = m_paramMap->find(m_roiMasterName);
		if (it == m_paramMap->end())
			ReportError("ERROR: image [%s] doesn't exist for %s\n", m_roiMasterName, desc);
		vx_image masterImage = (vx_image)it->second->GetVxObject();
		m_image = vxCreateImageFromChannel(masterImage, (vx_enum)channel);
	}
	else if (!_stricmp(objType, "image-from-handle")) {
		// syntax: image-from-handle:<image-format>,{<dim-x>;<dim-y>;<stride-x>;<stride-y>}[+...],<memory-type>[:<io-params>]
		ioParams = ScanParameters(ioParams, "<format>,{<dim-x>;<dim-y>;<stride-x>;<stride-y>}", "c,{d;d;d;d}", &m_format, 
											&m_addr[0].dim_x, &m_addr[0].dim_y, &m_addr[0].stride_x, &m_addr[0].stride_y);
		m_width = m_addr[0].dim_x;
		m_height = m_addr[0].dim_y;
		m_planes = 1;
		while (ioParams[0] == ';' && ioParams[1] == '{' && m_planes < 4) {
			ioParams = ScanParameters(ioParams, ",{<dim-x>;<dim-y>;<stride-x>;<stride-y>}", "+{d;d;d;d}", &m_format, 
												&m_addr[m_planes].dim_x, &m_addr[m_planes].dim_y, &m_addr[m_planes].stride_x, &m_addr[m_planes].stride_y);
			m_planes++;
		}
		char type_str[64];
		ioParams = ScanParameters(ioParams, "<memory-type>", ",s", type_str);
		vx_uint64 type_value = 0;
		if (GetScalarValueFromString(VX_TYPE_ENUM, type_str, &type_value) < 0)
			ReportError("ERROR: invalid channel enum: %s\n", type_str);
		int alloc_flags = 0;
		if (ioParams[0] == ',') {
			ioParams = ScanParameters(ioParams, "<alloc-flag>", ",d", &alloc_flags);
		}
		bool align_memory = false;
		m_memory_type = (vx_enum)type_value;
		if (m_memory_type == VX_MEMORY_TYPE_HOST) {
			if (alloc_flags == 1) {
				memset(m_memory_handle, 0, sizeof(m_memory_handle));
			}
			else {
				// allocate all handles on host
				for (int active_handle = 0; active_handle < 2; active_handle++) {
					for (vx_size plane = 0; plane < m_planes; plane++) {
						vx_size size = (vx_size) m_addr[plane].dim_y * m_addr[plane].stride_y;
						m_memory_handle[active_handle][plane] = malloc(size);
						if (!m_memory_handle[active_handle][plane])
							ReportError("ERROR: malloc(%d) failed\n", (int)size);
					}
				}
			}
		}
#if ENABLE_OPENCL
		else if (m_memory_type == VX_MEMORY_TYPE_OPENCL) {
			if (alloc_flags == 1) {
				memset(m_memory_handle, 0, sizeof(m_memory_handle));
			}
			else {
				// allocate all handles on opencl
				cl_context opencl_context = nullptr;
				vx_status status = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &opencl_context, sizeof(opencl_context));
				if (status)
					ReportError("ERROR: vxQueryContext(*,VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,...) failed (%d)\n", status);
				for (int active_handle = 0; active_handle < 2; active_handle++) {
					for (vx_size plane = 0; plane < m_planes; plane++) {
						vx_size size = m_addr[plane].dim_y * m_addr[plane].stride_y;
						cl_int err = CL_SUCCESS;
						m_memory_handle[active_handle][plane] = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, size, NULL, &err);
						if (!m_memory_handle[active_handle][plane] || err)
							ReportError("ERROR: clCreateBuffer(*,CL_MEM_READ_WRITE,%d,NULL,*) failed (%d)\n", (int)size, err);
					}
				}
			}
		}
#elif ENABLE_HIP
            else if (m_memory_type == VX_MEMORY_TYPE_HIP) {
			if (alloc_flags == 1) {
				memset(m_memory_handle, 0, sizeof(m_memory_handle));
			}
			else {
				// allocate all handles on opencl
				int hip_device = -1;
				vx_status status = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE, &hip_device, sizeof(hip_device));
				if (status || hip_device<0 )
					ReportError("ERROR: vxQueryContext(*,VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE,...) failed (%d, %d)\n", status, hip_device);
				for (int active_handle = 0; active_handle < 2; active_handle++) {
					for (vx_size plane = 0; plane < m_planes; plane++) {
						vx_size size = m_addr[plane].dim_y * m_addr[plane].stride_y;
						hipError_t err = hipMalloc(&m_memory_handle[active_handle][plane], size);
						if (!m_memory_handle[active_handle][plane] || err != hipSuccess)
							ReportError("ERROR: hipMalloc(%d) failed (%d)\n", (int)size, err);
					}
				}
			}
		}
#endif
		else ReportError("ERROR: invalid memory-type enum: %s\n", type_str);
		m_active_handle = 0;
		m_image = vxCreateImageFromHandle(context, m_format, m_addr, m_memory_handle[m_active_handle], m_memory_type);
	}
	else ReportError("ERROR: unsupported image type: %s\n", desc);
	vx_status ovxStatus = vxGetStatus((vx_reference)m_image);
	if (ovxStatus != VX_SUCCESS) {
		printf("ERROR: image creation failed => %d (%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		if (m_image) vxReleaseImage(&m_image);
		throw - 1;
	}
	m_vxObjRef = (vx_reference)m_image;

	// io initialize
	return InitializeIO(context, graph, m_vxObjRef, ioParams);
}

int CVxParamImage::InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params)
{
	// save reference object and get object attributes
	m_vxObjRef = ref;
	m_image = (vx_image)m_vxObjRef;
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_WIDTH, &m_width, sizeof(m_width)));
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &m_height, sizeof(m_height)));
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_FORMAT, &m_format, sizeof(m_format)));
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_PLANES, &m_planes, sizeof(m_planes)));

	// first-time initialization
	if (m_countInitializeIO == 0) {
		// initialize compare region to complete image
		m_rectCompare.start_x = 0;
		m_rectCompare.start_y = 0;
		m_rectCompare.end_x = m_width;
		m_rectCompare.end_y = m_height;
	}
	m_countInitializeIO++;

	// process I/O requests
	if (*io_params == ':') io_params++;
	while (*io_params) {
		char ioType[64], fileName[256];
		io_params = ScanParameters(io_params, "<io-operation>,<parameter>", "s,S", ioType, fileName);
		// get file extension position in fileName
		int extpos = (int)strlen(fileName) - 1;
		while (extpos > 0 && fileName[extpos] != '.')
			extpos--;
		if (!_stricmp(ioType, "read") || !_stricmp(ioType, "camera"))
		{ // read request syntax: read,<fileNameOrURL>[,frames{<start>[;<count>;repeat]}|no-resize] or camera,<deviceNumber>
			int cameraDevice = -1;
			if (!_stricmp(ioType, "camera"))
				cameraDevice = atoi(fileName);
			// get optional repeat frame count and starting frame
			m_repeatFrames = 0;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",frames{<start>[;<count>;repeat]}|no-resize", ",s", option);
				if (!_strnicmp(option, "frames{", 7)) {
					int startFrame = 0, count = 0; char repeat[64] = { 0 };
					if (sscanf(&option[7], "%d;%d;%s", &startFrame, &count, repeat) >= 1) {
						repeat[6] = 0; // truncate since scanf will read all characters till the end of string into repeat
						m_captureFrameStart = startFrame;
						if (!_stricmp(repeat, "repeat") && (count > 0))
							m_repeatFrames = count;
					}
					else ReportError("ERROR: invalid image read/camera option: %s\n", option);
				}
				else if (!_stricmp(option, "no-resize")) {
					m_doNotResizeCapturedImages = true;
				}
				else ReportError("ERROR: invalid image read/camera option: %s\n", option);
			}
			// check if openCV video capture need to be used
			if (!_stricmp(&fileName[extpos], ".mp4") || !_stricmp(&fileName[extpos], ".avi") ||
				!_stricmp(&fileName[extpos], ".jpg") || !_stricmp(&fileName[extpos], ".jpeg") ||
				!_stricmp(&fileName[extpos], ".jpe") || !_stricmp(&fileName[extpos], ".png") ||
				!_stricmp(&fileName[extpos], ".bmp") || !_stricmp(&fileName[extpos], ".tif") ||
				!_stricmp(&fileName[extpos], ".ppm") || !_stricmp(&fileName[extpos], ".tiff") ||
				!_stricmp(&fileName[extpos], ".pgm") || !_stricmp(&fileName[extpos], ".pbm") ||
				!_strnicmp(fileName, "file://", 7) || !_strnicmp(fileName, "http://", 7) || !_strnicmp(fileName, "https://", 8) ||
				cameraDevice >= 0)
			{ // need OpenCV to process these read I/O requests ////////////////////
#if ENABLE_OPENCV
				if (m_format == VX_DF_IMAGE_RGB || m_format == VX_DF_IMAGE_U8) {
					// pen video capture device and mark multi-frame capture
                    m_usingMultiFrameCapture = false;
					VideoCapture * pCap = nullptr;
					if (cameraDevice >= 0) {
						pCap = new VideoCapture(cameraDevice);
					}
					else {
						pCap = new VideoCapture(fileName);
						// if single .jpg are is specified, mark as single-frame capture
						if (strstr(fileName, "%") == NULL && (!_stricmp(&fileName[strlen(fileName) - 4], ".avi") || !_stricmp(&fileName[extpos], ".mp4"))) {
                            m_usingMultiFrameCapture = true;
						}
					}
					m_cvCapDev = pCap;
					if (!pCap->isOpened()) {
						printf("ERROR: OpenCV device capture(%s) failed\n", fileName);
						throw - 1;
					}
#if 0 // TBD: disabled the check to avoid errors with video files
					if (pCap->get(CV_CAP_PROP_FRAME_WIDTH) != m_width || pCap->get(CV_CAP_PROP_FRAME_HEIGHT) != m_height) {
						printf("ERROR: OpenCV capture(%s) device is %dx%d whereas requested image is %dx%d\n", fileName, pCap->get(CV_CAP_PROP_FRAME_WIDTH), pCap->get(CV_CAP_PROP_FRAME_HEIGHT), m_width, m_height);
						throw - 1;
					}
#endif
					m_cvReadEofOccured = false;
					int cvMatType = (m_format == VX_DF_IMAGE_RGB) ? CV_8UC3 : CV_8U;
					m_cvCapMat = new Mat(m_height, m_width, cvMatType); //Mat(row, column, type)
					strcpy(m_cameraName, fileName);
					g_numCvUse++;
					// skip frames if requested
					if (m_captureFrameStart > 0) {
						printf("OK: skipping %d frames from %s\n", m_captureFrameStart, fileName); fflush(stdout);
						for (vx_uint32 i = 0; i < m_captureFrameStart; i++) {
							if (!m_cvReadEofOccured) {
								*(VideoCapture *)m_cvCapDev >> *(Mat *)m_cvCapMat;
								if (!((Mat *)m_cvCapMat)->data) {
									m_cvReadEofOccured = true;
									break;
								}
							}
						}
					}
				}
#else
				printf("ERROR: This build doesn't support CAMERA option\n");
				throw - 1;
#endif
			}
			else
			{ // raw frames reading /////////////////////////
				if (m_fpRead) {
					fclose(m_fpRead);
					m_fpRead = nullptr;
				}
				m_fileNameRead.assign(RootDirUpdated(fileName));
				m_fileNameForReadHasIndex = (m_fileNameRead.find("%") != m_fileNameRead.npos) ? true : false;
				// mark multi-frame capture enabled
				m_usingMultiFrameCapture = true;
			}
		}
		else if (!_stricmp(ioType, "init")) {
			if (fileName[0])
			{ // initialize image by reading from a file
				if (m_fpRead) {
					fclose(m_fpRead);
					m_fpRead = nullptr;
				}
				m_fileNameRead.assign(RootDirUpdated(fileName));
				m_fileNameForReadHasIndex = (m_fileNameRead.find("%") != m_fileNameRead.npos) ? true : false;
				m_usingMultiFrameCapture = true;
				// read two images into handles
				m_rectFull.start_x = 0;
				m_rectFull.start_y = 0;
				m_rectFull.end_x = m_width;
				m_rectFull.end_y = m_height;
				m_fpRead = fopen(m_fileNameRead.c_str(), "rb");
				if (!m_fpRead) ReportError("ERROR: unable to open: %s\n", m_fileNameRead.c_str());
				if (ReadImage(m_image, &m_rectFull, m_fpRead))
					ReportError("ERROR: unable to initialize image(%s) fully from %s\n", m_vxObjName, fileName);
				if (m_memory_type != VX_MEMORY_TYPE_NONE)
				{ // need to read second image if image was created from handle
					m_active_handle = !m_active_handle;
					vx_status status = vxSwapImageHandle(m_image, m_memory_handle[m_active_handle], m_memory_handle[!m_active_handle], m_planes);
					if (status)
						ReportError("ERROR: vxSwapImageHandle(%s,*,*,%d) failed (%d)\n", m_vxObjName, (int)m_planes, status);
					if (ReadImage(m_image, &m_rectFull, m_fpRead))
						ReportError("ERROR: unable to initialize second image(%s) fully from %s\n", m_vxObjName, fileName);
				}
				// close the file
				if (m_fpRead) {
					fclose(m_fpRead);
					m_fpRead = nullptr;
				}
				m_fileNameRead = "";
				m_fileNameForReadHasIndex = false;
			}
			// mark that swap handles needs to be executed for images created from handle
			if (m_memory_type != VX_MEMORY_TYPE_NONE)
				m_swap_handles = true;
		}
		else if (!_stricmp(ioType, "view") || !_stricmp(ioType, "write"))
		{ // write or view request syntax: write,<fileNameOrURL> OR view,<window-name>[,color-index{<#>}|,radius{<#>}]
			bool needDisplay = false;
			while (*io_params == ',') {
				char option[64];
				io_params = ScanParameters(io_params, ",color-index{index}|radius{radius}", ",s", option);
				if (!_strnicmp(option, "color-index{", 12)) {
					int colorIndex = 0;
					if (sscanf(&option[12], "%d", &colorIndex) == 1) {
						m_colorIndexDefault = colorIndex;
					}
					else ReportError("ERROR: invalid image read/camera option: %s\n", option);
				}
				else if (!_strnicmp(option, "radius{", 7)) {
					float radius = 2.0;
					if (sscanf(&option[7], "%f", &radius) == 1) {
						m_radiusDefault = radius;
					}
					else ReportError("ERROR: invalid image read/camera option: %s\n", option);
				}
			}
			if (!_stricmp(ioType, "view") || !_stricmp(&fileName[extpos], ".mp4") || !_stricmp(&fileName[extpos], ".avi"))
			{ // need OpenCV to process these write I/O requests ////////////////////
#if ENABLE_OPENCV
				if (!_stricmp(ioType, "view")) {
					m_usingDisplay = true;
					m_displayName.assign(fileName);
					namedWindow(m_displayName, WINDOW_AUTOSIZE);
					g_numCvUse++;
				}
				else {
					if (m_fpWrite) {
						fclose(m_fpWrite);
						m_fpWrite = nullptr;
					}
					m_fileNameWrite.assign(RootDirUpdated(fileName));
					VideoWriter * writer = new VideoWriter(m_fileNameWrite.c_str(), -1, 30, Size(m_width, m_height));
					m_cvWriter = (void *)writer;
					m_usingWriter = true;
				}
				// create Mat object
				int cvMatType = CV_8UC1;
				if (m_format == VX_DF_IMAGE_U8 || m_format == VX_DF_IMAGE_U1_AMD) cvMatType = CV_8UC1;
				else if (m_format == VX_DF_IMAGE_S16) cvMatType = CV_16UC1; // CV_16SC1 is not supported
				else if (m_format == VX_DF_IMAGE_U16) cvMatType = CV_16UC1;
				else if (m_format == VX_DF_IMAGE_RGB) cvMatType = CV_8UC3;
				else if (m_format == VX_DF_IMAGE_RGBX) cvMatType = CV_8UC4;
				else if (m_format == VX_DF_IMAGE_F32_AMD) cvMatType = CV_32FC1;
				else {
					printf("ERROR: display of image type (%4.4s) is not support. Exiting.\n", (const char *)&m_format);
					throw - 1;
				}
				m_cvDispMat = new Mat(m_height, m_width, cvMatType);
#else
				printf("ERROR: this feature requires OpenCV missing in this build\n");
				throw - 1;
#endif
			}
			else {
				m_fileNameWrite.assign(RootDirUpdated(fileName));
				m_fileNameForWriteHasIndex = (m_fileNameWrite.find("%") != m_fileNameWrite.npos) ? true : false;
			}
		}
		else if (!_stricmp(ioType, "compare"))
		{ // compare syntax: compare,fileName[,rect{<start-x>;<start-y>;<end-x>;<end-y>}][,err{<min>;<max>}][,checksum|checksum-save-instead-of-test]
			if (m_fpCompare) {
				fclose(m_fpCompare);
				m_fpCompare = nullptr;
			}
			// save the reference image fileName
			m_fileNameCompare.assign(RootDirUpdated(fileName));
			m_fileNameForCompareHasIndex = (m_fileNameCompare.find("%") != m_fileNameCompare.npos) ? true : false;
			// initialize pixel error range to exact match
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
				else ReportError("ERROR: invalid image compare option: %s\n", option);
			}
		}
		else if (!_stricmp(ioType, "directive") && (!_stricmp(fileName, "VX_DIRECTIVE_AMD_COPY_TO_OPENCL") || !_stricmp(fileName, "sync-cl-write"))) {
			m_useSyncOpenCLWriteDirective = true;
		}
		else ReportError("ERROR: invalid image operation: %s\n", ioType);
		if (*io_params == ':') io_params++;
		else if (*io_params) ReportError("ERROR: unexpected character sequence in parameter specification: %s\n", io_params);
	}

	return 0;
}

int CVxParamImage::Finalize()
{
	// get object attributes
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_WIDTH, &m_width, sizeof(m_width)));
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &m_height, sizeof(m_height)));
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_FORMAT, &m_format, sizeof(m_format)));
	ERROR_CHECK(vxQueryImage(m_image, VX_IMAGE_ATTRIBUTE_PLANES, &m_planes, sizeof(m_planes)));

	// set m_rectFull to full image region
	m_rectFull.start_x = 0;
	m_rectFull.start_y = 0;
	m_rectFull.end_x = m_width;
	m_rectFull.end_y = m_height;

	// initialize other parameters
	m_compareCountMatches = 0;
	m_compareCountMismatches = 0;

	// Calculate image width for single plane image:
	vx_size width_in_bytes = (m_planes == 1) ? CalculateImageWidthInBytes(m_image) : 0;

	// compute frame size in bytes
	m_frameSize = 0;
	for (vx_uint32 plane = 0; plane < (vx_uint32)m_planes; plane++) {
		vx_rectangle_t rect = { 0, 0, m_width, m_height };
		vx_imagepatch_addressing_t addr = { 0 };
		vx_uint8 * dst = NULL;
		if (vxAccessImagePatch(m_image, &m_rectFull, plane, &addr, (void **)&dst, VX_READ_ONLY) == VX_SUCCESS) {
			vx_size width = (addr.dim_x * addr.scale_x) / VX_SCALE_UNITY;
			vx_size height = (addr.dim_y * addr.scale_y) / VX_SCALE_UNITY;
			if (addr.stride_x != 0)
				width_in_bytes = (width * addr.stride_x);
			m_frameSize += width_in_bytes * height;
			ERROR_CHECK(vxCommitImagePatch(m_image, &m_rectFull, plane, &addr, (void *)dst));
		}
	}

	if (m_useSyncOpenCLWriteDirective) {
		// process user requested directives (required for uniform images)
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_image, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

int CVxParamImage::SyncFrame(int frameNumber)
{
	if (m_swap_handles) {
		// swap handles if requested for images created from handle
		m_active_handle = !m_active_handle;
		vx_status status = vxSwapImageHandle(m_image, m_memory_handle[m_active_handle], m_memory_handle[!m_active_handle], m_planes);
		if (status)
			ReportError("ERROR: vxSwapImageHandle(%s,*,*,%d) failed (%d)\n", m_vxObjName, (int)m_planes, status);
	}
	return 0;
}

int CVxParamImage::ReadFrame(int frameNumber)
{
#if ENABLE_OPENCV
	if (m_cvCapMat && m_cvCapDev) {
		// read image from camera
		if (m_cvReadEofOccured) {
			// no data available, report that no more frames available
			return 1;
		}
		VideoCapture * pCap = (VideoCapture *)m_cvCapDev;
		Mat * pMat = (Mat *)m_cvCapMat;
        //Get Mat type, bevor get a new frame from VideoCapture.
        int type = pMat->type();
		int timeout = 0;
		*pCap >> *pMat;
        //change Mat type.
        if(type == CV_8U){ /*CV_8U convert to gray*/
            cvtColor( *pMat, *pMat, CV_BGR2GRAY );
        }
		if (!pMat->data) {
			// no data available, report that no more frames available
			m_cvReadEofOccured = true;
			return 1;
		}
		else if (!m_gotCaptureVideoSize) {
			m_captureWidth = pMat->cols;
			m_captureHeight = pMat->rows;
			m_gotCaptureVideoSize = true;
			bool doResize = !m_doNotResizeCapturedImages && (pMat->cols != m_width || pMat->rows != m_height);
			printf("OK: capturing %dx%d image(s) into %dx%d RGB image buffer%s\n", m_captureWidth, m_captureHeight, m_width, m_height, doResize ? " with resize" : "");
		}

		// resize image using bicubic interpolation, if needed
		bool doResize = !m_doNotResizeCapturedImages && (pMat->cols != m_width || pMat->rows != m_height);
		if (doResize) {
			// resize the captured video to specifed buffer size
			resize(*pMat, *pMat, Size(m_width, m_height), 0, 0, INTER_CUBIC);
		}

		// copy Mat into image
		// NOTE: currently only supports U8, S16, RGB, RGBX image formats
		if (m_format == VX_DF_IMAGE_U8 || m_format == VX_DF_IMAGE_S16 || m_format == VX_DF_IMAGE_RGB || m_format == VX_DF_IMAGE_RGBX) {
			vx_rectangle_t rect = { 0, 0, min(m_width, (vx_uint32)pMat->cols), min(m_height, (vx_uint32)pMat->rows) };
			vx_imagepatch_addressing_t addr = { 0 };
			vx_uint8 * dst = NULL;
			ERROR_CHECK(vxAccessImagePatch(m_image, &rect, 0, &addr, (void **)&dst, VX_WRITE_ONLY));
			vx_int32 rowSize = ((vx_int32)pMat->step < addr.stride_y) ? (vx_int32)pMat->step : addr.stride_y;
			for (vx_uint32 y = 0; y < rect.end_y; y++) {
				if (m_format == VX_DF_IMAGE_RGB) {
					// convert BGR to RGB
					vx_uint8 * pDst = (vx_uint8 *)dst + y * addr.stride_y;
					vx_uint8 * pSrc = (vx_uint8 *)pMat->data + y * pMat->step;
					for (vx_uint32 x = 0; x < m_width; x++) {
						pDst[0] = pSrc[2];
						pDst[1] = pSrc[1];
						pDst[2] = pSrc[0];
						pDst += 3;
						pSrc += 3;
					}
				}
				else {
					memcpy(dst + y * addr.stride_y, pMat->data + y * pMat->step, rowSize);
				}
			}
			ERROR_CHECK(vxCommitImagePatch(m_image, &rect, 0, &addr, dst));
		}
	}
	else if (m_cvImage) {
		// read image from camera
		VideoCapture * pCap = (VideoCapture *)m_cvCapDev;
		Mat * pMat = (Mat *)m_cvImage;
		int timeout = 0;
		*pCap >> *pMat;
		if (!pMat->data) {
			printf("ERROR: Can't read camera input. Camera is not supported.\n");
			return -1;
		}

		vx_imagepatch_addressing_t addr = { 0 };
		vx_uint8 * dst = NULL;
		ERROR_CHECK(vxAccessImagePatch(m_image, &m_rectFull, 0, &addr, (void **)&dst, VX_WRITE_ONLY));
		vx_int32 rowSize = ((vx_int32)pMat->step < addr.stride_y) ? (vx_int32)pMat->step : addr.stride_y;
		for (vx_uint32 y = 0; y < m_height; y++) {
			memcpy(dst + y * addr.stride_y, pMat->data + y * pMat->step, rowSize);
		}
		ERROR_CHECK(vxCommitImagePatch(m_image, &m_rectFull, 0, &addr, dst));
	}
#endif

	// make sure that input file is open when OpenCV camera is not active and input filename is specified
#if ENABLE_OPENCV
	if (!m_cvImage)
#endif
	if (!m_fpRead) {
		if (m_fileNameRead.length() > 0) {
			char fileName[MAX_FILE_NAME_LENGTH];
			snprintf(fileName, sizeof(fileName), m_fileNameRead.c_str(), frameNumber, m_width, m_height);
			m_fpRead = fopen(fileName, "rb"); if (!m_fpRead) ReportError("ERROR: unable to open: %s\n", fileName);
			if (!m_fileNameForReadHasIndex && m_captureFrameStart > 0) {
				// skip to specified frame when starting frame is specified
				fseek(m_fpRead, m_captureFrameStart*(long)m_frameSize, SEEK_SET);
			}
		}
	}

	if (m_fpRead) {
		// update m_countFrames to be able to repeat after every m_repeatFrames
		if (m_repeatFrames != 0) {
			if (m_countFrames == m_repeatFrames) {
				// seek back to beginning after every m_repeatFrames frames
				fseek(m_fpRead, m_captureFrameStart*(long)m_frameSize, SEEK_SET);
				m_countFrames = 0;
			}
			else {
				m_countFrames++;
			}
		}

		// read all image planes into vx_image and check if EOF has occured while reading
		bool eofDetected = ReadImage(m_image, &m_rectFull, m_fpRead) ? true : false;

		// close file if file names has indices (i.e., only one frame per file requested)
		if (m_fileNameForReadHasIndex) {
			fclose(m_fpRead);
			m_fpRead = nullptr;
		}

		if (eofDetected) {
			// report the caller that end of file has been detected -- no frames available in input
			return 1;
		}
	}

	// process user requested directives
	if (m_useSyncOpenCLWriteDirective) {
		ERROR_CHECK_AND_WARN(vxDirective((vx_reference)m_image, VX_DIRECTIVE_AMD_COPY_TO_OPENCL), VX_ERROR_NOT_ALLOCATED);
	}

	return 0;
}

#if ENABLE_OPENCV
int CVxParamImage::ViewFrame(int frameNumber)
{
	if (m_cvDispMat) {
		// NOTE: supports only U8, S16, RGB, RGBX, F32 formats
		if (m_format == VX_DF_IMAGE_U8 || m_format == VX_DF_IMAGE_S16 || m_format == VX_DF_IMAGE_RGB || m_format == VX_DF_IMAGE_RGBX || m_format == VX_DF_IMAGE_F32_AMD || m_format == VX_DF_IMAGE_U1_AMD) {
			// copy image into Mat
			Mat * pMat = (Mat *)m_cvDispMat;
			vx_imagepatch_addressing_t addr = { 0 };
			vx_uint8 * src = NULL;
			ERROR_CHECK(vxAccessImagePatch(m_image, &m_rectFull, 0, &addr, (void **)&src, VX_READ_ONLY));
			if (m_format == VX_DF_IMAGE_U1_AMD) {
				for (vx_uint32 y = 0; y < m_height; y++) {
					vx_uint8 * pDst = (vx_uint8 *)pMat->data + y * pMat->step;
					vx_uint8 * pSrc = (vx_uint8 *)src + y * addr.stride_y;
					for (vx_uint32 x = 0; x < m_width; x++) {
						pDst[x] = (pSrc[x >> 3] & (1 << (x & 3))) ? 255u : 0;
					}
				}
			}
			else if (m_format == VX_DF_IMAGE_RGB) {
				for (vx_uint32 y = 0; y < m_height; y++) {
					vx_uint8 * pDst = (vx_uint8 *)pMat->data + y * pMat->step;
					vx_uint8 * pSrc = (vx_uint8 *)src + y * addr.stride_y;
					for (vx_uint32 x = 0; x < m_width; x++) {
						pDst[0] = pSrc[2];
						pDst[1] = pSrc[1];
						pDst[2] = pSrc[0];
						pDst += 3;
						pSrc += 3;
					}
				}
			}
			else {
				vx_int32 rowSize = ((vx_int32)pMat->step < addr.stride_y) ? (vx_int32)pMat->step : addr.stride_y;
				for (vx_uint32 y = 0; y < m_height; y++) {
					memcpy(pMat->data + y * pMat->step, src + y * addr.stride_y, rowSize);
				}
			}
			ERROR_CHECK(vxCommitImagePatch(m_image, &m_rectFull, 0, &addr, src));
			// convert grayscale Mat pMat to RGB Mat convertedToRGB:
			//   this is done in order to be able to plot keypoints with different colors
			Mat convertedToRGB(pMat->rows, pMat->cols, CV_8UC3, Scalar(0, 0, 255));
			Mat *pOutputImage = pMat;
			if (pMat->type() == CV_8UC1) { // TBD: need to support S16 images here
				cvtColor(*pMat, convertedToRGB, CV_GRAY2RGB);
				pOutputImage = &convertedToRGB;
			}

			// color table for key-points
			static int colorTable[][3] = { { 0, 255, 0 }, { 255, 0, 0 }, { 0, 255, 255 }, { 51, 51, 255 }, { 0, 0, 102 }, { 255, 255, 255 } };
			int colorIndex = m_colorIndexDefault;

			// list of golbal list
			std::vector<ArrayItemForView> kpList;
			// process objects with same window name as the image
			int overlayOffsetX = 10, overlayOffsetY = 10;
			for (auto it = m_paramList.begin(); it != m_paramList.end(); it++)
			{
				if (*it != this && !m_displayName.compare((*it)->getDisplayName()))
				{ // name of the window matched
					vx_delay delay = nullptr;
					vx_enum delayObjType = VX_TYPE_INVALID;
					if ((*it)->GetVxObjectType() == VX_TYPE_DELAY)
					{ // display the slot[0] of delay object
						delay = (vx_delay)(*it)->GetVxObject();
						ERROR_CHECK(vxQueryDelay(delay, VX_DELAY_ATTRIBUTE_TYPE, &delayObjType, sizeof(delayObjType)));
					}

					if ((*it)->GetVxObjectType() == VX_TYPE_ARRAY || (delay && delayObjType == VX_TYPE_ARRAY))
					{ // view the array data (processed in two steps) //////////////////////////// 
						// get array and itemtype and numitems
						CVxParameter * paramArray = nullptr;
						vx_array arr = nullptr;
						if (delay && delayObjType == VX_TYPE_ARRAY) {
							// view the array from slot[0]
							arr = (vx_array)vxGetReferenceFromDelay(delay, 0);
						}
						else {
							paramArray = *it;
							arr = (vx_array)paramArray->GetVxObject();
						}
						vx_enum itemtype = VX_TYPE_INVALID;
						vx_size arrayNumItems = 0;
						ERROR_CHECK(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
						ERROR_CHECK(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arrayNumItems, sizeof(arrayNumItems)));
						if (itemtype != VX_TYPE_KEYPOINT && itemtype != VX_TYPE_RECTANGLE && itemtype != VX_TYPE_COORDINATES2D)
							ReportError("ERROR: doesn't support viewing of specified array type\n");
						// add data items to the global kpList
						if (paramArray && paramArray->GetArrayListForViewCount() > 0)
						{ // use data items from the shared list for view, if available
							size_t count = paramArray->GetArrayListForViewCount();
							int colorIndexMax = colorIndex;
							for (auto index = 0; index < count; index++) {
								ArrayItemForView kpItem = *paramArray->GetArrayListForViewItemAt(index);
								// update kpItem.colorIndex and colorIndexMax
								int id = colorIndex + kpItem.colorIndex;
								if (id >= int(sizeof(colorTable) / sizeof(colorTable[0])))
									id = int(sizeof(colorTable) / sizeof(colorTable[0]) - 1);
								colorIndexMax = max(id, colorIndexMax);
								kpItem.colorIndex = id;
								// add the item to global list
								kpList.push_back(kpItem);
							}
							// update colorIndex for next item
							colorIndex = colorIndexMax;
							if (colorIndex < int(sizeof(colorTable) / sizeof(colorTable[0]) - 1))
								colorIndex++;
							// reset the list
							paramArray->ResetArrayListForView();
						}
						else if (arrayNumItems > 0)
						{ // use the data items from the vx_array object
							// initialize keypoint with colorIndex and update colorIndex for next keypoint set
							ArrayItemForView kpItem = { itemtype, colorIndex, 0, 0, 0.0f, 0, 0 };
							if (colorIndex < int(sizeof(colorTable) / sizeof(colorTable[0]) - 1))
								colorIndex++;
							// compute strength bounds and binSize for plotted point radius
							vx_size stride = 0;
							void *base = NULL;
							ERROR_CHECK(vxAccessArrayRange(arr, 0, arrayNumItems, &stride, &base, VX_READ_ONLY));
							if (itemtype == VX_TYPE_KEYPOINT) {
								size_t arrayNumTracked = 0;
								for (size_t i = 0; i < arrayNumItems; i++) {
									vx_keypoint_t * kp = &vxArrayItem(vx_keypoint_t, base, i, stride);
									if (kp->tracking_status) {
										kpItem.strength = kp->strength;
										kpItem.x = kp->x;
										kpItem.y = kp->y;
										kpList.push_back(kpItem);
										arrayNumTracked++;
									}
								}
								char message[128]; snprintf(message, sizeof(message), "%s [tracked %d/%d]", (*it)->GetVxObjectName(), (int)arrayNumTracked, (int)arrayNumItems);
								int H = 20;
								cv::putText(*pOutputImage, message, Point(overlayOffsetX + 0, overlayOffsetY + H - 6), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255, 0), 2, 8, false);
								cv::putText(*pOutputImage, message, Point(overlayOffsetX + 2, overlayOffsetY + H - 8), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255, 0), 1, 8, false);
								overlayOffsetY += H;
							}
							else if (itemtype == VX_TYPE_RECTANGLE) {
								for (size_t i = 0; i < arrayNumItems; i++) {
									vx_rectangle_t * kp = &vxArrayItem(vx_rectangle_t, base, i, stride);
									kpItem.x = kp->start_x;
									kpItem.y = kp->start_y;
									kpItem.w = kp->end_x - kp->start_x;
									kpItem.h = kp->end_y - kp->start_y;
									kpList.push_back(kpItem);
								}
							}
							else if (itemtype == VX_TYPE_COORDINATES2D) {
								for (size_t i = 0; i < arrayNumItems; i++) {
									vx_coordinates2d_t * kp = &vxArrayItem(vx_coordinates2d_t, base, i, stride);
									kpItem.x = kp->x;
									kpItem.y = kp->y;
									kpList.push_back(kpItem);
								}
							}
							ERROR_CHECK(vxCommitArrayRange(arr, 0, arrayNumItems, base));
						}
					}
					else if ((*it)->GetVxObjectType() == VX_TYPE_DISTRIBUTION)
					{ // view the distribution data ////////////////////////////
						vx_distribution dist = (vx_distribution)(*it)->GetVxObject();
						vx_size numBins = 0;
						vx_uint32 * hist = nullptr;
						ERROR_CHECK(vxQueryDistribution(dist, VX_DISTRIBUTION_ATTRIBUTE_BINS, &numBins, sizeof(numBins)));
						ERROR_CHECK(vxAccessDistribution(dist, (void **)&hist, VX_READ_ONLY));
						vx_uint32 maxValue = 0;
						for (size_t bin = 0; bin < numBins; bin++) {
							maxValue = max(maxValue, hist[bin]);
						}
						Rect box(overlayOffsetX, overlayOffsetY, 256, 100); overlayOffsetY += (box.height + 8);
						rectangle(*pOutputImage, Rect(box.x - 2, box.y - 2, box.width + 4, box.height + 4), Scalar(0, 0, 255), 1, 8);
						rectangle(*pOutputImage, Rect(box.x - 1, box.y - 1, box.width + 2, box.height + 2), Scalar(255, 0, 0), 1, 8);
						if (maxValue > 0) {
							int barWidth = box.width / (int)numBins;
							for (int bin = 0; bin < (int)numBins; bin++) {
								int barHeight = box.height * hist[bin] / maxValue;
								Rect bar(box.x + bin*barWidth, box.y + box.height - barHeight, barWidth, barHeight);
								rectangle(*pOutputImage, bar, Scalar(0, 255, 255), CV_FILLED, 8);
							}
						}
						ERROR_CHECK(vxCommitDistribution(dist, hist));
						// show the name of the object to the right
						char message[128]; snprintf(message, sizeof(message), "%s (distribution)", (*it)->GetVxObjectName());
						int H = 20;
						cv::putText(*pOutputImage, message, Point(box.x + box.width + 10, box.y + H - 6), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255, 0), 2, 8, false);
						cv::putText(*pOutputImage, message, Point(box.x + box.width + 12, box.y + H - 8), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255, 0), 1, 8, false);
					}
					else if ((*it)->GetVxObjectType() == VX_TYPE_LUT)
					{ // view the lut data ////////////////////////////
						vx_lut lut = (vx_lut)(*it)->GetVxObject();
						vx_enum data_type;
						ERROR_CHECK(vxQueryLUT(lut, VX_LUT_ATTRIBUTE_TYPE, &data_type, sizeof(data_type)));
						if (data_type == VX_TYPE_UINT8)
						{ // only supports 8-bit look-up tables
							vx_size count;
							vx_uint8 * data = nullptr;
							ERROR_CHECK(vxQueryLUT(lut, VX_LUT_ATTRIBUTE_COUNT, &count, sizeof(count)));
							ERROR_CHECK(vxAccessLUT(lut, (void **)&data, VX_READ_ONLY));
							vx_uint32 maxValue = 255;
							Rect box(overlayOffsetX, overlayOffsetY, 256, 256); overlayOffsetY += (box.height + 8);
							rectangle(*pOutputImage, Rect(box.x - 2, box.y - 2, box.width + 4, box.height + 4), Scalar(255, 0, 255), 1, 8);
							rectangle(*pOutputImage, Rect(box.x - 1, box.y - 1, box.width + 2, box.height + 2), Scalar(255, 255, 0), 1, 8);
							int barWidth = box.width / (int)count;
							for (int bin = 0; bin < (int)count; bin++) {
								int barHeight = box.height * data[bin] / maxValue;
								Rect bar(box.x + bin*barWidth, box.y + box.height - barHeight, barWidth, barHeight);
								rectangle(*pOutputImage, bar, Scalar(0, 255, 255), CV_FILLED, 8);
							}
							ERROR_CHECK(vxCommitLUT(lut, data));
							// show the name of the object to the right
							char message[128]; snprintf(message, sizeof(message), "%s (lut)", (*it)->GetVxObjectName());
							int H = 20;
							cv::putText(*pOutputImage, message, Point(box.x + box.width + 10, box.y + H - 6), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0, 0), 2, 8, false);
							cv::putText(*pOutputImage, message, Point(box.x + box.width + 12, box.y + H - 8), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255, 0), 1, 8, false);
						}
					}
					else if ((*it)->GetVxObjectType() == VX_TYPE_SCALAR)
					{ // view the scalar data ////////////////////////////
						char value[MAX_SCALAR_TO_STRING_SIZE];
						vx_scalar scalar = (vx_scalar)(*it)->GetVxObject();
						ReadScalarToString(scalar, value);
						char message[2*MAX_SCALAR_TO_STRING_SIZE]; snprintf(message, sizeof(message), "%s = %s", (*it)->GetVxObjectName(), value);
						int H = 20;
						cv::putText(*pOutputImage, message, Point(overlayOffsetX+0, overlayOffsetY+H-6), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,   0, 255, 0), 2, 8, false);
						cv::putText(*pOutputImage, message, Point(overlayOffsetX+2, overlayOffsetY+H-8), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255, 0), 1, 8, false);
						overlayOffsetY += H;
					}
				}
			}
			// add keypoints from user specified file(s)
			for (auto it = m_viewKeypointFilenameList.begin(); it != m_viewKeypointFilenameList.end(); it++)
			{
				// initialize keypoint with colorIndex and update colorIndex for next keypoint set
				ArrayItemForView kpItem = { VX_TYPE_KEYPOINT, colorIndex, 0, 0, 0.0f, 0, 0 };
				if (colorIndex < int(sizeof(colorTable) / sizeof(colorTable[0]) - 1))
					colorIndex++;
				// get list of keypoints from the user specified file
				char fileName[512];
				snprintf(fileName, sizeof(fileName), it->c_str(), frameNumber);
				FILE * fp = fopen(fileName, "r");
				if (!fp) ReportError("ERROR: unable to open '%s'\n", fileName);
				char line[256];
				while (fgets(line, sizeof(line), fp) != NULL) {
					if (sscanf(line, "%d%d%f", &kpItem.x, &kpItem.y, &kpItem.strength) == 3) {
						kpList.push_back(kpItem);
					}
				}
				fclose(fp);
			}
			// compute strength bounds and binSize for computing keypoint radius
			float minStrength = FLT_MAX, maxStrength = FLT_MIN;
			for (auto it = kpList.begin(); it != kpList.end(); it++) {
				if (it->itemtype == VX_TYPE_KEYPOINT) {
					float strength = it->strength;
					minStrength = min(strength, minStrength);
					maxStrength = max(strength, maxStrength);
				}
			}
			float binSize = (maxStrength - minStrength) / 5;
			// plot the points
			for (auto it = kpList.begin(); it != kpList.end(); it++) {
				Scalar color(colorTable[it->colorIndex][0], colorTable[it->colorIndex][1], colorTable[it->colorIndex][2]);
				if (it->itemtype == VX_TYPE_KEYPOINT) {
					// compute the radius of point using strength and binSize
					float strength = it->strength;
					double radius = m_radiusDefault;
					if (strength > minStrength) {
						radius += 2.0 * floor((strength - minStrength) / binSize);
					}
					// plot the points with key-point location as center of circle
					Point center(it->x, it->y);
					circle(*pOutputImage, center, (int)radius, Scalar(0, 0, 0), 1, 8);
					circle(*pOutputImage, center, (int)radius + 1, color, 1, 8);
				}
				else if (it->itemtype == VX_TYPE_RECTANGLE) {
					// plot the rectangle
					Rect rec(it->x, it->y, it->w, it->h);
					rectangle(*pOutputImage, rec, color, 1, 8);
				}
				else if (it->itemtype == VX_TYPE_COORDINATES2D) {
					// plot the points with small circle
					float radius = m_radiusDefault;
					Point center(it->x, it->y);
					circle(*pOutputImage, center, (int)radius, Scalar(0, 0, 0), 1, 8);
					circle(*pOutputImage, center, (int)radius + 1, color, 1, 8);
				}
			}
			// show the image and points (if requested)
			if (m_usingDisplay) {
				imshow(m_displayName, *pOutputImage);
			}
			if (m_usingWriter) {
				((VideoWriter *)m_cvWriter)->write(*pOutputImage);
			}
		}
	}
	return 0;
}
#endif

int CVxParamImage::WriteFrame(int frameNumber)
{
#if ENABLE_OPENCV
	if (ViewFrame(frameNumber) < 0)
		return -1;
#endif

	if (!m_fpWrite) {
		if (m_fileNameWrite.length() > 0 && !m_usingWriter) {
			char fileName[MAX_FILE_NAME_LENGTH];
			snprintf(fileName, sizeof(fileName), m_fileNameWrite.c_str(), frameNumber, m_width, m_height);
#if ENABLE_OPENCV
            // check if openCV imwrite need to be used
            int extpos = (int)strlen(fileName) - 1;
            while (extpos > 0 && fileName[extpos] != '.')
                extpos--;
 
            if (!_stricmp(&fileName[extpos], ".jpg") || !_stricmp(&fileName[extpos], ".jpeg") ||
                !_stricmp(&fileName[extpos], ".jpe") || !_stricmp(&fileName[extpos], ".png") ||
                !_stricmp(&fileName[extpos], ".bmp") || !_stricmp(&fileName[extpos], ".tif") ||
                !_stricmp(&fileName[extpos], ".ppm") || !_stricmp(&fileName[extpos], ".tiff") ||
                !_stricmp(&fileName[extpos], ".pgm") || !_stricmp(&fileName[extpos], ".pbm"))
            {
                WriteImageCompressed(m_image, &m_rectFull,fileName);
                return 0;
            }
#endif
			m_fpWrite = fopen(fileName, "wb+");
			if (!m_fpWrite) ReportError("ERROR: unable to create: %s\n", fileName);
		}
	}

	if (m_fpWrite) {
		// write vx_image into file
		WriteImage(m_image, &m_rectFull, m_fpWrite);

		// close the file if one frame gets written per file
		if (m_fileNameForWriteHasIndex && m_fpWrite) {
			fclose(m_fpWrite);
			m_fpWrite = nullptr;
		}
	}

	return 0;
}

int CVxParamImage::CompareFrame(int frameNumber)
{
	// make sure that compare reference data is opened
	if (!m_fpCompare) {
		if (m_fileNameCompare.length() > 0) {
			snprintf(m_fileNameCompareCurrent, sizeof(m_fileNameCompareCurrent), m_fileNameCompare.c_str(), frameNumber, m_width, m_height);
			if (m_generateCheckSumForCompare) {
				m_fpCompare = fopen(m_fileNameCompareCurrent, "w");
				if (!m_fpCompare) ReportError("ERROR: unable to create: %s\n", m_fileNameCompareCurrent);
			}
			else {
				m_fpCompare = fopen(m_fileNameCompareCurrent, "rb");
				if (!m_fpCompare) ReportError("ERROR: unable to open: %s\n", m_fileNameCompareCurrent);
			}
		}
	}
	if (!m_fpCompare) return 0;

	if (m_generateCheckSumForCompare)
	{ // generate checksum //////////////////////////////////////////
		char checkSumString[64];
		ComputeChecksum(checkSumString, m_image, &m_rectCompare);
		fprintf(m_fpCompare, "%s\n", checkSumString);
	}
	else if (m_useCheckSumForCompare)
	{ // compare checksum //////////////////////////////////////////
		char checkSumStringRef[64] = { 0 };
		if (fscanf(m_fpCompare, "%s", checkSumStringRef) != 1) {
			printf("ERROR: image checksum missing for frame#%d in %s\n", frameNumber, m_fileNameCompareCurrent);
			throw - 1;
		}
		char checkSumString[64];
		ComputeChecksum(checkSumString, m_image, &m_rectCompare);
		if (!strcmp(checkSumString, checkSumStringRef)) {
			m_compareCountMatches++;
			if (m_verbose) printf("OK: image CHECKSUM MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, m_fileNameCompareCurrent);
		}
		else {
			m_compareCountMismatches++;
			printf("ERROR: image CHECKSUM MISMATCHED for %s with frame#%d of %s [%s instead of %s]\n", GetVxObjectName(), frameNumber, m_fileNameCompareCurrent, checkSumString, checkSumStringRef);
			if (!m_discardCompareErrors) return -1;
		}
	}
	else
	{ // compare raw frames //////////////////////////////////////////
		// make sure buffer has been allocated
		if (!m_bufForCompare) {
			NULLPTR_CHECK(m_bufForCompare = new vx_uint8[m_frameSize]);
		}
		// read data from frame
		if (m_frameSize != fread(m_bufForCompare, 1, m_frameSize, m_fpCompare)) {
			// no more data to compare
			ReportError("ERROR: image data missing for frame#%d in %s\n", frameNumber, m_fileNameCompareCurrent);
		}
		// compare image to reference from file
		size_t errorPixelCountTotal = CompareImage(m_image, &m_rectCompare, m_bufForCompare, m_comparePixelErrorMin, m_comparePixelErrorMax, frameNumber, m_fileNameCompareCurrent);
		if (!errorPixelCountTotal) {
			m_compareCountMatches++;
			if (m_verbose) printf("OK: image COMPARE MATCHED for %s with frame#%d of %s\n", GetVxObjectName(), frameNumber, m_fileNameCompareCurrent);
		}
		else {
			m_compareCountMismatches++;
			if (!m_discardCompareErrors) return -1;
		}
	}

	// close the file if user requested separate file for each compare data
	if (m_fileNameForCompareHasIndex) {
		fclose(m_fpCompare);
		m_fpCompare = nullptr;
	}

	return 0;
}
