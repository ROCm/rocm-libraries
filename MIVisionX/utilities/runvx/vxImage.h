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


#ifndef __VX_IMAGE_H__
#define __VX_IMAGE_H__

#include "vxParameter.h"
#include "vxParamHelper.h"
#include "vxUtils.h"

class CVxParamImage : public CVxParameter
{
public:
	CVxParamImage();
	virtual ~CVxParamImage();
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);
	virtual int Finalize();
	virtual int SyncFrame(int frameNumber);
	virtual int ReadFrame(int frameNumber);
	virtual int WriteFrame(int frameNumber);
	virtual int CompareFrame(int frameNumber);
	virtual int Shutdown();
	virtual void DisableWaitForKeyPress();

protected:
#if ENABLE_OPENCV
	int ViewFrame(int frameNumber);
#endif

private:
	// vx configuration
	vx_df_image m_format;
	vx_uint32 m_width;
	vx_uint32 m_height;
	vx_size m_planes;
	// I/O configuration
	int m_repeatFrames;
	int m_countFrames;
	char m_cameraName[256];
	vx_enum m_memory_type;
	int m_active_handle;
	vx_imagepatch_addressing_t m_addr[4];
	void * m_memory_handle[2][4];
	bool m_swap_handles;
#if ENABLE_OPENCV
	void * m_cvCapDev;
	void * m_cvCapMat;
	void * m_cvWriter;
	cv::Mat * m_cvDispMat;
	cv::Mat * m_cvImage;
	bool m_cvReadEofOccured;
#endif
	// vx object
	vx_image m_image;
	char m_roiMasterName[64];      // name of ROI image master
	vx_rectangle_t m_roiRegion;    // rectangle used to save ROI image dimensions
	vx_rectangle_t m_rectFull;     // rectangle with full image size for use by access/commit
	vx_pixel_value_t m_uniformValue; // uniform image value

	// image I/O
	size_t m_frameSize;
	std::list<std::string> m_viewKeypointFilenameList;
	float m_comparePixelErrorMin;
	float m_comparePixelErrorMax;
	vx_rectangle_t m_rectCompare;  // rectangle used to save rectangular region used for compare
	vx_uint8 * m_bufForCompare;
	bool m_useCheckSumForCompare;
	bool m_generateCheckSumForCompare;
	char m_fileNameCompareCurrent[256];
	int m_compareCountMatches;
	int m_compareCountMismatches;
	bool m_disableWaitForKeyPress;
	bool m_usingDisplay;
	bool m_usingWriter;
	bool m_gotCaptureVideoSize;
	bool m_doNotResizeCapturedImages;
	vx_uint32 m_captureWidth;
	vx_uint32 m_captureHeight;
	int m_countInitializeIO;
	int m_colorIndexDefault;
	float m_radiusDefault;
};


#endif /* __VX_IMAGE_H__ */
