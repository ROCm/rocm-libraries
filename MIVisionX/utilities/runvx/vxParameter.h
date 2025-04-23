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


#ifndef __VX_PARAMETER_H__
#define __VX_PARAMETER_H__

#include "vxUtils.h"

// common constants
#define MAX_BUFFER_HANDLES        16

// constants local to vxParameter.h/cpp
#define MAX_FILE_NAME_LENGTH    1024
#define MAX_MODE_STRING_LENGTH    16

// CVxParameter: base-class for all type of objects
class CVxParameter
{
public:
	// constructor and destructor
	CVxParameter();
	virtual ~CVxParameter();

	// mechanism to pass global (vxEngine) parameter map to help access other objects by name
	void SetParamMap(std::map<std::string, CVxParameter *> * paramMap) { m_paramMap = paramMap; }
	void SetUserStructMap(std::map<std::string, vx_enum> * userStructMap){ m_userStructMap = userStructMap;  }
	bool IsUsingMultiFrameCapture(){ return m_usingMultiFrameCapture; }
	void SetCaptureFrameStart(vx_uint32 frameStart) { m_captureFrameStart = frameStart; }
	void SetVerbose(bool verbose) { m_verbose = verbose; }
	void SetDiscardCompareErrors(bool discardCompareErrors) { m_discardCompareErrors = discardCompareErrors; }
	bool IsVirtualObject() { return m_isVirtualObject; }

	// Initialize: create OpenVX object and further uses InitializeIO to input/output initialization
	//   desc: object description as specified on command-line or in script
	//   returns 0 on SUCCESS, else error code
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc) = 0;

	// InitializeIO: performs I/O initialization using the OpenVX object already created
	//   ref: OpenVX object already created
	//   io_params: I/O description as specified on command-line or in script
	//   returns 0 on SUCCESS, else error code
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params) = 0;

	// Finalize: for final initialization after vxVerifyGraph
	//   meant for querying object parameters which are not available until vxVerifyGraph
	virtual int Finalize() = 0;

	// get OpenVX object reference
	vx_reference& GetVxObject() { return m_vxObjRef; }
	const char * GetVxObjectName();

	// get OpenVX object type (e.g., VX_TYPE_IMAGE, VX_TYPE_SCALAR, ...)
	vx_enum GetVxObjectType() { return m_vxObjType; }

	// frame-level sync, read, write, and compare
	//   returns 0 on SUCCESS, else error code
	//   ReadFrame() returns +ve value to indicate data unavailability
	virtual int SyncFrame(int frameNumber);
	virtual int ReadFrame(int frameNumber) = 0;
	virtual int WriteFrame(int frameNumber) = 0;
	virtual int CompareFrame(int frameNumber) = 0;

	// helper functions
	//   GetDisplayName -- returns DISPLAY name specified as part of ":W,DISPLAY-<name>" I/O request
	//   DisableWaitForKeyPress -- mark that there is no need to wait at the end 
	//                             showing the last image output on OpenCV window
	// TBD: change getDisplayName to GetDisplayName
	// TBD: remove DisableWaitForKeyPress and have the final wait in top-level (i.e,. vxEngine)
	string getDisplayName() { return m_displayName; }
	virtual void DisableWaitForKeyPress();

protected:
	// global parameter map to access VX objects by name
	std::map<std::string, CVxParameter *> * m_paramMap;
	// keep track of objects for cross referencing across them (e.g., image needs arrays for displaying keypoints)
	static list<CVxParameter *> m_paramList;
	// global user defined struct map to access user defined structs
	std::map<std::string, vx_enum> * m_userStructMap;
	// DISPLAY name specified as part of ":W,DISPLAY-<name>" I/O request
	// NOTE: when not specified, this will be an empty string
	string m_displayName;
	// VX Object Type
	vx_enum m_vxObjType;
	// VX Object Reference
	vx_reference m_vxObjRef;
	char m_vxObjName[64];
	// I/O variables
	// TBD: add comment describing purpose of each of the variables below
	string m_fileNameRead;
	string m_fileNameWrite;
	string m_fileNameCompare;
	bool m_fileNameForReadHasIndex;
	bool m_fileNameForWriteHasIndex;
	bool m_fileNameForCompareHasIndex;
	FILE * m_fpRead;
	FILE * m_fpWrite;
	FILE * m_fpCompare;
	bool m_verbose;
	bool m_discardCompareErrors;
	bool m_isVirtualObject;
	bool m_useSyncOpenCLWriteDirective;
	// for multi-frame capture support
	bool m_usingMultiFrameCapture;
	vx_uint32 m_captureFrameStart;
	// Data shared for viewing
	struct ArrayItemForView { vx_enum itemtype; int colorIndex; int x, y; float strength; int w, h; };
	std::vector<ArrayItemForView> m_arrayListForView;

public:
	// utility functions for m_arrayListForView
	void ResetArrayListForView();
	void AddToArrayListForView(int colorIndex, int x, int y, float strength); // adds keypoint
	void AddToArrayListForView(int colorIndex, int x, int y); // adds coordinates2d
	size_t GetArrayListForViewCount() { return m_arrayListForView.size(); }
	const ArrayItemForView * GetArrayListForViewItemAt(size_t index) { return &m_arrayListForView[index]; }
};

// CVxParamDelay for vx_delay object
// TBD: this needs to be moved to separate file
class CVxParamDelay : public CVxParameter
{
public:
	CVxParamDelay();
	virtual ~CVxParamDelay();
	virtual int Initialize(vx_context context, vx_graph graph, const char * desc);
	virtual int InitializeIO(vx_context context, vx_graph graph, vx_reference ref, const char * io_params);
	virtual int Finalize();
	virtual int ReadFrame(int frameNumber);
	virtual int WriteFrame(int frameNumber);
	virtual int CompareFrame(int frameNumber);
	virtual int Shutdown();

private:
	// vx configuration
	vx_size m_count;
	// vx object
	vx_delay m_delay;
};

// parse the description of a data object and create parameter object: this function
// creates different kinds of CVxParamTYPE class objects depending upon the prefix
// in desc - for example when desc is "image:..." an object of type CVxParamImage
// will be created and initialized. It will return nullptr on error.
CVxParameter * CreateDataObject(vx_context context, vx_graph graph, map<string, CVxParameter *> * m_paramMap, map<string, vx_enum> * m_userStructMap, const char * desc, vx_uint32 captureFrameStart);
CVxParameter * CreateDataObject(vx_context context, vx_graph graph, vx_reference ref, const char * params, vx_uint32 captureFrameStart);

/*! \brief Parse parameter strings.
* \details This creates a top-level object context for OpenVX.
* \param [in] s The input string.
* \param [in] syntax The syntax description for error messaging.
* \param [in] fmt The format string: d(32-bit integer) D(64-bit integer) f(float) F(double) c(color-format) s(string upto 64-chars) S(string upto 256-chars).
* \param [in] ... Pointers to list of parameters.
* \return pointer to input string after processing the all the parameters
*/
const char * ScanParameters(const char * s, const char * syntax, const char * fmt, ...);

#endif /* __VX_PARAMETER_H__ */
