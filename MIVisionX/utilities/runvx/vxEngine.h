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


#ifndef CVX_ENGINE_H
#define CVX_ENGINE_H

#include "vxParameter.h"

class CVxEngine {
public:
	CVxEngine();
	virtual ~CVxEngine();
	int Initialize(int paramCount, int defaultTargetAffinity, int defaultTargetInfo, bool enableScheduleGraph, bool disableVirtual, bool enableFullProfile, bool disableNodeFlushForCL, std::string discardCommandList);
	void SetConfigOptions(bool verbose, bool discardCompareErrors, bool enableDumpProfile, bool enableDumpGDF, int waitKeyDelayInMilliSeconds);
	void SetFrameCountOptions(bool enableMultiFrameProcessing, bool framesEofRequested, bool frameCountSpecified, int frameStart, int frameEnd);
	int SetGraphOptimizerFlags(vx_uint32 graph_optimizer_flags);
	void SetDumpDataConfig(std::string dumpDataConfig);
	int SetParameter(int index, const char * param);
	int Shell(int level, FILE * fp = nullptr);
	int BuildAndProcessGraph(int level, char * graphScript, bool importMode);
	int SetImportedData(vx_reference ref, const char * name, const char * params);
	int Shutdown();
	void DisableWaitForKeyPress();

protected:
	vx_context getContext();
	void viewParameters();
	int BuildAndProcessGraphFromLine(int level, char * line);
	int ProcessGraph(std::vector<const char *> * graphNameList = nullptr, size_t beginIndex = 0);
	int DumpInternalGDF();
	int DumpGraphInfo(const char * graphName = nullptr);
	int SyncFrame(int frameNumber);
	int ReadFrame(int frameNumber);
	int WriteFrame(int frameNumber);
	int CompareFrame(int frameNumber);
	void MeasureFrame(int frameNumber, int status, std::vector<vx_graph>& graphList);
	float GetMedianRunTime();
	void PerformanceStatistics(int status, std::vector<vx_graph>& graphList);
	bool IsUsingMultiFrameCapture();
	void ReleaseAllVirtualObjects();
	int RenameData(const char * oldName, const char * newName);

private:
	// implementation specific data
	// m_paramMap - holds names and pointers to all data objects
	// m_paramCount - number of data objects on command-line
	map<string, CVxParameter *> m_paramMap;
	map<string, vx_enum> m_userStructMap;
	int m_paramCount;
	vx_context m_context;
	vx_graph m_graph;
	bool m_enableScheduleGraph;
	std::vector<float> m_timeMeasurements;
	std::vector<std::string> m_graphAutoAgeList;
	std::map<std::string, vx_graph> m_graphNameListForObj;
	std::map<std::string, std::vector<std::string> > m_graphNameListForAge;
	// configuration flags
	bool m_usingMultiFrameCapture;
	bool m_disableVirtual;
	bool m_verbose;
	bool m_discardCompareErrors;
	bool m_enableDumpProfile;
	bool m_enableDumpGDF;
	bool m_enableMultiFrameProcessing;
	bool m_framesEofRequested;
	bool m_frameCountSpecified;
	int m_frameStart;
	int m_frameEnd;
	int m_waitKeyDelayInMilliSeconds;
	bool m_disableCompare;
	int m_numGraphProcessed;
	bool m_graphVerified;
	bool m_dumpDataEnabled;
	std::string m_dumpDataFilePrefix;
	std::string m_dumpDataObjectList;
	int m_dumpDataCount;
	std::string m_discardCommandList;
	bool m_setBorderMode;
	std::string m_cmdBorderMode;
};

void PrintHelpGDF(const char * command = nullptr);

#endif /* CVX_ENGINE_H*/
