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
#include "vxEngine.h"
#include "vxEngineUtil.h"
#include "vxParamHelper.h"

#define MAX_GDF_LEVELS  4
#define NANO2MILLISECONDS(t) (((float)t)*0.000001f)

enum {
	BUILD_GRAPH_SUCCESS = 0,
	BUILD_GRAPH_LAUNCHED = 1,
	BUILD_GRAPH_EXIT = 2,
	BUILD_GRAPH_ABORT = 3,
	BUILD_GRAPH_FAILURE = -1,
};

void RemoveVirtualKeywordFromParamDescription(std::string& paramDesc)
{
	size_t pos = paramDesc.find("-virtual:");
	if (pos != std::string::npos) {
		paramDesc.erase(pos, 8);
	}
	else if ((pos = paramDesc.find("virtual-")) != std::string::npos) {
		paramDesc.erase(pos, 8);
	}
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
	size_t len = strlen(string);
	if (len > 0) {
		printf("%s", string);
		if (string[len - 1] != '\n')
			printf("\n");
		fflush(stdout);
	}
}

CVxEngine::CVxEngine()
{
	m_paramCount = 0;
	m_usingMultiFrameCapture = false;
	m_disableVirtual = false;
	m_verbose = false;
	m_discardCompareErrors = false;
	m_enableScheduleGraph = false;
	m_enableDumpProfile = false;
	m_enableDumpGDF = false;
	m_enableMultiFrameProcessing = false;
	m_framesEofRequested = false;
	m_frameCountSpecified = false;
	m_frameStart = 0;
	m_frameEnd = 0;
	m_waitKeyDelayInMilliSeconds = 1; // default is 1ms
	m_disableCompare = false;
	m_numGraphProcessed = 0;
	m_graphVerified = false;
	m_dumpDataEnabled = false;
	m_dumpDataCount = 0;
	m_setBorderMode = false;
}

CVxEngine::~CVxEngine()
{
	Shutdown();
}

int CVxEngine::Initialize(int argCount, int defaultTargetAffinity, int defaultTargetInfo, bool enableScheduleGraph, bool disableVirtual, bool enableFullProfile, bool disableNodeFlushForCL, std::string discardCommandList)
{
	// save configuration
	m_paramCount = argCount;
	m_enableScheduleGraph = enableScheduleGraph;
	vx_status ovxStatus = VX_SUCCESS;
	m_disableVirtual = disableVirtual;
	// add ','s at the end of command list to be discarded to ease string search
	m_discardCommandList = ",";
	m_discardCommandList += discardCommandList;
	m_discardCommandList += ",";

	// create OpenVX context, register log_callback, and show implementation
	vxRegisterLogCallback(nullptr, log_callback, vx_false_e); // to receive log messages from vxCreateContext (if any)
	m_context = vxCreateContext();
	if (vxGetStatus((vx_reference)m_context)) { printf("ERROR: vxCreateContext failed\n"); throw - 1; }
	vxRegisterLogCallback(m_context, log_callback, vx_false_e);
	char name[VX_MAX_IMPLEMENTATION_NAME];
	ovxStatus = vxQueryContext(m_context, VX_CONTEXT_ATTRIBUTE_IMPLEMENTATION, name, VX_MAX_IMPLEMENTATION_NAME);
	if (ovxStatus) {
		printf("ERROR: vxQueryContext(VX_CONTEXT_ATTRIBUTE_IMPLEMENTATION) failed (%d:%s)\n", ovxStatus, ovxEnum2Name(ovxStatus));
		return -1;
	}
	printf("OK: using %s\n", name);

	// set default target affinity, if request
	if (defaultTargetAffinity) {
		AgoTargetAffinityInfo attr_affinity = { 0 };
		attr_affinity.device_type = defaultTargetAffinity;
		attr_affinity.device_info = defaultTargetInfo;
		vx_status status = vxSetContextAttribute(m_context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &attr_affinity, sizeof(attr_affinity));
		if (status) {
			printf("ERROR: vxSetContextAttribute(VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,%d) failed (%d:%s)\n", defaultTargetAffinity, status, ovxEnum2Name(status));
			throw - 1;
		}
	}

	// create graph
	m_graphVerified = false;
	m_graph = vxCreateGraph(m_context);
	if (vxGetStatus((vx_reference)m_graph)){ printf("ERROR: vxCreateGraph failed\n"); throw - 1; }

	// select graph options
	if(enableFullProfile) {
	    vxDirective((vx_reference)m_graph, VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE);
	}
	if(disableNodeFlushForCL) {
	    vxDirective((vx_reference)m_graph, VX_DIRECTIVE_AMD_DISABLE_GPU_FLUSH);
	}

	return 0;
}

int CVxEngine::SetGraphOptimizerFlags(vx_uint32 graph_optimizer_flags)
{
	// set optimizer flags
	vx_status status = vxSetGraphAttribute(m_graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &graph_optimizer_flags, sizeof(graph_optimizer_flags));
	if (status)
		ReportError("ERROR: vxSetGraphAttribute(*,VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS,%d) failed (%d:%s)\n", graph_optimizer_flags, status, ovxEnum2Name(status));
	return 0;
}

void CVxEngine::SetDumpDataConfig(std::string dumpDataConfig)
{
	m_dumpDataEnabled = false;
	// extract dumpDataFilePrefix and dumpDataObjectList (with added ',' at the end)
	// to check if an object type is specified, use dumpDataObjectList.find(",<object-type>,") != npos
	size_t pos = dumpDataConfig.find(",");
	if(pos == std::string::npos) return;
	m_dumpDataFilePrefix = dumpDataConfig.substr(0,pos);
	m_dumpDataObjectList = dumpDataConfig.substr(pos) + ",";
	m_dumpDataEnabled = true;
}

vx_context CVxEngine::getContext()
{
	return m_context;
}

int CVxEngine::SetParameter(int index, const char * param)
{
	std::string paramDesc;
	if (m_disableVirtual) {
		paramDesc = param;
		RemoveVirtualKeywordFromParamDescription(paramDesc);
		param = paramDesc.c_str();
	}
	if(!_strnicmp(param, "attr:border_mode:", 17)){
		m_setBorderMode = true;
		m_cmdBorderMode = param;
	}
	else{
		CVxParameter * parameter = CreateDataObject(m_context, m_graph, &m_paramMap, &m_userStructMap, param, m_frameStart);
		if (!parameter) {
			printf("ERROR: unable to create parameter# %d\nCheck syntax: %s\n", index, param);
			return -1;
		}
	
		vx_reference ref;
		char name[16];
		snprintf(name, sizeof(name), "$%d", index+1);
		m_paramMap.insert(pair<string, CVxParameter *>(name, parameter));
		ref = m_paramMap[name]->GetVxObject();
		vxSetReferenceName(ref, name);
	}
	return 0;
}

int CVxEngine::SetImportedData(vx_reference ref, const char * name, const char * params)
{
	if (params) {
		CVxParameter * obj = CreateDataObject(m_context, m_graph, ref, params, m_frameStart);
		if (!obj) {
			printf("ERROR: CreateDataObject(*,*,*,%s) failed\n", params);
			return -1;
		}
		m_paramMap.insert(pair<string, CVxParameter *>(name, obj));
		vxSetReferenceName(ref, name);
	}
	return 0;
}

void VX_CALLBACK CVxEngine_data_registry_callback_f(void * obj, vx_reference ref, const char * name, const char * params)
{
	int status = ((CVxEngine *)obj)->SetImportedData(ref, name, params);
	if (status) {
		printf("ERROR: SetImportedData(*,%s,%s) failed (%d)\n", name, params, status);
		throw -1;
	}
}

void CVxEngine::ReleaseAllVirtualObjects()
{
	// close all virtual objects
	std::vector<std::string> virtualNameList;
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it) {
		if (it->second->IsVirtualObject()) {
			virtualNameList.push_back(it->first);
			delete it->second;
		}
	}
	for (auto i = 0; i < virtualNameList.size(); i++) {
		m_paramMap.erase(virtualNameList[i]);
	}
}

int CVxEngine::DumpInternalGDF()
{
	vx_reference ref[64] = { 0 };
	int num_ref = 0;
	for (int i = 0; i < 64; i++) {
		char name[16]; snprintf(name, sizeof(name), "$%d", i + 1);
		if (m_paramMap.find(name) == m_paramMap.end())
			break;
		ref[num_ref++] = m_paramMap[name]->GetVxObject();
	}
	AgoGraphExportInfo info = { 0 };
	strcpy(info.fileName, "stdout");
	info.ref = ref;
	info.num_ref = num_ref;
	strcpy(info.comment, "internal");
	vx_status status = vxSetGraphAttribute(m_graph, VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT, &info, sizeof(info));
	if (status != VX_SUCCESS)
		ReportError("ERROR: vxSetGraphAttribute(...,VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT,...) failed (%d)\n", status);
	fflush(stdout);
	return 0;
}

int CVxEngine::DumpGraphInfo(const char * graphName)
{
	vx_graph graph = m_graph;
	if (!graphName) {
		printf("graph info: %s\n", !m_graphVerified ? "current (not verified)" : "current");
		for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it) {
			if (it->second->IsVirtualObject()) {
				printf("  active virtual object : %s\n", it->first.c_str());
			}
		}
		for (auto it = m_graphAutoAgeList.begin(); it != m_graphAutoAgeList.end(); ++it) {
			printf("  auto-age delay object : %s\n", it->c_str());
		}
	}
	else {
		graph = m_graphNameListForObj[graphName];
		printf("graph info: %s\n", graphName);
		for (auto it = m_graphNameListForAge[graphName].begin(); it != m_graphNameListForAge[graphName].end(); ++it) {
			printf("  auto-age delay object : %s\n", it->c_str());
		}
	}
	if (graph) {
		// device affinity
		AgoTargetAffinityInfo attr_affinity = { 0 };
		vx_status status = vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &attr_affinity, sizeof(attr_affinity));
		if (status != VX_SUCCESS)
			ReportError("ERROR: vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY) failed (%d:%s)\n", status, ovxEnum2Name(status));
		printf("  graph device affinity :");
		if (!attr_affinity.device_type) printf(" unspecified");
		else if (attr_affinity.device_type == AGO_TARGET_AFFINITY_CPU) printf(" cpu");
		else if (attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU) printf(" gpu");
		else printf(" unknown");
		if (attr_affinity.device_info > 0) printf(" %d", attr_affinity.device_info);
		printf("\n");
		// optimizer flags
		vx_uint32 graph_optimizer_flags = 0;
		status = vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &graph_optimizer_flags, sizeof(graph_optimizer_flags));
		if (status)
			ReportError("ERROR: vxQueryGraph(*,VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS) failed (%d:%s)\n", status, ovxEnum2Name(status));
		if (graph_optimizer_flags)
			printf("  graph optimizer_flags : %d (0x%08d)\n", graph_optimizer_flags, graph_optimizer_flags);
	}
	return 0;
}

int CVxEngine::ProcessGraph(std::vector<const char *> * graphNameList, size_t beginIndex)
{
	// update graph process count
	m_numGraphProcessed++;

	if (!graphNameList && !m_graphVerified)
	{ // verify the graph
		vx_status status = vxVerifyGraph(m_graph); fflush(stdout);
		if (status != VX_SUCCESS)
			ReportError("ERROR: vxVerifyGraph(graph) failed (%d:%s)\n", status, ovxEnum2Name(status));

		// mark that graph has been verified
		m_graphVerified = true;

		// dump optimized graph (if requested)
		if (m_enableDumpGDF) {
			DumpInternalGDF();
		}
	}

	// Finalize() on all objects in graph and check if multi-frame capture is enabled
	m_usingMultiFrameCapture = m_enableMultiFrameProcessing;
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		m_usingMultiFrameCapture |= it->second->IsUsingMultiFrameCapture();
		it->second->SetVerbose(m_verbose);
		it->second->SetDiscardCompareErrors(m_discardCompareErrors);
		it->second->Finalize();
	}
	if (m_frameCountSpecified) {
		m_usingMultiFrameCapture = false;
	}

	// get graph and delay object list
	std::vector<vx_graph> graphObjList;
	std::vector<vx_delay> delayObjList;
	if (!graphNameList) {
		graphObjList.push_back(m_graph);
		for (auto it = m_graphAutoAgeList.begin(); it != m_graphAutoAgeList.end(); it++) {
			auto item = m_paramMap.find(*it);
			if (item == m_paramMap.end())
				ReportError("ERROR: graph auto-age: unable to find delay object: %s\n", it->c_str());
			vx_delay delay = (vx_delay)item->second->GetVxObject();
			delayObjList.push_back(delay);
			if (m_verbose) printf("> auto-age delay %s\n", it->c_str());
		}
	}
	else {
		for (size_t i = beginIndex; i < graphNameList->size(); i++) {
			const char * graphName = (*graphNameList)[i];
			auto it = m_graphNameListForObj.find(graphName);
			if (it == m_graphNameListForObj.end())
				ReportError("ERROR: %s is not a valid graph name\n", graphName);
			vx_graph graph = it->second;
			graphObjList.push_back(graph);
			auto jt = m_graphNameListForAge.find(graphName);
			for (auto it = jt->second.begin(); it != jt->second.end(); it++) {
				auto item = m_paramMap.find(*it);
				if (item == m_paramMap.end())
					ReportError("ERROR: graph auto-age: unable to find delay object: %s (graph %s)\n", it->c_str(), graphName);
				vx_delay delay = (vx_delay)item->second->GetVxObject();
				bool found = false;
				for (size_t j = 0; j < delayObjList.size(); j++) {
					if (delayObjList[j] == delay) {
						found = true;
						break;
					}
				}
				if(!found) delayObjList.push_back(delay);
				if (m_verbose) printf("> auto-age delay %s for graph %s\n", it->c_str(), graphName);
			}
		}
	}

	if (graphObjList.size() < 2) {
		printf("csv,HEADER ,STATUS, COUNT,cur-ms,avg-ms,min-ms,clenqueue-ms,clwait-ms,clwrite-ms,clread-ms\n");
	}
	else {
		printf("csv,HEADER ,STATUS, COUNT");
		for (size_t i = 0; i < graphObjList.size(); i++) {
			printf(",%s", (*graphNameList)[beginIndex + i]);
		}
		printf("\n");
	}
	fflush(stdout);

	// execute the graph for all requested frames
	bool abortRequested = false;
	int count = 0, status = 0;
	m_timeMeasurements.clear();
	int64_t start_time = utilGetClockCounter();
	for (int frameNumber = m_frameStart; m_usingMultiFrameCapture || frameNumber < m_frameEnd; frameNumber++, count++){
		// sync frame
		if ((status = SyncFrame(frameNumber)) < 0) throw - 1;
		// read input data, when specified
		if ((status = ReadFrame(frameNumber)) < 0) throw - 1;
		if (m_framesEofRequested && status > 0) {
			// data is not available
			if (frameNumber == m_frameStart) {
				ReportError("ERROR: insufficient input data -- check input files\n");
			}
			else break;
		}
		// execute graph for current frame
		if (graphObjList.size() < 2 && !m_enableScheduleGraph) {
			status = vxProcessGraph(graphObjList[0]);
			if (status != VX_SUCCESS && status != VX_ERROR_GRAPH_ABANDONED)
				ReportError("ERROR: vxScheduleGraph(%s) failed (%d:%s)\n", !graphNameList ? "" : (*graphNameList)[beginIndex], status, ovxEnum2Name(status));
		}
		else {
			// schedule all graph
			for (size_t i = 0; i < graphObjList.size(); i++) {
				status = vxScheduleGraph(graphObjList[i]);
				if (status)
					ReportError("ERROR: vxScheduleGraph(%s) failed (%d:%s)\n", !graphNameList ? "" : (*graphNameList)[beginIndex+i], status, ovxEnum2Name(status));
			}
			// wait for all graphs to complete
			bool abandoned = false;
			for (size_t i = 0; i < graphObjList.size(); i++) {
				status = vxWaitGraph(graphObjList[i]);
				if (status == VX_ERROR_GRAPH_ABANDONED)
					abandoned = true;
				else if (status)
					ReportError("ERROR: vxScheduleGraph(%s) failed (%d:%s)\n", !graphNameList ? "" : (*graphNameList)[beginIndex + i], status, ovxEnum2Name(status));
			}
			if (abandoned)
				status = VX_ERROR_GRAPH_ABANDONED;
		}
		if (status == VX_ERROR_GRAPH_ABANDONED) {
#if _DEBUG
			printf("WARNING: graph aborted with VX_ERROR_GRAPH_ABANDONED\n");
#endif
			break; // don't report graph abandoned as an error
		}
		if (status < 0) throw - 1;
		// get frame level performance measurements
		MeasureFrame(frameNumber, status, graphObjList);

		if (!m_disableCompare) {
			// compare output data, when requested
			status = CompareFrame(frameNumber);
		}
		// write output data, when requested
		if (WriteFrame(frameNumber) < 0) throw - 1;

		// auto-age delays associated with graphs
		for (size_t i = 0; i < delayObjList.size(); i++) {
			ERROR_CHECK(vxAgeDelay(delayObjList[i]));
		}
		if (status < 0) throw - 1;
		else if (status) break;
		// display refresh
		if (ProcessCvWindowKeyRefresh(m_waitKeyDelayInMilliSeconds) > 0) {
			DisableWaitForKeyPress();
			abortRequested = true;
			break;
		}
	}
	// print the execution time statistics
	int64_t end_time = utilGetClockCounter();
	int64_t frequency = utilGetClockFrequency();
	float elapsed_time = (float)(end_time - start_time) / frequency;
	PerformanceStatistics(status, graphObjList);
	printf("> total elapsed time: %6.2f sec\n", (float)elapsed_time);
	if (m_enableDumpProfile) {
		for (size_t i = 0; i < graphObjList.size(); i++) {
			printf("> graph profile: %s\n", !graphNameList ? "" : (*graphNameList)[beginIndex + i]);
			char fileName[] = "stdout";
			ERROR_CHECK(vxQueryGraph(graphObjList[i], VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_PROFILE, fileName, 0));
		}
	}
	fflush(stdout);

	return abortRequested ? BUILD_GRAPH_EXIT : BUILD_GRAPH_SUCCESS;
}

const char * RemoveWhiteSpacesAndComment(char * line)
{
	static char buf[4096];
	int pos = 0;
	for (char c = ' ', *p = line; *p; c = *p++) {
		if (*p == '\t' || *p == '\r' || *p == '\n') *p = ' ';
		if (*p != ' ' || c != ' ')
			buf[pos++] = *p;
		else if (*p == '#' && c == ' ')
			break;
	}
	buf[pos] = 0;
	return buf;
}

int CVxEngine::RenameData(const char * oldName, const char * newName)
{
	auto itOld = m_paramMap.find(oldName);
	auto itNew = m_paramMap.find(newName);
	if (itNew != m_paramMap.end()) {
		ReportError("ERROR: data object with name '%s' already exists\n", newName);
	}
	else if (itOld == m_paramMap.end()) {
		ReportError("ERROR: data object with name '%s' doesn't exist\n", oldName);
	}
	else {
		auto obj = itOld->second;
		m_paramMap.erase(itOld);
		m_paramMap.insert(pair<string, CVxParameter *>(newName, obj));
	}
	return 0;
}

int CVxEngine::BuildAndProcessGraphFromLine(int level, char * line)
{
	// remove whitespaces and save original text for error reporting
	std::string originalLine = RemoveWhiteSpacesAndComment(line);
	const char * originalText = originalLine.c_str();
	strcpy(line, originalText);
	if (m_enableDumpGDF || m_verbose) {
		if (m_verbose) printf(">>");
		printf("%s\n", originalText);
		fflush(stdout);
	}

	// split the line into words
	std::vector<const char *> wordList;
	for (char *s = line; *s;) {
		// find end of string
		char *t = s;
		while (*t && *t != ' ')
			t++;
		char c = *t; *t = '\0';
		wordList.push_back(s);
		if (c == '\0')
			break;
		s = t + 1;
	}
	if (!wordList.size())
		return BUILD_GRAPH_SUCCESS;

	// discard command if listed in discardCommandList
	std::string commandKey = ",";
	commandKey += wordList[0];
	commandKey += ",";
	if (m_discardCommandList.find(commandKey) != std::string::npos)
		return BUILD_GRAPH_SUCCESS;

	// process a GDF statement
	if (!_stricmp(wordList[0], "help"))
	{ // syntax: exit
		PrintHelpGDF((wordList.size() > 1) ? wordList[1] : nullptr);
	}
	else if (!_stricmp(wordList[0], "exit")) {
		return BUILD_GRAPH_EXIT;
	}
	else if (!_stricmp(wordList[0], "quit")) {
		throw (int)BUILD_GRAPH_ABORT;
	}
	else if (!_stricmp(wordList[0], "include") && wordList.size() > 1) {
		if (level >= MAX_GDF_LEVELS) ReportError("Too many levels of recursion from inside GDF\n");
		const char * fileName = RootDirUpdated(wordList[1]);
		CFileBuffer txt(fileName);
		char * txtBuffer = (char *)txt.GetBuffer();
		if (!txtBuffer)
			ReportError("ERROR: unable to open: %s\n", fileName);
		return BuildAndProcessGraph(level + 1, txtBuffer, true);
	}
	else if (!_stricmp(wordList[0], "shell")) {
		if (level >= MAX_GDF_LEVELS) ReportError("Too many levels of recursion from inside GDF\n");
		return Shell(level + 1);
	}
	else if (!_stricmp(wordList[0], "set") && wordList.size() > 1) {
		if (!_stricmp(wordList[1], "verbose"))
		{ // syntax: set verbose [on|off]
			if (wordList.size() > 2) {
				m_verbose = true;
				if (!_stricmp(wordList[2], "off"))
					m_verbose = false;
			}
			printf("> current settings for verbose: %s\n", m_verbose ? "on" : "off");
		}
		else if (!_stricmp(wordList[1], "frames"))
		{ // syntax: set frames [[<start-frame>:]<end-frame>|eof|live|default]
			if (wordList.size() > 2) {
				if (!_stricmp(wordList[2], "default")) {
					m_enableMultiFrameProcessing = false;
					m_frameCountSpecified = false;
					m_framesEofRequested = true;
					m_frameStart = 0;
					m_frameEnd = 1;
				}
				else if (!_stricmp(wordList[2], "live")) {
					m_enableMultiFrameProcessing = true;
				}
				else if (!_stricmp(wordList[2], "eof")) {
					m_framesEofRequested = true;
				}
				else {
					if (sscanf(wordList[2], "%d:%d", &m_frameStart, &m_frameEnd) == 1) {
						m_frameEnd = m_frameStart;
						m_frameStart = 0;
					}
					m_frameCountSpecified = true;
					m_enableMultiFrameProcessing = false;
				}
			}
			if (wordList.size() == 2 || m_verbose) {
				printf("> current settings for frames:");
				if (m_frameCountSpecified) {
					if (m_frameStart) printf(" %d:%d", m_frameStart, m_frameEnd);
					else printf(" %d", m_frameEnd);
				}
				if (m_framesEofRequested) printf(" eof");
				if (m_enableMultiFrameProcessing) printf(" live");
				printf("\n");
			}
		}
		else if (!_stricmp(wordList[1], "dump-profile"))
		{ // syntax: set dump-profile [on|off]
			if (wordList.size() > 2) {
				m_enableDumpProfile = true;
				if (!_stricmp(wordList[2], "off"))
					m_enableDumpProfile = false;
			}
			printf("> current settings for dump-profile: %s\n", m_enableDumpProfile ? "on" : "off");
		}
		else if (!_stricmp(wordList[1], "wait"))
		{ // syntax: set wait [key|<milliseconds>]
			if (wordList.size() > 2) {
				m_waitKeyDelayInMilliSeconds = atoi(wordList[2]);
			}
			if (wordList.size() == 2 || m_verbose) {
				if (m_waitKeyDelayInMilliSeconds == 0)
					printf("> current settings for frame-level wait: key\n");
				else
					printf("> current settings for frame-level wait: %d milliseconds\n", m_waitKeyDelayInMilliSeconds);
			}
		}
		else if (!_stricmp(wordList[1], "compare"))
		{ // syntax: set compare [on|discard-errors|off]
			if (wordList.size() > 2) {
				m_discardCompareErrors = false;
				m_disableCompare = false;
				if (!_stricmp(wordList[2], "off"))
					m_disableCompare = true;
				else if (!_stricmp(wordList[2], "discard-errors"))
					m_discardCompareErrors = true;
			}
			if (wordList.size() == 2 || m_verbose) {
				printf("> current settings for compare: %s\n", m_disableCompare ? "off" : (m_discardCompareErrors ? "discard-errors" : "on"));
			}
		}
		else if (!_stricmp(wordList[1], "use-schedule-graph"))
		{ // syntax: set use-schedule-graph [on|off]
			if (wordList.size() > 2) {
				m_enableScheduleGraph = true;
				if (!_stricmp(wordList[2], "off"))
					m_enableScheduleGraph = false;
			}
			if (wordList.size() == 2 || m_verbose) {
				printf("> current settings for use-schedule-graph: %s\n", m_enableScheduleGraph ? "on" : "off");
			}
		}
		else if (!_stricmp(wordList[1], "dump-gdf"))
		{ // syntax: set dump-gdf [on|off]
			if (wordList.size() > 2) {
				m_enableDumpGDF = true;
				if (!_stricmp(wordList[2], "off"))
					m_enableDumpGDF = false;
			}
			printf("> current settings for dump-gdf: %s\n", m_enableDumpGDF ? "on" : "off");
		}
		else if (!_stricmp(wordList[1], "dump-data-config"))
		{ // syntax: set dump-data-config [<dumpFilePrefix>,<obj-type>[,<obj-type>[...]]]
			std::string dumpDataConfig = (wordList.size() > 2) ? wordList[2] : "";
			SetDumpDataConfig(dumpDataConfig);
			printf("> current settings for dump-data-config: %s\n", dumpDataConfig.c_str());
		}
		else ReportError("ERROR: syntax error: %s\n" "See help for details.\n", originalText);
	}
	else if (!_stricmp(wordList[0], "graph") && wordList.size() > 1)
	{
		if (!_stricmp(wordList[1], "launch"))
		{ // syntax: graph launch [<graphName(s)>]
			if (m_verbose) {
				if (wordList.size() > 2) {
					printf("> launching graph:");
					for (size_t i = 2; i < wordList.size(); i++) printf(" %s", wordList[i]);
					printf("\n");
				}
				else printf("> launching current graph\n");
			}
			int status = ProcessGraph(wordList.size() > 2 ? &wordList : nullptr, 2);
			if (status == BUILD_GRAPH_SUCCESS)
				status = BUILD_GRAPH_LAUNCHED;
			return status;
		}
		else if (!_stricmp(wordList[1], "reset"))
		{ // syntax: graph reset [<graphName(s)>]
			if (wordList.size() > 2) {
				for (size_t item = 2; item < wordList.size(); item++) {
					auto it = m_graphNameListForObj.find(wordList[item]);
					if (it == m_graphNameListForObj.end())
						ReportError("ERROR: syntax error: %s # <graphName> %s doesn't exists.\n", originalText, wordList[item]);
					// release the graph and remove the entries in the graphName list
					ERROR_CHECK(vxReleaseGraph(&it->second));
					m_graphNameListForObj.erase(wordList[item]);
					m_graphNameListForAge.erase(wordList[item]);
					if (m_verbose) printf("> released and removed graph: %s\n", wordList[item]);
				}
			}
			else {
				// empty virtual object list and delay age-list; release current graph and create a new graph
				ReleaseAllVirtualObjects();
				ERROR_CHECK(vxReleaseGraph(&m_graph));
				m_graphAutoAgeList.clear();
				// open a new graph
				m_graphVerified = false;
				m_graph = vxCreateGraph(m_context);
				vx_status status = vxGetStatus((vx_reference)m_graph);
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxCreateGraph(context) failed (%d:%s)\n", status, ovxEnum2Name(status));
				if (m_verbose) printf("> reset current graph to empty\n");
			}
		}
		else if (!_stricmp(wordList[1], "save-and-reset") && wordList.size() > 2)
		{ // syntax: graph save-and-reset <graphName>
			if (m_graphNameListForObj.find(wordList[2]) != m_graphNameListForObj.end())
				ReportError("ERROR: syntax error: %s # <graphName> %s already used.\n", originalText, wordList[2]);
			// verify the graph and save
			vx_status status = vxVerifyGraph(m_graph); fflush(stdout);
			if (status != VX_SUCCESS)
				ReportError("ERROR: vxVerifyGraph(graph) failed (%d:%s)\n", status, ovxEnum2Name(status));
			if (m_enableDumpGDF) {
				DumpInternalGDF();
			}
			m_graphNameListForObj.insert(pair<string, vx_graph>(wordList[2], m_graph));
			m_graphNameListForAge.insert(pair<string, std::vector<std::string> >(wordList[2], m_graphAutoAgeList));
			// open a new graph with empty virtual object list and delay age-list
			ReleaseAllVirtualObjects();
			m_graphAutoAgeList.clear();
			m_graphVerified = false;
			m_graph = vxCreateGraph(m_context);
			status = vxGetStatus((vx_reference)m_graph);
			if (status != VX_SUCCESS)
				ReportError("ERROR: vxCreateGraph(context) failed (%d:%s)\n", status, ovxEnum2Name(status));
			if (m_verbose) printf("> verified current graph as %s and created a new empty graph\n", wordList[2]);
		}
		else if (!_stricmp(wordList[1], "auto-age"))
		{ // syntax: graph auto-age [<delayName> [<delayName> ...]]
			for (size_t item = 2; item < wordList.size(); item++) {
				auto it = m_paramMap.find(wordList[item]);
				if (it == m_paramMap.end())
					ReportError("ERROR: syntax error: %s # %s is not a valid delay object\n", originalText, wordList[item]);
				m_graphAutoAgeList.push_back(wordList[item]);
			}
			if (wordList.size() == 2 || m_verbose) {
				printf("> current graph auto-age delay objects:");
				for (auto it = m_graphAutoAgeList.begin(); it != m_graphAutoAgeList.end(); it++)
					printf(" %s", it->c_str());
				printf("\n");
			}
		}
		else if (!_stricmp(wordList[1], "affinity"))
		{ // syntax: graph affinity [CPU|GPU[<device-index>]]
			AgoTargetAffinityInfo attr_affinity = { 0 };
			if (wordList.size() > 2) {
				const char * target = wordList[2];
				if (!_strnicmp(target, "cpu", 3))
					attr_affinity.device_type = AGO_TARGET_AFFINITY_CPU;
				else if (!_strnicmp(target, "gpu", 3))
					attr_affinity.device_type = AGO_TARGET_AFFINITY_GPU;
				else
					ReportError("ERROR: syntax error: %s\n" "unsupported target affinity specified.\n", originalText);
				if (target[3] >= '0' && target[3] <= '9')
					attr_affinity.device_info = atoi(&target[3]);
				vx_status status = vxSetGraphAttribute(m_graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &attr_affinity, sizeof(attr_affinity));
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, \"%s\") failed (%d:%s)\n", target, status, ovxEnum2Name(status));
			}
			if (wordList.size() == 2 || m_verbose) {
				vx_status status = vxQueryGraph(m_graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &attr_affinity, sizeof(attr_affinity));
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY) failed (%d:%s)\n", status, ovxEnum2Name(status));
				printf("> current settings for affinity:");
				if (!attr_affinity.device_type) printf(" unspecified");
				else if (attr_affinity.device_type == AGO_TARGET_AFFINITY_CPU) printf(" cpu");
				else if (attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU) printf(" gpu");
				else printf(" unknown");
				if (attr_affinity.device_info > 0) printf(" %d", attr_affinity.device_info);
				printf("\n");
			}
		}
		else if (!_stricmp(wordList[1], "optimizer"))
		{ // syntax: graph optimizer [<flags>]
			if (wordList.size() > 2) {
				vx_uint32 graph_optimizer_flags = atoi(wordList[2]);
				vx_status status = vxSetGraphAttribute(m_graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &graph_optimizer_flags, sizeof(graph_optimizer_flags));
				if (status)
					ReportError("ERROR: vxSetGraphAttribute(*,VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS,%d) failed (%d:%s)\n", graph_optimizer_flags, status, ovxEnum2Name(status));
			}
			if (wordList.size() == 2 || m_verbose) {
				vx_uint32 graph_optimizer_flags = 0;
				vx_status status = vxQueryGraph(m_graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &graph_optimizer_flags, sizeof(graph_optimizer_flags));
				if (status)
					ReportError("ERROR: vxQueryGraph(*,VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS) failed (%d:%s)\n", status, ovxEnum2Name(status));
				printf("> current graph optimizer flags: %d (0x%08d)\n", graph_optimizer_flags, graph_optimizer_flags);
			}
		}
		else if (!_stricmp(wordList[1], "info"))
		{ // syntax: graph info [<graphName(s)>]
			if (m_verbose) {
				if (wordList.size() > 2) {
					printf("> graph info:");
					for (size_t i = 2; i < wordList.size(); i++) printf(" %s", wordList[i]);
					printf("\n");
				}
				else printf("> graph info: current\n");
			}
			for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it) {
				if (!it->second->IsVirtualObject()) {
					printf("active data object      : %s\n", it->first.c_str());
				}
			}
			if (wordList.size() > 2) {
				for (size_t i = 2; i < wordList.size(); i++)
					DumpGraphInfo(wordList[i]);
			}
			else DumpGraphInfo();
		}
		else ReportError("ERROR: syntax error: %s\n" "See help for details.\n", originalText);
	}
	else if (!_stricmp(wordList[0], "pause"))
	{ // syntax: pause
		// wait for keyboard input
		if (ProcessCvWindowKeyRefresh(0) > 0) {
			DisableWaitForKeyPress();
			return BUILD_GRAPH_EXIT;
		}
	}
	else if (!_stricmp(wordList[0], "import"))
	{ // syntax: import <libraryName>
		if (wordList.size() != 2)
			ReportError("ERROR: syntax error: %s\n" "valid sytax: import <libraryName>\n", originalText);
		vx_status status = vxLoadKernels(m_context, wordList[1]);
		if (status)
			ReportError("ERROR: vxLoadKernels(context,\"%s\") failed (%d:%s)\n", wordList[1], status, ovxEnum2Name(status));
	}
	else if (!_stricmp(wordList[0], "type"))
	{ // syntax: type <typeName> userstruct:<size-in-bytes>
		vx_size type_size = 0;
		if (wordList.size() != 3 || _strnicmp(wordList[2], "userstruct:", 11) != 0 || sscanf(wordList[2]+11, "%i", (int *)&type_size) != 1)
			ReportError("ERROR: syntax error: %s\n" "valid sytax: type <typeName> userstruct:<size-in-bytes>\n", originalText);
		for (auto it = m_userStructMap.begin(); it != m_userStructMap.end(); ++it)
			if (strcmp(wordList[1], it->first.c_str()) == 0)
				ReportError("ERROR: syntax error: %s # <typeName> %s already used.\n", originalText, wordList[1]);
		vx_enum type_enum = vxRegisterUserStruct(m_context, type_size);
		if (type_enum < VX_TYPE_USER_STRUCT_START || type_enum > VX_TYPE_USER_STRUCT_END)
			ReportError("ERROR: vxRegisterUserStruct(context,%d) failed (%d:%s)\n", (int)type_size, type_enum, ovxEnum2Name(type_enum));
		m_userStructMap.insert(pair<string, vx_enum>(wordList[1], type_enum));
	}
	else if (!_stricmp(wordList[0], "data") && wordList.size() == 4 && !strcmp(wordList[2], "="))
	{ // syntax: data <name> = <data-description>[:<io-operations>]
		for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it)
			if (strcmp(wordList[1], it->first.c_str()) == 0)
				ReportError("ERROR: syntax error: %s # <dataName> %s already used.\n", originalText, wordList[1]);
		std::string objDesc = wordList[3];
		if (m_disableVirtual)
			RemoveVirtualKeywordFromParamDescription(objDesc);
		const char * objDescText = objDesc.c_str();
		CVxParameter * obj = CreateDataObject(m_context, m_graph, &m_paramMap, &m_userStructMap, objDescText, m_frameStart);
		if (!obj)
			ReportError("ERROR: syntax error: %s\n" "invalid object description\n", originalText);
		m_paramMap.insert(pair<string, CVxParameter *>(wordList[1], obj));
		vx_reference ref = m_paramMap[wordList[1]]->GetVxObject();
		vx_status status = vxSetReferenceName(ref, wordList[1]);
		if (status != VX_SUCCESS)
			ReportError("ERROR: vxSetReferenceName(%s) failed (%d:%s)\n", wordList[1], status, ovxEnum2Name(status));
		// check if dump-data-config is enabled for non-virtual data objects
		if(m_dumpDataEnabled && objDesc.find("virtual") == std::string::npos) {
			// check if object type is requeted for dump
			std::string objType = objDesc.substr(0,objDesc.find(":"));
			std::string objTypeKey = ","; objTypeKey += objType + ",";
			if(m_dumpDataObjectList.find(objTypeKey) != std::string::npos) {
				// issue an InitializeIO command with write to dump into a file
				char io_params[256];
				snprintf(io_params, sizeof(io_params), "write,%sdump_%04d_%s_%s.raw", m_dumpDataFilePrefix.c_str(), m_dumpDataCount, objType.c_str(), wordList[1]);
				int status = obj->InitializeIO(m_context, m_graph, obj->GetVxObject(), io_params);
				if (status < 0)
					ReportError("ERROR: dump-data-config for %s failed: %s\n", wordList[1], io_params);
				m_dumpDataCount++;
			}
		}
	}
	else if (!_stricmp(wordList[0], "rename") && wordList.size() == 3)
	{ // syntax: rename <old-name> <new-name>
		RenameData(wordList[1], wordList[2]);
	}
	else if (wordList.size() > 2 && (!_stricmp(wordList[0], "init") || 
		!_stricmp(wordList[0], "read") || !_stricmp(wordList[0], "write") || 
		!_stricmp(wordList[0], "view") || !_stricmp(wordList[0], "compare") || 
		!_stricmp(wordList[0], "camera") || !_stricmp(wordList[0], "directive")))
	{ // syntax: init|read|camera|write|view|compare|directive <dataName> <parameters> [...]
		CVxParameter * obj = m_paramMap[wordList[1]];
		if (!obj) {
			ReportError("ERROR: syntax error: %s # <dataName> %s doesn't exist.\n", originalText, wordList[1]);
		}
		// construct a comma-separated string with command and parameters
		std::string io_params;
		io_params = wordList[0];
		for (size_t i = 2; i < wordList.size(); i++) {
			io_params += ",";
			io_params += wordList[i];
		}
		// use InitializeIO call to process the command
		int status = obj->InitializeIO(m_context, m_graph, obj->GetVxObject(), io_params.c_str());
		if (status < 0) {
			ReportError("ERROR: syntax error: %s # <dataName> %s initialization failed.\n", originalText, wordList[1]);
		}
	}
	else if (!_stricmp(wordList[0], "node") && wordList.size() >= 2)
	{ // syntax: node <kernelName> [<argument(s)>]
		// create kernel object
		const char * kernelName = wordList[1];
		vx_kernel kernel = vxGetKernelByName(m_context, kernelName);
		vx_status status = vxGetStatus((vx_reference)kernel);
		if (status != VX_SUCCESS)
			ReportError("ERROR: vxGetKernelByName(context,\"%s\") failed (%d:%s)\n", kernelName, status, ovxEnum2Name(status));
		// create node object and release kernel object
		vx_node node = vxCreateGenericNode(m_graph, kernel);
		status = vxGetStatus((vx_reference)node);
		if (status != VX_SUCCESS)
			ReportError("ERROR: vxCreateGenericNode(graph,node(\"%s\")) failed (%d:%s)\n", kernelName, status, ovxEnum2Name(status));
		ERROR_CHECK(vxReleaseKernel(&kernel));
		if(m_setBorderMode){
			wordList.pop_back();
			const char * paramDesc = m_cmdBorderMode.c_str();
			if (!_strnicmp(paramDesc, "attr:border_mode:", 17)) {
				char mode[128];
				const char * p = ScanParameters(&paramDesc[17], "<border-mode>", "s", mode);
				if (!_stricmp(mode, "UNDEFINED") || !_stricmp(mode, "REPLICATE") || !_stricmp(mode, "CONSTANT")) {
					char item[256];
					snprintf(item, sizeof(item), "VX_BORDER_MODE_%s", mode);
					strcpy(mode, item);
				}
				vx_border_mode_t border_mode = { 0 };
				border_mode.mode = ovxName2Enum(mode);
				if (*p == ',') {
					if (p[1] == '{') {
						p++;
						for (int index = 0; index < 4 && (*p == '{' || *p == ';');) {
							int value = 0;
							p = ScanParameters(&p[1], "<byte>", "d", &value);
							border_mode.constant_value.reserved[index++] = (vx_uint8)value;
						}
					}
					else (void)sscanf(p + 1, "%i", &border_mode.constant_value.U32);
				}
				status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border_mode, sizeof(border_mode));
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxSetNodeAttribute(node(\"%s\"), VX_NODE_ATTRIBUTE_BORDER_MODE, \"%s\") failed (%d:%s)\n", kernelName, &paramDesc[17], status, ovxEnum2Name(status));
			}		
		}
		// set node arguments
		vx_uint32 index = 0;
		for (size_t arg = 2; arg < wordList.size(); arg++) {
			const char * paramDesc = wordList[arg];
			if (!_strnicmp(paramDesc, "attr:border_mode:", 17)) {
				char mode[128];
				const char * p = ScanParameters(&paramDesc[17], "<border-mode>", "s", mode);
				if (!_stricmp(mode, "UNDEFINED") || !_stricmp(mode, "REPLICATE") || !_stricmp(mode, "CONSTANT")) {
					// add prefix "VX_BORDER_MODE_"
					// TBD: this needs to be removed
					char item[256];
					snprintf(item, sizeof(item), "VX_BORDER_MODE_%s", mode);
					strcpy(mode, item);
				}
				vx_border_mode_t border_mode = { 0 };
				border_mode.mode = ovxName2Enum(mode);
				if (*p == ',') {
					if (p[1] == '{') {
						// scan get 8-bit values for RGB/RGBX/YUV formats as {R;G;B}/{R;G;B;X}/{Y;U;V}
						p++;
						for (int index = 0; index < 4 && (*p == '{' || *p == ';');) {
							int value = 0;
							p = ScanParameters(&p[1], "<byte>", "d", &value);
							border_mode.constant_value.reserved[index++] = (vx_uint8)value;
						}
					}
					else (void)sscanf(p + 1, "%i", &border_mode.constant_value.U32);
				}
				status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border_mode, sizeof(border_mode));
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxSetNodeAttribute(node(\"%s\"), VX_NODE_ATTRIBUTE_BORDER_MODE, \"%s\") failed (%d:%s)\n", kernelName, &paramDesc[17], status, ovxEnum2Name(status));
			}
			else if (!_strnicmp(paramDesc, "attr:affinity:", 14)) {
				const char * target = &paramDesc[14];
				AgoTargetAffinityInfo attr_affinity = { 0 };
				if (!_strnicmp(target, "cpu", 3))
					attr_affinity.device_type = AGO_TARGET_AFFINITY_CPU;
				else if (!_strnicmp(target, "gpu", 3))
					attr_affinity.device_type = AGO_TARGET_AFFINITY_GPU;
				else
					ReportError("ERROR: syntax error: %s\n" "unsupported target affinity specified.\n", originalText);
				if (target[3] >= '0' && target[3] <= '9')
					attr_affinity.device_info = atoi(&target[3]);
				status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_AMD_AFFINITY, &attr_affinity, sizeof(attr_affinity));
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxSetNodeAttribute(node(\"%s\"), VX_NODE_ATTRIBUTE_AMD_AFFINITY, \"%s\") failed (%d:%s)\n", kernelName, target, status, ovxEnum2Name(status));
			}
			else if (paramDesc[0] == '!') {
				// create scalar object
				vx_enum value = ovxName2Enum(&paramDesc[1]);
				vx_scalar scalar = vxCreateScalar(m_context, VX_TYPE_ENUM, &value);
				status = vxGetStatus((vx_reference)scalar);
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxCreateScalar(context,VX_TYPE_ENUM,0x%08x) failed (%d:%s)\n", value, status, ovxEnum2Name(status));
				// set node argument and release the scalar object
				status = vxSetParameterByIndex(node, index, (vx_reference)scalar);
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxSetParameterByIndex(node(%s),%d,scalar:enum) failed (%d:%s)\n", kernelName, index, status, ovxEnum2Name(status));
				ERROR_CHECK(vxReleaseScalar(&scalar));
				index++;
			}
			else if (!_stricmp(paramDesc, "null")) {
				// don't set the node argument -- useful for optional arguments
				index++;
			}
			else {
				// get object name and reference object
				char name[64]; strncpy(name, paramDesc, sizeof(name) - 1);
				const char * paramDescIndex = strstr(paramDesc, "[");
				if (paramDescIndex)
					name[paramDescIndex - paramDesc] = '\0';
				if (m_paramMap.find(name) == m_paramMap.end())
					ReportError("ERROR: syntax error: %s\n" "invalid object name for parameter#%d: %s\n", originalText, index, paramDesc);
				vx_reference ref = m_paramMap[name]->GetVxObject();
				// get subobject
				bool needToReleaseImage = false;
				while (paramDescIndex && *paramDescIndex == '[') {
					int subObjIndex = 0;
					paramDescIndex = ScanParameters(paramDescIndex, "[<index>]", "[d]", &subObjIndex);
					// get sub-indexed object
					vx_enum type; ERROR_CHECK(vxQueryReference(ref, VX_REFERENCE_TYPE, &type, sizeof(type)));
					if (type == VX_TYPE_PYRAMID) {
						ref = (vx_reference)vxGetPyramidLevel((vx_pyramid)ref, (vx_uint32)subObjIndex);
						status = vxGetStatus(ref);
						if (status != VX_SUCCESS)
							ReportError("ERROR: vxGetPyramidLevel(%s,%d) failed (%d:%s)\n", name, subObjIndex, status, ovxEnum2Name(status));
						needToReleaseImage = true;
					}
					else if (type == VX_TYPE_DELAY) {
						ref = vxGetReferenceFromDelay((vx_delay)ref, (vx_int32)subObjIndex);
						status = vxGetStatus(ref);
						if (status != VX_SUCCESS)
							ReportError("ERROR: vxGetReferenceFromDelay(%s,%d) failed (%d:%s)\n", name, subObjIndex, status, ovxEnum2Name(status));
					}
					else
						ReportError("ERROR: syntax error: %s\n" "can not index into parameter#%d: %s\n", originalText, index, paramDesc);
				}
				// set node parameter
				status = vxSetParameterByIndex(node, index, ref);
				if (status != VX_SUCCESS)
					ReportError("ERROR: vxSetParameterByIndex(node(%s),%d,obj(%s)) failed (%d:%s)\n", kernelName, index, paramDesc, status, ovxEnum2Name(status));
				if (needToReleaseImage)
					ERROR_CHECK(vxReleaseImage((vx_image *)&ref));
				index++;
			}
		}
		// release node object
		ERROR_CHECK(vxReleaseNode(&node));
	}
	else ReportError("ERROR: syntax error: %s\n" "See help for details.\n", originalText);

	return BUILD_GRAPH_SUCCESS;
}

int CVxEngine::BuildAndProcessGraph(int level, char * graphScript, bool importMode)
{
	// remove replace whitespace, comments, and line-endings with SP in GDF
#define CHECK_BACKSLASH_AT_LINE_ENDING(s,i) ((s[i] == ' ' || s[i] == '\t') && s[i + 1] == '\\' && (s[i + 2] == '\n' || (s[i + 2] == '\r' && s[i + 3] == '\n')))
	char * s = graphScript;
	for (size_t i = 0; s[i]; i++) {
		if (CHECK_BACKSLASH_AT_LINE_ENDING(s, i)) {
			// replace line-endings with SP
			if (s[i + 2] == '\r') s[i + 3] = ' ';
			s[i] = s[i + 1] = s[i + 2] = ' ';
		}
		else if (s[i] == '\t' || s[i] == '\r')
			s[i] = ' '; // replace TAB/CR with SP
		else if (s[i] == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\n')) {
			// remove comment with SP
			while (s[i] && s[i] != '\n' && !CHECK_BACKSLASH_AT_LINE_ENDING(s, i)) {
				s[i++] = ' ';
			}
			if (CHECK_BACKSLASH_AT_LINE_ENDING(s, i)) {
				// remove SP BACKSLASH [CR] LF with blanks
				s[i++] = ' '; s[i++] = ' ';
				if (s[i] == '\r') s[i++] = ' ';
				s[i] = ' ';
			}
		}
	}
#undef CHECK_BACKSLASH_AT_LINE_ENDING

	// process GDF line by line
	int lastBuildAndProcessStatus = BUILD_GRAPH_SUCCESS;
	for (char * s = graphScript, *t = graphScript;; t++) {
		char c = *t;
		if (c == '\n' || c == '\0') {
			// process the current line
			*t = '\0';
			bool hasText = false;
			for (char * p = s; *p; p++) {
				if (*p != ' ') {
					hasText = true;
					break;
				}
			}
			if (hasText) {
				lastBuildAndProcessStatus = BuildAndProcessGraphFromLine(level, s);
				if (lastBuildAndProcessStatus == BUILD_GRAPH_FAILURE)
					return -1;
				else if (lastBuildAndProcessStatus == BUILD_GRAPH_EXIT) {
					return 0;
				}
			}
			// mark next line beginning
			s = t + 1;
		}
		if (c == '\0')
			break;
	}
	fflush(stdout);

	// process the graph, if there are some non-processed statements pending
	if (!importMode && !m_numGraphProcessed) {
		if (ProcessGraph() < 0)
			return -1;
	}
	return 0;
}

int CVxEngine::Shell(int level, FILE * fp)
{
	char line[4096];
	if (!fp) { printf("%% "); fflush(stdout); }
	while (fgets(line, sizeof(line) - 1, fp ? fp : stdin) != nullptr) {
		strcpy(line, RemoveWhiteSpacesAndComment(line));
		int status = 0;
		try {
			status = BuildAndProcessGraphFromLine(level, line);
		}
		catch (int code) {
			if (code >= 0)
				throw code;
		}
		fflush(stdout);
		if (status == BUILD_GRAPH_FAILURE)
			return -1;
		else if (status == BUILD_GRAPH_EXIT)
			return 0;
		if (!fp) { printf("%% "); fflush(stdout); }
	}
	return 0;
}

int CVxEngine::SyncFrame(int frameNumber)
{
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		int status = it->second->SyncFrame(frameNumber);
		if (status)
			return status;
	}
	return 0;
}

int CVxEngine::ReadFrame(int frameNumber)
{
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		int status = it->second->ReadFrame(frameNumber);
		if (status)
			return status;
	}
	return 0;
}

int CVxEngine::WriteFrame(int frameNumber)
{
	
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		int status = it->second->WriteFrame(frameNumber);
		if (status)
			return status;
	}
	return 0;
}

int CVxEngine::CompareFrame(int frameNumber)
{
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		int status = it->second->CompareFrame(frameNumber);
		if (status)
			return status;
	}
	return 0;
}

void CVxEngine::SetFrameCountOptions(bool enableMultiFrameProcessing, bool framesEofRequested, bool frameCountSpecified, int frameStart, int frameEnd)
{
	m_enableMultiFrameProcessing = enableMultiFrameProcessing;
	m_framesEofRequested = framesEofRequested;
	m_frameCountSpecified = frameCountSpecified;
	m_frameStart = frameStart;
	m_frameEnd = frameEnd;
}

void CVxEngine::SetConfigOptions(bool verbose, bool discardCompareErrors, bool enableDumpProfile, bool enableDumpGDF, int waitKeyDelayInMilliSeconds)
{
	m_verbose = verbose;
	m_discardCompareErrors = discardCompareErrors;
	m_enableDumpProfile = enableDumpProfile;
	m_enableDumpGDF = enableDumpGDF;
	if (waitKeyDelayInMilliSeconds >= 0)
		m_waitKeyDelayInMilliSeconds = waitKeyDelayInMilliSeconds;
}

void CVxEngine::MeasureFrame(int frameNumber, int status, std::vector<vx_graph>& graphList)
{
	if (m_verbose) printf("csv,FRAME  ,  %s,%6d", (status == 0 ? "PASS" : "FAIL"), frameNumber);
	if (graphList.size() < 2) {
		vx_perf_t perf = { 0 };
		ERROR_CHECK(vxQueryGraph(graphList[0], VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));
		m_timeMeasurements.push_back(NANO2MILLISECONDS(perf.tmp));
		if (m_verbose) {
			AgoGraphPerfInternalInfo iperf = { 0 };
			ERROR_CHECK(vxQueryGraph(graphList[0], VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_LAST, &iperf, sizeof(iperf)));
			printf(",%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f",
				NANO2MILLISECONDS(perf.tmp),
				NANO2MILLISECONDS(perf.avg),
				NANO2MILLISECONDS(perf.min),
				NANO2MILLISECONDS(iperf.kernel_enqueue),
				NANO2MILLISECONDS(iperf.kernel_wait),
				NANO2MILLISECONDS(iperf.buffer_write),
				NANO2MILLISECONDS(iperf.buffer_read));
		}
	}
	else {
		for (size_t i = 0; i < graphList.size(); i++) {
			vx_perf_t perf = { 0 };
			ERROR_CHECK(vxQueryGraph(graphList[i], VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));
			m_timeMeasurements.push_back(NANO2MILLISECONDS(perf.tmp)); // TBD: need to track median separately for each graph
			if (m_verbose) printf(",%6.2f", NANO2MILLISECONDS(perf.tmp));
		}
	}
	if (m_verbose) { printf("\n"); fflush(stdout); }
}


float CVxEngine::GetMedianRunTime()
{
	float median = 0.0;
	size_t count = m_timeMeasurements.size();
	if (count > 0) {
		sort(m_timeMeasurements.begin(), m_timeMeasurements.end());
		median = (m_timeMeasurements[(count - 1) >> 1] + m_timeMeasurements[count >> 1]) * 0.5f;
	}
	return median;
}

void CVxEngine::PerformanceStatistics(int status, std::vector<vx_graph>& graphList)
{
	if (graphList.size() < 2) {
		printf("csv,OVERALL,  %s,", (status >= 0 ? "PASS" : "FAIL"));
		vx_perf_t perf = { 0 };
		ERROR_CHECK(vxQueryGraph(graphList[0], VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));
		printf("%6d,      ,%6.2f,%6.2f", (int)perf.num, NANO2MILLISECONDS(perf.avg), NANO2MILLISECONDS(perf.min));
		AgoGraphPerfInternalInfo iperf = { 0 };
		ERROR_CHECK(vxQueryGraph(graphList[0], VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_AVG, &iperf, sizeof(iperf)));
		printf(",%6.2f,%6.2f,%6.2f,%6.2f (median %.3f)\n",
			NANO2MILLISECONDS(iperf.kernel_enqueue),
			NANO2MILLISECONDS(iperf.kernel_wait),
			NANO2MILLISECONDS(iperf.buffer_write),
			NANO2MILLISECONDS(iperf.buffer_read),
			GetMedianRunTime());
	}
	else {
		printf("csv,OVERALL,  %s,", (status >= 0 ? "PASS" : "FAIL"));
		for (size_t i = 0; i < graphList.size(); i++) {
			vx_perf_t perf = { 0 };
			ERROR_CHECK(vxQueryGraph(graphList[i], VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));
			if (i == 0) printf(",%6d", (int)perf.num);
			printf(",%6.2f", NANO2MILLISECONDS(perf.avg));
		}
		printf(" (median %.3f)\n", GetMedianRunTime());
	}
	fflush(stdout);
}

int CVxEngine::Shutdown()
{
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		if (it->second){
			delete it->second;
		}
	}
	m_paramMap.clear();
	if (m_graph){
		vxReleaseGraph(&m_graph);
		m_graph = nullptr;
	}

	if (m_context) {
		vxReleaseContext(&m_context);
		m_context = nullptr;
	}
	return 0;
}

void CVxEngine::DisableWaitForKeyPress()
{
	for (auto it = m_paramMap.begin(); it != m_paramMap.end(); ++it){
		if (it->second){
			it->second->DisableWaitForKeyPress();
		}
	}
}

bool CVxEngine::IsUsingMultiFrameCapture(){
	return m_usingMultiFrameCapture;
}

void PrintHelpGDF(const char * command)
{
	if (!command) {
		command = "";
		printf("The available GDF commands are:\n");
	}
	if (strstr("import", command)) printf(
		"  import <libraryName>\n"
		"      Import kernels in a library using vxLoadKernel API.\n"
		"\n"
		);
	if (strstr("type", command)) printf(
		"  type <typeName> userstruct:<size-in-bytes>\n"
		"      Create an OpenVX user defined structure using vxRegisterUserStruct API.\n"
		"      The <typeName> can be used as a type in array object.\n"
		"\n"
		);
	if (strstr("data", command)) printf(
		"  data <dataName> = <data-description>\n"
		"      Create an OpenVX data object in context using the below syntax for\n"
		"      <data-description>:\n"
		"          array:<data-type>,<capacity>\n"
		"          convolution:<columns>,<rows>\n"
		"          distribution:<numBins>,<offset>,<range>\n"
		"          delay:<exemplar>,<slots>\n"
		"          image:<width>,<height>,<image-format>[,<range>][,<space>]\n"
		"          uniform-image:<width>,<height>,<image-format>,<uniform-pixel-value>\n"
		"          image-from-roi:<master-image>,rect{<start-x>;<start-y>;<end-x>;<end-y>}\n"
		"          image-from-handle:<image-format>,{<dim-x>;<dim-y>;<stride-x>;<stride-y>}[+...],<memory-type>\n"
		"          image-from-channel:<master-image>,<channel>\n"
		"          lut:<data-type>,<count>\n"
		"          matrix:<data-type>,<columns>,<rows>\n"
		"          pyramid:<numLevels>,half|orb|<scale-factor>,<width>,<height>,<image-format>\n"
		"          remap:<srcWidth>,<srcHeight>,<dstWidth>,<dstHeight>\n"
		"          scalar:<data-type>,<value>\n"
		"          threshold:<thresh-type>,<data-type>\n"
		"          tensor:<num-of-dims>,{<dim0>,<dim1>,...},<data-type>,<fixed-point-pos>\n"
		"          tensor-from-roi:<master-tensor>,<num-of-dims>,{<start0>,<start1>,...},{<end0>,<end1>,...}\n"
		"          tensor-from-handle:<num-of-dims>,{<dim0>,<dim1>,...},<data-type>,<fixed-point-pos>,{<stride0>,<stride1>,...},<num-alloc-handles>,<memory-type>\n"
		"      For virtual object in default graph use the below syntax for\n"
		"      <data-description>:\n"
		"          virtual-array:<data-type>,<capacity>\n"
		"          virtual-image:<width>,<height>,<image-format>\n"
		"          virtual-pyramid:<numLevels>,half|orb|<scale-factor>,<width>,<height>,<image-format>\n"
		"          virtual-tensor:<num-of-dims>,{<dim0>,<dim1>,...},<data-type>,<fixed-point-pos>\n"
		"\n"
		"      where:\n"
		"          <master-image> can be name of a image data object (including $1, $2, ...)\n"
		"          <master-tensor> can be name of a tensor data object (including $1, $2, ...)\n"
		"          <exemplar> can be name of a data object (including $1, $2, ...)\n"
		"          <thresh-type> can be BINARY,RANGE\n"
		"          <uniform-pixel-value> can be an integer or {<byte>;<byte>;...}\n"
		"          <image-format> can be RGB2,RGBX,IYUV,NV12,U008,S016,U001,F032,...\n"
		"          <data-type> can be UINT8,INT16,INT32,UINT32,FLOAT32,ENUM,BOOL,SIZE,\n"
		"                             KEYPOINT,COORDINATES2D,RECTANGLE,<typeName>,...\n"
		"          <range> can be vx_channel_range_e enums FULL or RESTRICTED\n"
		"          <space> can be vx_color_space_e enums BT709 or BT601_525 or BT601_625\n"
		"\n"
		);
	if (strstr("node", command)) printf(
		"  node <kernelName> [<argument(s)>]\n"
		"      Create a node of specified kernel in the default graph with specified\n"
		"      node arguments. Node arguments have to be OpenVX data objects created\n"
		"      earlier in GDF or data objects specified on command-line accessible as\n"
		"      $1, $2, etc. For scalar enumerations as node arguments, use !<enumName>\n"
		"      syntax (e.g., !VX_CHANNEL_Y for channel_extract node).\n"
		"\n"
		);
	if (strstr("include", command)) printf(
		"  include <file.gdf>\n"
		"      Specify inclusion of another GDF file.\n"
		"\n"
		);
	if (strstr("shell", command)) printf(
		"  shell\n"
		"      Start a shell command session.\n"
		"\n"
		);
	if (strstr("set", command)) printf(
		"  set <option> [<value>]\n"
		"      Specify or query the following global options:\n"
		"          set verbose [on|off]\n"
		"              Turn on/off verbose option.\n"
		"          set frames [[<start-frame>:]<end-frame>|eof|live|default]\n"
		"              Specify input frames to be processed. Here are some examples:\n"
		"                  set frames 10      # process frames 0 through 9\n"
		"                  set frames 1:10    # process frames 1 through 9\n"
		"                  set frames eof     # process all frames till end-of-file\n"
		"                  set frames live    # input is live until terminated by user\n"
		"                  set frames default # process all frames specified on input\n"
		"          set dump-profile [on|off]\n"
		"              Turn on/off profiler output.\n"
		"          set wait [key|<milliseconds>]\n"
		"              Specify wait time between frame processing to give extra time\n"
		"              for viewing. Or wait for key press between frames.\n"
		"          set compare [on|off|discard-errors]\n"
		"              Turn on/off data compares or just discard data compare errors.\n"
		"          set use-schedule-graph [on|off]\n"
		"              Turn on/off use of vxScheduleGraph instead of vxProcessGraph.\n"
		"          set dump-data-config [<dumpFilePrefix>,<obj-type>[,<obj-type>[...]]]\n"
		"              Specify dump data config for portion of the graph. To disable\n"
		"              don't specify any config.\n"
		"\n"
		);
	if (strstr("graph", command)) printf(
		"  graph <command> [<arguments> ...]\n"
		"      Specify below graph specific commands:\n"
		"          graph auto-age [<delayName> [<delayName> ...]]\n"
		"              Make the default graph use vxAgeDelay API for the specified\n"
		"              delay objects after processing each frame.\n"
		"          graph affinity [CPU|GPU[<device-index>]]\n"
		"              Specify graph affinity to CPU or GPU.\n"
		"          graph save-and-reset <graphName>\n"
		"              Verify the default graph and save it as <graphName>. Then\n"
		"              create a new graph as the default graph. Note that the earlier\n"
		"              virtual data object won't be available after graph reset.\n"
		"          graph reset [<graphName(s)>]\n"
		"              Reset the default or specified graph(s). Note that the earlier\n"
		"              virtual data object won't be available after graph reset.\n"
		"          graph launch [<graphName(s)>]\n"
		"              Launch the default or specified graph(s).\n"
		"          graph info [<graphName(s)>]\n"
		"              Show graph details for debug.\n"
		"\n"
		);
	if (strstr("rename", command)) printf(
		"  rename <dataNameOld> <dataNameNew>\n"
		"      Rename a data object\n"
		"\n"
		);
	if (strstr("init", command)) printf(
		"  init <dataName> <initial-value>\n"
		"      Initialize data object with specified value.\n"
		"      - convolution object initial values can be:\n"
		"          {<value1>;<value2>;...<valueN>}\n"
		"          scale{<scale>}\n"
		"      - matrix object initial values can be:\n"
		"          {<value1>;<value2>;...<valueN>}\n"
		"      - remap object initial values can be:\n"
		"          dst is same as src: same\n"
		"          dst is 90 degree rotation of src: rotate-90\n"
		"          dst is 180 degree rotation of src: rotate-180\n"
		"          dst is 270 degree rotation of src: rotate-270\n"
		"          dst is horizontal flip of src: hflip\n"
		"          dst is vertical flip of src: vflip\n"
		"      - threshold object initial values can be:\n"
		"          For VX_THRESHOLD_TYPE_BINARY: <value>\n"
		"          For VX_THRESHOLD_TYPE_RANGE: {<lower>;<upper>}\n"
		"      - image object initial values can be:\n"
		"          Binary file with image data. For images created from handle,\n"
		"          the vxSwapHandles API will be invoked before executing the graph.\n"
		"      - tensor object initial values can be:\n"
		"          Binary file with tensor data.\n"
		"          To replicate a file multiple times, use @repeat~N~<fileName>.\n"
		"          To fill the tensor with a value, use @fill~f32~<float-value>,\n"
		"          @fill~i32~<int-value>, @fill~i16~<int-value>, or @fill~u8~<uint-value>.\n"
		"\n"
		);
	if (strstr("read", command)) printf(
		"  read <dataName> <fileName> [ascii|binary] [<option(s)>]\n"
		"      Read frame-level data from the specified <fileName>.\n"
		"      - images can be read from containers (such as, .jpg, .avi, .mp4, etc.)\n"
		"        as well as raw binary files\n"
		"      - certain raw data formats support reading data for all frames from a\n"
		"        single file (such as, video.yuv, video.rgb, video.avi etc.)\n"
		"        The data objects that support this feature are image, scalar, and\n"
		"        threshold data objects.\n"
		"      - certain data formats support printf format-syntax (e.g., joy_%%04d.yuv)\n"
		"        to read individual data from separate files. Note that scalar and\n"
		"        threshold data objects doesn't support this feature. Also note that\n"
		"        pyramid objects expect all frames of each level in separate files.\n"
		"      - convolution objects support the option: scale\n"
		"        This will read scale value as the first 32-bit integer in file(s).\n"
		"\n"
		);
	if (strstr("write", command)) printf(
		"  write <dataName> <fileName> [ascii|binary] [<option(s)>]\n"
		"      Write frame-level data to the specified <fileName>.\n"
		"      - certain raw data formats support writing data for all frames into a\n"
		"        single file (such as, video.yuv, video.rgb, video.u8, etc.)\n"
		"        The data objects that support this feature are image, scalar, and\n"
		"        threshold data objects.\n"
		"      - certain data formats support printf format-syntax (e.g., joy_%%04d.yuv)\n"
		"        to write individual data from separate files. Note that scalar and\n"
		"        threshold data objects doesn't support this feature. Also note that\n"
		"        pyramid objects expect all frames of each level in separate files.\n"
		"      - convolution objects support the option: scale\n"
		"        This will write scale value as the first 32-bit integer in file(s).\n"
		"\n"
		);
	if (strstr("compare", command)) printf(
		"  compare <dataName> <fileName> [ascii|binary] [<option(s)>]\n"
		"      Compare frame-level data from the specified <fileName>.\n"
		"      - certain raw data formats support comparing data for all frames from a\n"
		"        single file (such as, video.yuv, video.rgb, video.u8, etc.)\n"
		"        The data objects that support this feature are image, scalar, and\n"
		"        threshold data objects.\n"
		"      - certain data formats support printf format-syntax (e.g., joy_%%04d.yuv)\n"
		"        to read individual data from separate files. Note that scalar and\n"
		"        threshold data objects doesn't support this feature.\n"
		"      - array objects with VX_TYPE_KEYPOINT data type support the options:\n"
		"          specify tolerance: err{<x>;<y>;<strength>[;<%%mismatch>]}\n"
		"          specify compare log file: log{<fileName>}\n"
		"      - array objects with VX_TYPE_COORDINATES2D data type support the options:\n"
		"          specify tolerance: err{<x>;<y>[;<%%mismatch>]}\n"
		"          specify compare log file: log{<fileName>}\n"
		"      - convolution objects support the option:\n"
		"          read scale value as the first 32-bit integer in file(s): scale\n"
		"      - image and pyramid objects support the options:\n"
		"          specify compare region: rect{<start-x>;<start-y>;<end-x>;<end-y>}\n"
		"          specify valid pixel difference: err{<min>;<max>}\n"
		"          specify pixel checksum to compare: checksum\n"
		"          specify generate checksum: checksum-save-instead-of-test\n"
		"      - matrix objects support the options:\n"
		"          specify tolerance: err{<tolerance>}\n"
		"      - remap objects support the options:\n"
		"          specify tolerance: err{<x>;<y>}\n"
		"      - scalar objects support the option:\n"
		"          specify that file specifies inclusive range of valid values: range\n"
		"\n"
		);
	if (strstr("view", command)) printf(
		"  view <dataName> <windowName>\n"
		"      Display frame-level data in a window with title <windowName>. Each window\n"
		"      can display an image data object and optionally additional other data\n"
		"      objects overlaid on top of the image.\n"
		"      - supported data object types are: array, distribution, image, lut,\n"
		"        scalar, and delay.\n"
		"      - display of array, distribution, lut, and scalar objects are\n"
		"        overlaid on top of an image with the same <windowName>.\n"
		"      - delay object displays reference in the slot#0 of current time.\n"
		"\n"
		);
	if (strstr("directive", command)) printf(
		"  directive <dataName> <directive>\n"
		"      Specify a directive to data object. Only a few directives are supported:\n"
		"      - Use sync-cl-write directive to issue VX_DIRECTIVE_AMD_COPY_TO_OPENCL\n"
		"        directive whenever data object is updated using init or read commands.\n"
		"        Supported for array, image, lut, and remap data objects only.\n"
		"      - Use readonly directive to issue VX_DIRECTIVE_AMD_READ_ONLY directive\n"
		"        that informs the OpenVX framework that object won't be updated after\n"
		"        init command. Supported for convolution and matrix data objects only.\n"
		"\n"
		);
	if (strstr("pause", command)) printf(
		"  pause\n"
		"      Wait until a key is pressed before processing next GDF command.\n"
		"\n"
		);
	if (strstr("help", command)) printf(
		"  help [command]\n"
		"      Show the GDF command help.\n"
		"\n"
		);
	if (strstr("exit", command)) printf(
		"  exit\n"
		"      Exit from shell or included GDF file.\n"
		"\n"
		);
	if (strstr("quit", command)) printf(
		"  quit\n"
		"      Abort the application.\n"
		"\n"
		);
}
