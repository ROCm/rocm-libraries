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
#include "vxParamHelper.h"
#include "vxEngineUtil.h"
#include "vxEngine.h"
#include <iostream>

char buf[4096];
// program and version
#define RUNVX_VERSION "1.0.0"
#if _WIN32
#include <direct.h>
#define cwd _getcwd
#define cd _chdir
#define RUNVX_PROGRAM "runvx.exe"
#else
#include "unistd.h"
#define cwd getcwd
#define cd chdir
#define RUNVX_PROGRAM "runvx"
#endif

void show_usage(const char * program, bool detail)
{
	printf("\n");
	printf("Usage:\n");
	printf("  %s [options] [file] <file.gdf> [argument(s)]\n", RUNVX_PROGRAM);
	printf("  %s [options] node <kernelName> [argument(s)]\n", RUNVX_PROGRAM);
	printf("  %s [options] shell [argument(s)]\n", RUNVX_PROGRAM);
	printf("\n");
	printf("The argument(s) are data objects created using <data-description> syntax.\n");
	printf("These arguments can be accessed from inside GDF as $1, $2, etc.\n");
	printf("\n");
	printf("The available command-line options are:\n");
	printf("  -h\n");
	printf("      Show full help.\n");
	printf("  -v\n");
	printf("      Turn on verbose logs.\n");
	printf("  -root:<directory>\n");
	printf("      Replace ~ in filenames with <directory> in the command-line and\n");
	printf("      GDF file. The default value of '~' is current working directory.\n");
	printf("  -frames:[<start>:]<end>|eof|loop|live\n");
	printf("      Run the graph/node for specified frames or until eof or just as live.\n");
	printf("      Use live to indicate that input is live until aborted by user.\n");
	printf("      Use <number of frames>:loop to indicate that input should be looped the amount of times stated with number of frames.\n");
	printf("  -affinity:CPU|GPU[<device-index>]\n");
	printf("      Set context affinity to CPU or GPU.\n");
	printf("  -dump-profile\n");
	printf("      Print performance profiling information after graph launch.\n");
	printf("  -enable-profile\n");
	printf("      use directive VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE when graph is created\n");
	printf("  -discard-compare-errors\n");
	printf("      Continue graph processing even if compare mismatches occur.\n");
	printf("  -disable-virtual\n");
	printf("      Replace all virtual data types in GDF with non-virtual data types.\n");
	printf("      Use of this flag (i.e. for debugging) can make a graph run slower.\n");
	printf("  -dump-data-config:<dumpFilePrefix>,<object-type>[,object-type[...]]\n");
	printf("      Automatically dump all non-virtual objects of specified object types\n");
	printf("      into files '<dumpFilePrefix>dumpdata_####_<object-type>_<object-name>.raw'.\n");
	printf("  -discard-commands:<cmd>[,cmd[...]]\n");
	printf("      Discard the listed commands.\n");
	printf("\n");

	if (!detail) return;
	PrintHelpGDF();
}

int main(int argc, char * argv[])
{	
	printf("%s %s\n", RUNVX_PROGRAM, RUNVX_VERSION);
	// process command-line options
	const char * program = RUNVX_PROGRAM;
	bool verbose = false;
	bool enableMultiFrameProcessing = false;
	bool framesEofRequested = true;
	bool enableDumpGDF = false, enableScheduleGraph = false;
	bool pauseBeforeExit = false, noPauseBeforeExit = false;
	bool enableDumpProfile = false;
	bool disableVirtual = false;
	bool discardCompareErrors = false;
	vx_uint32 defaultTargetAffinity = 0;
	vx_uint32 defaultTargetInfo = 0;
	bool doSetGraphOptimizerFlags = false;
	vx_uint32 graphOptimizerFlags = 0;
	int arg, frameStart = 0, frameEnd = 1;
	bool frameCountSpecified = false;
	int waitKeyDelayInMilliSeconds = -1; // -ve indicates no user preference
	bool enableFullProfile = false, disableNodeFlushForCL = false;
	std::string dumpDataConfig = "";
	std::string discardCommandList = "";
	for (arg = 1; arg < argc; arg++){
		if (argv[arg][0] == '-'){
			if (!_stricmp(argv[arg], "-h")) {
				show_usage(program, true);
				exit(0);
			}
			else if (!_stricmp(argv[arg], "-v")) {
				verbose ^= true;
			}
			else if (!strncmp(argv[arg], "--", 2)) { // skip specified number of arguments: --[#] (default just skip --)
				arg += atoi(&argv[arg][2]);
			}
			else if (!_strnicmp(argv[arg], "-root:", 6)) {
				SetRootDir(argv[arg] + 6);
			}
			else if (!_strnicmp(argv[arg], "-frames:", 8)) {
				int spos = 8;
				while (argv[arg][spos]) {
					if (argv[arg][spos] == ',')
						spos++;
					else if (!_strnicmp(&argv[arg][spos], "live", 4)) {
						enableMultiFrameProcessing = true;
						spos += 4;
					}
					else if (!_strnicmp(&argv[arg][spos], "eof", 3)) {
						framesEofRequested = true;
						spos += 3;
					}
					else if (!_strnicmp(&argv[arg][spos], "loop", 4)) {
						framesEofRequested = false;
						spos += 4;
					}
					else {
						int k = sscanf(&argv[arg][spos], "%d:%d", &frameStart, &frameEnd);
						if (k == 1) { frameEnd = frameStart, frameStart = 0; }
						else if (k != 2) { printf("ERROR: invalid -frames option\n"); return -1; }
						frameCountSpecified = true;
						while (argv[arg][spos] && argv[arg][spos] != ',')
							spos++;
					}
				}
			}
			else if (!_strnicmp(argv[arg], "-affinity:", 10)) {
				if (!_strnicmp(&argv[arg][10], "cpu", 3)) defaultTargetAffinity = AGO_TARGET_AFFINITY_CPU;
				else if (!_strnicmp(&argv[arg][10], "gpu", 3)) defaultTargetAffinity = AGO_TARGET_AFFINITY_GPU;
				else { printf("ERROR: unsupported affinity target: %s\n", &argv[arg][10]); return -1; }
				if (argv[arg][13] >= '0' && argv[arg][13] <= '9')
					defaultTargetInfo = atoi(&argv[arg][13]);
			}
			else if (!_stricmp(argv[arg], "-dump-profile")) {
				enableDumpProfile = true;
			}
			else if (!_stricmp(argv[arg], "-enable-profile")) {
				enableFullProfile = true;
			}
			else if (!_stricmp(argv[arg], "-disable-opencl-node-flush")) {
				disableNodeFlushForCL = true;
			}
			else if (!_stricmp(argv[arg], "-dump-gdf") || !_stricmp(argv[arg], "-ago-dump")) { // TBD: remove -ago-dump
				enableDumpGDF = true;
			}
			else if (!_stricmp(argv[arg], "-discard-compare-errors")) {
				discardCompareErrors = true;
			}
			else if (!_stricmp(argv[arg], "-use-schedule-graph")) {
				enableScheduleGraph = true;
			}
			else if (!_stricmp(argv[arg], "-disable-virtual")) {
				disableVirtual = true;
			}
			else if (!_strnicmp(argv[arg], "-dump-data-config:", 18)) {
				dumpDataConfig = &argv[arg][18];
			}
			else if (!_strnicmp(argv[arg], "-discard-commands:", 18)) {
				discardCommandList = &argv[arg][18];
			}
			else if (!_strnicmp(argv[arg], "-graph-optimizer-flags:", 23)) {
				if (sscanf(&argv[arg][23], "%i", &graphOptimizerFlags) == 1) {
					doSetGraphOptimizerFlags = true;
				}
				else { printf("ERROR: invalid graph optimizer flags: %s\n", argv[arg]); return -1; }
			}
			else if (!_strnicmp(argv[arg], "-key-wait-delay:", 16)) {
				(void)sscanf(&argv[arg][16], "%i", &waitKeyDelayInMilliSeconds);
			}
			else if (!_stricmp(argv[arg], "-pause")) {
				pauseBeforeExit = true;
			}
			else if (!_stricmp(argv[arg], "-no-pause")) {
				noPauseBeforeExit = true;
			}
			else { printf("ERROR: invalid option: %s\n", argv[arg]); return -1; }
		}
		else break;
	}
	if (arg == argc) { show_usage(program, false); return -1; }
	int argCount = argc - arg - 1;
	int argParamOffset = 1;
	if (!_stricmp(argv[arg], "node") || !_stricmp(argv[arg], "file")) {
		argParamOffset++;
		argCount--;
	}
	fflush(stdout);

	CVxEngine engine;
	int errorCode = 0;
	try {
		// initialize engine
		if (engine.Initialize(argCount, defaultTargetAffinity, defaultTargetInfo, enableScheduleGraph, disableVirtual, enableFullProfile, disableNodeFlushForCL, discardCommandList) < 0) throw - 1;
		if (doSetGraphOptimizerFlags) {
			engine.SetGraphOptimizerFlags(graphOptimizerFlags);
		}
		if (dumpDataConfig.find(",") != std::string::npos) {
			engine.SetDumpDataConfig(dumpDataConfig);
		}
		engine.SetConfigOptions(verbose, discardCompareErrors, enableDumpProfile, enableDumpGDF, waitKeyDelayInMilliSeconds);
		engine.SetFrameCountOptions(enableMultiFrameProcessing, framesEofRequested, frameCountSpecified, frameStart, frameEnd);
		fflush(stdout);
		// pass parameters to the engine: note that shell takes no extra parameters whereas node and file take extra parameter
		for (int i = 0, j = 0; i < argCount; i++) {
			char * param = argv[arg + argParamOffset + i];
			if (engine.SetParameter(j++, param) < 0)
				throw -1;
		}
		fflush(stdout);
		// get full GDF text
		char * fullText = nullptr;
		if (!_stricmp(argv[arg], "node")) {
			if ((arg + 1) == argc)
				ReportError("ERROR: missing kernel name on command-line (see help for details)\n");
			int paramCount = argc - arg - 2;
			arg++;
			size_t size = strlen("node") + 1 + strlen(argv[arg]) + paramCount*6 + 1;
			fullText = new char[size];
			snprintf(fullText, size, "node %s", argv[arg]);
			for (int i = 0, j = 0; i < paramCount; i++)
				snprintf(fullText + strlen(fullText), (size - strlen(fullText)), " $%d", j++ + 1);
		}
		else if (!_stricmp(argv[arg], "shell")) {
			// nothing to do
		}
		else {
			if (!_stricmp(argv[arg], "file")) {
				if ((arg + 1) == argc)
					ReportError("ERROR: missing file name on command-line (see help for details)\n");
				arg++;
			}
            std::string fileNameToParse_s;
            std::string addToDir_s;
            std::string gdfName_s;
            std::string delimiter = "/";

            fileNameToParse_s.assign(argv[arg]);

            int pos = 0;
            std::string token;
            while((pos = (int)(fileNameToParse_s.find(delimiter))) != std::string::npos){
                token = fileNameToParse_s.substr(0, pos);
                addToDir_s.append(token);
                addToDir_s.append("/");
                fileNameToParse_s.erase(0, pos + delimiter.length());
            }
            gdfName_s.assign(fileNameToParse_s);

            int chdir_status = 0;

            chdir_status = cd(addToDir_s.c_str());
            const char * fileName = RootDirUpdated(gdfName_s.c_str());

			size_t size = strlen("include") + 1 + strlen(fileName) + 1;
			fullText = new char[size];
			snprintf(fullText, size, "include %s", fileName);
		}

		if (fullText) {
			// process the GDF
			if (engine.BuildAndProcessGraph(0, fullText, false) < 0)
				throw - 1;
			delete[] fullText;
		}
		else {
			// run shell
			if (engine.Shell(0) < 0)
				throw - 1;
		}
		fflush(stdout);

		if (engine.Shutdown() < 0) throw -1;
		fflush(stdout);
	}
	catch (int errorCode_) {
		fflush(stdout);
		engine.DisableWaitForKeyPress();
		errorCode = errorCode_;
	}
	if (pauseBeforeExit) {
		fflush(stdout);
		printf("Press ENTER to exit ...\n");
		while (getchar() != '\n')
			;
		engine.DisableWaitForKeyPress();
	}
	else if (noPauseBeforeExit) {
		engine.DisableWaitForKeyPress();
	}
	return errorCode;
}
