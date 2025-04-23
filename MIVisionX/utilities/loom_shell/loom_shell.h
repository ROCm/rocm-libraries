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
#ifndef __LOOM_SHELL_H__
#define __LOOM_SHELL_H__

#include <stdio.h>
#include <string>
#include <map>

#include "live_stitch_api.h"
#if __APPLE__
#include <cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif

#define VERSION          "0.9.9"
#define SCRIPT_EXTENSION ".lss"
#if _WIN32
#define PROGRAM_NAME     "loom_shell.exe"
#else
#define PROGRAM_NAME     "loom_shell"
#endif

#define DEFAULT_VX_CONTEXT_COUNT     4     // default number of OpenVX context count
#define DEFAULT_CL_CONTEXT_COUNT     4     // default number of OpenCL context count
#define DEFAULT_CL_BUFFER_COUNT     16     // default number of OpenCL buffer count
#define DEFAULT_LS_CONTEXT_COUNT    16     // default number of ls_context count

class CCommandLineParser
{
public:
	CCommandLineParser();
	void SetVerbose(bool verbose);
	int Run(const char * fileName);

protected:
	void Message(const char * format, ...);
	void Terminate(int code, const char * format, ...);
	int Error(const char * format, ...);
	bool Verbose();
	virtual int OnCommand();

protected:
	std::string cmd_;

private:
	bool GetCommand(FILE * fp);

private:
	bool verbose_;
	int lineNum_;
	const char * fileName_;
	int includeLevels_;
};

class CLoomShellParser : public CCommandLineParser
{
public:
	CLoomShellParser();
	~CLoomShellParser();
	void help(bool detailed);

protected:
	virtual int OnCommand();

private:
	const char * ParseIndex(const char * s, const char * prefix, vx_uint32& index, vx_uint32 count);
	const char * ParseUInt(const char * s, vx_uint32& value);
	const char * ParseInt(const char * s, vx_int32& value);
	const char * ParseFloat(const char * s, vx_float32& value);
	const char * ParseWord(const char * s, char * value, size_t size);
	const char * ParseString(const char * s, char * value, size_t size);
	const char * ParseSkipPattern(const char * s, const char * pattern);
	const char * ParseSkip(const char * s, const char * charList);
	const char * ParseEndOfLine(const char * s);
	const char * ParseContextWithErrorCheck(const char * s, vx_uint32& index, const char * syntaxError);
	const char * ParseFormat(const char * s, vx_df_image& format);
	int ReleaseAllResources();

private:
	bool decl_ls_disabled, decl_vx_disabled, decl_cl_disabled, decl_buf_disabled;
	char name_ls[64], name_vx[64], name_cl[64], name_buf[64];
	vx_uint32 num_context_, num_openvx_context_, num_opencl_context_, num_opencl_buf_;
	ls_context * context_;
	vx_context * openvx_context_;
	cl_context * opencl_context_;
	bool * openvx_context_allocated_;
	bool * opencl_context_allocated_;
	cl_mem * opencl_buf_mem_;
	std::map<std::string, camera_params> camParList;
	std::map<std::string, rig_params> rigParList;
	vx_float32 attr_buf_[LIVE_STITCH_ATTR_MAX_COUNT];
};

#endif // __LOOM_SHELL_H__
