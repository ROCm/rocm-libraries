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
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "loom_shell.h"
#include "loom_shell_util.h"
#include <stdarg.h>
#if _WIN32
#include <direct.h>
#include <Windows.h>
#else
#include <unistd.h>
#include <strings.h>
#define _strnicmp strncasecmp
#define _stricmp  strcasecmp
#define _chdir    chdir
#endif

void stitch_log_callback(const char * message)
{
	printf("%s", message);
	fflush(stdout);
}

CLoomShellParser::CLoomShellParser()
{
	// initialize default counts
	decl_ls_disabled = false;
	decl_vx_disabled = false;
	decl_cl_disabled = false;
	decl_buf_disabled = false;
	num_context_ = DEFAULT_LS_CONTEXT_COUNT;
	num_openvx_context_ = DEFAULT_VX_CONTEXT_COUNT;
	num_opencl_context_ = DEFAULT_CL_CONTEXT_COUNT;
	num_opencl_buf_ = DEFAULT_CL_BUFFER_COUNT;
	// set log callback
	lsGlobalSetLogCallback(stitch_log_callback);
	// name of contexts
	strcpy(name_ls, "ls");
	strcpy(name_vx, "vx");
	strcpy(name_cl, "cl");
	strcpy(name_buf, "buf");
	// create array for contexts, cmd_queues, and buffers
	context_ = new ls_context[num_context_]();
	openvx_context_ = new vx_context[num_openvx_context_]();
	opencl_context_ = new cl_context[num_opencl_context_]();
	openvx_context_allocated_ = new bool[num_openvx_context_]();
	opencl_context_allocated_ = new bool[num_opencl_context_]();
	opencl_buf_mem_ = new cl_mem[num_opencl_buf_]();
	// misc
	memset(attr_buf_, 0, sizeof(attr_buf_));
}

CLoomShellParser::~CLoomShellParser()
{
	if (context_) delete[] context_;
	if (opencl_buf_mem_) delete[] opencl_buf_mem_;
	if (openvx_context_) delete[] openvx_context_;
	if (opencl_context_) delete[] opencl_context_;
	if (openvx_context_allocated_) delete[] openvx_context_allocated_;
	if (opencl_context_allocated_) delete[] opencl_context_allocated_;
}

const char * CLoomShellParser::ParseIndex(const char * s, const char * prefix, vx_uint32& index, vx_uint32 count)
{
	// skip prefix
	if (prefix) {
		s = ParseSkipPattern(s, prefix);
		if (!s) return nullptr;
	}
	if (count < 2) index = 0;
	else {
		// skip initial bracket
		if (*s++ != '[') return nullptr;
		// get index
		s = ParseUInt(s, index);
		if (!s) return nullptr;
		// skip last bracked
		if (*s++ != ']') return nullptr;
	}
	return s;
}

const char * CLoomShellParser::ParseUInt(const char * s, vx_uint32& value)
{
	bool gotValue = false;
	vx_uint32 multiplier = 1;
	while (*s >= '0' && *s <= '9') {
		for (value = 0; *s >= '0' && *s <= '9'; s++) {
			value = value * 10 + (*s - '0');
			gotValue = true;
		}
		value *= multiplier;
		if (*s == '*') {
			// continue processing after multiplication
			s++;
			multiplier = value;
		}
	}
	if (!gotValue) return nullptr;
	return s;
}

const char * CLoomShellParser::ParseInt(const char * s, vx_int32& value)
{
	bool negative = false, gotValue = false;
	if (*s == '-' || *s == '+') {
		negative = (*s == '-') ? true : false;
		s++;
	}
	for (value = 0; *s >= '0' && *s <= '9'; s++) {
		value = value * 10 + (*s - '0');
		gotValue = true;
	}
	if (negative)
		value = -value;
	if (!gotValue) return nullptr;
	return s;
}

const char * CLoomShellParser::ParseFloat(const char * s, vx_float32& value)
{
	bool negative = false, gotValue = false;
	if (*s == '-' || *s == '+') {
		negative = (*s == '-') ? true : false;
		s++;
	}
	for (value = 0.0f; *s >= '0' && *s <= '9'; s++) {
		value = value * 10.0f + (vx_float32)(*s - '0');
		gotValue = true;
	}
	if (*s == '.') {
		vx_float32 f = 1.0f;
		for (s++; *s >= '0' && *s <= '9'; s++) {
			value = value * 10.0f + (vx_float32)(*s - '0');
			f = f * 10.0f;
			gotValue = true;
		}
		value /= f;
	}
	if (negative)
		value = -value;
	if (!gotValue) return nullptr;
	return s;
}

const char * CLoomShellParser::ParseWord(const char * s, char * value, size_t size)
{
	bool gotValue = false;
	// copy word until separator (i.e., SPACE or end paranthesis or end brace)
	for (size_t len = 1; len < size && *s && *s != ' ' && *s != ';' && *s != ',' && *s != '{' && *s != '(' && *s != '[' && *s != ']' && *s != ')' && *s != '}'; len++) {
		*value++ = *s++;
		gotValue = true;
	}
	*value = '\0';
	if (!gotValue) return nullptr;
	return s;
}

const char * CLoomShellParser::ParseString(const char * s, char * value, size_t size)
{
	// skip initial quote
	if (*s++ != '"') return nullptr;
	// copy string
	for (size_t len = 1; len < size && *s && *s != '"'; len++)
		*value++ = *s++;
	*value = '\0';
	// skip last quote
	if (*s++ != '"') return nullptr;
	return s;
}

const char * CLoomShellParser::ParseSkipPattern(const char * s, const char * pattern)
{
	// skip whitespace
	while (*s && *s == ' ')
		s++;
	// skip pattern
	for (; *pattern; pattern++, s++) {
		if (*pattern != *s)
			return nullptr;
	}
	// skip whitespace
	while (*s && *s == ' ')
		s++;
	return s;
}

const char * CLoomShellParser::ParseSkip(const char * s, const char * charList)
{
	// skip whitespace
	while (*s && *s == ' ')
		s++;
	// skip pattern
	for (; *charList; charList++) {
		if (*charList != *s)
			return nullptr;
		s++;
		if (*s == ' ') s++;
	}
	// skip whitespace
	while (*s && *s == ' ')
		s++;
	return s;
}

const char * CLoomShellParser::ParseEndOfLine(const char * s)
{
	// skip whitespace and ';'
	bool foundSemicolon = false;
	while (*s && *s == ' ') s++;
	if (*s == ';') {
		foundSemicolon = true;
		s++;
	}
	while (*s && *s == ' ') s++;
	// check for end-of-line
	if (*s != '\0') {
		Error("ERROR: unexpected character '%c' at the end", *s);
		return nullptr;
	}
	if (!foundSemicolon) {
		Error("ERROR: missing ';' at the end", *s);
		return nullptr;
	}
	return s;
}

const char * CLoomShellParser::ParseContextWithErrorCheck(const char * s, vx_uint32& index, const char * syntaxError)
{
	s = ParseIndex(s, name_ls, index, num_context_);
	if (!s) {
		Error(syntaxError);
		return nullptr;
	}
	if (index >= num_context_) {
		Error("ERROR: context out-of-range: expects: 0..%d", num_context_ - 1);
		return nullptr;
	}
	if (!context_[index]) {
		Error("ERROR: %s[%d] doesn't exist", name_ls, index);
		return nullptr;
	}
	return s;
}

const char * CLoomShellParser::ParseFormat(const char * s, vx_df_image& format)
{
	char word[64];
	s = ParseWord(s, word, sizeof(word));
	if (s) {
		if (!_stricmp(word, "VX_DF_IMAGE_RGB")) format = VX_DF_IMAGE_RGB;
		else if (!_stricmp(word, "VX_DF_IMAGE_RGBX")) format = VX_DF_IMAGE_RGBX;
		else if (!_stricmp(word, "VX_DF_IMAGE_UYVY")) format = VX_DF_IMAGE_UYVY;
		else if (!_stricmp(word, "VX_DF_IMAGE_YUYV")) format = VX_DF_IMAGE_YUYV;
		else {
			if (strlen(word) != 4) {
				Error("ERROR: image format should have FOUR characters");
				return nullptr;
			}
			format = VX_DF_IMAGE(word[0], word[1], word[2], word[3]);
		}
	}
	return s;
}

int CLoomShellParser::ReleaseAllResources()
{
	if (ClearCmdqCache())
		Terminate(1, "ERROR: ClearCmdqCache() failed\n");
	if (opencl_buf_mem_) {
		for (vx_uint32 i = 0; i < num_opencl_context_; i++) {
			if (opencl_buf_mem_[i]) {
				cl_int status = clReleaseMemObject(opencl_buf_mem_[i]);
				if (status < 0) Terminate(1, "ERROR: clReleaseMemObject(%s[%d]) failed (%d)\n", name_buf, i, status);
				opencl_buf_mem_[i] = nullptr;
				Message("..released %s[%d]\n", name_buf, i);
			}
		}
	}
	if (context_) {
		for (vx_uint32 i = 0; i < num_context_; i++) {
			if (context_[i]) {
				vx_status status = lsReleaseContext(&context_[i]);
				if (status < 0) Terminate(1, "ERROR: lsReleaseContext(%s[%d]) failed (%d)\n", name_ls, i, status);
				Message("..released %s[%d]\n", name_ls, i);
			}
		}
	}
	if (openvx_context_ && openvx_context_allocated_) {
		for (vx_uint32 i = 0; i < num_openvx_context_; i++) {
			if (openvx_context_[i]) {
				if (openvx_context_allocated_[i]) {
					vx_status status = vxReleaseContext(&openvx_context_[i]);
					if (status < 0) Terminate(1, "ERROR: vxReleaseContext(%s[%d]) failed (%d)\n", name_vx, i, status);
					Message("..released %s[%d]\n", name_vx, i);
				}
				else {
					openvx_context_[i] = nullptr;
				}
			}
		}
	}
	if (opencl_context_ && opencl_context_allocated_) {
		for (vx_uint32 i = 0; i < num_opencl_context_; i++) {
			if (opencl_context_[i]) {
				if (opencl_context_allocated_[i]) {
					cl_int status = clReleaseContext(opencl_context_[i]);
					if (status < 0) Terminate(1, "ERROR: clReleaseContext(%s[%d]) failed (%d)\n", name_cl, i, status);
					Message("..released %s[%d]\n", name_cl, i);
				}
				opencl_context_[i] = nullptr;
			}
		}
	}
	return 0;
}

int CLoomShellParser::OnCommand()
{
#define SYNTAX_CHECK(call)                  s = call; if(!s) return Error(invalidSyntax);
#define SYNTAX_CHECK_WITH_MESSAGE(call,msg) s = call; if(!s) return Error(msg);
	const char *s = cmd_.c_str();
	// get command and skip whitespace
	char command[64];
	s = ParseWord(s, command, sizeof(command)); if (!s || !command[0]) return Error("ERROR: valid command missing");
	s = ParseSkip(s, "");
	// process command
	if (!_stricmp(command, name_ls)) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: ls[#] = lsCreateContext()";
		SYNTAX_CHECK(ParseIndex(s, "", contextIndex, num_context_));
		SYNTAX_CHECK(ParseSkip(s, "="));
		SYNTAX_CHECK(ParseSkipPattern(s, "lsCreateContext"));
		SYNTAX_CHECK(ParseSkip(s, "()"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (contextIndex >= num_context_) return Error("ERROR: context out-of-range: expects: 0..%d", num_context_ - 1);
		if (context_[contextIndex]) return Error("ERROR: context %s[%d] already exists", name_ls, contextIndex);
		// process the command
		context_[contextIndex] = lsCreateContext();
		if (!context_[contextIndex]) return Error("ERROR: lsCreateContext() failed");
		Message("..lsCreateContext: created context %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsReleaseContext")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsReleaseContext(&ls[#])";
		SYNTAX_CHECK(ParseSkip(s, "(&"));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsReleaseContext(&context_[contextIndex]);
		if (status) return Error("ERROR: lsReleaseContext(%s[%d]) failed (%d)", name_ls, contextIndex, status);
		Message("..lsReleaseContext: released context %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOpenVXContext")) {
		// parse the command
		vx_uint32 contextIndex = 0, vxIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOpenVXContext(ls[#],vx[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseIndex(s, name_vx, vxIndex, num_openvx_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (vxIndex >= num_openvx_context_) return Error("ERROR: OpenVX context out-of-range: expects: 0..%d", num_openvx_context_ - 1);
		// process the command
		vx_status status = lsSetOpenVXContext(context_[contextIndex], openvx_context_[vxIndex]);
		if (status) return Error("ERROR: lsSetOpenVXContext(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_vx, vxIndex, status);
		Message("..lsSetOpenVXContext: set OpenVX context %s[%d] for %s[%d]\n", name_vx, vxIndex, name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOpenCLContext")) {
		// parse the command
		vx_uint32 contextIndex = 0, clIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOpenCLContext(ls[#],cl[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseIndex(s, name_cl, clIndex, num_opencl_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (clIndex >= num_opencl_context_) return Error("ERROR: OpenCL context out-of-range: expects: 0..%d", num_opencl_context_ - 1);
		// process the command
		vx_status status = lsSetOpenCLContext(context_[contextIndex], opencl_context_[clIndex]);
		if (status) return Error("ERROR: lsSetOpenCLContext(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_cl, clIndex, status);
		Message("..lsSetOpenCLContext: set OpenCL context %s[%d] for %s[%d]\n", name_cl, clIndex, name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetRigParams")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		rig_params rig_par = { 0 };
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetRigParams(context,&rig_par)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ",&"));
		SYNTAX_CHECK(ParseSkip(s, ""));
		if (*s == '{') {
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "{"), "ERROR: missing '{'");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.yaw), "ERROR: invalid yaw value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing pitch value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.pitch), "ERROR: invalid pitch value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing roll value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.roll), "ERROR: invalid roll value");
			SYNTAX_CHECK(ParseSkip(s, ""));
			if (*s == ',') {
				SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing d value");
				SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.d), "ERROR: invalid d value");
			}
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "})"), "ERROR: missing '})'");
			SYNTAX_CHECK(ParseEndOfLine(s));
		}
		else {
			char parName[64];
			SYNTAX_CHECK(ParseWord(s, parName, sizeof(parName)));
			SYNTAX_CHECK(ParseSkip(s, ")"));
			SYNTAX_CHECK(ParseEndOfLine(s));
			if (rigParList.find(parName) == rigParList.end()) return Error("ERROR: rig_params %s not defined", parName);
			memcpy(&rig_par, &rigParList[parName], sizeof(rig_par));
		}
		// process the command
		vx_status status = lsSetRigParams(context_[contextIndex], &rig_par);
		if (status) return Error("ERROR: lsSetRigParams(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetRigParams: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetCameraConfig")) {
		// parse the command
		vx_uint32 contextIndex = 0, camera_rows = 0, camera_cols = 0, buffer_width = 0, buffer_height = 0; vx_df_image buffer_format = VX_DF_IMAGE_VIRT;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetCameraConfig(ls[#],rows,cols,format,width,height)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, camera_rows));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, camera_cols));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFormat(s, buffer_format));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_width));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_height));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetCameraConfig(context_[contextIndex], camera_rows, camera_cols,
			buffer_format, buffer_width, buffer_height);
		if (status) return Error("ERROR: lsSetCameraConfig(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetCameraConfig: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOutputConfig")) {
		// parse the command
		vx_uint32 contextIndex = 0, buffer_width = 0, buffer_height = 0; vx_df_image buffer_format = VX_DF_IMAGE_VIRT;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOutputConfig(ls[#],format,width,height)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFormat(s, buffer_format));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_width));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_height));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetOutputConfig(context_[contextIndex],
			buffer_format, buffer_width, buffer_height);
		if (status) return Error("ERROR: lsSetOutputConfig(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetOutputConfig: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOverlayConfig")) {
		// parse the command
		vx_uint32 contextIndex = 0, overlay_rows = 0, overlay_cols = 0, buffer_width = 0, buffer_height = 0; vx_df_image buffer_format = VX_DF_IMAGE_VIRT;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOverlayConfig(ls[#],rows,cols,format,width,height)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, overlay_rows));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, overlay_cols));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFormat(s, buffer_format));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_width));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_height));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetOverlayConfig(context_[contextIndex], overlay_rows, overlay_cols,
			buffer_format, buffer_width, buffer_height);
		if (status) return Error("ERROR: lsSetOverlayConfig(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetOverlayConfig: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetCameraParams") || !_stricmp(command, "lsSetOverlayParams")) {
		bool isCamera = !_stricmp(command, "lsSetCameraParams") ? true : false;
		// parse the command
		vx_uint32 contextIndex = 0, index = 0;
		camera_params camera_par = { 0 };
		const char * invalidSyntax = isCamera ?
			"ERROR: invalid syntax: expects: lsSetCameraParams(ls[#],index,&cam_par)" :
			"ERROR: invalid syntax: expects: lsSetOverlayParams(ls[#],index,&cam_par)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, index));
		SYNTAX_CHECK(ParseSkip(s, ",&"));
		SYNTAX_CHECK(ParseSkip(s, ""));
		if (*s == '{') {
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "{{"), "ERROR: missing '{{'");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.yaw), "ERROR: invalid yaw value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing pitch value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.pitch), "ERROR: invalid pitch value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing roll value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.roll), "ERROR: invalid roll value");
			SYNTAX_CHECK(ParseSkip(s, ""));
			if (*s == ',') {
				SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing tx value");
				SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.tx), "ERROR: invalid tx value");
				SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing ty value");
				SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.ty), "ERROR: invalid ty value");
				SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing tz value");
				SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.tz), "ERROR: invalid tz value");
			}
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "},{"), "ERROR: missing separator before lens_type");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.hfov), "ERROR: invalid hfov value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing haw value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.haw), "ERROR: invalid haw value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing r_crop value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.r_crop), "ERROR: invalid r_crop value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing du0 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.du0), "ERROR: invalid du0 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing dv0 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.dv0), "ERROR: invalid dv0 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing lens_type value");
			char lensType[64];
			SYNTAX_CHECK_WITH_MESSAGE(ParseWord(s, lensType, sizeof(lensType)), "ERROR: invalid lens_type value");
			if (!_stricmp(lensType, "ptgui_lens_rectilinear")) camera_par.lens.lens_type = ptgui_lens_rectilinear;
			else if (!_stricmp(lensType, "ptgui_lens_fisheye_ff")) camera_par.lens.lens_type = ptgui_lens_fisheye_ff;
			else if (!_stricmp(lensType, "ptgui_lens_fisheye_circ")) camera_par.lens.lens_type = ptgui_lens_fisheye_circ;
			else if (!_stricmp(lensType, "adobe_lens_rectilinear")) camera_par.lens.lens_type = adobe_lens_rectilinear;
			else if (!_stricmp(lensType, "adobe_lens_fisheye")) camera_par.lens.lens_type = adobe_lens_fisheye;
			else return Error("ERROR: invalid lens_type value: see help for valid values");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing k1 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.k1), "ERROR: invalid k1 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing k2 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.k2), "ERROR: invalid k2 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing k3 value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.k3), "ERROR: invalid k3 value");
			for (vx_uint32 i = 0; i < 7; i++) {
				s = ParseSkip(s, ""); if (*s != ',') break;
				s = ParseSkip(s, ",");
				SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.reserved[i]), "ERROR: invalid reserved field value");
			}
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "}})"), "ERROR: missing '}})'");
			SYNTAX_CHECK(ParseEndOfLine(s));
		}
		else {
			char parName[64];
			SYNTAX_CHECK(ParseWord(s, parName, sizeof(parName)));
			SYNTAX_CHECK(ParseSkip(s, ")"));
			SYNTAX_CHECK(ParseEndOfLine(s));
			if (camParList.find(parName) == camParList.end()) return Error("ERROR: camera_params %s not defined", parName);
			memcpy(&camera_par, &camParList[parName], sizeof(camera_par));
		}
		// process the command
		if (isCamera) {
			vx_status status = lsSetCameraParams(context_[contextIndex], index, &camera_par);
			if (status) return Error("ERROR: lsSetCameraParams(%s[%d],%d,*) failed (%d)", name_ls, contextIndex, index, status);
			Message("..lsSetCameraParams: successful for %s[%d] and camera#%d\n", name_ls, contextIndex, index);
		}
		else {
			vx_status status = lsSetOverlayParams(context_[contextIndex], index, &camera_par);
			if (status) return Error("ERROR: lsSetOverlayParams(%s[%d],%d,*) failed (%d)", name_ls, contextIndex, index, status);
			Message("..lsSetOverlayParams: successful for %s[%d] and overlay#%d\n", name_ls, contextIndex, index);
		}
	}
	else if (!_stricmp(command, "lsSetCameraBufferStride")) {
		// parse the command
		vx_uint32 contextIndex = 0, buffer_stride_in_bytes = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetCameraBufferStride(ls[#],stride_in_bytes)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_stride_in_bytes));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetCameraBufferStride(context_[contextIndex], buffer_stride_in_bytes);
		if (status) return Error("ERROR: lsSetCameraBufferStride(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetCameraBufferStride: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOutputBufferStride")) {
		// parse the command
		vx_uint32 contextIndex = 0, buffer_stride_in_bytes = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOutputBufferStride(ls[#],stride_in_bytes)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_stride_in_bytes));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetOutputBufferStride(context_[contextIndex], buffer_stride_in_bytes);
		if (status) return Error("ERROR: lsSetOutputBufferStride(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetOutputBufferStride: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOverlayBufferStride")) {
		// parse the command
		vx_uint32 contextIndex = 0, buffer_stride_in_bytes = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOverlayBufferStride(ls[#],stride_in_bytes)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_stride_in_bytes));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetOverlayBufferStride(context_[contextIndex], buffer_stride_in_bytes);
		if (status) return Error("ERROR: lsSetOverlayBufferStride(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetOverlayBufferStride: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetCameraModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		char module[256], kernelName[64], kernelArguments[1024];
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetCameraModule(ls[#],\"module\",\"kernelName\",\"kernelArguments\")";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, module, sizeof(module)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelName, sizeof(kernelName)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelArguments, sizeof(kernelArguments)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetCameraModule(context_[contextIndex], module, kernelName, kernelArguments);
		if (status) return Error("ERROR: lsSetCameraModule(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetCameraModule: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOutputModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		char module[256], kernelName[64], kernelArguments[1024];
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOutputModule(ls[#],\"module\",\"kernelName\",\"kernelArguments\")";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, module, sizeof(module)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelName, sizeof(kernelName)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelArguments, sizeof(kernelArguments)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetOutputModule(context_[contextIndex], module, kernelName, kernelArguments);
		if (status) return Error("ERROR: lsSetOutputModule(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetOutputModule: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetOverlayModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		char module[256], kernelName[64], kernelArguments[1024];
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOverlayModule(ls[#],\"module\",\"kernelName\",\"kernelArguments\")";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, module, sizeof(module)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelName, sizeof(kernelName)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelArguments, sizeof(kernelArguments)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetOverlayModule(context_[contextIndex], module, kernelName, kernelArguments);
		if (status) return Error("ERROR: lsSetOverlayModule(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetOverlayModule: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetViewingModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		char module[256], kernelName[64], kernelArguments[1024];
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetViewingModule(ls[#],\"module\",\"kernelName\",\"kernelArguments\")";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, module, sizeof(module)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelName, sizeof(kernelName)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, kernelArguments, sizeof(kernelArguments)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsSetViewingModule(context_[contextIndex], module, kernelName, kernelArguments);
		if (status) return Error("ERROR: lsSetViewingModule(%s[%d],*) failed (%d)", name_ls, contextIndex, status);
		Message("..lsSetViewingModule: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsInitialize")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsInitialize(ls[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		double clk2msec = 1000.0 / GetClockFrequency();
		int64_t clk = GetClockCounter();
		vx_status status = lsInitialize(context_[contextIndex]);
		double msec = clk2msec * (GetClockCounter() - clk);
		if (status) return Error("ERROR: lsInitialize(%s[%d]) failed (%d)", name_ls, contextIndex, status);
		Message("..lsInitialize: successful for %s[%d] (%7.3lf ms)\n", name_ls, contextIndex, msec);
	}
	else if (!_stricmp(command, "lsReinitialize")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsReinitialize(ls[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		double clk2msec = 1000.0 / GetClockFrequency();
		int64_t clk = GetClockCounter();
		vx_status status = lsReinitialize(context_[contextIndex]);
		double msec = clk2msec * (GetClockCounter() - clk);
		if (status) return Error("ERROR: lsReinitialize(%s[%d]) failed (%d)", name_ls, contextIndex, status);
		Message("..lsReinitialize: successful for %s[%d] (%7.3lf ms)\n", name_ls, contextIndex, msec);
	}
	else if (!_stricmp(command, "lsScheduleFrame")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsScheduleFrame(ls[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsScheduleFrame(context_[contextIndex]);
		if (status) return Error("ERROR: lsScheduleFrame(%s[%d]) failed (%d)", name_ls, contextIndex, status);
		Message("..lsScheduleFrame: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsWaitForCompletion")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsWaitForCompletion(ls[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsWaitForCompletion(context_[contextIndex]);
		if (status) return Error("ERROR: lsWaitForCompletion(%s[%d]) failed (%d)", name_ls, contextIndex, status);
		Message("..lsWaitForCompletion: successful for %s[%d]\n", name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsSetCameraBuffer")) {
		// parse the command
		vx_uint32 contextIndex = 0, bufIndex = 0;
		vx_uint32 num_buffers;
		bool useNull = false;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetCameraBuffer(ls[#],&buf[#]|NULL)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		if (!_stricmp(s, ",null)")) {
			useNull = true;
		}
		else {
			SYNTAX_CHECK(ParseSkip(s, ",&"));
			SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
			if (*s == ','){
				SYNTAX_CHECK(ParseSkip(s, ","));
				SYNTAX_CHECK(ParseUInt(s, num_buffers));
			}
			SYNTAX_CHECK(ParseSkip(s, ")"));
		}
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		// process the command
		if (useNull) {
			vx_status status = lsSetCameraBuffer(context_[contextIndex], nullptr);
			if (status) return Error("ERROR: lsSetCameraBuffer(%s[%d],NULL) failed (%d)", name_ls, contextIndex, status);
			Message("..lsSetCameraBuffer: set NULL for %s[%d]\n", name_ls, contextIndex);
		}
		else {
			vx_status status = lsSetCameraBuffer(context_[contextIndex], &opencl_buf_mem_[bufIndex]);
			if (status) return Error("ERROR: lsSetCameraBuffer(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_buf, bufIndex, status);
			Message("..lsSetCameraBuffer: set OpenCL buffer %s[%d] for %s[%d]\n", name_buf, bufIndex, name_ls, contextIndex);
		}
	}
	else if (!_stricmp(command, "lsSetOutputBuffer")) {
		// parse the command
		vx_uint32 contextIndex = 0, bufIndex = 0;
		vx_uint32 num_buffers;
		bool useNull = false;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOutputBuffer(ls[#],&buf[#]|NULL)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		if (!_stricmp(s, ",null)")) {
			useNull = true;
		}
		else {
			SYNTAX_CHECK(ParseSkip(s, ",&"));
			SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
			if (*s == ','){
				SYNTAX_CHECK(ParseSkip(s, ","));
				SYNTAX_CHECK(ParseUInt(s, num_buffers));
			}
			SYNTAX_CHECK(ParseSkip(s, ")"));
		}
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		// process the command
		if (useNull) {
			vx_status status = lsSetOutputBuffer(context_[contextIndex], nullptr);
			if (status) return Error("ERROR: lsSetOutputBuffer(%s[%d],NULL) failed (%d)", name_ls, contextIndex, status);
			Message("..lsSetOutputBuffer: set NULL for %s[%d]\n", name_ls, contextIndex);
		}
		else {
			vx_status status = lsSetOutputBuffer(context_[contextIndex], &opencl_buf_mem_[bufIndex]);
			if (status) return Error("ERROR: lsSetOutputBuffer(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_buf, bufIndex, status);
			Message("..lsSetOutputBuffer: set OpenCL buffer %s[%d] for %s[%d]\n", name_buf, bufIndex, name_ls, contextIndex);
		}
	}
	else if (!_stricmp(command, "lsSetOverlayBuffer")) {
		// parse the command
		vx_uint32 contextIndex = 0, bufIndex = 0;
		vx_uint32 num_buffers;
		bool useNull = false;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetOverlayBuffer(ls[#],&buf[#]|NULL)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		if (!_stricmp(s, ",null)")) {
			useNull = true;
		}
		else {
			SYNTAX_CHECK(ParseSkip(s, ",&"));
			SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
			if (*s == ','){
				SYNTAX_CHECK(ParseSkip(s, ","));
				SYNTAX_CHECK(ParseUInt(s, num_buffers));
			}
			SYNTAX_CHECK(ParseSkip(s, ")"));
		}
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		// process the command
		if (useNull) {
			vx_status status = lsSetOverlayBuffer(context_[contextIndex], nullptr);
			if (status) return Error("ERROR: lsSetOverlayBuffer(%s[%d],NULL) failed (%d)", name_ls, contextIndex, status);
			Message("..lsSetOverlayBuffer: set NULL for %s[%d]\n", name_ls, contextIndex);
		}
		else {
			vx_status status = lsSetOverlayBuffer(context_[contextIndex], &opencl_buf_mem_[bufIndex]);
			if (status) return Error("ERROR: lsSetOverlayBuffer(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_buf, bufIndex, status);
			Message("..lsSetOverlayBuffer: set OpenCL buffer %s[%d] for %s[%d]\n", name_buf, bufIndex, name_ls, contextIndex);
		}
	}
	else if (!_stricmp(command, "lsSetChromaKeyBuffer")) {
		// parse the command
		vx_uint32 contextIndex = 0, bufIndex = 0;
		vx_uint32 num_buffers;
		bool useNull = false;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsSetChromaKeyBuffer(ls[#],&buf[#]|NULL)";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		if (!_stricmp(s, ",null)")) {
			useNull = true;
		}
		else {
			SYNTAX_CHECK(ParseSkip(s, ",&"));
			SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
			if (*s == ','){
				SYNTAX_CHECK(ParseSkip(s, ","));
				SYNTAX_CHECK(ParseUInt(s, num_buffers));
			}
			SYNTAX_CHECK(ParseSkip(s, ")"));
		}
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		// process the command
		if (useNull) {
			vx_status status = lsSetChromaKeyBuffer(context_[contextIndex], nullptr);
			if (status) return Error("ERROR: lsSetChromaKeyBuffer(%s[%d],NULL) failed (%d)", name_ls, contextIndex, status);
			Message("..lsSetChromaKeyBuffer: set NULL for %s[%d]\n", name_ls, contextIndex);
		}
		else {
			vx_status status = lsSetChromaKeyBuffer(context_[contextIndex], &opencl_buf_mem_[bufIndex]);
			if (status) return Error("ERROR: lsSetChromaKeyBuffer(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_buf, bufIndex, status);
			Message("..lsSetChromaKeyBuffer: set OpenCL buffer %s[%d] for %s[%d]\n", name_buf, bufIndex, name_ls, contextIndex);
		}
	}
	else if (!_stricmp(command, "setGlobalAttribute")) {
		// parse the command
		vx_uint32 attr_offset = 0; float value = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: setGlobalAttribute(offset,value);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseUInt(s, attr_offset));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFloat(s, value));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (setGlobalAttribute(attr_offset, value) != VX_SUCCESS)
			return -1;
	}
	else if (!_stricmp(command, "setAttribute")) {
		// parse the command
		vx_uint32 contextIndex = 0, attr_offset = 0; float value = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: setAttribute(context,offset,value);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_offset));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFloat(s, value));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (setAttribute(context_[contextIndex], attr_offset, value) != VX_SUCCESS)
			return -1;
	}
	else if (!_stricmp(command, "showGlobalAttributes")) {
		// parse the command
		vx_uint32 attr_offset = 0, attr_count = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showGlobalAttributes(offset,count);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseUInt(s, attr_offset));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_count));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (showGlobalAttributes(attr_offset, attr_count) != VX_SUCCESS)
			return -1;
	}
	else if (!_stricmp(command, "showAttributes")) {
		// parse the command
		vx_uint32 contextIndex = 0, attr_offset = 0, attr_count = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showAttributes(context,offset,count);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_offset));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_count));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (showAttributes(context_[contextIndex], attr_offset, attr_count) != VX_SUCCESS)
			return -1;
	}
	else if (!_stricmp(command, "loadAttributes") || !_stricmp(command, "saveAttributes")) {
		// parse the command
		vx_uint32 contextIndex = 0, attr_offset = 0, attr_count = 0;
		char fileName[256] = { 0 };
		const char * invalidSyntax = !_stricmp(command, "loadAttributes") ?
			"ERROR: invalid syntax: expects: loadAttributes(context,offset,count,\"attr.txt\");" :
			"ERROR: invalid syntax: expects: saveAttributes(context,offset,count,\"attr.txt\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_offset));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_count));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (!_stricmp(command, "loadAttributes")) {
			if (loadAttributes(context_[contextIndex], attr_offset, attr_count, fileName) != VX_SUCCESS)
				return -1;
		}
		else {
			if (saveAttributes(context_[contextIndex], attr_offset, attr_count, fileName) != VX_SUCCESS)
				return -1;
		}
	}
	else if (!_stricmp(command, "loadGlobalAttributes") || !_stricmp(command, "saveGlobalAttributes")) {
		// parse the command
		vx_uint32 attr_offset = 0, attr_count = 0;
		char fileName[256] = { 0 };
		const char * invalidSyntax = !_stricmp(command, "loadGlobalAttributes") ?
			"ERROR: invalid syntax: expects: loadGlobalAttributes(offset,count,\"attr.txt\");" :
			"ERROR: invalid syntax: expects: saveGlobalAttributes(offset,count,\"attr.txt\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseUInt(s, attr_offset));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, attr_count));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (!_stricmp(command, "loadGlobalAttributes")) {
			if (loadGlobalAttributes(attr_offset, attr_count, fileName) != VX_SUCCESS)
				return -1;
		}
		else {
			if (saveGlobalAttributes(attr_offset, attr_count, fileName) != VX_SUCCESS)
				return -1;
		}
	}
	else if (!_stricmp(command, "lsGetOpenVXContext")) {
		// parse the command
		vx_uint32 contextIndex = 0, vxIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsGetOpenVXContext(ls[#],&vx[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ",&"));
		SYNTAX_CHECK(ParseIndex(s, name_vx, vxIndex, num_openvx_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (vxIndex >= num_openvx_context_) return Error("ERROR: OpenVX context out-of-range: expects: 0..%d", num_openvx_context_ - 1);
		if (openvx_context_allocated_[vxIndex]) return Error("ERROR: OpenVX context %s[%d] already created\n", name_vx, vxIndex);
		// process the command
		vx_status status = lsGetOpenVXContext(context_[contextIndex], &openvx_context_[vxIndex]);
		if (status) return Error("ERROR: lsGetOpenVXContext(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_vx, vxIndex, status);
		Message("..lsGetOpenVXContext: get OpenVX context %s[%d] from %s[%d]\n", name_vx, vxIndex, name_ls, contextIndex);
	}
	else if (!_stricmp(command, "lsGetOpenCLContext")) {
		// parse the command
		vx_uint32 contextIndex = 0, clIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsGetOpenCLContext(ls[#],&cl[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ",&"));
		SYNTAX_CHECK(ParseIndex(s, name_cl, clIndex, num_opencl_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (clIndex >= num_opencl_context_) return Error("ERROR: OpenCL context out-of-range: expects: 0..%d", num_opencl_context_ - 1);
		if (opencl_context_allocated_[clIndex]) return Error("ERROR: OpenCL context %s[%d] already created\n", name_cl, clIndex);
		// process the command
		vx_status status = lsGetOpenCLContext(context_[contextIndex], &opencl_context_[clIndex]);
		if (status) return Error("ERROR: lsGetOpenCLContext(%s[%d],%s[%d]) failed (%d)", name_ls, contextIndex, name_cl, clIndex, status);
		Message("..lsGetOpenCLContext: get OpenCL context %s[%d] from %s[%d]\n", name_cl, clIndex, name_ls, contextIndex);
	}
	else if (!_stricmp(command, "showRigParams")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showRigParams(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showRigParams(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showCameraConfig")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showCameraConfig(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showCameraConfig(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showOutputConfig")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showOutputConfig(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showOutputConfig(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showOverlayConfig")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showOverlayConfig(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showOverlayConfig(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showCameraBufferStride")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showCameraBufferStride(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showCameraBufferStride(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showOutputBufferStride")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showOutputBufferStride(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showOutputBufferStride(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showOverlayBufferStride")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showOverlayBufferStride(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showOverlayBufferStride(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showCameraModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showCameraModule(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showCameraModule(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showOutputModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showOutputModule(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showOutputModule(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showOverlayModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showOverlayModule(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showOverlayModule(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showViewingModule")) {
		// parse the command
		vx_uint32 contextIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showViewingModule(context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showViewingModule(context_[contextIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "showConfiguration")) {
		// parse the command
		vx_uint32 contextIndex = 0; char exportType[128], fileName[256] = { 0 };
		const char * invalidSyntax = "ERROR: invalid syntax: expects: showConfiguration(context,\"<exportType>\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, exportType, sizeof(exportType)));
		SYNTAX_CHECK(ParseSkip(s, ""));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = showConfiguration(context_[contextIndex], exportType);
		if (status) return status;
	}
	else if (!_stricmp(command, "lsExportConfiguration")) {
		// parse the command
		vx_uint32 contextIndex = 0; char exportType[128], fileName[256] = { 0 };
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsExportConfiguration(context,\"<exportType>\",\"<fileName>\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, exportType, sizeof(exportType)));
		SYNTAX_CHECK(ParseSkip(s, ""));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsExportConfiguration(context_[contextIndex], exportType, fileName);
		if (status) return status;
	}
	else if (!_stricmp(command, "lsImportConfiguration")) {
		// parse the command
		vx_uint32 contextIndex = 0; char importType[128], fileName[256];
		const char * invalidSyntax = "ERROR: invalid syntax: expects: lsImportConfiguration(context,\"<importType>\",\"<fileName>\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, importType, sizeof(importType)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = lsImportConfiguration(context_[contextIndex], importType, fileName);
		if (status) return status;
	}
	else if (!_stricmp(command, "createOpenVXContext")) {
		// parse the command
		vx_uint32 vxIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: createOpenVXContext(&openvx_context);";
		SYNTAX_CHECK(ParseSkip(s, "(&"));
		SYNTAX_CHECK(ParseIndex(s, name_vx, vxIndex, num_openvx_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (vxIndex >= num_openvx_context_) return Error("ERROR: OpenVX context out-of-range: expects: 0..%d", num_openvx_context_ - 1);
		if (openvx_context_[vxIndex] && openvx_context_allocated_[vxIndex]) return Error("ERROR: OpenVX context %s[%d] already created", name_vx, vxIndex);
		// process the command
		cl_int status = createOpenVXContext(&openvx_context_[vxIndex]);
		if (status) return status;
		openvx_context_allocated_[vxIndex] = true;
	}
	else if (!_stricmp(command, "releaseOpenVXContext")) {
		// parse the command
		vx_uint32 vxIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: releaseOpenVXContext(&openvx_context)";
		SYNTAX_CHECK(ParseSkip(s, "(&"));
		SYNTAX_CHECK(ParseIndex(s, name_vx, vxIndex, num_openvx_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (vxIndex >= num_openvx_context_) return Error("ERROR: OpenVX context out-of-range: expects: 0..%d", num_openvx_context_ - 1);
		if (!openvx_context_[vxIndex]) return Error("ERROR: OpenVX context %s[%d] doesn't exist", name_vx, vxIndex);
		if (!openvx_context_allocated_[vxIndex]) return Error("ERROR: attempted to release OpenVX context not created here", name_vx, vxIndex);
		// process the command
		cl_int status = releaseOpenVXContext(&openvx_context_[vxIndex]);
		if (status) return status;
		openvx_context_allocated_[vxIndex] = false;
	}
	else if (!_stricmp(command, "createOpenCLContext")) {
		// parse the command
		char platform[64], device[64];
		vx_uint32 clIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: createOpenCLContext(\"<platform>\",\"<device>\",&opencl_context);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseString(s, platform, sizeof(platform)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, device, sizeof(device)));
		SYNTAX_CHECK(ParseSkip(s, ",&"));
		SYNTAX_CHECK(ParseIndex(s, name_cl, clIndex, num_opencl_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (clIndex >= num_opencl_context_) return Error("ERROR: OpenCL context out-of-range: expects: 0..%d", num_opencl_context_ - 1);
		if (opencl_context_[clIndex] && opencl_context_allocated_[clIndex]) return Error("ERROR: OpenCL context %s[%d] already created", name_cl, clIndex);
		// process the command
		cl_int status = createOpenCLContext(platform, device, &opencl_context_[clIndex]);
		if (status) return status;
		opencl_context_allocated_[clIndex] = true;
	}
	else if (!_stricmp(command, "releaseOpenCLContext")) {
		// parse the command
		vx_uint32 clIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: releaseOpenCLContext(&opencl_context)";
		SYNTAX_CHECK(ParseSkip(s, "(&"));
		SYNTAX_CHECK(ParseIndex(s, name_cl, clIndex, num_opencl_context_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (clIndex >= num_opencl_context_) return Error("ERROR: OpenCL context out-of-range: expects: 0..%d", num_opencl_context_ - 1);
		if (!opencl_context_[clIndex]) return Error("ERROR: OpenCL context %s[%d] doesn't exist", name_cl, clIndex);
		if (!opencl_context_allocated_[clIndex]) return Error("ERROR: attempted to release OpenCL context not created here", name_cl, clIndex);
		// process the command
		cl_int status = releaseOpenCLContext(&opencl_context_[clIndex]);
		if (status) return status;
		opencl_context_allocated_[clIndex] = false;
	}
	else if (!_stricmp(command, "createBuffer")) {
		// parse the command
		vx_uint32 clIndex = 0, bufIndex = 0, bufSize = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: createBuffer(opencl_context,<size-in-bytes>,&buf[#])";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseIndex(s, name_cl, clIndex, num_opencl_context_));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, bufSize));
		SYNTAX_CHECK(ParseSkip(s, ",&"));
		SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (clIndex >= num_opencl_context_) return Error("ERROR: OpenCL context out-of-range: expects: 0..%d", num_opencl_context_ - 1);
		if (!opencl_context_[clIndex]) return Error("ERROR: OpenCL context %s[%d] doesn't exist", name_cl, clIndex);
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		if (opencl_buf_mem_[bufIndex]) return Error("ERROR: OpenCL buffer %s[%d] already exists", name_buf, bufIndex);
		// process the command
		cl_int status = createBuffer(opencl_context_[clIndex], bufSize, &opencl_buf_mem_[bufIndex]);
		if (status) return status;
		// initialize buffer
		char textBuffer[1024];
		cl_int LS_INITIALIZE_CLMEM = 0;
		if (GetEnvVariable("LS_INITIALIZE_CLMEM", textBuffer, sizeof(textBuffer))){ LS_INITIALIZE_CLMEM = (cl_int)atoi(textBuffer); }
		if (LS_INITIALIZE_CLMEM >= 0){
			status = initializeBuffer(opencl_buf_mem_[bufIndex], bufSize, LS_INITIALIZE_CLMEM);
			if (status) return status;
		}
	}
	else if (!_stricmp(command, "releaseBuffer")) {
		// parse the command
		vx_uint32 bufIndex = 0;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: releaseBuffer(&buf[#]);";
		SYNTAX_CHECK(ParseSkip(s, "(&"));
		SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		if (!opencl_buf_mem_[bufIndex]) return Error("ERROR: OpenCL buffer %s[%d] doesn't exist", name_buf, bufIndex);
		// process the command
		cl_int status = releaseBuffer(&opencl_buf_mem_[bufIndex]);
		if (status) return status;
	}
	else if (!_stricmp(command, "loadBufferFromImage") || !_stricmp(command, "saveBufferToImage")) {
		// parse the command
		vx_uint32 bufIndex = 0; char fileName[256];
		vx_df_image buffer_format = VX_DF_IMAGE_VIRT;
		vx_uint32 buffer_width = 0, buffer_height = 0, stride_in_bytes = 0;
		const char * invalidSyntax = !_stricmp(command, "loadBufferFromImage") ?
			"ERROR: invalid syntax: expects: loadBufferFromImage(buf,\"fileName.bin\",format,width,height,stride_in_bytes);" :
			"ERROR: invalid syntax: expects: saveBufferToImage(buf,\"fileName.bin\",format,width,height,stride_in_bytes);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFormat(s, buffer_format));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_width));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_height));
		if (*s == ',') {
			SYNTAX_CHECK(ParseSkip(s, ","));
			SYNTAX_CHECK(ParseUInt(s, stride_in_bytes));
		}
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		if (!opencl_buf_mem_[bufIndex]) return Error("ERROR: OpenCL buffer %s[%d] doesn't exist", name_buf, bufIndex);
		// process the command
		if (!_stricmp(command, "loadBufferFromImage")) {
			vx_status status = loadBufferFromImage(opencl_buf_mem_[bufIndex], fileName, buffer_format, buffer_width, buffer_height, stride_in_bytes);
			if (status) return status;
		}
		else {
			vx_status status = saveBufferToImage(opencl_buf_mem_[bufIndex], fileName, buffer_format, buffer_width, buffer_height, stride_in_bytes);
			if (status) return status;
		}
	}
	else if (!_stricmp(command, "loadBufferFromMultipleImages") || !_stricmp(command, "saveBufferToMultipleImages")) {
		// parse the command
		vx_uint32 bufIndex = 0; char fileName[256];
		vx_uint32 num_rows = 0, num_cols = 0;
		vx_df_image buffer_format = VX_DF_IMAGE_VIRT;
		vx_uint32 buffer_width = 0, buffer_height = 0, stride_in_bytes = 0;
		const char * invalidSyntax = !_stricmp(command, "loadBufferFromMultipleImages") ?
			"ERROR: invalid syntax: expects: loadBufferFromMultipleImages(buf,\"fileName.bin\",num_rows,num_cols,format,width,height,stride_in_bytes);" :
			"ERROR: invalid syntax: expects: saveBufferToMultipleImages(buf,\"fileName.bin\",num_rows,num_cols,format,width,height,stride_in_bytes);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, num_rows));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, num_cols));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseFormat(s, buffer_format));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_width));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, buffer_height));
		SYNTAX_CHECK(ParseSkip(s, ""));
		if (*s == ',') {
			SYNTAX_CHECK(ParseSkip(s, ","));
			SYNTAX_CHECK(ParseUInt(s, stride_in_bytes));
		}
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		if (!opencl_buf_mem_[bufIndex]) return Error("ERROR: OpenCL buffer %s[%d] doesn't exist", name_buf, bufIndex);
		// process the command
		if (!_stricmp(command, "loadBufferFromMultipleImages")) {
			vx_status status = loadBufferFromMultipleImages(opencl_buf_mem_[bufIndex], fileName, num_rows, num_cols, buffer_format, buffer_width, buffer_height, stride_in_bytes);
			if (status) return status;
		}
		else {
			vx_status status = saveBufferToMultipleImages(opencl_buf_mem_[bufIndex], fileName, num_rows, num_cols, buffer_format, buffer_width, buffer_height, stride_in_bytes);
			if (status) return status;
		}
	}
	else if (!_stricmp(command, "loadBuffer") || !_stricmp(command, "saveBuffer")) {
		// parse the command
		vx_uint32 bufIndex = 0; char fileName[256]; vx_uint32 offsetOrFlag = 0;
		const char * invalidSyntax = !_stricmp(command, "loadBuffer") ?
			"ERROR: invalid syntax: expects: loadBuffer(buf,\"fileName.bin\");" :
			"ERROR: invalid syntax: expects: saveBuffer(buf,\"fileName.bin\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseIndex(s, name_buf, bufIndex, num_opencl_buf_));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		if (*s == ',') {
			SYNTAX_CHECK(ParseSkip(s, ","));
			SYNTAX_CHECK(ParseUInt(s, offsetOrFlag));
		}
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (bufIndex >= num_opencl_buf_) return Error("ERROR: OpenCL buffer out-of-range: expects: 0..%d", num_opencl_buf_ - 1);
		if (!opencl_buf_mem_[bufIndex]) return Error("ERROR: OpenCL buffer %s[%d] doesn't exist", name_buf, bufIndex);
		// process the command
		if (!_stricmp(command, "loadBuffer")) {
			vx_status status = loadBuffer(opencl_buf_mem_[bufIndex], fileName, offsetOrFlag);
			if (status) return status;
		}
		else {
			vx_status status = saveBuffer(opencl_buf_mem_[bufIndex], fileName, offsetOrFlag);
			if (status) return status;
		}
	}
	else if (!_stricmp(command, "run")) {
		// parse the command
		vx_uint32 contextIndex = 0, frameCount = 1;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: run(context,frameCount);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, frameCount));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = run(context_[contextIndex], frameCount);
		if (status) return status;
	}
	else if (!_stricmp(command, "runParallel")) {
		// parse the command
		vx_uint32 contextCount = 0, frameCount = 1;
		const char * invalidSyntax = "ERROR: invalid syntax: expects: runParallel(contextArray,contextCount,frameCount);";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseSkipPattern(s, name_ls));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, contextCount));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, frameCount));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (num_context_ < 2 || contextCount < 2 || contextCount > num_context_) return Error("ERROR: can't run specified contexts in parallel");
		// process the command
		vx_status status = runParallel(context_, contextCount, frameCount);
		if (status) return status;
	}
	else if (!_stricmp(command, "ls_context")) {
		// sanity check
		if (decl_ls_disabled) return Error("ERROR: ls_context declaration should be specified earlier\n");
		// parse the command
		const char * invalidSyntax = "ERROR: invalid syntax: expects: ls_context ls|ls[count]";
		SYNTAX_CHECK(ParseWord(s, name_ls, sizeof(name_ls)));
		SYNTAX_CHECK(ParseSkip(s, ""));
		num_context_ = 1;
		if (*s == '[') {
			SYNTAX_CHECK(ParseIndex(s, "", num_context_, 2));
		}
		// process the command
		Message("..ls_context %s[%d] created\n", name_ls, num_context_);
	}
	else if (!_stricmp(command, "vx_context")) {
		// sanity check
		if (decl_vx_disabled) return Error("ERROR: vx_context declaration should be specified earlier\n");
		// parse the command
		const char * invalidSyntax = "ERROR: invalid syntax: expects: vx_context vx|vx[count]";
		SYNTAX_CHECK(ParseWord(s, name_vx, sizeof(name_vx)));
		SYNTAX_CHECK(ParseSkip(s, ""));
		num_openvx_context_ = 1;
		if (*s == '[') {
			SYNTAX_CHECK(ParseIndex(s, "", num_openvx_context_, 2));
		}
		// process the command
		Message("..vx_context %s[%d] created\n", name_vx, num_openvx_context_);
	}
	else if (!_stricmp(command, "cl_context")) {
		// sanity check
		if (decl_cl_disabled) return Error("ERROR: cl_context declaration should be specified earlier\n");
		// parse the command
		const char * invalidSyntax = "ERROR: invalid syntax: expects: cl_context cl|cl[count]";
		SYNTAX_CHECK(ParseWord(s, name_cl, sizeof(name_cl)));
		SYNTAX_CHECK(ParseSkip(s, ""));
		num_opencl_context_ = 1;
		if (*s == '[') {
			SYNTAX_CHECK(ParseIndex(s, "", num_opencl_context_, 2));
		}
		// process the command
		Message("..cl_context %s[%d] created\n", name_cl, num_opencl_context_);
	}
	else if (!_stricmp(command, "cl_mem")) {
		// sanity check
		if (decl_buf_disabled) return Error("ERROR: cl_mem declaration should be specified earlier\n");
		// parse the command
		const char * invalidSyntax = "ERROR: invalid syntax: expects: cl_mem buf|buf[count]";
		SYNTAX_CHECK(ParseWord(s, name_buf, sizeof(name_buf)));
		SYNTAX_CHECK(ParseSkip(s, ""));
		num_opencl_buf_ = 1;
		if (*s == '[') {
			SYNTAX_CHECK(ParseIndex(s, "", num_opencl_buf_, 2));
		}
		// process the command
		Message("..cl_mem %s[%d] created\n", name_buf, num_opencl_buf_);
	}
	else if (!_stricmp(command, "camera_params")) {
		// parse command
		const char * invalidSyntax = "ERROR: invalid syntax: expects: camera_params par = {{yaw,pitch,roll,tx,ty,tz},{hfov,haw,r_crop,du0,dv0,lens_type,k1,k2,k3}};";
		char parName[64], lensType[64]; camera_params camera_par = { 0 };
		SYNTAX_CHECK(ParseWord(s, parName, sizeof(parName)));
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "={{"), "ERROR: missing '={{'");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.yaw), "ERROR: invalid yaw value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing pitch value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.pitch), "ERROR: invalid pitch value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing roll value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.roll), "ERROR: invalid roll value");
		SYNTAX_CHECK(ParseSkip(s, ""));
		if (*s == ',') {
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing tx value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.tx), "ERROR: invalid tx value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing ty value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.ty), "ERROR: invalid ty value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing tz value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.focal.tz), "ERROR: invalid tz value");
		}
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "},{"), "ERROR: missing separator before lens_type");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.hfov), "ERROR: invalid hfov value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing haw value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.haw), "ERROR: invalid haw value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing r_crop value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.r_crop), "ERROR: invalid r_crop value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing du0 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.du0), "ERROR: invalid du0 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing dv0 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.dv0), "ERROR: invalid dv0 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing lens_type value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseWord(s, lensType, sizeof(lensType)), "ERROR: invalid lens_type value");
		if (!_stricmp(lensType, "ptgui_lens_rectilinear")) camera_par.lens.lens_type = ptgui_lens_rectilinear;
		else if (!_stricmp(lensType, "ptgui_lens_fisheye_ff")) camera_par.lens.lens_type = ptgui_lens_fisheye_ff;
		else if (!_stricmp(lensType, "ptgui_lens_fisheye_circ")) camera_par.lens.lens_type = ptgui_lens_fisheye_circ;
		else if (!_stricmp(lensType, "adobe_lens_rectilinear")) camera_par.lens.lens_type = adobe_lens_rectilinear;
		else if (!_stricmp(lensType, "adobe_lens_fisheye")) camera_par.lens.lens_type = adobe_lens_fisheye;
		else return Error("ERROR: invalid lens_type value: see help for valid values");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing k1 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.k1), "ERROR: invalid k1 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing k2 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.k2), "ERROR: invalid k2 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing k3 value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.k3), "ERROR: invalid k3 value");
		for (vx_uint32 i = 0; i < 7; i++) {
			s = ParseSkip(s, ""); if (*s != ',') break;
			s = ParseSkip(s, ",");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, camera_par.lens.reserved[i]), "ERROR: invalid reserved field value");
		}
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "}}"), "ERROR: missing '}}'");
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (camParList.find(parName) != camParList.end()) return Error("ERROR: camera_params %s already exists", parName);
		// process the command
		camParList[parName] = camera_par;
		Message("..camera_params %s declared\n", parName);
	}
	else if (!_stricmp(command, "rig_params")) {
		// parse command
		const char * invalidSyntax = "ERROR: invalid syntax: expects: rig_params par = {yaw,pitch,roll,d};";
		char parName[64]; rig_params rig_par = { 0 };
		SYNTAX_CHECK(ParseWord(s, parName, sizeof(parName)));
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "={"), "ERROR: missing '={'");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.yaw), "ERROR: invalid yaw value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing pitch value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.pitch), "ERROR: invalid pitch value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing roll value");
		SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.roll), "ERROR: invalid roll value");
		SYNTAX_CHECK(ParseSkip(s, ""));
		if (*s == ',') {
			SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, ","), "ERROR: missing d value");
			SYNTAX_CHECK_WITH_MESSAGE(ParseFloat(s, rig_par.d), "ERROR: invalid d value");
		}
		SYNTAX_CHECK_WITH_MESSAGE(ParseSkip(s, "}"), "ERROR: missing '}'");
		SYNTAX_CHECK(ParseEndOfLine(s));
		if (rigParList.find(parName) != rigParList.end()) return Error("ERROR: rig_params %s already exists", parName);
		// process the command
		rigParList[parName] = rig_par;
		Message("..rig_params %s declared\n", parName);
	}
	else if (!_stricmp(command, "loadExpCompGains") || !_stricmp(command, "saveExpCompGains") || !_stricmp(command, "showExpCompGains")) {
		// parse the command
		vx_uint32 contextIndex = 0; char fileName[256] = { 0 };
		const char * invalidSyntax = !_stricmp(command, "loadExpCompGains") ?
			"ERROR: invalid syntax: expects: loadExpCompGains(context,num_entries,\"gains.txt\");" :
			"ERROR: invalid syntax: expects: saveExpCompGains(context,num_entries,\"gains.txt\");";
		if (!_stricmp(command, "showExpCompGains"))
			invalidSyntax = "ERROR: invalid syntax: expects: saveExpCompGains(context,num_entries);";
		vx_uint32 num_entries;
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseUInt(s, num_entries));
		if (_stricmp(command, "showExpCompGains") != 0) {
			SYNTAX_CHECK(ParseSkip(s, ","));
			SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		}
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		if (!_stricmp(command, "loadExpCompGains")) {
			vx_status status = loadExpCompGains(context_[contextIndex], num_entries, fileName);
			if (status) return status;
		}
		else if (!_stricmp(command, "saveExpCompGains")) {
			vx_status status = saveExpCompGains(context_[contextIndex], num_entries, fileName);
			if (status) return status;
		}
		else {
			vx_status status = showExpCompGains(context_[contextIndex], num_entries);
			if (status) return status;
		}
	}
	else if (!_stricmp(command, "loadBlendWeights")) {
		// parse the command
		vx_uint32 contextIndex = 0; char fileName[256] = { 0 };
		const char * invalidSyntax = "ERROR: invalid syntax: expects: loadBlendWeights(context,\"blend-weights.raw\");";
		SYNTAX_CHECK(ParseSkip(s, "("));
		SYNTAX_CHECK(ParseContextWithErrorCheck(s, contextIndex, invalidSyntax));
		SYNTAX_CHECK(ParseSkip(s, ","));
		SYNTAX_CHECK(ParseString(s, fileName, sizeof(fileName)));
		SYNTAX_CHECK(ParseSkip(s, ")"));
		SYNTAX_CHECK(ParseEndOfLine(s));
		// process the command
		vx_status status = loadBlendWeights(context_[contextIndex], fileName);
		if (status) return status;
	}
	else if (!_stricmp(command, "help")) {
		Message("..help\n");
		help(false);
	}
	else return Error("ERROR: invalid command: '%s'", command);
	return 0;
#undef SYNTAX_CHECK
#undef SYNTAX_CHECK_WITH_MESSAGE
}

CCommandLineParser::CCommandLineParser()
{
	verbose_ = false;
	lineNum_ = 0;
	fileName_ = nullptr;
	includeLevels_ = 0;
}

void CCommandLineParser::SetVerbose(bool verbose)
{
	verbose_ = verbose;
}

bool CCommandLineParser::Verbose()
{
	return verbose_;
}

void CCommandLineParser::Message(const char * format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

int CCommandLineParser::Error(const char * format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	printf(" @%s#%d\n", fileName_ ? fileName_ : "console", lineNum_);
	fflush(stdout);
	return -1;
}

void CCommandLineParser::Terminate(int code, const char * format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	exit(code);
}

bool CCommandLineParser::GetCommand(FILE * fp)
{
	cmd_ = "";
	int c = EOF;
	enum { BEGIN, MIDDLE, SPACE, ESCAPE, COMMENT } state = BEGIN;
	if (fp == stdin) Message("> ");
	while ((c = getc(fp)) != EOF) {
		// increment lineNum
		if (c == '\n') lineNum_++;
		// process begin, space, escape, and comments
		if (state == BEGIN) {
			if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
				// skip white space
				continue;
			}
			state = MIDDLE;
		}
		else if (state == ESCAPE) {
			if (c == '\r') {
				// skip CR and anticipate LF
				state = ESCAPE;
				continue;
			}
			else if (c == '\n') {
				// skip LF
				state = (cmd_.length() > 0) ? MIDDLE : BEGIN;
				continue;
			}
			state = MIDDLE;
		}
		else if (state == COMMENT) {
			if (c == '\n')
				state = (cmd_.length() > 0) ? MIDDLE : BEGIN;
			continue;
		}
		else if (state == SPACE) {
			if (c == ' ' || c == '\t' || c == '\r')
				continue;
			else if (c == '\\') {
				state = ESCAPE;
				continue;
			}
			else if (c == '#') {
				state = COMMENT;
				continue;
			}
			else if (c == '\n') {
				break;
			}
			state = MIDDLE;
			cmd_ += " ";
		}
		// detect space, escape, comments, and end-of-line
		if (c == '\\') {
			state = ESCAPE;
			continue;
		}
		else if (c == '#') {
			state = COMMENT;
			continue;
		}
		else if (c == ' ' || c == '\t' || c == '\r') {
			state = SPACE;
			continue;
		}
		else if (c == '\n') {
			break;
		}
		// add character to cmd
		cmd_ += c;
	}
	return cmd_.length() > 0 ? true : false;
}

int CCommandLineParser::Run(const char * fileName)
{
	int status = 0;

	// open the input script file (or use stdin)
	FILE * fp = stdin;
	if (fileName) fp = fopen(fileName, "r");
	if (!fp) Terminate(1, "ERROR: unable to open: %s\n", fileName);
	Message("... processing commands from %s\n", fileName ? fileName : "console");

	// process one command at a time
	lineNum_ = 0;
	fileName_ = fileName;
	while (GetCommand(fp)) {
		const char * cmd = cmd_.c_str();
		// verbose
		if (verbose_) Message("> %s\n", cmd);
		// process built-in commands
		if (!_stricmp(cmd, "verbose on")) {
			Message("... verbose ON\n");
			SetVerbose(true);
		}
		else if (!_stricmp(cmd, "verbose off")) {
			Message("... verbose OFF\n");
			SetVerbose(false);
		}
		else if (!_stricmp(cmd, "quit")) {
			Message("... quit from %s\n", fileName ? fileName : "console");
			exit(0);
		}
		else if (!_stricmp(cmd, "exit")) {
			break;
		}
		else if (!_strnicmp(cmd, "chdir", 5) || !_strnicmp(cmd, "copy", 4)) {
			bool isCmdChdir = false, isCmdCopy = false, isCmdCopyIfMissing = false;
			const char * commandName = "", * usage = "";
			if (!_strnicmp(cmd, "chdir", 5)) {
				isCmdChdir = true;
				commandName = "chdir";
				usage = "chdir(\"directory\")";
			}
			else if (!_strnicmp(cmd, "copy", 4)) {
				isCmdCopy = true;
				commandName = "copy";
				usage = "copy(\"srcFileName\",\"dstFileName\")";
				if (!_strnicmp(cmd, "copyifmissing", 13)) {
					isCmdCopyIfMissing = true;
					commandName = "copyifmissing";
					usage = "copyifmissing(\"srcFileName\",\"dstFileName\")";
				}
			}
			char path[256] = { 0 }, file[256] = { 0 };
			bool syntaxError = false;
			const char * s = cmd + strlen(commandName);
			while (*s && *s == ' ') s++; if (*s == '(') s++; else syntaxError = true;
			while (*s && *s == ' ') s++; if (*s == '"') s++; else syntaxError = true;
			if (!syntaxError) {
				for (vx_uint32 i = 0; i < (sizeof(path)-1) && *s && *s != '"'; i++, s++)
					path[i] = *s;
				if (*s == '"') s++; else syntaxError = true;
				while (*s && *s == ' ') s++;
				if (isCmdCopy && !syntaxError && *s == ',') {
					s++;
					while (*s && *s == ' ') s++; if (*s == '"') s++; else syntaxError = true;
					for (vx_uint32 i = 0; i < (sizeof(file) - 1) && *s && *s != '"'; i++, s++)
						file[i] = *s;
					if (strlen(file) < 1) syntaxError = true;
					if (*s == '"') s++; else syntaxError = true;
				}
				while (*s && *s == ' ') s++; if (*s == ')') s++; else syntaxError = true;
				while (*s && *s == ' ') s++; if (*s == ';') s++; else syntaxError = true;
			}
			if (!syntaxError && strlen(path) > 0) {
				if (isCmdChdir) {
					if (_chdir(path)) {
						Error("ERROR: unable to change directory to %s\n", path);
						exit(1);
					}
					Message("... changed directory to %s\n", path);
				}
				else if (isCmdCopy) {
					bool copyDisabled = false;
					if (isCmdCopyIfMissing) {
						FILE * fp = fopen(file, "rb");
						if (fp) {
							fclose(fp);
							copyDisabled = true;
						}
					}
					if (!copyDisabled) {
						FILE * fp = fopen(path, "rb"); if (!fp) { Error("ERROR: %s: unable to open: %s\n", commandName, path); exit(1); }
						FILE * fo = fopen(file, "wb"); if (!fp) { Error("ERROR: %s: unable to create: %s\n", commandName, file); exit(1); }
						char buf[8192];
						for (size_t size; (size = fread(buf, 1, sizeof(buf), fp)) > 0;)
							fwrite(buf, 1, size, fo);
						fclose(fo);
						fclose(fp);
						Message("... copied %s to %s\n", path, file);
					}
				}
			}
			else {
				Error("ERROR: incorrect usage: expects: %s; - got '%s'\n", usage, cmd);
				exit(1);
			}
		}
		else if (!_strnicmp(cmd, "include ", 8)) {
			if (includeLevels_ > 5) {
				Error("ERROR: include depth reached stack limitation - try to avoid deep include from within scripts\n");
				exit(1);
			}
			// continue running script and make sure to save and restore fileName_ and lineNum_ for Error()
			int lineNum = lineNum_;
			char scriptFileName[256] = { 0 };
			if (cmd[8] == '"' && cmd[strlen(cmd)-1] == '"' && strlen(&cmd[8]) > 2) {
				strncpy(scriptFileName, &cmd[8+1], sizeof(scriptFileName) - 1);
				scriptFileName[strlen(scriptFileName) - 1] = '\0';
				includeLevels_++;
				status = Run(scriptFileName);
				includeLevels_--;
			}
			else if (!_stricmp(&cmd[8], "prompt")) {
				status = Run(nullptr);
			}
			else {
				Error("ERROR: incorrect usage: expects: include \"script.lss\"|prompt - got '%s'\n", cmd);
				exit(1);
			}
			fileName_ = fileName;
			lineNum_ = lineNum;
			// check for error
			if (status < 0)
				break;
		}
		else {
			// invoke command processor
			status = OnCommand();
			if (status < 0)
				break;
		}
	}

	// close input script file
	if (fp != stdin) fclose(fp);
	Message("... exit from %s\n", fileName ? fileName : "console");
	return status;
}

int CCommandLineParser::OnCommand()
{
	return 0;
}

int main(int argc, char *argv[])
{
	printf("%s %s [loomsl %s]\n", PROGRAM_NAME, VERSION, lsGetVersion());

	// process command-line options
	bool verbose = false;
	for (; argv[1] && argv[1][0] == '-'; ++argv) {
		if (!_stricmp(argv[1], "-v")) {
			verbose = true;
		}
		else if (!_stricmp(argv[1], "-help")) {
			CLoomShellParser parser;
			parser.help(true);
			return 0;
		}
		else {
			printf("ERROR: invalid command-line option: %s (use -help see all options)\n", argv[1]);
			return -1;
		}
	}

	// run loom-shell parser
	CLoomShellParser parser;
	parser.SetVerbose(verbose);
	return parser.Run(argv[1]);
}
