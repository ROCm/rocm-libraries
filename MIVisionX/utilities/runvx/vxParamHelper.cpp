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

int g_numCvUse = 0;

// input track bars connected to scalar objects
#define MAX_INPUT_TRACK_BARS  12
static int g_trackBarActive = 0;
static vx_reference g_trackBarObj[MAX_INPUT_TRACK_BARS] = { 0 };
static float g_trackBarValueMin[MAX_INPUT_TRACK_BARS], g_trackBarValueMax[MAX_INPUT_TRACK_BARS], g_trackBarValueInc[MAX_INPUT_TRACK_BARS];
static float g_trackBarValueR[MAX_INPUT_TRACK_BARS] = { 0.0f }, g_trackBarAngle[MAX_INPUT_TRACK_BARS][3] = { 0.0f, 0.0f, 0.0f };

/////////////////////////////////
// input track bars connected to scalar objects
int GuiTrackBarInitializeScalar(vx_reference obj, int id, float valueMin, float valueMax, float valueInc)
{
	if (id < 0 || id >= MAX_INPUT_TRACK_BARS)
		return -1;
	if (!obj) {
		g_trackBarObj[id] = nullptr;
		return 0;
	}
	if (g_trackBarObj[id] && g_trackBarObj[id] != obj)
		return -1;
	g_trackBarObj[id] = obj;
	g_trackBarValueMin[id] = valueMin;
	g_trackBarValueMax[id] = valueMax;
	g_trackBarValueInc[id] = valueInc;
	return 0;
}
int GuiTrackBarInitializeMatrix(vx_reference obj, int id, float valueR, float valueInc)
{
	if (id < 0 || id >= MAX_INPUT_TRACK_BARS)
		return -1;
	if (!obj) {
		g_trackBarObj[id] = nullptr;
		return 0;
	}
	if (g_trackBarObj[id] && g_trackBarObj[id] != obj)
		return -1;
	g_trackBarObj[id] = obj;
	g_trackBarValueR[id] = valueR;
	g_trackBarValueInc[id] = valueInc;
	return 0;
}
int GuiTrackBarShutdown(vx_reference obj)
{
	for (int id = 0; id < MAX_INPUT_TRACK_BARS; id++) {
		if (g_trackBarObj[id] == obj) {
			g_trackBarObj[id] = nullptr;
			return 0;
		}
	}
	return -1;
}
int GuiTrackBarProcessKey(int key)
{
	int keyInc = '+', keyDec = '-';
	int id = g_trackBarActive;
	if (key >= 0x00700000 && key <= 0x007b0000)
	{ // use F1..F12 to select UIs
		id = (key >> 16) & 15;
		if (id >= 0 && id < MAX_INPUT_TRACK_BARS) {
			g_trackBarActive = id;
		}
		return 0;
	}
	if (g_trackBarObj[id]) {
		vx_enum obj_type = VX_ERROR_INVALID_TYPE;
		vxQueryReference(g_trackBarObj[id], VX_REFERENCE_TYPE, &obj_type, sizeof(obj_type));
		if (obj_type == VX_TYPE_SCALAR) {
			if (key == 0x00250000) id = 0, key = '-'; // left arrow: hardcoded to id#0 (F1) dec
			else if (key == 0x00260000) id = 1, key = '+'; // up arrow: hardcoded to id#1 (F2) inc
			else if (key == 0x00270000) id = 0, key = '+'; // right arrow: hardcoded to id#0 (F1) inc
			else if (key == 0x00280000) id = 1, key = '-'; // down arrow: hardcoded to id#1 (F2) dec
			else if (key == '_') key = '-'; // easy keys to avoid Shift dependency
			else if (key == '=') key = '+'; // easy keys to avoid Shift dependency
			if (key == keyInc || key == keyDec) {
				vx_enum format = VX_TYPE_FLOAT32;
				vxQueryScalar((vx_scalar)g_trackBarObj[id], VX_SCALAR_ATTRIBUTE_TYPE, &format, sizeof(format));
				float value = g_trackBarValueMin[id];
				if (format == VX_TYPE_FLOAT32) { vxReadScalarValue((vx_scalar)g_trackBarObj[id], &value); }
				else if (format == VX_TYPE_INT32) { vx_int32 v;  vxReadScalarValue((vx_scalar)g_trackBarObj[id], &v); value = (vx_float32)v; }
				else if (format == VX_TYPE_UINT32) { vx_uint32 v;  vxReadScalarValue((vx_scalar)g_trackBarObj[id], &v); value = (vx_float32)v; }
				float value_earlier = value;
				if (key == keyInc) value += g_trackBarValueInc[id];
				else if (key == keyDec) value -= g_trackBarValueInc[id];
				if (value < g_trackBarValueMin[id]) value = g_trackBarValueMin[id];
				else if (value > g_trackBarValueMax[id]) value = g_trackBarValueMax[id];
				if (format == VX_TYPE_FLOAT32) { vxWriteScalarValue((vx_scalar)g_trackBarObj[id], &value); }
				else if (format == VX_TYPE_INT32) { vx_int32 v;  vxWriteScalarValue((vx_scalar)g_trackBarObj[id], &v); value = (vx_float32)v; }
				else if (format == VX_TYPE_UINT32) { vx_uint32 v;  vxWriteScalarValue((vx_scalar)g_trackBarObj[id], &v); value = (vx_float32)v; }
				if (value != value_earlier) printf("OK: Scalar:UI,F%-2d => %g\n", id + 1, value);
			}
		}
		else if (obj_type == VX_TYPE_MATRIX) {
			if (key == 0x00250000) g_trackBarAngle[id][0] -= g_trackBarValueInc[id]; // left arrow: H(yaw) dec
			else if (key == 0x00270000) g_trackBarAngle[id][0] += g_trackBarValueInc[id]; // right arrow: H(yaw) inc
			else if (key == 0x00280000) g_trackBarAngle[id][1] -= g_trackBarValueInc[id]; // down arrow: P(pitch) dec
			else if (key == 0x00260000) g_trackBarAngle[id][1] += g_trackBarValueInc[id]; // up arrow: P(pitch) inc
			else if (key == '-' || key == '_') g_trackBarAngle[id][2] -= g_trackBarValueInc[id]; // B(round) dec
			else if (key == '+' || key == '=') g_trackBarAngle[id][2] += g_trackBarValueInc[id]; // B(round) inc
			// convert angles to matrix
			float H = g_trackBarAngle[id][0];
			float P = g_trackBarAngle[id][1];
			float B = g_trackBarAngle[id][2];
			printf("OK: Matrix:UI,F%-2d => H:%g P:%g B:%g\n", id + 1, H, P, B);
			H *= (float)M_PI / 180.0f;
			P *= (float)M_PI / 180.0f;
			B *= (float)M_PI / 180.0f;
			vx_float32 mat[3][3] = { { 0.0f } };
			vxReadMatrix((vx_matrix)g_trackBarObj[id], &mat);
			// create perspective transform using H/P/B
			// TBD
			mat[0][0] = cosf(H);
			mat[0][1] = sinf(H);
			mat[0][2] = 0.0f;
			mat[1][0] = -sinf(H);
			mat[1][1] = cosf(H);
			mat[1][2] = 0.0f;
			mat[2][0] = 0.0f;
			mat[2][1] = 0.0f;
			mat[2][2] = 1.0f;
			vxWriteMatrix((vx_matrix)g_trackBarObj[id], &mat);
		}
	}
	return 0;
}

/////////////////////////////////
// global OpenCV image count and specified read inputs count
int ProcessCvWindowKeyRefresh(int waitKeyDelayInMilliSeconds)
{
#if ENABLE_OPENCV
	if (g_numCvUse > 0) {
		// process keyboard
		int key = waitKey(waitKeyDelayInMilliSeconds);
		if (key == 'q' || key == 27 || key == ' ') {
			if (key == ' ' && waitKeyDelayInMilliSeconds != 0) {
				printf("Paused: Press spacebar to continue...\n"); fflush(stdout);
				while ((key = waitKey(0)) != ' ' && key != 'q' && key != 27)
					;
			}
			if (key == 'q' || key == 27) {
				return 1;
			}
		}
		else if (key >= 0) {
			GuiTrackBarProcessKey(key);
		}
	}
#endif
	return 0;
}
