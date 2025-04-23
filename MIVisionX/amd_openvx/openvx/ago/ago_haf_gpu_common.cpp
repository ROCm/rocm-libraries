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


#include "ago_haf_gpu.h"

#if ENABLE_OPENCL

#define ENABLE_UINT4_FOR_LOCAL_MEMORY_LOADS      1  // 0:disable 1:enable uint4 for local memory loads
#define ENABLE_UINT8_FOR_LOCAL_MEMORY_LOADS      1  // 0:disable 1:enable uint8 for local memory loads

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code to load into local memory:
//   this code assumes following variables created by caller in "code"
//     gx   - global work item [0]
//     gy   - global work item [1]
//     gbuf - global buffer
//     lx   - local work item [0]
//     ly   - local work item [1]
//     lbuf - local buffer
//
int HafGpu_Load_Local(int WGWidth, int WGHeight, int LMWidth, int LMHeight, int gxoffset, int gyoffset, std::string& code)
{
	char item[1024];

	// configuration parameters
	int LMdivWGWidthShift = leftmostbit(LMWidth / WGWidth);
	int LMWidthRemain = LMWidth - (WGWidth << LMdivWGWidthShift);
	if (LMdivWGWidthShift < 2) {
		agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_Load_Local(%dx%d,%dx%d,(%d,%d)): doesn't support LMdivWGWidthShift=%d\n", WGWidth, WGHeight, LMWidth, LMHeight, gxoffset, gyoffset, LMdivWGWidthShift);
		return -1;
	}

	// identify load data type
	const char * dType = "uint";
	int dTypeShift = 2;
	if ((LMdivWGWidthShift > 2)) {
		dType = "uint2";
		dTypeShift = 3;
#if ENABLE_UINT4_FOR_LOCAL_MEMORY_LOADS
		if (LMdivWGWidthShift > 3) {
			dType = "uint4";
			dTypeShift = 4;
#if ENABLE_UINT8_FOR_LOCAL_MEMORY_LOADS
			if (LMdivWGWidthShift > 4) {
				dType = "uint8";
				dTypeShift = 5;
			}
#endif
		}
#endif
	}
	int dGroupsShift = LMdivWGWidthShift - dTypeShift;
	int dGroups = 1 << dGroupsShift;
	bool use_vload = ((dTypeShift > 2) && (gxoffset & ((1 << dTypeShift) - 1))) ? true : false;

	// generate code
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"  { // load %dx%d bytes into local memory using %dx%d workgroup\n" // LMWidth, LMHeight, WGWidth, WGHeight
		"    int loffset = ly * %d + (lx << %d);\n" // LMWidth, dTypeShift
		"    int goffset = (gy - %d) * gstride + (gx << %d) - %d;\n" // gyoffset, dTypeShift, gxoffset
		), LMWidth, LMHeight, WGWidth, WGHeight, LMWidth, dTypeShift, gyoffset, dTypeShift, gxoffset);
	code += item;
	int LMHeightRemain = LMHeight - WGHeight;
	int LMRemain = (LMWidthRemain * LMHeight + (LMWidth - LMWidthRemain) * LMHeightRemain) >> dTypeShift;
	if (dGroups == 1 && LMWidthRemain > 0 && LMHeightRemain > 0 && LMRemain < (WGWidth * WGHeight)) {
		// sanity check
		if (LMWidthRemain & ((1 << dTypeShift) - 1)) {
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_Load_Local(%dx%d,%dx%d,(%d,%d)): doesn't support LMWidthRemain=%d with %s\n", WGWidth, WGHeight, LMWidth, LMHeight, gxoffset, gyoffset, LMWidthRemain, dType);
			return -1;
		}
		if (use_vload) {
			snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = vload%c(0, (__global uint *)(gbuf + goffset));\n", dType, dType[4]);
		}
		else {
			snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = *(__global %s *)(gbuf + goffset);\n", dType, dType);
		}
		code += item;
		// get configuration for extra load
		int dWidth = LMWidthRemain >> dTypeShift;
		int dHeight = LMHeight;
		int dSize = dWidth * dHeight;
		int dWidthShift = leftmostbit(dWidth);
		if (dWidth != (1 << dWidthShift)) dWidthShift = -1;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"    bool doExtraLoad = false;\n"
			"    if (ly < %d) {\n" // LMHeight - WGHeight
			"      loffset += %d * %d;\n" // WGHeight, LMWidth
			"      goffset += %d * gstride;\n" // WGHeight
			"      doExtraLoad = true;\n"
			"    }\n"
			"    else {\n"
			"      int id = (ly - %d) * %d + lx;\n" // LMHeight - WGHeight, WGWidth
			"      int ry = id %s %d;\n" // (id / dWidth) or (id >> dWidthShift)
			"      int rx = id %s %d;\n" // (id - ry * dWidth) or (id & (dWidth-1))
			"      loffset = ry * %d + (rx << %d) + %d;\n" // LMWidth, dTypeShift
			"      goffset = (gy - ly + ry - %d) * gstride + ((gx - lx + rx) << %d) + %d;\n" // gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset
			"      doExtraLoad = (ry < %d) ? true : false;\n" // LMHeight
			"    }\n"
			"    if (doExtraLoad) {\n")
			, LMHeight - WGHeight, WGHeight, LMWidth, WGHeight
			, LMHeight - WGHeight, WGWidth, (dWidthShift < 0) ? "/" : ">>", (dWidthShift < 0) ? dWidth : dWidthShift, (dWidthShift < 0) ? "- ry *" : "&", (dWidthShift < 0) ? dWidth : dWidth - 1
			, LMWidth, dTypeShift, (WGWidth << LMdivWGWidthShift)
			, gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset, LMHeight);
		code += item;
		if (use_vload) {
			snprintf(item, sizeof(item), "      *(__local %s *)(lbuf + loffset) = vload%c(0, (__global uint *)(gbuf + goffset));\n", dType, dType[4]);
		}
		else {
			snprintf(item, sizeof(item), "      *(__local %s *)(lbuf + loffset) = *(__global %s *)(gbuf + goffset);\n", dType, dType);
		}
		code += item;
		code += "    }\n";
	}
	else {
		for (int y = 0; y < LMHeight; y += WGHeight) {
			if ((LMHeight - y) < WGHeight) {
				snprintf(item, sizeof(item), "   if (ly < %d) {\n", LMHeight - y);
				code += item;
			}
			if (y > 0) {
				snprintf(item, sizeof(item),
					"    loffset += %d * %d;\n" // WGHeight, LMWidth
					"    goffset += %d * gstride;\n" // WGHeight
					, WGHeight, LMWidth, WGHeight);
				code += item;
			}
			if (use_vload) {
				snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = vload%c(0, (__global uint *)(gbuf + goffset));\n", dType, dType[4]);
			}
			else {
				snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = *(__global %s *)(gbuf + goffset);\n", dType, dType);
			}
			code += item;
			if (dGroups > 1) {
				if (y > 0) {
					code +=
						"    loffset_t = loffset;\n"
						"    goffset_t = goffset;\n";
				}
				else {
					code +=
						"    int loffset_t = loffset;\n"
						"    int goffset_t = goffset;\n";
				}
				for (int ix = 1; ix < dGroups; ix++) {
					snprintf(item, sizeof(item),
						"    loffset_t += %d;\n" // WGWidth << dTypeShift
						"    goffset_t += %d;\n" // WGWidth << dTypeShift
						, WGWidth << dTypeShift, WGWidth << dTypeShift);
					code += item;
					if (use_vload) {
						snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset_t) = vload%c(0, (__global uint *)(gbuf + goffset_t));\n", dType, dType[4]);
					}
					else {
						snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset_t) = *(__global %s *)(gbuf + goffset_t);\n", dType, dType);
					}
					code += item;
				}
			}
			if ((LMHeight - y) < WGHeight) {
				code += "   }\n";
			}
		}
		if (LMWidthRemain > 0) {
			// sanity check
			if (LMWidthRemain & ((1 << dTypeShift) - 1)) {
				agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_Load_Local(%dx%d,%dx%d,(%d,%d)): doesn't support LMWidthRemain=%d with %s\n", WGWidth, WGHeight, LMWidth, LMHeight, gxoffset, gyoffset, LMWidthRemain, dType);
				return -1;
			}
			// get configuration
			int dWidth = LMWidthRemain >> dTypeShift;
			int dHeight = LMHeight;
			int dSize = dWidth * dHeight;
			int dWidthShift = leftmostbit(dWidth);
			if (dWidth != (1 << dWidthShift)) dWidthShift = -1;
			// compute start addresses
			snprintf(item, sizeof(item),
				"    __local uchar * lbufptr = lbuf + %d;\n" // (WGWidth << LMdivWGWidthShift)
				"    goffset = (gy - ly - %d) * gstride + ((gx - lx) << %d) + %d;\n" // gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset
				, (WGWidth << LMdivWGWidthShift), gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset);
			code += item;
			// load memory
			for (int dCount = 0; dCount < dSize; dCount += WGWidth * WGHeight) {
				// compute rx & ry of remaining region
				if (dCount == 0) snprintf(item, sizeof(item), "    int id = ly * %d + lx, rx, ry;\n", WGWidth);
				else             snprintf(item, sizeof(item), "    id += %d;\n", WGWidth * WGHeight);
				code += item;
				if (dWidthShift < 0) {
					snprintf(item, sizeof(item),
						"    ry = id / %d;\n" // dWidth
						"    rx = id - ry * %d;\n" // dWidth
						, dWidth, dWidth);
				}
				else {
					snprintf(item, sizeof(item),
						"    ry = id >> %d;\n" // dWidthShift
						"    rx = id & %d;\n" // dWidth-1
						, dWidthShift, dWidth - 1);
				}
				code += item;
				if ((dSize - dCount) < (WGWidth * WGHeight)) {
					snprintf(item, sizeof(item), "   if (ry < %d) {\n", dHeight);
					code += item;
				}
				if (use_vload) {
					snprintf(item, sizeof(item), "    *(__local %s *)(lbufptr + ry * %d + (rx << %d)) = vload%c(0, (__global uint *)(gbuf + goffset + ry * gstride + (rx << %d)));\n", dType, LMWidth, dTypeShift, dType[4], dTypeShift);
				}
				else {
					snprintf(item, sizeof(item), "    *(__local %s *)(lbufptr + ry * %d + (rx << %d)) = *(__global %s *)(gbuf + goffset + ry * gstride + (rx << %d));\n", dType, LMWidth, dTypeShift, dType, dTypeShift);
				}
				code += item;
				if ((dSize - dCount) < (WGWidth * WGHeight)) {
					code += "   }\n";
				}
			}
		}
	}

	code +=
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"  }\n";

	return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code to load into local memory based on image size:
//   this code assumes following variables created by caller in "code"
//     gx   - global work item [0]
//     gy   - global work item [1]
//     gbuf - global buffer
//     lx   - local work item [0]
//     ly   - local work item [1]
//     lbuf - local buffer
//
int HafGpu_Load_Local_ImageSize(int WGWidth, int WGHeight, int LMWidth, int LMHeight, int gxoffset, int gyoffset, std::string& code, int srcImageBufferSize)
{
	char item[1024];

	// configuration parameters
	int LMdivWGWidthShift = leftmostbit(LMWidth / WGWidth);
	int LMWidthRemain = LMWidth - (WGWidth << LMdivWGWidthShift);
	if (LMdivWGWidthShift < 2) {
		agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_Load_Local_ImageSize(%dx%d,%dx%d,(%d,%d)): doesn't support LMdivWGWidthShift=%d\n", WGWidth, WGHeight, LMWidth, LMHeight, gxoffset, gyoffset, LMdivWGWidthShift);
		return -1;
	}

	// identify load data type
	const char * dType = "uint";
	int dTypeShift = 2;
	if ((LMdivWGWidthShift > 2)) {
		dType = "uint2";
		dTypeShift = 3;
#if ENABLE_UINT4_FOR_LOCAL_MEMORY_LOADS
		if (LMdivWGWidthShift > 3) {
			dType = "uint4";
			dTypeShift = 4;
#if ENABLE_UINT8_FOR_LOCAL_MEMORY_LOADS
			if (LMdivWGWidthShift > 4) {
				dType = "uint8";
				dTypeShift = 5;
			}
#endif
		}
#endif
	}
	int dGroupsShift = LMdivWGWidthShift - dTypeShift;
	int dGroups = 1 << dGroupsShift;
	bool use_vload = ((dTypeShift > 2) && (gxoffset & ((1 << dTypeShift) - 1))) ? true : false;

	// generate code
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"  { // load %dx%d bytes into local memory using %dx%d workgroup\n" // LMWidth, LMHeight, WGWidth, WGHeight
		"    int loffset = ly * %d + (lx << %d);\n" // LMWidth, dTypeShift
		"    int goffset = (gy - %d) * gstride + (gx << %d) - %d;\n" // gyoffset, dTypeShift, gxoffset
		), LMWidth, LMHeight, WGWidth, WGHeight, LMWidth, dTypeShift, gyoffset, dTypeShift, gxoffset);
	code += item;
	int LMHeightRemain = LMHeight - WGHeight;
	int LMRemain = (LMWidthRemain * LMHeight + (LMWidth - LMWidthRemain) * LMHeightRemain) >> dTypeShift;
	if (dGroups == 1 && LMWidthRemain > 0 && LMHeightRemain > 0 && LMRemain < (WGWidth * WGHeight)) {
		// sanity check
		if (LMWidthRemain & ((1 << dTypeShift) - 1)) {
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_Load_Local_ImageSize(%dx%d,%dx%d,(%d,%d)): doesn't support LMWidthRemain=%d with %s\n", WGWidth, WGHeight, LMWidth, LMHeight, gxoffset, gyoffset, LMWidthRemain, dType);
			return -1;
		}
		if (use_vload) {
			snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = vload%c(0, (__global uint *)(gbuf + goffset));\n", dType, dType[4]);
		}
		else {
			snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = *(__global %s *)(gbuf + goffset);\n", dType, dType);
		}
		code += item;
		// get configuration for extra load
		int dWidth = LMWidthRemain >> dTypeShift;
		int dHeight = LMHeight;
		int dSize = dWidth * dHeight;
		int dWidthShift = leftmostbit(dWidth);
		if (dWidth != (1 << dWidthShift)) dWidthShift = -1;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"    bool doExtraLoad = false;\n"
			"    if (ly < %d) {\n" // LMHeight - WGHeight
			"      loffset += %d * %d;\n" // WGHeight, LMWidth
			"      goffset += %d * gstride;\n" // WGHeight
			"      doExtraLoad = true;\n"
			"    }\n"
			"    else {\n"
			"      int id = (ly - %d) * %d + lx;\n" // LMHeight - WGHeight, WGWidth
			"      int ry = id %s %d;\n" // (id / dWidth) or (id >> dWidthShift)
			"      int rx = id %s %d;\n" // (id - ry * dWidth) or (id & (dWidth-1))
			"      loffset = ry * %d + (rx << %d) + %d;\n" // LMWidth, dTypeShift
			"      goffset = (gy - ly + ry - %d) * gstride + ((gx - lx + rx) << %d) + %d;\n" // gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset
			"      doExtraLoad = (ry < %d) ? true : false;\n" // LMHeight
			"    }\n"
			"    if (doExtraLoad) {\n")
			, LMHeight - WGHeight, WGHeight, LMWidth, WGHeight
			, LMHeight - WGHeight, WGWidth, (dWidthShift < 0) ? "/" : ">>", (dWidthShift < 0) ? dWidth : dWidthShift, (dWidthShift < 0) ? "- ry *" : "&", (dWidthShift < 0) ? dWidth : dWidth - 1
			, LMWidth, dTypeShift, (WGWidth << LMdivWGWidthShift)
			, gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset, LMHeight);
		code += item;
		if (use_vload) {
			snprintf(item, sizeof(item), "      *(__local %s *)(lbuf + loffset) = vload%c(0, (__global uint *)(gbuf + goffset));\n", dType, dType[4]);
		}
		else {
			snprintf(item, sizeof(item), "      *(__local %s *)(lbuf + loffset) = *(__global %s *)(gbuf + goffset);\n", dType, dType);
		}
		code += item;
		code += "    }\n";
	}
	else {
		for (int y = 0; y < LMHeight; y += WGHeight) {
			if ((LMHeight - y) < WGHeight) {
				snprintf(item, sizeof(item), "    if (ly < %d) {\n", LMHeight - y);
				code += item;
			}
			if (y > 0) {
				snprintf(item, sizeof(item),
					"    loffset += %d * %d;\n" // WGHeight, LMWidth
					"    goffset += %d * gstride;\n" // WGHeight
					"    int sidx = gbuf + goffset - p1_buf - p1_offset;\n"  // gbuf = p1_buf + p1_offset + (((gy - ly) << 1) + 1) * gstride + ((gx - lx) << 3). Subtract p1_buf and p1_offset to get the actual offset
					"    if (sidx < %d)\n" // srcImageBufferSize
					, WGHeight, LMWidth, WGHeight, srcImageBufferSize);
				code += item;
			}
			if (use_vload) {
				snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = vload%c(0, (__global uint *)(gbuf + goffset));\n", dType, dType[4]);
			}
			else {
				snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset) = *(__global %s *)(gbuf + goffset);\n", dType, dType);
			}
			code += item;
			if (dGroups > 1) {
				if (y > 0) {
					code +=
						"    loffset_t = loffset;\n"
						"    goffset_t = goffset;\n";
				}
				else {
					code +=
						"    int loffset_t = loffset;\n"
						"    int goffset_t = goffset;\n";
				}
				for (int ix = 1; ix < dGroups; ix++) {
					snprintf(item, sizeof(item),
						"    loffset_t += %d;\n" // WGWidth << dTypeShift
						"    goffset_t += %d;\n" // WGWidth << dTypeShift
						, WGWidth << dTypeShift, WGWidth << dTypeShift);
					code += item;
					if (use_vload) {
						snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset_t) = vload%c(0, (__global uint *)(gbuf + goffset_t));\n", dType, dType[4]);
					}
					else {
						snprintf(item, sizeof(item), "    *(__local %s *)(lbuf + loffset_t) = *(__global %s *)(gbuf + goffset_t);\n", dType, dType);
					}
					code += item;
				}
			}
			if ((LMHeight - y) < WGHeight) {
				code += "    }\n";
			}
		}
		if (LMWidthRemain > 0) {
			// sanity check
			if (LMWidthRemain & ((1 << dTypeShift) - 1)) {
				agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_Load_Local_ImageSize(%dx%d,%dx%d,(%d,%d)): doesn't support LMWidthRemain=%d with %s\n", WGWidth, WGHeight, LMWidth, LMHeight, gxoffset, gyoffset, LMWidthRemain, dType);
				return -1;
			}
			// get configuration
			int dWidth = LMWidthRemain >> dTypeShift;
			int dHeight = LMHeight;
			int dSize = dWidth * dHeight;
			int dWidthShift = leftmostbit(dWidth);
			if (dWidth != (1 << dWidthShift)) dWidthShift = -1;
			// compute start addresses
			snprintf(item, sizeof(item),
				"    __local uchar * lbufptr = lbuf + %d;\n" // (WGWidth << LMdivWGWidthShift)
				"    goffset = (gy - ly - %d) * gstride + ((gx - lx) << %d) + %d;\n" // gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset
				, (WGWidth << LMdivWGWidthShift), gyoffset, dTypeShift, (WGWidth << LMdivWGWidthShift) - gxoffset);
			code += item;
			// load memory
			for (int dCount = 0; dCount < dSize; dCount += WGWidth * WGHeight) {
				// compute rx & ry of remaining region
				if (dCount == 0) snprintf(item, sizeof(item), "    int id = ly * %d + lx, rx, ry;\n", WGWidth);
				else             snprintf(item, sizeof(item), "    id += %d;\n", WGWidth * WGHeight);
				code += item;
				if (dWidthShift < 0) {
					snprintf(item, sizeof(item),
						"    ry = id / %d;\n" // dWidth
						"    rx = id - ry * %d;\n" // dWidth
						, dWidth, dWidth);
				}
				else {
					snprintf(item, sizeof(item),
						"    ry = id >> %d;\n" // dWidthShift
						"    rx = id & %d;\n" // dWidth-1
						, dWidthShift, dWidth - 1);
				}
				code += item;
				snprintf(item, sizeof(item), 
					"    int sidx2 = gbuf + goffset - p1_buf - p1_offset + ry * gstride + (rx << %d);\n" // gbuf = p1_buf + p1_offset + (((gy - ly) << 1) + 1) * gstride + ((gx - lx) << 3). Subtract p1_buf and p1_offset to get the actual offset
					, dTypeShift);
				code += item;
				if ((dSize - dCount) < (WGWidth * WGHeight)) {
					snprintf(item, sizeof(item), "    if (ry < %d && (sidx2 < %d)) {\n", dHeight, srcImageBufferSize);
					code += item;
				}
				if (use_vload) {
					snprintf(item, sizeof(item), "    *(__local %s *)(lbufptr + ry * %d + (rx << %d)) = vload%c(0, (__global uint *)(gbuf + goffset + ry * gstride + (rx << %d)));\n", dType, LMWidth, dTypeShift, dType[4], dTypeShift);
				}
				else {
					snprintf(item, sizeof(item), "    *(__local %s *)(lbufptr + ry * %d + (rx << %d)) = *(__global %s *)(gbuf + goffset + ry * gstride + (rx << %d));\n", dType, LMWidth, dTypeShift, dType, dTypeShift);
				}
				code += item;
				if ((dSize - dCount) < (WGWidth * WGHeight)) {
					code += "    }\n";
				}
			}
		}
	}

	code +=
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"  }\n";

	return VX_SUCCESS;
}

#endif