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

#ifndef __EXP_COMP_H__
#define __EXP_COMP_H__

#include "kernels.h"

#define MAX_NUM_IMAGES_IN_STITCHED_OUTPUT	16
#define USE_LUMA_VALUES_FOR_GAIN			1

typedef struct _block_gain_info
{
	vx_uint32   b_dstX      : 16;
	vx_uint32   b_dstY      : 16;
	vx_uint16   Count[MAX_NUM_IMAGES_IN_STITCHED_OUTPUT][MAX_NUM_IMAGES_IN_STITCHED_OUTPUT];
	vx_uint8    Sum[MAX_NUM_IMAGES_IN_STITCHED_OUTPUT][MAX_NUM_IMAGES_IN_STITCHED_OUTPUT];
}block_gain_info;

class CExpCompensator
{
public:
	CExpCompensator(int rows = 0, int columns = 0);
	virtual ~CExpCompensator();
	virtual vx_status Process();
	virtual vx_status ProcessBlockGains(vx_array ArrBlkGains);
	virtual vx_status Initialize(vx_node node, vx_float32 alpha, vx_float32 beta, vx_array valid_roi, vx_image input, vx_image output, vx_array blockgain_arr = nullptr, vx_int32 channel=-1);
	virtual vx_status DeInitialize();
	virtual vx_status SolveForGains(vx_float32 alpha, vx_float32 beta, vx_uint32 *IMat, vx_uint32 *NMat, vx_uint32 num_images, vx_array pGains, vx_uint32 rows, vx_uint32 cols);
	vx_uint32 *m_pIMat, *m_pNMat;

protected:
	vx_uint32	m_numImages;
	vx_node		m_node;
	vx_uint32	m_width, m_height, m_stride,m_stride_x;
	vx_uint32   m_blockgainsStride;
	vx_float32	m_alpha, m_beta;
	vx_int32   m_channel;
	vx_image	m_InputImage, m_OutputImage;
	vx_array	m_valid_roi;
	vx_int32	m_bUseRGBgains;
	vx_rectangle_t m_pRoi_rect[MAX_NUM_IMAGES_IN_STITCHED_OUTPUT][MAX_NUM_IMAGES_IN_STITCHED_OUTPUT];	// assuming 
	block_gain_info *m_pblockgainInfo;
	vx_uint32 **m_NMat;
	vx_float32  **m_IMat, **m_IMatG, **m_IMatB;
	vx_float64 **m_AMat;
	vx_float32 *m_Gains, *m_GainsG, *m_GainsB;
	vx_rectangle_t mValidRect[MAX_NUM_IMAGES_IN_STITCHED_OUTPUT];
	vx_float32 *m_block_gain_buf;       // for block based exposure control


// functions
	virtual vx_status CompensateGains();
	virtual vx_status CompensateGainsRGB(vx_int32 ref_img);
	virtual vx_status CompensateBlockGains();
	virtual vx_status ApplyGains(void *in_base_addr);
	virtual vx_status ApplyBlockGains(void *in_base_addr);

private:
	void solve_gauss(vx_float64 **A, vx_float32* g, int num);
	vx_status applygains_thread_func(vx_int32 img_num, char *in_base_addr);
	vx_status applyblockgains_thread_func(vx_int32 img_num, char *in_base_addr);
};

vx_status Compute_StitchExpCompCalcEntry(vx_rectangle_t *pValid_roi, vx_array ExpCompOut, int numCameras);
vx_status Compute_StitchExpCompCalcValidEntry(vx_rectangle_t *pValid_roi, vx_array pExpCompOut, int numCameras, int Dst_height);
vx_status Compute_StitchBlendCalcValidEntry(vx_rectangle_t *pValid_roi, vx_array blendOffs, int numCameras);

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status exposure_compensation_publish(vx_context context);

#endif // __EXP_COMP_H__
