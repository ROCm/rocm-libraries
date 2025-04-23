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


#include "ago_kernel_api.h"

// for argConfig[]
#define AIN                                    ( AGO_KERNEL_ARG_INPUT_FLAG )
#define AINOUT                                 ( AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG )
#define AOUT                                   ( AGO_KERNEL_ARG_OUTPUT_FLAG )
#define AOPTIN                                 ( AGO_KERNEL_ARG_OPTIONAL_FLAG | AGO_KERNEL_ARG_INPUT_FLAG )
#define AOPTOUT                                ( AGO_KERNEL_ARG_OPTIONAL_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG )
#define AIN_AOUT                               { AIN, AOUT }
#define AINx2_AOUT                             { AIN, AIN, AOUT }
#define AINx3_AOUT                             { AIN, AIN, AIN, AOUT }
#define AINx4_AOUT                             { AIN, AIN, AIN, AIN, AOUT }
#define AINx5_AOUT                             { AIN, AIN, AIN, AIN, AIN, AOUT }
#define AINx2_AOPTINx2_AOUT                    { AIN, AIN, AOPTIN, AOPTIN, AOUT }
#define AIN_AOPTOUTx2                          { AIN, AOPTOUT, AOPTOUT }
#define AIN_AOUT_AIN                           { AIN, AOUT, AIN }
#define AIN_AOUTx2                             { AIN, AOUT, AOUT }
#define AIN_AINOUT                             { AIN, AINOUT }
#define AINx2_AINOUT                           { AIN, AIN, AINOUT }
#define AIN_AOUTx2_AOPTOUTx4                   { AIN, AOUT, AOUT, AOPTOUT, AOPTOUT, AOPTOUT, AOPTOUT }
#define AIN_AOUT_AINx2                         { AIN, AOUT, AIN, AIN }
#define AINx3_AOUT_AOPTOUT                     { AIN, AIN, AIN, AOUT, AOPTOUT }
#define AINx6_AOUT_AOPTOUT                     { AIN, AIN, AIN, AIN, AIN, AIN, AOUT, AOPTOUT }
#define AINx4_AOUT_AINx5                       { AIN, AIN, AIN, AIN, AOUT, AIN, AIN, AIN, AIN, AIN }
#define AOUT_AIN                               { AOUT, AIN }
#define AOUT_AIN_AOPTIN                        { AOUT, AIN, AOPTIN }
#define AOUT_AINx2                             { AOUT, AIN, AIN }
#define AOUT_AINx2_AOPTIN                      { AOUT, AIN, AIN, AOPTIN }
#define AOUT_AINx3                             { AOUT, AIN, AIN, AIN }
#define AOUT_AINx4                             { AOUT, AIN, AIN, AIN, AIN }
#define AOUT_AINx8                             { AOUT, AIN, AIN, AIN, AIN, AIN, AIN, AIN, AIN }
#define AOUT_AINx9                             { AOUT, AIN, AIN, AIN, AIN, AIN, AIN, AIN, AIN, AIN }
#define AOUTx2_AIN                             { AOUT, AOUT, AIN }
#define AOUTx2_AINx2                           { AOUT, AOUT, AIN, AIN }
#define AOUTx2_AINx2_AOPTIN                    { AOUT, AOUT, AIN, AIN, AOPTIN }
#define AOUTx2_AINx3                           { AOUT, AOUT, AIN, AIN, AIN }
#define AOUTx3_AIN                             { AOUT, AOUT, AOUT, AIN }
#define AOUTx3_AINx2                           { AOUT, AOUT, AOUT, AIN, AIN }
#define AOUTx4_AIN                             { AOUT, AOUT, AOUT, AOUT, AIN }
#define AOUTx4_AINx2                           { AOUT, AOUT, AOUT, AOUT, AIN, AIN }
#define AOUTx3_AINx3                           { AOUT, AOUT, AOUT, AIN, AIN, AIN }
#define AOUTx2_AIN_AOPTINx7                    { AOUT, AOUT, AIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN }
#define AOUTx3_AIN_AOPTINx6                    { AOUT, AOUT, AOUT, AIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN }
#define AOUTx2_AINx2_AOPTINx6                  { AOUT, AOUT, AIN, AIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN }
#define AOUT_AIN_AOPTINx8                      { AOUT, AIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN }
#define AINOUT_AIN                             { AINOUT, AIN }
#define AINOUT_AINx2                           { AINOUT, AIN, AIN }
#define AOUTx3_AINx2_AOPTINx5                  { AOUT, AOUT, AOUT, AIN, AIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN }
#define AOUTx4_AINx2_AOPTINx4                  { AOUT, AOUT, AOUT, AOUT, AIN, AIN, AOPTIN, AOPTIN, AOPTIN, AOPTIN }
#define AOUTx5_AINx2_AOPTINx2                  { AOUT, AOUT, AOUT, AOUT, AOUT, AIN, AIN, AOPTIN, AOPTIN }
#define AOUTx6_AINx2_AOPTINx2                  { AOUT, AOUT, AOUT, AOUT, AOUT, AOUT, AIN, AIN, AOPTIN, AOPTIN }
#define AOUT_AOPTOUT_AINx2                     { AOUT, AOPTOUT, AIN, AIN }
#define AOUT_AOPTOUT_AINx4                     { AOUT, AOPTOUT, AIN, AIN, AIN, AIN }
#define AOUT_AOPTOUTx2_AINx2                   { AOUT, AOPTOUT, AOPTOUT, AIN, AIN }
#define AOUTx2_AOPTOUTx2_AINx2                 { AOUT, AOUT, AOPTOUT, AOPTOUT, AIN, AIN }

// for argType[]
#define ATYPE_I                                { VX_TYPE_IMAGE }
#define ATYPE_II                               { VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_III                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_IIII                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_IIIII                            { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_IIIIII                           { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_IIS                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR }
#define ATYPE_ISI                              { VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_IIL                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_LUT }
#define ATYPE_ILI                              { VX_TYPE_IMAGE, VX_TYPE_LUT, VX_TYPE_IMAGE }
#define ATYPE_ID                               { VX_TYPE_IMAGE, VX_TYPE_DISTRIBUTION }
#define ATYPE_ISS                              { VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_IIT                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_THRESHOLD }
#define ATYPE_IITS                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_THRESHOLD, VX_TYPE_SCALAR }
#define ATYPE_ITI                              { VX_TYPE_IMAGE, VX_TYPE_THRESHOLD, VX_TYPE_IMAGE }
#define ATYPE_IIC                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_CONVOLUTION }
#define ATYPE_IIICC                            { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_CONVOLUTION, VX_TYPE_CONVOLUTION }
#define ATYPE_ICI                              { VX_TYPE_IMAGE, VX_TYPE_CONVOLUTION, VX_TYPE_IMAGE }
#define ATYPE_IP                               { VX_TYPE_IMAGE, VX_TYPE_PYRAMID }
#define ATYPE_IPI                              { VX_TYPE_IMAGE, VX_TYPE_PYRAMID, VX_TYPE_IMAGE }
#define ATYPE_PII                              { VX_TYPE_PYRAMID, VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_IIP                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_PYRAMID }
#define ATYPE_ISSAASS                          { VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_IISSSI                           { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_IISI                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_ISII                             { VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_IMAGE, VX_TYPE_IMAGE }
#define ATYPE_IISS                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_IISSS                            { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_ITSSI                            { VX_TYPE_IMAGE, VX_TYPE_THRESHOLD, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_IMSI                             { VX_TYPE_IMAGE, VX_TYPE_MATRIX, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_ISSSSSAS                         { VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_ARRAY, VX_TYPE_SCALAR }
#define ATYPE_ISSAS                            { VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_ARRAY, VX_TYPE_SCALAR }
#define ATYPE_PPAAASSSSS                       { VX_TYPE_PYRAMID, VX_TYPE_PYRAMID, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_IRSI                             { VX_TYPE_IMAGE, VX_TYPE_REMAP, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_IIIS                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR }
#define ATYPE_SISI                             { VX_TYPE_SCALAR, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_IMAGE }
#define ATYPE_SIMI                             { VX_TYPE_SCALAR, VX_TYPE_IMAGE, VX_TYPE_MATRIX, VX_TYPE_IMAGE }
#define ATYPE_AI                               { VX_TYPE_ARRAY, VX_TYPE_IMAGE }
#define ATYPE_AIS                              { VX_TYPE_ARRAY, VX_TYPE_IMAGE, VX_TYPE_SCALAR }
#define ATYPE_ASIS                             { VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_IMAGE, VX_TYPE_SCALAR }
#define ATYPE_ASASSS                           { VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_IcIT                             { VX_TYPE_IMAGE, AGO_TYPE_CANNY_STACK, VX_TYPE_IMAGE, VX_TYPE_THRESHOLD }
#define ATYPE_IcITS                            { VX_TYPE_IMAGE, AGO_TYPE_CANNY_STACK, VX_TYPE_IMAGE, VX_TYPE_THRESHOLD, VX_TYPE_SCALAR }
#define ATYPE_IIR                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_REMAP }
#define ATYPE_IIM                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_MATRIX }
#define ATYPE_IIIMM                            { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_MATRIX, VX_TYPE_MATRIX }
#define ATYPE_IIRS                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_REMAP, VX_TYPE_SCALAR }
#define ATYPE_IIMS                             { VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_MATRIX, VX_TYPE_SCALAR }
#define ATYPE_IIx                              { VX_TYPE_IMAGE, VX_TYPE_IMAGE, AGO_TYPE_SCALE_MATRIX }
#define ATYPE_AAA                              { VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY }
#define ATYPE_AAAAS                            { VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_SCALAR }
#define ATYPE_AAIISSSSS                        { VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_IMAGE, VX_TYPE_IMAGE, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_APPAASSSSS                       { VX_TYPE_ARRAY, VX_TYPE_PYRAMID, VX_TYPE_PYRAMID, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_SCALAR }
#define ATYPE_ASAAAAAAAA                       { VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY }
#define ATYPE_Ic                               { VX_TYPE_IMAGE, AGO_TYPE_CANNY_STACK }
#define ATYPE_DI                               { VX_TYPE_DISTRIBUTION, VX_TYPE_IMAGE }
#define ATYPE_mI                               { AGO_TYPE_MINMAXLOC_DATA, VX_TYPE_IMAGE }
#define ATYPE_sI                               { AGO_TYPE_MEANSTDDEV_DATA, VX_TYPE_IMAGE }
#define ATYPE_LDDDDDDDDD                       { VX_TYPE_LUT, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION }
#define ATYPE_DDDDDDDDDD                       { VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION, VX_TYPE_DISTRIBUTION }
#define ATYPE_SSssssssss                       { VX_TYPE_SCALAR, VX_TYPE_SCALAR, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA, AGO_TYPE_MEANSTDDEV_DATA }
#define ATYPE_SSmmmmmmmm                       { VX_TYPE_SCALAR, VX_TYPE_SCALAR, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA, AGO_TYPE_MINMAXLOC_DATA }
#define ATYPE_SIm                              { VX_TYPE_SCALAR, VX_TYPE_IMAGE, AGO_TYPE_MINMAXLOC_DATA }
#define ATYPE_SSIm                             { VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_IMAGE, AGO_TYPE_MINMAXLOC_DATA }
#define ATYPE_ASIm                             { VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_IMAGE, AGO_TYPE_MINMAXLOC_DATA }
#define ATYPE_ASSIm                            { VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_IMAGE, AGO_TYPE_MINMAXLOC_DATA }
#define ATYPE_AASSIm                           { VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_SCALAR, VX_TYPE_SCALAR, VX_TYPE_IMAGE, AGO_TYPE_MINMAXLOC_DATA }
#define ATYPE_SAAAAAAAAA                       { VX_TYPE_SCALAR, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY, VX_TYPE_ARRAY }
#define ATYPE_RR                               { VX_TYPE_REFERENCE, VX_TYPE_REFERENCE }
#define ATYPE_SRRR                             { VX_TYPE_SCALAR, VX_TYPE_REFERENCE, VX_TYPE_REFERENCE, VX_TYPE_REFERENCE }
#define ATYPE_RSRR                             { VX_TYPE_REFERENCE, VX_TYPE_SCALAR, VX_TYPE_REFERENCE, VX_TYPE_REFERENCE }
#define ATYPE_IMIS                             { VX_TYPE_IMAGE, VX_TYPE_MATRIX, VX_TYPE_IMAGE, VX_TYPE_SCALAR }

// for kernOpType & kernOpInfo
#define KOP_UNKNOWN    AGO_KERNEL_OP_TYPE_UNKNOWN,         0,
#define KOP_ELEMWISE   AGO_KERNEL_OP_TYPE_ELEMENT_WISE,    0,
#define KOP_FIXED(N)   AGO_KERNEL_OP_TYPE_FIXED_NEIGHBORS, N,

// list of all built-in kernels
static struct {
	vx_enum id;
	int(*func)(AgoNode * node, AgoKernelCommand cmd);
	const char * name;
	vx_uint32 flags;
	vx_uint8  argConfig[AGO_MAX_PARAMS];
	vx_enum argType[AGO_MAX_PARAMS];
	vx_uint8 kernOpType;
	vx_uint8 kernOpInfo;
} ago_kernel_list[] = {
#define OVX_KERNEL_ENTRY(kernel_id,name,kname,argCfg,argType,validRectReset) \
	{                                                               \
		kernel_id, ovxKernel_ ## name, "org.khronos.openvx." kname, \
		AGO_KERNEL_FLAG_GROUP_OVX10 | \
		(validRectReset ? AGO_KERNEL_FLAG_VALID_RECT_RESET : 0), argCfg, argType \
	}
#define AGO_KERNEL_ENTRY(kernel_id,cpu_avail,gpu_avail,name,argCfg,argType,kernOp,validRectReset) \
	{                                                               \
		kernel_id, agoKernel_ ## name, "com.amd.openvx." #name,     \
		AGO_KERNEL_FLAG_GROUP_AMDLL | (cpu_avail ? AGO_KERNEL_FLAG_DEVICE_CPU : 0) | (gpu_avail ? AGO_KERNEL_FLAG_DEVICE_GPU : 0) | \
		(validRectReset ? AGO_KERNEL_FLAG_VALID_RECT_RESET : 0), argCfg, argType, kernOp \
	}
	// OpenVX 1.x built-in kernels
	OVX_KERNEL_ENTRY( VX_KERNEL_COLOR_CONVERT         , ColorConvert, "color_convert",             		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_CHANNEL_EXTRACT       , ChannelExtract, "channel_extract",         		AINx2_AOUT,           ATYPE_ISI          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_CHANNEL_COMBINE       , ChannelCombine, "channel_combine",         		AINx2_AOPTINx2_AOUT,  ATYPE_IIIII        , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_SOBEL_3x3             , Sobel3x3, "sobel_3x3",                     		AIN_AOPTOUTx2,        ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_MAGNITUDE             , Magnitude, "magnitude",                    		AINx2_AOUT,           ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_PHASE                 , Phase, "phase",                            		AINx2_AOUT,           ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_SCALE_IMAGE           , ScaleImage, "scale_image",                 		AIN_AOUT_AIN,         ATYPE_IIS          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_TABLE_LOOKUP          , TableLookup, "table_lookup",               		AINx2_AOUT,           ATYPE_ILI          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_HISTOGRAM             , Histogram, "histogram",                    		AIN_AOUT,             ATYPE_ID           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_EQUALIZE_HISTOGRAM    , EqualizeHistogram, "equalize_histogram",   		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_ABSDIFF               , AbsDiff, "absdiff",                        		AINx2_AOUT,           ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_MEAN_STDDEV           , MeanStdDev, "mean_stddev",                 		AIN_AOUTx2,           ATYPE_ISS          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_THRESHOLD             , Threshold, "threshold",                    		AINx2_AOUT,           ATYPE_ITI          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_INTEGRAL_IMAGE        , IntegralImage, "integral_image",           		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_DILATE_3x3            , Dilate3x3, "dilate_3x3",                   		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_ERODE_3x3             , Erode3x3, "erode_3x3",                     		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_MEDIAN_3x3            , Median3x3, "median_3x3",                   		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_BOX_3x3               , Box3x3, "box_3x3",                         		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_GAUSSIAN_3x3          , Gaussian3x3, "gaussian_3x3",               		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_CUSTOM_CONVOLUTION    , CustomConvolution, "custom_convolution",   		AINx2_AOUT,           ATYPE_ICI          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_GAUSSIAN_PYRAMID      , GaussianPyramid, "gaussian_pyramid",       		AIN_AOUT,             ATYPE_IP           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_ACCUMULATE            , Accumulate, "accumulate",                  		AIN_AINOUT,           ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_ACCUMULATE_WEIGHTED   , AccumulateWeighted, "accumulate_weighted", 		AINx2_AINOUT,         ATYPE_ISI          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_ACCUMULATE_SQUARE     , AccumulateSquare, "accumulate_square",     		AINx2_AINOUT,         ATYPE_ISI          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_MINMAXLOC             , MinMaxLoc, "minmaxloc",                    		AIN_AOUTx2_AOPTOUTx4, ATYPE_ISSAASS      , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_CONVERTDEPTH          , ConvertDepth, "convertdepth",              		AIN_AOUT_AINx2,       ATYPE_IISS         , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_CANNY_EDGE_DETECTOR   , CannyEdgeDetector, "canny_edge_detector",  		AINx4_AOUT,           ATYPE_ITSSI        , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_AND                   , And, "and",                                		AINx2_AOUT,           ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_OR                    , Or, "or",                                  		AINx2_AOUT,           ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_XOR                   , Xor, "xor",                                		AINx2_AOUT,           ATYPE_III          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_NOT                   , Not, "not",                                		AIN_AOUT,             ATYPE_II           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_MULTIPLY              , Multiply, "multiply",                      		AINx5_AOUT,           ATYPE_IISSSI       , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_ADD                   , Add, "add",                                		AINx3_AOUT,           ATYPE_IISI         , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_SUBTRACT              , Subtract, "subtract",                      		AINx3_AOUT,           ATYPE_IISI         , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_WARP_AFFINE           , WarpAffine, "warp_affine",                 		AINx3_AOUT,           ATYPE_IMSI         , true  ),
	OVX_KERNEL_ENTRY( VX_KERNEL_WARP_PERSPECTIVE      , WarpPerspective, "warp_perspective",       		AINx3_AOUT,           ATYPE_IMSI         , true  ),
	OVX_KERNEL_ENTRY( VX_KERNEL_HARRIS_CORNERS        , HarrisCorners, "harris_corners",           		AINx6_AOUT_AOPTOUT,   ATYPE_ISSSSSAS     , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_FAST_CORNERS          , FastCorners, "fast_corners",               		AINx3_AOUT_AOPTOUT,   ATYPE_ISSAS        , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_OPTICAL_FLOW_PYR_LK   , OpticalFlowPyrLK, "optical_flow_pyr_lk",   		AINx4_AOUT_AINx5,     ATYPE_PPAAASSSSS   , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_REMAP                 , Remap, "remap",                            		AINx3_AOUT,           ATYPE_IRSI         , true  ),
	OVX_KERNEL_ENTRY( VX_KERNEL_HALFSCALE_GAUSSIAN    , HalfScaleGaussian, "halfscale_gaussian",   		AIN_AOUT_AIN,         ATYPE_IIS          , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_COPY                  , Copy, "copy",                              		AIN_AOUT,             ATYPE_RR           , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_SELECT                , Select, "select",                          		AINx3_AOUT,           ATYPE_SRRR         , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_WEIGHTED_AVERAGE      , WeightedAverage, "weighted_average",        		AINx3_AOUT,		      ATYPE_ISII         , false ),
	OVX_KERNEL_ENTRY( VX_KERNEL_NON_LINEAR_FILTER     , NonLinearFilter, "non_linear_filter",      		AINx3_AOUT,	     	  ATYPE_SIMI         , false ),	
	OVX_KERNEL_ENTRY( VX_KERNEL_LAPLACIAN_PYRAMID     , LaplacianPyramid, "laplacian_pyramid",     		AINx2_AOUT,	     	  ATYPE_IPI        	 , false ),	
	OVX_KERNEL_ENTRY( VX_KERNEL_LAPLACIAN_RECONSTRUCT , LaplacianReconstruct, "laplacian_reconstruct",  AINx2_AOUT,	     	  ATYPE_PII        	 , false ),	
	// AMD low-level kernel primitives
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SET_00_U8                                               , 1, 1, Set00_U8, { AOUT },                                           ATYPE_I                 , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SET_FF_U8                                               , 1, 1, SetFF_U8, { AOUT },                                           ATYPE_I                 , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOT_U8_U8                                               , 1, 1, Not_U8_U8, AOUT_AIN,                                          ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOT_U8_U1                                               , 1, 1, Not_U8_U1, AOUT_AIN,                                          ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOT_U1_U8                                               , 1, 1, Not_U1_U8, AOUT_AIN,                                          ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOT_U1_U1                                               , 1, 1, Not_U1_U1, AOUT_AIN,                                          ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_LUT_U8_U8                                               , 1, 1, Lut_U8_U8, AOUT_AINx2,                                        ATYPE_IIL               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_LUT_S16_S16                                             , 1, 1, Lut_S16_S16, AOUT_AINx2,                                      ATYPE_IIL               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_U8_U8_BINARY                                  , 1, 1, Threshold_U8_U8_Binary, AOUT_AINx2,                           ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_U8_U8_RANGE                                   , 1, 1, Threshold_U8_U8_Range, AOUT_AINx2,                            ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_U1_U8_BINARY                                  , 1, 1, Threshold_U1_U8_Binary, AOUT_AINx2,                           ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_U1_U8_RANGE                                   , 1, 1, Threshold_U1_U8_Range, AOUT_AINx2,                            ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_U8_S16_BINARY                                 , 1, 1, Threshold_U8_S16_Binary, AOUT_AINx2,                          ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_U8_S16_RANGE                                  , 1, 1, Threshold_U8_S16_Range, AOUT_AINx2,                           ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_BINARY                              , 1, 1, ThresholdNot_U8_U8_Binary, AOUT_AINx2,                        ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_RANGE                               , 1, 1, ThresholdNot_U8_U8_Range, AOUT_AINx2,                         ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_NOT_U1_U8_BINARY                              , 1, 1, ThresholdNot_U1_U8_Binary, AOUT_AINx2,                        ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_NOT_U1_U8_RANGE                               , 1, 1, ThresholdNot_U1_U8_Range, AOUT_AINx2,                         ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_NOT_U8_S16_BINARY                             , 1, 1, ThresholdNot_U8_S16_Binary, AOUT_AINx2,                       ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_THRESHOLD_NOT_U8_S16_RANGE                              , 1, 1, ThresholdNot_U8_S16_Range, AOUT_AINx2,                        ATYPE_IIT               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_DEPTH_U8_S16_WRAP                                 , 1, 1, ColorDepth_U8_S16_Wrap, AOUT_AINx2,                           ATYPE_IIS               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_DEPTH_U8_S16_SAT                                  , 1, 1, ColorDepth_U8_S16_Sat, AOUT_AINx2,                            ATYPE_IIS               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_DEPTH_S16_U8                                      , 1, 1, ColorDepth_S16_U8, AOUT_AINx2,                                ATYPE_IIS               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_U8_U8U8_WRAP                                        , 1, 1, Add_U8_U8U8_Wrap, AOUT_AINx2,                                 ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_U8_U8U8_SAT                                         , 1, 1, Add_U8_U8U8_Sat, AOUT_AINx2,                                  ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_U8_U8U8_WRAP                                        , 1, 1, Sub_U8_U8U8_Wrap, AOUT_AINx2,                                 ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_U8_U8U8_SAT                                         , 1, 1, Sub_U8_U8U8_Sat, AOUT_AINx2,                                  ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_U8_U8U8_WRAP_TRUNC                                  , 1, 1, Mul_U8_U8U8_Wrap_Trunc, AOUT_AINx3,                           ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_U8_U8U8_WRAP_ROUND                                  , 1, 1, Mul_U8_U8U8_Wrap_Round, AOUT_AINx3,                           ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_U8_U8U8_SAT_TRUNC                                   , 1, 1, Mul_U8_U8U8_Sat_Trunc, AOUT_AINx3,                            ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_U8_U8U8_SAT_ROUND                                   , 1, 1, Mul_U8_U8U8_Sat_Round, AOUT_AINx3,                            ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U8_U8U8                                             , 1, 1, And_U8_U8U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U8_U8U1                                             , 1, 1, And_U8_U8U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U8_U1U8                                             , 1, 1, And_U8_U1U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U8_U1U1                                             , 1, 1, And_U8_U1U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U1_U8U8                                             , 1, 1, And_U1_U8U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U1_U8U1                                             , 1, 1, And_U1_U8U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U1_U1U8                                             , 1, 1, And_U1_U1U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_AND_U1_U1U1                                             , 1, 1, And_U1_U1U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U8_U8U8                                              , 1, 1, Or_U8_U8U8, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U8_U8U1                                              , 1, 1, Or_U8_U8U1, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U8_U1U8                                              , 1, 1, Or_U8_U1U8, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U8_U1U1                                              , 1, 1, Or_U8_U1U1, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U1_U8U8                                              , 1, 1, Or_U1_U8U8, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U1_U8U1                                              , 1, 1, Or_U1_U8U1, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U1_U1U8                                              , 1, 1, Or_U1_U1U8, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OR_U1_U1U1                                              , 1, 1, Or_U1_U1U1, AOUT_AINx2,                                       ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U8_U8U8                                             , 1, 1, Xor_U8_U8U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U8_U8U1                                             , 1, 1, Xor_U8_U8U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U8_U1U8                                             , 1, 1, Xor_U8_U1U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U8_U1U1                                             , 1, 1, Xor_U8_U1U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U1_U8U8                                             , 1, 1, Xor_U1_U8U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U1_U8U1                                             , 1, 1, Xor_U1_U8U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U1_U1U8                                             , 1, 1, Xor_U1_U1U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XOR_U1_U1U1                                             , 1, 1, Xor_U1_U1U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U8_U8U8                                            , 1, 1, Nand_U8_U8U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U8_U8U1                                            , 1, 1, Nand_U8_U8U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U8_U1U8                                            , 1, 1, Nand_U8_U1U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U8_U1U1                                            , 1, 1, Nand_U8_U1U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U1_U8U8                                            , 1, 1, Nand_U1_U8U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U1_U8U1                                            , 1, 1, Nand_U1_U8U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U1_U1U8                                            , 1, 1, Nand_U1_U1U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NAND_U1_U1U1                                            , 1, 1, Nand_U1_U1U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U8_U8U8                                             , 1, 1, Nor_U8_U8U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U8_U8U1                                             , 1, 1, Nor_U8_U8U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U8_U1U8                                             , 1, 1, Nor_U8_U1U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U8_U1U1                                             , 1, 1, Nor_U8_U1U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U1_U8U8                                             , 1, 1, Nor_U1_U8U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U1_U8U1                                             , 1, 1, Nor_U1_U8U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U1_U1U8                                             , 1, 1, Nor_U1_U1U8, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NOR_U1_U1U1                                             , 1, 1, Nor_U1_U1U1, AOUT_AINx2,                                      ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U8_U8U8                                            , 1, 1, Xnor_U8_U8U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U8_U8U1                                            , 1, 1, Xnor_U8_U8U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U8_U1U8                                            , 1, 1, Xnor_U8_U1U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U8_U1U1                                            , 1, 1, Xnor_U8_U1U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U1_U8U8                                            , 1, 1, Xnor_U1_U8U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U1_U8U1                                            , 1, 1, Xnor_U1_U8U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U1_U1U8                                            , 1, 1, Xnor_U1_U1U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_XNOR_U1_U1U1                                            , 1, 1, Xnor_U1_U1U1, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ABS_DIFF_U8_U8U8                                        , 1, 1, AbsDiff_U8_U8U8, AOUT_AINx2,                                  ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ACCUMULATE_WEIGHTED_U8_U8U8                             , 1, 1, AccumulateWeighted_U8_U8U8, AINOUT_AINx2,                     ATYPE_IIS               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_S16_U8U8                                            , 1, 1, Add_S16_U8U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_U8U8                                            , 1, 1, Sub_S16_U8U8, AOUT_AINx2,                                     ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_U8U8_WRAP_TRUNC                                 , 1, 1, Mul_S16_U8U8_Wrap_Trunc, AOUT_AINx3,                          ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_U8U8_WRAP_ROUND                                 , 1, 1, Mul_S16_U8U8_Wrap_Round, AOUT_AINx3,                          ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_U8U8_SAT_TRUNC                                  , 1, 1, Mul_S16_U8U8_Sat_Trunc, AOUT_AINx3,                           ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_U8U8_SAT_ROUND                                  , 1, 1, Mul_S16_U8U8_Sat_Round, AOUT_AINx3,                           ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_S16_S16U8_WRAP                                      , 1, 1, Add_S16_S16U8_Wrap, AOUT_AINx2,                               ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_S16_S16U8_SAT                                       , 1, 1, Add_S16_S16U8_Sat, AOUT_AINx2,                                ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ACCUMULATE_S16_S16U8_SAT                                , 1, 1, Accumulate_S16_S16U8_Sat, AINOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_S16U8_WRAP                                      , 1, 1, Sub_S16_S16U8_Wrap, AOUT_AINx2,                               ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_S16U8_SAT                                       , 1, 1, Sub_S16_S16U8_Sat, AOUT_AINx2,                                ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16U8_WRAP_TRUNC                                , 1, 1, Mul_S16_S16U8_Wrap_Trunc, AOUT_AINx3,                         ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16U8_WRAP_ROUND                                , 1, 1, Mul_S16_S16U8_Wrap_Round, AOUT_AINx3,                         ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16U8_SAT_TRUNC                                 , 1, 1, Mul_S16_S16U8_Sat_Trunc, AOUT_AINx3,                          ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16U8_SAT_ROUND                                 , 1, 1, Mul_S16_S16U8_Sat_Round, AOUT_AINx3,                          ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ACCUMULATE_SQUARED_S16_S16U8_SAT                        , 1, 1, AccumulateSquared_S16_S16U8_Sat, AINOUT_AINx2,                ATYPE_IIS               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_U8S16_WRAP                                      , 1, 1, Sub_S16_U8S16_Wrap, AOUT_AINx2,                               ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_U8S16_SAT                                       , 1, 1, Sub_S16_U8S16_Sat, AOUT_AINx2,                                ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ABS_DIFF_S16_S16S16_SAT                                 , 1, 1, AbsDiff_S16_S16S16_Sat, AOUT_AINx2,                           ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_S16_S16S16_WRAP                                     , 1, 1, Add_S16_S16S16_Wrap, AOUT_AINx2,                              ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ADD_S16_S16S16_SAT                                      , 1, 1, Add_S16_S16S16_Sat, AOUT_AINx2,                               ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_S16S16_WRAP                                     , 1, 1, Sub_S16_S16S16_Wrap, AOUT_AINx2,                              ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SUB_S16_S16S16_SAT                                      , 1, 1, Sub_S16_S16S16_Sat, AOUT_AINx2,                               ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16S16_WRAP_TRUNC                               , 1, 1, Mul_S16_S16S16_Wrap_Trunc, AOUT_AINx3,                        ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16S16_WRAP_ROUND                               , 1, 1, Mul_S16_S16S16_Wrap_Round, AOUT_AINx3,                        ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16S16_SAT_TRUNC                                , 1, 1, Mul_S16_S16S16_Sat_Trunc, AOUT_AINx3,                         ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_S16_S16S16_SAT_ROUND                                , 1, 1, Mul_S16_S16S16_Sat_Round, AOUT_AINx3,                         ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MAGNITUDE_S16_S16S16                                    , 1, 1, Magnitude_S16_S16S16, AOUT_AINx2,                             ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_PHASE_U8_S16S16                                         , 1, 1, Phase_U8_S16S16, AOUT_AINx2,                                  ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COPY_U8_U8                                      , 1, 1, ChannelCopy_U8_U8, AOUT_AIN,                                  ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COPY_U8_U1                                      , 1, 1, ChannelCopy_U8_U1, AOUT_AIN,                                  ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COPY_U1_U8                                      , 1, 1, ChannelCopy_U1_U8, AOUT_AIN,                                  ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COPY_U1_U1                                      , 1, 1, ChannelCopy_U1_U1, AOUT_AIN,                                  ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS0                             , 1, 1, ChannelExtract_U8_U16_Pos0, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS1                             , 1, 1, ChannelExtract_U8_U16_Pos1, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS0                             , 1, 1, ChannelExtract_U8_U24_Pos0, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS1                             , 1, 1, ChannelExtract_U8_U24_Pos1, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS2                             , 1, 1, ChannelExtract_U8_U24_Pos2, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0                             , 1, 1, ChannelExtract_U8_U32_Pos0, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1                             , 1, 1, ChannelExtract_U8_U32_Pos1, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2                             , 1, 1, ChannelExtract_U8_U32_Pos2, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3                             , 1, 1, ChannelExtract_U8_U32_Pos3, AOUT_AIN,                         ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8U8U8_U24                              , 1, 1, ChannelExtract_U8U8U8_U24, AOUTx3_AIN,                        ATYPE_IIII              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8U8U8_U32                              , 1, 1, ChannelExtract_U8U8U8_U32, AOUTx3_AIN,                        ATYPE_IIII              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_EXTRACT_U8U8U8U8_U32                            , 1, 1, ChannelExtract_U8U8U8U8_U32, AOUTx4_AIN,                      ATYPE_IIIII             , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COMBINE_U16_U8U8                                , 1, 1, ChannelCombine_U16_U8U8, AOUT_AINx2,                          ATYPE_III               , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COMBINE_U24_U8U8U8_RGB                          , 1, 1, ChannelCombine_U24_U8U8U8_RGB, AOUT_AINx3,                    ATYPE_IIII              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_UYVY                         , 1, 1, ChannelCombine_U32_U8U8U8_UYVY, AOUT_AINx3,                   ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_YUYV                         , 1, 1, ChannelCombine_U32_U8U8U8_YUYV, AOUT_AINx3,                   ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8U8_RGBX                       , 1, 1, ChannelCombine_U32_U8U8U8U8_RGBX, AOUT_AINx4,                 ATYPE_IIIII             , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_U24_U24U8_SAT_ROUND                                 , 1, 1, Mul_U24_U24U8_Sat_Round, AOUT_AINx3,                          ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MUL_U32_U32U8_SAT_ROUND                                 , 1, 1, Mul_U32_U32U8_Sat_Round, AOUT_AINx3,                          ATYPE_IIIS              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGB_RGBX                                  , 1, 1, ColorConvert_RGB_RGBX, AOUT_AIN,                              ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGB_UYVY                                  , 1, 1, ColorConvert_RGB_UYVY, AOUT_AIN,                              ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGB_YUYV                                  , 1, 1, ColorConvert_RGB_YUYV, AOUT_AIN,                              ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGB_IYUV                                  , 1, 1, ColorConvert_RGB_IYUV, AOUT_AINx3,                            ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV12                                  , 1, 1, ColorConvert_RGB_NV12, AOUT_AINx2,                            ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV21                                  , 1, 1, ColorConvert_RGB_NV21, AOUT_AINx2,                            ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGBX_RGB                                  , 1, 1, ColorConvert_RGBX_RGB, AOUT_AIN,                              ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGBX_UYVY                                 , 1, 1, ColorConvert_RGBX_UYVY, AOUT_AIN,                             ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGBX_YUYV                                 , 1, 1, ColorConvert_RGBX_YUYV, AOUT_AIN,                             ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGBX_IYUV                                 , 1, 1, ColorConvert_RGBX_IYUV, AOUT_AINx3,                           ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV12                                 , 1, 1, ColorConvert_RGBX_NV12, AOUT_AINx2,                           ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV21                                 , 1, 1, ColorConvert_RGBX_NV21, AOUT_AINx2,                           ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_YUV4_RGB                                  , 1, 1, ColorConvert_YUV4_RGB, AOUTx3_AIN,                            ATYPE_IIII              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_YUV4_RGBX                                 , 1, 1, ColorConvert_YUV4_RGBX, AOUTx3_AIN,                           ATYPE_IIII              , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_UP_2x2_U8_U8                                      , 1, 1, ScaleUp2x2_U8_U8, AOUT_AIN,                                   ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_UV_UV12                                  , 1, 1, FormatConvert_UV_UV12, AOUTx2_AIN,                            ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGB                                  , 1, 1, ColorConvert_IYUV_RGB, AOUTx3_AIN,                            ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGBX                                 , 1, 1, ColorConvert_IYUV_RGBX, AOUTx3_AIN,                           ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_UYVY                                , 1, 1, FormatConvert_IYUV_UYVY, AOUTx3_AIN,                          ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_YUYV                                , 1, 1, FormatConvert_IYUV_YUYV, AOUTx3_AIN,                          ATYPE_IIII              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_IUV_UV12                                 , 1, 1, FormatConvert_IUV_UV12, AOUTx2_AIN,                           ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB                                  , 1, 1, ColorConvert_NV12_RGB, AOUTx2_AIN,                            ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX                                 , 1, 1, ColorConvert_NV12_RGBX, AOUTx2_AIN,                           ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_NV12_UYVY                                , 1, 1, FormatConvert_NV12_UYVY, AOUTx2_AIN,                          ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_NV12_YUYV                                , 1, 1, FormatConvert_NV12_YUYV, AOUTx2_AIN,                          ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FORMAT_CONVERT_UV12_IUV                                 , 1, 1, FormatConvert_UV12_IUV, AOUT_AINx2,                           ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB                                     , 1, 1, ColorConvert_Y_RGB, AOUT_AIN,                                 ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX                                    , 1, 1, ColorConvert_Y_RGBX, AOUT_AIN,                                ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_U_RGB                                     , 1, 1, ColorConvert_U_RGB, AOUT_AIN,                                 ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_U_RGBX                                    , 1, 1, ColorConvert_U_RGBX, AOUT_AIN,                                ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_V_RGB                                     , 1, 1, ColorConvert_V_RGB, AOUT_AIN,                                 ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_V_RGBX                                    , 1, 1, ColorConvert_V_RGBX, AOUT_AIN,                                ATYPE_II                , KOP_ELEMWISE  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB                                    , 1, 1, ColorConvert_IU_RGB, AOUT_AIN,                                ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX                                   , 1, 1, ColorConvert_IU_RGBX, AOUT_AIN,                               ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB                                    , 1, 1, ColorConvert_IV_RGB, AOUT_AIN,                                ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX                                   , 1, 1, ColorConvert_IV_RGBX, AOUT_AIN,                               ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGB                                   , 1, 1, ColorConvert_IUV_RGB, AOUTx2_AIN,                             ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGBX                                  , 1, 1, ColorConvert_IUV_RGBX, AOUTx2_AIN,                            ATYPE_III               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB                                  , 1, 1, ColorConvert_UV12_RGB, AOUT_AIN,                              ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX                                 , 1, 1, ColorConvert_UV12_RGBX, AOUT_AIN,                             ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_BOX_U8_U8_3x3                                           , 1, 1, Box_U8_U8_3x3, AOUT_AIN,                                      ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_DILATE_U8_U8_3x3                                        , 1, 1, Dilate_U8_U8_3x3, AOUT_AIN,                                   ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ERODE_U8_U8_3x3                                         , 1, 1, Erode_U8_U8_3x3, AOUT_AIN,                                    ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MEDIAN_U8_U8_3x3                                        , 1, 1, Median_U8_U8_3x3, AOUT_AIN,                                   ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_GAUSSIAN_U8_U8_3x3                                      , 1, 1, Gaussian_U8_U8_3x3, AOUT_AIN,                                 ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_3x3                           , 1, 1, ScaleGaussianHalf_U8_U8_3x3, AOUT_AIN,                        ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_5x5                           , 1, 1, ScaleGaussianHalf_U8_U8_5x5, AOUT_AIN,                        ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_GAUSSIAN_ORB_U8_U8_5x5                            , 1, 1, ScaleGaussianOrb_U8_U8_5x5, AOUT_AIN,                         ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CONVOLVE_U8_U8                                          , 1, 1, Convolve_U8_U8, AOUT_AINx2,                                   ATYPE_IIC               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CONVOLVE_S16_U8                                         , 1, 1, Convolve_S16_U8, AOUT_AINx2,                                  ATYPE_IIC               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_LINEAR_FILTER_ANY_ANY                                   , 1, 1, LinearFilter_ANY_ANY, AOUT_AINx2,                             ATYPE_IIM               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_LINEAR_FILTER_ANYx2_ANY                                 , 1, 1, LinearFilter_ANYx2_ANY, AOUTx2_AINx3,                         ATYPE_IIIMM             , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SOBEL_MAGNITUDE_S16_U8_3x3                              , 1, 1, SobelMagnitude_S16_U8_3x3, AOUT_AIN,                          ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SOBEL_PHASE_U8_U8_3x3                                   , 1, 1, SobelPhase_U8_U8_3x3, AOUT_AIN,                               ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SOBEL_MAGNITUDE_PHASE_S16U8_U8_3x3                      , 1, 1, SobelMagnitudePhase_S16U8_U8_3x3, AOUTx2_AIN,                 ATYPE_III               , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY                                 , 1, 1, Sobel_S16S16_U8_3x3_GXY, AOUTx2_AIN,                          ATYPE_III               , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GX                                     , 1, 1, Sobel_S16_U8_3x3_GX, AOUT_AIN,                                ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GY                                     , 1, 1, Sobel_S16_U8_3x3_GY, AOUT_AIN,                                ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_DILATE_U1_U8_3x3                                        , 1, 1, Dilate_U1_U8_3x3, AOUT_AIN,                                   ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ERODE_U1_U8_3x3                                         , 1, 1, Erode_U1_U8_3x3, AOUT_AIN,                                    ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_DILATE_U1_U1_3x3                                        , 1, 1, Dilate_U1_U1_3x3, AOUT_AIN,                                   ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ERODE_U1_U1_3x3                                         , 1, 1, Erode_U1_U1_3x3, AOUT_AIN,                                    ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_DILATE_U8_U1_3x3                                        , 1, 1, Dilate_U8_U1_3x3, AOUT_AIN,                                   ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_ERODE_U8_U1_3x3                                         , 1, 1, Erode_U8_U1_3x3, AOUT_AIN,                                    ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FAST_CORNERS_XY_U8_SUPRESSION                           , 1, 1, FastCorners_XY_U8_Supression, AOUT_AOPTOUT_AINx2,             ATYPE_ASIS              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FAST_CORNERS_XY_U8_NOSUPRESSION                         , 1, 1, FastCorners_XY_U8_NoSupression, AOUT_AOPTOUT_AINx2,           ATYPE_ASIS              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_3x3                                 , 1, 1, HarrisSobel_HG3_U8_3x3, AOUT_AIN,                             ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_5x5                                 , 1, 1, HarrisSobel_HG3_U8_5x5, AOUT_AIN,                             ATYPE_II                , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_7x7                                 , 1, 1, HarrisSobel_HG3_U8_7x7, AOUT_AIN,                             ATYPE_II                , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_3x3                                , 1, 1, HarrisScore_HVC_HG3_3x3, AOUT_AINx4,                          ATYPE_IISSS             , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_5x5                                , 1, 1, HarrisScore_HVC_HG3_5x5, AOUT_AINx4,                          ATYPE_IISSS             , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_7x7                                , 1, 1, HarrisScore_HVC_HG3_7x7, AOUT_AINx4,                          ATYPE_IISSS             , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_3x3_L1NORM             , 0, 1, CannySobelSuppThreshold_U8_U8_3x3_L1NORM, AOUT_AINx2,         ATYPE_IIT               , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_3x3_L2NORM             , 0, 1, CannySobelSuppThreshold_U8_U8_3x3_L2NORM, AOUT_AINx2,         ATYPE_IIT               , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_5x5_L1NORM             , 0, 1, CannySobelSuppThreshold_U8_U8_5x5_L1NORM, AOUT_AINx2,         ATYPE_IIT               , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_5x5_L2NORM             , 0, 1, CannySobelSuppThreshold_U8_U8_5x5_L2NORM, AOUT_AINx2,         ATYPE_IIT               , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_7x7_L1NORM             , 0, 1, CannySobelSuppThreshold_U8_U8_7x7_L1NORM, AOUT_AINx2,         ATYPE_IIT               , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_7x7_L2NORM             , 0, 1, CannySobelSuppThreshold_U8_U8_7x7_L2NORM, AOUT_AINx2,         ATYPE_IIT               , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_3x3_L1NORM           , 1, 0, CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM, AOUTx2_AINx2,     ATYPE_IcIT              , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_3x3_L2NORM           , 1, 0, CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM, AOUTx2_AINx2,     ATYPE_IcIT              , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_5x5_L1NORM           , 1, 0, CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM, AOUTx2_AINx2,     ATYPE_IcIT              , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_5x5_L2NORM           , 1, 0, CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM, AOUTx2_AINx2,     ATYPE_IcIT              , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_7x7_L1NORM           , 1, 0, CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM, AOUTx2_AINx2,     ATYPE_IcIT              , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_7x7_L2NORM           , 1, 0, CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM, AOUTx2_AINx2,     ATYPE_IcIT              , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L1NORM                           , 1, 1, CannySobel_U16_U8_3x3_L1NORM, AOUT_AIN,                       ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L2NORM                           , 1, 1, CannySobel_U16_U8_3x3_L2NORM, AOUT_AIN,                       ATYPE_II                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L1NORM                           , 1, 1, CannySobel_U16_U8_5x5_L1NORM, AOUT_AIN,                       ATYPE_II                , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L2NORM                           , 1, 1, CannySobel_U16_U8_5x5_L2NORM, AOUT_AIN,                       ATYPE_II                , KOP_FIXED(5)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM                           , 1, 1, CannySobel_U16_U8_7x7_L1NORM, AOUT_AIN,                       ATYPE_II                , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L2NORM                           , 1, 1, CannySobel_U16_U8_7x7_L2NORM, AOUT_AIN,                       ATYPE_II                , KOP_FIXED(7)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8_U16_3x3                         , 0, 1, CannySuppThreshold_U8_U16_3x3, AOUT_AINx2_AOPTIN,             ATYPE_IITS              , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3                       , 1, 1, CannySuppThreshold_U8XY_U16_3x3, AOUTx2_AINx2_AOPTIN,         ATYPE_IcITS             , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_7x7                       , 1, 1, CannySuppThreshold_U8XY_U16_7x7, AOUTx2_AINx2_AOPTIN,         ATYPE_IcITS             , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NON_MAX_SUPP_XY_ANY_3x3                                 , 0, 1, NonMaxSupp_XY_ANY_3x3, AOUT_AIN,                              ATYPE_AI                , KOP_FIXED(3)  , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U8_U8_NEAREST                                     , 1, 1, Remap_U8_U8_Nearest, AOUT_AINx2,                              ATYPE_IIR               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U8_U8_NEAREST_CONSTANT                            , 1, 1, Remap_U8_U8_Nearest_Constant, AOUT_AINx3,                     ATYPE_IIRS              , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U8_U8_BILINEAR                                    , 1, 1, Remap_U8_U8_Bilinear, AOUT_AINx2,                             ATYPE_IIR               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U8_U8_BILINEAR_CONSTANT                           , 1, 1, Remap_U8_U8_Bilinear_Constant, AOUT_AINx3,                    ATYPE_IIRS              , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U24_U24_BILINEAR                                  , 1, 1, Remap_U24_U24_Bilinear, AOUT_AINx2,                           ATYPE_IIR               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U24_U32_BILINEAR                                  , 1, 1, Remap_U24_U32_Bilinear, AOUT_AINx2,                           ATYPE_IIR               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_REMAP_U32_U32_BILINEAR                                  , 1, 1, Remap_U32_U32_Bilinear, AOUT_AINx2,                           ATYPE_IIR               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_AFFINE_U8_U8_NEAREST                               , 1, 1, WarpAffine_U8_U8_Nearest, AOUT_AINx2,                         ATYPE_IIM               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_AFFINE_U8_U8_NEAREST_CONSTANT                      , 1, 1, WarpAffine_U8_U8_Nearest_Constant, AOUT_AINx3,                ATYPE_IIMS              , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_AFFINE_U8_U8_BILINEAR                              , 1, 1, WarpAffine_U8_U8_Bilinear, AOUT_AINx2,                        ATYPE_IIM               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_AFFINE_U8_U8_BILINEAR_CONSTANT                     , 1, 1, WarpAffine_U8_U8_Bilinear_Constant, AOUT_AINx3,               ATYPE_IIMS              , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_NEAREST                          , 1, 1, WarpPerspective_U8_U8_Nearest, AOUT_AINx2,                    ATYPE_IIM               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_NEAREST_CONSTANT                 , 1, 1, WarpPerspective_U8_U8_Nearest_Constant, AOUT_AINx3,           ATYPE_IIMS              , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_BILINEAR                         , 1, 1, WarpPerspective_U8_U8_Bilinear, AOUT_AINx2,                   ATYPE_IIM               , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_BILINEAR_CONSTANT                , 1, 1, WarpPerspective_U8_U8_Bilinear_Constant, AOUT_AINx3,          ATYPE_IIMS              , KOP_UNKNOWN   , true  ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST                               , 1, 1, ScaleImage_U8_U8_Nearest, AOUT_AIN,                           ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR                              , 1, 1, ScaleImage_U8_U8_Bilinear, AOUT_AIN,                          ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR_REPLICATE                    , 1, 1, ScaleImage_U8_U8_Bilinear_Replicate, AOUT_AIN,                ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR_CONSTANT                     , 1, 1, ScaleImage_U8_U8_Bilinear_Constant, AOUT_AINx2,               ATYPE_IIS               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA                                  , 1, 1, ScaleImage_U8_U8_Area, AOUT_AIN,                              ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OPTICAL_FLOW_PYR_LK_XY_XY                               , 1, 1, OpticalFlowPyrLK_XY_XY, AOUT_AINx9,                           ATYPE_APPAASSSSS        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OPTICAL_FLOW_PREPARE_LK_XY_XY                           , 1, 1, OpticalFlowPrepareLK_XY_XY, AOUT_AINx4,                       ATYPE_AAAAS             , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OPTICAL_FLOW_IMAGE_LK_XY_XY                             , 1, 1, OpticalFlowImageLK_XY_XY, AOUT_AINx8,                         ATYPE_AAIISSSSS         , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_OPTICAL_FLOW_FINAL_LK_XY_XY                             , 1, 1, OpticalFlowFinalLK_XY_XY, AOUT_AINx2,                         ATYPE_AAA               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_MERGE_SORT_AND_PICK_XY_HVC                       , 1, 0, HarrisMergeSortAndPick_XY_HVC, AOUT_AOPTOUT_AINx2,            ATYPE_ASIS              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HARRIS_MERGE_SORT_AND_PICK_XY_XYS                       , 1, 0, HarrisMergeSortAndPick_XY_XYS, AOUT_AOPTOUT_AINx4,            ATYPE_ASASSS            , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_FAST_CORNER_MERGE_XY_XY                                 , 1, 0, FastCornerMerge_XY_XY, AOUTx2_AIN_AOPTINx7,                   ATYPE_ASAAAAAAAA        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8                                  , 1, 0, CannyEdgeTrace_U8_U8, AINOUT_AIN,                             ATYPE_Ic                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8XY                                , 1, 0, CannyEdgeTrace_U8_U8XY, AINOUT_AIN,                           ATYPE_Ic                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_INTEGRAL_IMAGE_U32_U8                                   , 1, 0, IntegralImage_U32_U8, AOUT_AIN,                               ATYPE_II                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HISTOGRAM_DATA_U8                                       , 1, 0, Histogram_DATA_U8, AOUT_AIN,                                  ATYPE_DI                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MEAN_STD_DEV_DATA_U8                                    , 1, 0, MeanStdDev_DATA_U8, AOUT_AIN,                                 ATYPE_sI                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MEAN_STD_DEV_DATA_U1                                    , 1, 0, MeanStdDev_DATA_U1, AOUT_AIN,                                 ATYPE_sI                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_DATA_U8                                         , 1, 0, MinMax_DATA_U8, AOUT_AIN,                                     ATYPE_mI                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_DATA_S16                                        , 1, 0, MinMax_DATA_S16, AOUT_AIN,                                    ATYPE_mI                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_EQUALIZE_DATA_DATA                                      , 1, 0, Equalize_DATA_DATA, AOUT_AIN_AOPTINx8,                        ATYPE_LDDDDDDDDD        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_HISTOGRAM_MERGE_DATA_DATA                               , 1, 0, HistogramMerge_DATA_DATA, AOUT_AIN_AOPTINx8,                  ATYPE_DDDDDDDDDD        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MEAN_STD_DEV_MERGE_DATA_DATA                            , 1, 0, MeanStdDevMerge_DATA_DATA, AOUTx2_AIN_AOPTINx7,               ATYPE_SSssssssss        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_MERGE_DATA_DATA                                 , 1, 0, MinMaxMerge_DATA_DATA, AOUTx3_AIN_AOPTINx6,                   ATYPE_SSmmmmmmmm        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_NONE_COUNT_MIN              , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min, AOUT_AINx2,         ATYPE_SIm               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_NONE_COUNT_MAX              , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max, AOUT_AINx2,         ATYPE_SIm               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_NONE_COUNT_MINMAX           , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_None_Count_MinMax, AOUTx2_AINx2,    ATYPE_SSIm              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MIN_COUNT_MIN               , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_Min_Count_Min, AOUT_AOPTOUT_AINx2,  ATYPE_ASIm              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MIN_COUNT_MINMAX            , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax, AOUT_AOPTOUTx2_AINx2, ATYPE_ASSIm         , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MAX_COUNT_MAX               , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max, AOUT_AOPTOUT_AINx2,  ATYPE_ASIm              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MAX_COUNT_MINMAX            , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax, AOUT_AOPTOUTx2_AINx2, ATYPE_ASSIm         , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MINMAX_COUNT_MINMAX         , 1, 0, MinMaxLoc_DATA_U8DATA_Loc_MinMax_Count_MinMax, AOUTx2_AOPTOUTx2_AINx2, ATYPE_AASSIm   , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_NONE_COUNT_MIN             , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min, AOUT_AINx2,        ATYPE_SIm               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_NONE_COUNT_MAX             , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max, AOUT_AINx2,        ATYPE_SIm               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_NONE_COUNT_MINMAX          , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax, AOUTx2_AINx2,   ATYPE_SSIm              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MIN_COUNT_MIN              , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min, AOUT_AOPTOUT_AINx2, ATYPE_ASIm              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MIN_COUNT_MINMAX           , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax, AOUT_AOPTOUTx2_AINx2, ATYPE_ASSIm        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MAX_COUNT_MAX              , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max, AOUT_AOPTOUT_AINx2, ATYPE_ASIm              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MAX_COUNT_MINMAX           , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax, AOUT_AOPTOUTx2_AINx2, ATYPE_ASSIm        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MINMAX_COUNT_MINMAX        , 1, 0, MinMaxLoc_DATA_S16DATA_Loc_MinMax_Count_MinMax, AOUTx2_AOPTOUTx2_AINx2, ATYPE_AASSIm  , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_MIN_MAX_LOC_MERGE_DATA_DATA                             , 1, 0, MinMaxLocMerge_DATA_DATA, AOUTx2_AIN_AOPTINx7,                ATYPE_SAAAAAAAAA        , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_COPY_DATA_DATA                                          , 1, 1, Copy_DATA_DATA, AOUT_AIN,                                     ATYPE_RR                , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_SELECT_DATA_DATA_DATA                                   , 1, 1, Select_DATA_DATA_DATA, AOUT_AIN,                              ATYPE_RSRR              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_WEIGHTED_AVERAGE_U8_U8_U8                               , 1, 0, WeightedAverage_U8_U8_U8, AOUT_AINx3,                         ATYPE_IISI              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_NON_LINEAR_FILTER_DATA_DATA_DATA                        , 1, 0, NonLinearFilter_DATA_DATA_DATA, AOUT_AINx3,                   ATYPE_IMIS              , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_LAPLACIAN_PYRAMID_DATA_DATA_DATA                        , 1, 0, LaplacianPyramid_DATA_DATA_DATA, AOUT_AINx2,                  ATYPE_IPI               , KOP_UNKNOWN   , false ),
	AGO_KERNEL_ENTRY( VX_KERNEL_AMD_LAPLACIAN_RECONSTRUCT_DATA_DATA_DATA                    , 1, 0, LaplacianReconstruct_DATA_DATA_DATA, AOUT_AINx2,              ATYPE_IIP               , KOP_UNKNOWN   , false ),
#undef AGO_KERNEL_ENTRY
#undef OVX_KERNEL_ENTRY
};
size_t ago_kernel_count = sizeof(ago_kernel_list) / sizeof(ago_kernel_list[0]);

int agoPublishKernels(AgoContext * acontext)
{
	int ovxKernelCount = 0;
	int agoKernelCount = 0, agoKernelCountCpu = 0, agoKernelCountGpu = 0;
	for (vx_size i = 0; i < ago_kernel_count; i++)
	{
		AgoKernel * kernel = new AgoKernel;
		agoResetReference(&kernel->ref, VX_TYPE_KERNEL, acontext, NULL);
		kernel->id = ago_kernel_list[i].id;
		kernel->func = ago_kernel_list[i].func;
		kernel->flags = ago_kernel_list[i].flags;
		kernel->kernOpType = ago_kernel_list[i].kernOpType;
		kernel->kernOpInfo = ago_kernel_list[i].kernOpInfo;
		kernel->finalized = true;
		kernel->ref.internal_count = 1;
		kernel->user_kernel = false;
		strcpy(kernel->name, ago_kernel_list[i].name);
		memcpy(kernel->argConfig, ago_kernel_list[i].argConfig, sizeof(kernel->argConfig));
		kernel->argCount = 0;
		for (vx_uint32 j = 0; j < AGO_MAX_PARAMS; j++) {
			// if arg[j] is valid, then there are atleast j+1 arguments
			if (kernel->argConfig[j])
				kernel->argCount = j + 1;
		}
		memcpy(kernel->argType, ago_kernel_list[i].argType, sizeof(kernel->argType));
		for (vx_uint32 j = 0; j < kernel->argCount; j++) {
			// initialize for vx_parameter use
			agoResetReference(&kernel->parameters[j].ref, VX_TYPE_PARAMETER, acontext, &kernel->ref);
			kernel->parameters[j].index = j;
			kernel->parameters[j].direction = VX_INPUT;
			if (kernel->argConfig[j] & AGO_KERNEL_ARG_OUTPUT_FLAG)
				kernel->parameters[j].direction = (kernel->argConfig[j] & AGO_KERNEL_ARG_INPUT_FLAG) ? (vx_direction_e)VX_BIDIRECTIONAL : VX_OUTPUT;
			kernel->parameters[j].type = ago_kernel_list[i].argType[j];
			kernel->parameters[j].state = (kernel->argConfig[j] & AGO_KERNEL_ARG_OPTIONAL_FLAG) ? VX_PARAMETER_STATE_OPTIONAL : VX_PARAMETER_STATE_REQUIRED;
			kernel->parameters[j].scope = &kernel->ref;
		}
		agoAddKernel(&acontext->kernelList, kernel);
		int kernelGroup = kernel->flags & AGO_KERNEL_FLAG_GROUP_MASK;
		if (kernelGroup == AGO_KERNEL_FLAG_GROUP_OVX10) ovxKernelCount++;
		else if (kernelGroup == AGO_KERNEL_FLAG_GROUP_AMDLL) {
			agoKernelCount++;
			if (kernel->flags & AGO_KERNEL_FLAG_DEVICE_CPU) agoKernelCountCpu++;
			if (kernel->flags & AGO_KERNEL_FLAG_DEVICE_GPU) agoKernelCountGpu++;
		}
	}
#if ENABLE_DEBUG_MESSAGES
	printf("OK: ago imported %d(VX) + %d(AMD:[cpu-%d][gpu-%d]) kernels\n", ovxKernelCount, agoKernelCount, agoKernelCountCpu, agoKernelCountGpu); 
#endif
	return 0;
}
