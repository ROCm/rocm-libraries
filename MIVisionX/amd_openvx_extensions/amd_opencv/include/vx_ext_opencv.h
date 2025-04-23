
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

/*!
 * \file
 * \brief The AMD OpenVX OpenCV Nodes Extension Library.
 *
 * \defgroup group_opencv Extension: AMD OpenCV Interop API
 * \brief AMD OpenVX OpenCV Interop Nodes for OpenCV Functionality within OpenVX Graph.
 */

#include "VX/vx.h"
#include <VX/vx_compatibility.h>

#include "opencv2/opencv.hpp"
#if USE_OPENCV_CONTRIB
#include "opencv2/xfeatures2d.hpp"
#endif
#if USE_OPENCV_4
#define CV_DIST_L1 DIST_L1
#endif

#ifndef dimof
/*! \def dimof(x)
 *  \brief A macro to get the number of elements in an array.
 *  \param [in] x The array whose size is to be determined.
 *  \return The number of elements in the array.
 */
#define dimof(x) (sizeof(x) / sizeof(x[0]))
#endif

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
/*! \def SHARED_PUBLIC
 *  \brief A macro to specify public visibility for shared library symbols.
 */
#define SHARED_PUBLIC __attribute__((visibility("default")))
#endif

/*! \brief Creates a node in a graph using a predefined kernel structure.
 *  \param [in] graph The handle to the graph.
 *  \param [in] kernelenum The enum value representing the kernel to be used.
 *  \param [in] params An array of parameter references for the kernel.
 *  \param [in] num The number of parameters in the params array.
 *  \return A handle to the created node.
 */
vx_node vxCreateNodeByStructure(vx_graph graph, vx_enum kernelenum, vx_reference params[], vx_uint32 num);

#ifdef __cplusplus
extern "C"
{
#endif

	/*!***********************************************************************************************************
						OpenCV VX_API_ENTRY C Function NODE
	*************************************************************************************************************/

	/*! \brief [Graph] Creates a OpenCV blur function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> or <tt>\ref VX_DF_IMAGE_U16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] kwidth The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set normalized box filter width.
	 * \param [in] kheight The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set normalized box filter height.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set anchor point x.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set anchor point y.
	 * \param [in] Bordertype The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set borderType.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_blur(vx_graph graph, vx_image input, vx_image output, vx_uint32 kwidth, vx_uint32 kheight, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 Bordertype);

	/*! \brief [Graph] Creates a OpenCV boxFilter function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> or <tt>\ref VX_DF_IMAGE_U16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ddepth The input <tt>\ref VX_TYPE_INT32</tt> scalar to set output image depth.
	 * \param [in] kwidth The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set box filter width.
	 * \param [in] kheight The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set box filter height.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set anchor point x.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set anchor point y.
	 * \param [in] Normalized The input <tt>\ref VX_TYPE_BOOL</tt> scalar for Normalized argument.
	 * \param [in] Bordertype The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set borderType.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_boxFilter(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_uint32 kwidth, vx_uint32 kheight, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_bool Normalized, vx_int32 Bordertype);

	/*! \brief [Graph] Creates a OpenCV GaussianBlur function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> or <tt>\ref VX_DF_IMAGE_U16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] kwidth The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set gaussian filter width.
	 * \param [in] kheight The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set gaussian filter height.
	 * \param [in] sigmaX The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set gaussian filter standard deviation in x-direction.
	 * \param [in] sigmaY The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set gaussian filter standard deviation in y-direction.
	 * \param [in] border_mode The input <tt>\ref VX_TYPE_ENUM</tt> scalar to set border mode.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_gaussianBlur(vx_graph graph, vx_image input, vx_image output, vx_uint32 kwidth, vx_uint32 kheight, vx_float32 sigmaX, vx_float32 sigmaY, vx_int32 border_mode);

	/*! \brief [Graph] Creates a OpenCV medianBlur function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_U16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ksize The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set the aperture linear size.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_medianBlur(vx_graph graph, vx_image input, vx_image output, vx_uint32 ksize);

	/*! \brief [Graph] Creates a OpenCV filter2D function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ddepth The input <tt>\ref VX_TYPE_INT32</tt> scalar to set ddepth.
	 * \param [in] Kernel The input <tt>\ref vx_matrix</tt> scalar to set Kernel.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_X.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_Y.
	 * \param [in] delta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set delta.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_filter2D(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_float32 delta, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV sepFilter2D function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ddepth The input <tt>\ref VX_TYPE_INT32</tt> scalar to set ddepth.
	 * \param [in] KernelX The input <tt>\ref vx_matrix</tt> matrix to set KernelX.
	 * \param [in] KernelY The input <tt>\ref vx_matrix</tt> matrix to set KernelY.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_X.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_Y.
	 * \param [in] delta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set delta.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_sepFilter2D(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_matrix KernelX, vx_matrix KernelY, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_float32 delta, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV BilateralFilter function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] d The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set K width.
	 * \param [in] Sigma_Color The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set sigmaX.
	 * \param [in] Sigma_Space The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set sigmaY.
	 * \param [in] border_mode mode The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set border mode.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bilateralFilter(vx_graph graph, vx_image input, vx_image output, vx_uint32 d, vx_float32 Sigma_Color, vx_float32 Sigma_Space, vx_int32 border_mode);

	/*! \brief [Graph] Creates a OpenCV BRISK compute node to detect keypoints and optionally compute descriptors.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [out] output_des The output descriptors <tt>\ref vx_array</tt> of user defined data type with 64/128 byte element size depending upon extended argument (optional).
	 * \param [in] thresh The input <tt>\ref VX_TYPE_INT32</tt> scalar for thresh argument.
	 * \param [in] octaves The input <tt>\ref VX_TYPE_INT32</tt> scalar for octaves argument.
	 * \param [in] patternScale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for patternScale argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_briskCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des, vx_int32 thresh, vx_int32 octaves, vx_float32 patternScale);

	/*! \brief [Graph] Creates a OpenCV BRISK detector node to detect keypoints and optionally compute descriptors.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] thresh The input <tt>\ref VX_TYPE_INT32</tt> scalar for thresh argument.
	 * \param [in] octaves The input <tt>\ref VX_TYPE_INT32</tt> scalar for octaves argument.
	 * \param [in] patternScale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for patternScale argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_briskDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_int32 thresh, vx_int32 octaves, vx_float32 patternScale);

	/*! \brief [Graph] Creates a OpenCV FAST feature detector node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] threshold The input <tt>\ref VX_TYPE_INT32</tt> scalar for threshold argument.
	 * \param [in] nonmaxSuppression The input <tt>\ref VX_TYPE_BOOL</tt> scalar for nonmaxSuppression argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_fast(vx_graph graph, vx_image input, vx_array output_kp, vx_int32 threshold, vx_bool nonmaxSuppression);

	/*! \brief [Graph] Creates a OpenCV GoodFeaturesToTrack detector node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] maxCorners The input <tt>\ref VX_TYPE_INT32</tt> scalar for maxCorners argument.
	 * \param [in] qualityLevel The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for qualityLevel argument.
	 * \param [in] minDistance The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for minDistance argument.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [in] blockSize The input <tt>\ref VX_TYPE_INT32</tt> scalar for blockSize argument.
	 * \param [in] useHarrisDetector The input <tt>\ref VX_TYPE_BOOL</tt> scalar for useHarrisDetector argument.
	 * \param [in] k The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for k argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_goodFeaturesToTrack(vx_graph graph, vx_image input, vx_array output_kp, vx_int32 maxCorners, vx_float32 qualityLevel, vx_float32 minDistance, vx_image mask, vx_int32 blockSize, vx_bool useHarrisDetector, vx_float32 k);

	/*! \brief [Graph] Creates a OpenCV MSER feature detector node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format.
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [in] delta The input <tt>\ref VX_TYPE_INT32</tt> scalar for delta argument.
	 * \param [in] min_area The input <tt>\ref VX_TYPE_INT32</tt> scalar for min_area argument.
	 * \param [in] max_area The input <tt>\ref VX_TYPE_INT32</tt> scalar for max_area argument.
	 * \param [in] max_variation The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for max_variation argument.
	 * \param [in] min_diversity The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for min_diversity argument.
	 * \param [in] max_evolution The input <tt>\ref VX_TYPE_INT32</tt> scalar for max_evolution argument.
	 * \param [in] area_threshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for area_threshold argument.
	 * \param [in] min_margin The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for min_margin argument.
	 * \param [in] edge_blur_size The input <tt>\ref VX_TYPE_INT32</tt> scalar for max_area argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_mserDetect(vx_graph graph, vx_image input, vx_array output_kp, vx_image mask, vx_int32 delta, vx_int32 min_area, vx_int32 max_area, vx_float32 max_variation, vx_float32 min_diversity, vx_int32 max_evolution, vx_float32 area_threshold, vx_float32 min_margin, vx_int32 edge_blur_size);

	/*! \brief [Graph] Creates a OpenCV ORB Compute node to detect keypoints and optionally compute descriptors.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [in] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [out] output_des The output descriptors <tt>\ref vx_array</tt> of user defined data type with 64/128 byte element size depending upon extended argument.
	 * \param [in] nfeatures The input <tt>\ref VX_TYPE_INT32</tt> scalar for nfeatures argument.
	 * \param [in] scaleFactor The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for scaleFactor argument.
	 * \param [in] nlevels The input <tt>\ref VX_TYPE_INT32</tt> scalar for nlevels argument.
	 * \param [in] edgeThreshold The input <tt>\ref VX_TYPE_INT32</tt> scalar for edgeThreshold argument.
	 * \param [in] firstLevel The input <tt>\ref VX_TYPE_INT32</tt> scalar for firstLevel argument.
	 * \param [in] WTA_K The input <tt>\ref VX_TYPE_INT32</tt> scalar for WTA_K argument.
	 * \param [in] scoreType The input <tt>\ref VX_TYPE_INT32</tt> scalar for scoreType argument.
	 * \param [in] patchSize The input <tt>\ref VX_TYPE_INT32</tt> scalar for patchSize argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_orbCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des, vx_int32 nfeatures, vx_float32 scaleFactor, vx_int32 nlevels, vx_int32 edgeThreshold, vx_int32 firstLevel, vx_int32 WTA_K, vx_int32 scoreType, vx_int32 patchSize);

	/*! \brief [Graph] Creates a OpenCV ORB detector node to detect keypoints and optionally compute descriptors.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] nfeatures The input <tt>\ref VX_TYPE_INT32</tt> scalar for nfeatures argument.
	 * \param [in] scaleFactor The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for scaleFactor argument.
	 * \param [in] nlevels The input <tt>\ref VX_TYPE_INT32</tt> scalar for nlevels argument.
	 * \param [in] edgeThreshold The input <tt>\ref VX_TYPE_INT32</tt> scalar for edgeThreshold argument.
	 * \param [in] firstLevel The input <tt>\ref VX_TYPE_INT32</tt> scalar for firstLevel argument.
	 * \param [in] WTA_K The input <tt>\ref VX_TYPE_INT32</tt> scalar for WTA_K argument.
	 * \param [in] scoreType The input <tt>\ref VX_TYPE_INT32</tt> scalar for scoreType argument.
	 * \param [in] patchSize The input <tt>\ref VX_TYPE_INT32</tt> scalar for patchSize argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_orbDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_int32 nfeatures, vx_float32 scaleFactor, vx_int32 nlevels, vx_int32 edgeThreshold, vx_int32 firstLevel, vx_int32 WTA_K, vx_int32 scoreType, vx_int32 patchSize);

	/*! \brief [Graph] Creates a OpenCV SIFT Compute node to compute descriptor from specified keypoints.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [in] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [out] output_des The output descriptors <tt>\ref vx_array</tt> of user defined data type with 128 byte element size.
	 * \param [in] nfeatures The input <tt>\ref VX_TYPE_INT32</tt> scalar for nfeatures argument.
	 * \param [in] nOctaveLayers The input <tt>\ref VX_TYPE_INT32</tt> scalar for nOctaveLayers argument.
	 * \param [in] contrastThreshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for contrastThreshold argument.
	 * \param [in] edgeThreshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for edgeThreshold argument.
	 * \param [in] sigma The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for sigma argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_siftCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des, vx_int32 nfeatures, vx_int32 nOctaveLayers, vx_float32 contrastThreshold, vx_float32 edgeThreshold, vx_float32 sigma);

	/*! \brief [Graph] Creates a OpenCV SIFT detector node to detect keypoints and optionally compute descriptors.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] nfeatures The input <tt>\ref VX_TYPE_INT32</tt> scalar for nfeatures argument. Set this to capacity of output_kp to retain best keypoints.
	 * \param [in] nOctaveLayers The input <tt>\ref VX_TYPE_INT32</tt> scalar for nOctaveLayers argument.
	 * \param [in] contrastThreshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for contrastThreshold argument.
	 * \param [in] edgeThreshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for edgeThreshold argument.
	 * \param [in] sigma The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for sigma argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_siftDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_int32 nfeatures, vx_int32 nOctaveLayers, vx_float32 contrastThreshold, vx_float32 edgeThreshold, vx_float32 sigma);

	/*! \brief [Graph] Creates a OpenCV Simple Blob detector node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format.
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_simpleBlobDetector(vx_graph graph, vx_image input, vx_array output_kp, vx_image mask);

	/*! \brief [Graph] Creates a OpenCV Star feature detector node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format.
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [in] maxSize The input <tt>\ref VX_TYPE_INT32</tt> scalar for maxSize argument.
	 * \param [in] responseThreshold The input <tt>\ref VX_TYPE_INT32</tt> scalar for responseThreshold argument.
	 * \param [in] lineThresholdProjected The input <tt>\ref VX_TYPE_INT32</tt> scalar for lineThresholdProjected argument.
	 * \param [in] lineThresholdBinarized The input <tt>\ref VX_TYPE_INT32</tt> scalar for lineThresholdBinarized argument.
	 * \param [in] suppressNonmaxSize The input <tt>\ref VX_TYPE_INT32</tt> scalar for suppressNonmaxSize argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_starFeatureDetector(vx_graph graph, vx_image input, vx_array output_kp, vx_image mask, vx_int32 maxSize, vx_int32 responseThreshold, vx_int32 lineThresholdProjected, vx_int32 lineThresholdBinarized, vx_int32 suppressNonmaxSize);

	/*! \brief [Graph] Creates a OpenCV SURF Compute node to compute descriptor from specified keypoints.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [in] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [out] output_des The output descriptors <tt>\ref vx_array</tt> of user defined data type with 64/128 byte element size depending upon extended argument.
	 * \param [in] hessianThreshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for hessianThreshold argument.
	 * \param [in] nOctaves The input <tt>\ref VX_TYPE_INT32</tt> scalar for nOctaves argument.
	 * \param [in] nOctaveLayers The input <tt>\ref VX_TYPE_INT32</tt> scalar for nOctaveLayers argument.
	 * \param [in] extended The input <tt>\ref VX_TYPE_BOOL</tt> scalar for extended argument.
	 * \param [in] upright The input <tt>\ref VX_TYPE_BOOL</tt> scalar for upright argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_surfCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des, vx_float32 hessianThreshold, vx_int32 nOctaves, vx_int32 nOctaveLayers, vx_bool extended, vx_bool upright);

	/*! \brief [Graph] Creates a OpenCV SURF detector node to detect keypoints and optionally compute descriptors.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [in] mask The mask image in <tt>\ref VX_DF_IMAGE_U8</tt> format (optional).
	 * \param [out] output_kp The output keypoints <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>.
	 * \param [out] output_des The output descriptors <tt>\ref vx_array</tt> of user defined data type with 64/128 byte element size depending upon extended argument (optional).
	 * \param [in] hessianThreshold The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar for hessianThreshold argument.
	 * \param [in] nOctaves The input <tt>\ref VX_TYPE_INT32</tt> scalar for nOctaves argument.
	 * \param [in] nOctaveLayers The input <tt>\ref VX_TYPE_INT32</tt> scalar for nOctaveLayers argument.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_surfDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des, vx_float32 hessianThreshold, vx_int32 nOctaves, vx_int32 nOctaveLayers);

	/*! \brief [Graph] Creates a OpenCV Flip function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] FlipCode The input <tt>\ref VX_TYPE_INT32</tt> scalar to set FlipCode.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_flip(vx_graph graph, vx_image input, vx_image output, vx_int32 FlipCode);

	/*! \brief [Graph] Creates a OpenCV transpose function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> or <tt>\ref VX_DF_IMAGE_U16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_transpose(vx_graph graph, vx_image input, vx_image output);

	/*! \brief [Graph] Creates a OpenCV integral function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U32</tt> or <tt>\ref VX_DF_IMAGE_S32</tt> format.
	 * \param [in] sdepth The input <tt>\ref VX_TYPE_INT32</tt> scalar to set sdepth.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_integral(vx_graph graph, vx_image input, vx_image output, vx_int32 sdepth);

	/*! \brief [Graph] Creates a OpenCV norm function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] norm_value The output <tt>\ref VX_TYPE_FLOAT32</tt> scalar norm_value.
	 * \param [in] norm_type The input <tt>\ref VX_TYPE_INT32</tt> scalar to set norm_type.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_norm(vx_graph graph, vx_image input, vx_float32 norm_value, vx_int32 norm_type);

	/*! \brief [Graph] Creates a OpenCV countNonZero function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] non_zero The output <tt>\ref VX_TYPE_INT32</tt> scalar non_zero.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_countNonZero(vx_graph graph, vx_image input, vx_int32 non_zero);

	/*! \brief [Graph] Creates a OpenCV Multiply function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] scale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set scale.
	 * \param [in] dtype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dtype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_multiply(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output, vx_float32 scale, vx_int32 dtype);

	/*! \brief [Graph] Creates a OpenCV Divide function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] scale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set scale.
	 * \param [in] dtype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dtype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_divide(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output, vx_float32 scale, vx_int32 dtype);

	/*! \brief [Graph] Creates a OpenCV ADD function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_add(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output);

	/*! \brief [Graph] Creates a OpenCV Subtract function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_subtract(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output);

	/*! \brief [Graph] Creates a OpenCV absdiff function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_absDiff(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output);

	/*! \brief [Graph] Creates a OpenCV addWeighted function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] aplha The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set aplha.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] beta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set beta.
	 * \param [in] gamma The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set gamma.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] dtype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dtype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_addWeighted(vx_graph graph, vx_image input_1, vx_float32 aplha, vx_image input_2, vx_float32 beta, vx_float32 gamma, vx_image output, vx_int32 dtype);

	/*! \brief [Graph] Creates a OpenCV adaptiveThreshold function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] maxValue The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set maxValue.
	 * \param [in] adaptiveMethod The input <tt>\ref VX_TYPE_INT32</tt> scalar to set adaptiveMethod.
	 * \param [in] thresholdType The input <tt>\ref VX_TYPE_INT32</tt> scalar to set thresholdType.
	 * \param [in] blockSize The input <tt>\ref VX_TYPE_INT32</tt> scalar to set blockSize.
	 * \param [in] c The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set c.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_adaptiveThreshold(vx_graph graph, vx_image input, vx_image output, vx_float32 maxValue, vx_int32 adaptiveMethod, vx_int32 thresholdType, vx_int32 blockSize, vx_float32 c);

	/*! \brief [Graph] Creates a OpenCV threshold function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] thresh The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set thresh.
	 * \param [in] maxVal The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set maxVal.
	 * \param [in] type The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set type.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_threshold(vx_graph graph, vx_image input, vx_image output, vx_float32 thresh, vx_float32 maxVal, vx_int32 type);

	/*! \brief [Graph] Creates a OpenCV distanceTransform function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_distanceTransform(vx_graph graph, vx_image input, vx_image output);

	/*! \brief [Graph] Creates a OpenCV cvtColor function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] CODE The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set FlipCode.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_cvtColor(vx_graph graph, vx_image input, vx_image output, vx_uint32 CODE);

	/*! \brief [Graph] Creates a OpenCV fastNlMeansDenoising function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] h The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set h.
	 * \param [in] template_ws The input <tt>\ref VX_TYPE_INT32</tt> scalar to set template_ws.
	 * \param [in] search_ws The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set search_ws.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_fastNlMeansDenoising(vx_graph graph, vx_image input, vx_image output, vx_float32 h, vx_int32 template_ws, vx_int32 search_ws);

	/*! \brief [Graph] Creates a OpenCV fastNlMeansDenoisingColored function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_RGB</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] h The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set h.
	 * \param [in] h_color The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set h_color.
	 * \param [in] template_ws The input <tt>\ref VX_TYPE_INT32</tt> scalar to set template_ws.
	 * \param [in] search_ws The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set search_ws.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_fastNlMeansDenoisingColored(vx_graph graph, vx_image input, vx_image output, vx_float32 h, vx_float32 h_color, vx_int32 template_ws, vx_int32 search_ws);

	/*! \brief [Graph] Creates a OpenCV Resize function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] Size_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Size_X.
	 * \param [in] Size_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Size_Y.
	 * \param [in] FX The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set FX.
	 * \param [in] FY The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set FY.
	 * \param [in] interpolation The input <tt>\ref VX_TYPE_INT32</tt> scalar to set interpolation.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_resize(vx_graph graph, vx_image input, vx_image output, vx_int32 Size_X, vx_int32 Size_Y, vx_float32 FX, vx_float32 FY, vx_int32 interpolation);

	/*! \brief [Graph] Creates a OpenCV pyrup function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] Swidth The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set Swidth.
	 * \param [in] Sheight The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set Sheight.
	 * \param [in] bordertype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set bordertype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_pyrUp(vx_graph graph, vx_image input, vx_image output, vx_uint32 Swidth, vx_uint32 Sheight, vx_int32 bordertype);

	/*! \brief [Graph] Creates a OpenCV pyrdown function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] Swidth The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set Swidth.
	 * \param [in] Sheight The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set Sheight.
	 * \param [in] bordertype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set bordertype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_pyrDown(vx_graph graph, vx_image input, vx_image output, vx_uint32 Swidth, vx_uint32 Sheight, vx_int32 bordertype);

	/*! \brief [Graph] Creates a OpenCV buildPyramid function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output Pyramid.
	 * \param [in] maxLevel The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set maxLevel.
	 * \param [in] border The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_buildPyramid(vx_graph graph, vx_image input, vx_pyramid output, vx_uint32 maxLevel, vx_uint32 border);

	/*! \brief [Graph] Creates a OpenCV buildOpticalFlowPyramid function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output Pyramid.
	 * \param [in] S_width The input <tt>\ref VX_TYPE_INT32</tt> scalar to set S_width.
	 * \param [in] S_height The input <tt>\ref VX_TYPE_INT32</tt> scalar to set S_height.
	 * \param [in] WinSize The input <tt>\ref VX_TYPE_INT32</tt> scalar to set WinSize.
	 * \param [in] WithDerivatives The input <tt>\ref VX_TYPE_BOOL</tt> scalar to set withDerivatives.
	 * \param [in] Pyr_border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Pyr_border.
	 * \param [in] derviBorder The input <tt>\ref VX_TYPE_INT32</tt> scalar to set derviBorder.
	 * \param [in] tryReuse The input <tt>\ref VX_TYPE_BOOL</tt> scalar to set tryReuseInputImage.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_buildOpticalFlowPyramid(vx_graph graph, vx_image input, vx_pyramid output, vx_uint32 S_width, vx_uint32 S_height, vx_int32 WinSize, vx_bool WithDerivatives, vx_int32 Pyr_border, vx_int32 derviBorder, vx_bool tryReuse);

	/*! \brief [Graph] Creates a OpenCV Dilate function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] Kernel The input <tt>\ref vx_matrix</tt> matrix to set Kernel.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_X.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_Y.
	 * \param [in] iterations The input <tt>\ref VX_TYPE_INT32</tt> scalar to set iterations.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_dilate(vx_graph graph, vx_image input, vx_image output, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 iterations, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV Erode function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] Kernel The input <tt>\ref vx_matrix</tt> matrix to set Kernel.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_X.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_Y.
	 * \param [in] iterations The input <tt>\ref VX_TYPE_INT32</tt> scalar to set iterations.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_erode(vx_graph graph, vx_image input, vx_image output, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 iterations, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV warpAffine function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] M The input <tt>\ref vx_matrix</tt> matrix to set M.
	 * \param [in] Size_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Size_X.
	 * \param [in] Size_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Size_Y.
	 * \param [in] flags The input <tt>\ref VX_TYPE_INT32</tt> scalar to set flags.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_warpAffine(vx_graph graph, vx_image input, vx_image output, vx_matrix M, vx_int32 Size_X, vx_int32 Size_Y, vx_int32 flags, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV warpPerspective function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] M The input <tt>\ref vx_matrix</tt> matrix to set M.
	 * \param [in] Size_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Size_X.
	 * \param [in] Size_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Size_Y.
	 * \param [in] flags The input <tt>\ref VX_TYPE_INT32</tt> scalar to set flags.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_warpPerspective(vx_graph graph, vx_image input, vx_image output, vx_matrix M, vx_int32 Size_X, vx_int32 Size_Y, vx_int32 flags, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV morphologyEX function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] OP The input <tt>\ref VX_TYPE_INT32</tt> scalar to set OP.
	 * \param [in] Kernel The input <tt>\ref vx_matrix</tt> matrix to set Kernel.
	 * \param [in] Anchor_X The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_X.
	 * \param [in] Anchor_Y The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Anchor_Y.
	 * \param [in] iterations The input <tt>\ref VX_TYPE_INT32</tt> scalar to set iterations.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_morphologyEX(vx_graph graph, vx_image input, vx_image output, vx_int32 OP, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 iterations, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV Bitwise And function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseAnd(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output);

	/*! \brief [Graph] Creates a OpenCV Bitwise NOT function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseNot(vx_graph graph, vx_image input, vx_image output);

	/*! \brief [Graph] Creates a OpenCV Bitwise OR function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseOr(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output);

	/*! \brief [Graph] Creates a OpenCV Bitwise XOR function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseXor(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output);

	/*! \brief [Graph] Creates a OpenCV Canny function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>  format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] threshold1 The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set threshold1.
	 * \param [in] threshold2 The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set threshold2.
	 * \param [in] aperture_size The input <tt>\ref VX_TYPE_INT32</tt> scalar to set aperture_size.
	 * \param [in] L2_Gradient The input <tt>\ref VX_TYPE_BOOL</tt> scalar to set L2_Gradient.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_canny(vx_graph graph, vx_image input, vx_image output, vx_float32 threshold1, vx_float32 threshold2, vx_int32 aperture_size, vx_bool L2_Gradient);

	/*! \brief [Graph] Creates a OpenCV Compare function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input_1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [in] input_2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] cmpop The input value in vx_int32 format.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_compare(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output, vx_int32 cmpop);

	/*! \brief [Graph] Creates a OpenCV convertScaleAbs function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] image_in The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] image_out The output image is as same size and type of input.
	 * \param [in] alpha The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set alpha.
	 * \param [in] beta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set beta.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_convertScaleAbs(vx_graph graph, vx_image image_in, vx_image image_out, vx_float32 alpha, vx_float32 beta);

	/*! \brief [Graph] Creates a OpenCV cornerHarris function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] blocksize The input <tt>\ref VX_TYPE_INT32</tt> scalar to set blocksize.
	 * \param [in] ksize The input <tt>\ref VX_TYPE_INT32</tt> scalar to set ksize.
	 * \param [in] k The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set K.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_cornerHarris(vx_graph graph, vx_image input, vx_image output, vx_int32 blocksize, vx_int32 ksize, vx_float32 k, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV cornerMinEigenVal function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] blockSize The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set blocksize.
	 * \param [in] ksize The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set ksize.
	 * \param [in] border The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_cornerMinEigenVal(vx_graph graph, vx_image input, vx_image output, vx_uint32 blockSize, vx_uint32 ksize, vx_int32 border);

	/*! \brief [Graph] Creates a OpenCV Laplacian function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ddepth The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set ddepth.
	 * \param [in] ksize The input <tt>\ref VX_TYPE_UINT32</tt> scalar to set ksize.
	 * \param [in] scale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set scale.
	 * \param [in] delta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set delta.
	 * \param [in] border_mode The input <tt>\ref VX_TYPE_INT32</tt> scalar to set border_mode.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_laplacian(vx_graph graph, vx_image input, vx_image output, vx_uint32 ddepth, vx_uint32 ksize, vx_float32 scale, vx_float32 delta, vx_int32 border_mode);

	/*! \brief [Graph] Creates a OpenCV Scharr function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ddepth The input <tt>\ref VX_TYPE_INT32</tt> scalar to set ddepth.
	 * \param [in] dx The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dx.
	 * \param [in] dy The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dy.
	 * \param [in] scale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set scale.
	 * \param [in] delta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set delta.
	 * \param [in] bordertype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set bordertype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_scharr(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_int32 dx, vx_int32 dy, vx_float32 scale, vx_float32 delta, vx_int32 bordertype);

	/*! \brief [Graph] Creates a OpenCV Sobel function node.
	 * \ingroup group_opencv
	 * \param [in] graph The reference to the graph.
	 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
	 * \param [out] output The output image is as same size and type of input.
	 * \param [in] ddepth The input <tt>\ref VX_TYPE_INT32</tt> scalar to set ddepth.
	 * \param [in] dx The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dx.
	 * \param [in] dy The input <tt>\ref VX_TYPE_INT32</tt> scalar to set dy.
	 * \param [in] Ksize The input <tt>\ref VX_TYPE_INT32</tt> scalar to set Ksize.
	 * \param [in] scale The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set scale.
	 * \param [in] delta The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar to set delta.
	 * \param [in] bordertype The input <tt>\ref VX_TYPE_INT32</tt> scalar to set bordertype.
	 * \return <tt>\ref vx_node</tt>.
	 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>*/
	extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_sobel(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_int32 dx, vx_int32 dy, vx_int32 Ksize, vx_float32 scale, vx_float32 delta, vx_int32 bordertype);

#ifdef __cplusplus
}
#endif
