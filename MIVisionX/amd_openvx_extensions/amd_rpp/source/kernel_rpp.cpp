/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "internal_publishKernels.h"
#include "vx_ext_rpp.h"

vx_uint32 getGraphAffinity(vx_graph graph)
{
    AgoTargetAffinityInfo affinity;
    vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    ;
    if (affinity.device_type != AGO_TARGET_AFFINITY_GPU && affinity.device_type != AGO_TARGET_AFFINITY_CPU)
        affinity.device_type = AGO_TARGET_AFFINITY_CPU;
    // std::cerr<<"\n affinity "<<affinity.device_type;
    return affinity.device_type;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array alpha, vx_array beta, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)alpha,
            (vx_reference)beta,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESSBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array gamma, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)gamma,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlendbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array alpha, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)alpha,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BLENDBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlurbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BLURBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ContrastbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array min, vx_array max, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)min,
            (vx_reference)max,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CONTRASTBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PixelatebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_PIXELATEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_JitterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_JITTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SnowbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array snowValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)snowValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SNOWBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NoisebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array noiseProbability, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)noiseProbability,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NOISEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_array numberOfShadows, vx_array maxSizeX, vx_array maxSizeY, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)x2,
            (vx_reference)y2,
            (vx_reference)numberOfShadows,
            (vx_reference)maxSizeX,
            (vx_reference)maxSizeY,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RANDOMSHADOWBATCHPD, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FogbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array fogValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)fogValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FOGBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RainbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array rainValue, vx_array rainWidth, vx_array rainHeight, vx_array rainTransperancy, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)rainValue,
            (vx_reference)rainWidth,
            (vx_reference)rainHeight,
            (vx_reference)rainTransperancy,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RAINBATCHPD, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)x2,
            (vx_reference)y2,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExposurebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array exposureValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)exposureValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_EXPOSUREBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_array alpha, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)alpha,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AddbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ADDBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SubtractbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SUBTRACTBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MAGNITUDEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MultiplybatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MULTIPLYBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PhasebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_PHASEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD, params, 5);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BITWISEANDBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BITWISENOTBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_EXCLUSIVEORBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_INCLUSIVEORBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Histogram(vx_graph graph, vx_image pSrc, vx_array outputHistogram, vx_scalar bins)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)outputHistogram,
            (vx_reference)bins,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HISTOGRAM, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array min, vx_array max, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)min,
            (vx_reference)max,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_THRESHOLDINGBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MaxbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MAXBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MINBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinMaxLoc(vx_graph graph, vx_image pSrc, vx_scalar min, vx_scalar max, vx_scalar minLoc, vx_scalar maxLoc)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)min,
            (vx_reference)max,
            (vx_reference)minLoc,
            (vx_reference)maxLoc,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MINMAXLOC, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MeanStddev(vx_graph graph, vx_image pSrc, vx_scalar mean, vx_scalar stdDev)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)mean,
            (vx_reference)stdDev,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MEANSTDDEV, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FlipbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array flipAxis, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)flipAxis,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FLIPBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZEBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Resizetensor(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_int32 interpolation_type, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_scalar INTERPOLATION_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &interpolation_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)INTERPOLATION_TYPE,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZETENSOR, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)x2,
            (vx_reference)y2,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROPBATCHPD, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RotatebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array angle, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)angle,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ROTATEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array affine, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)affine,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_WARPAFFINEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FisheyebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FISHEYEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array strength, vx_array zoom, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)strength,
            (vx_reference)zoom,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTIONBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ScalebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array percentage, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)percentage,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SCALEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array perspective, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)perspective,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DilatebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_DILATEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ErodebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ERODEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HuebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array hueShift, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)hueShift,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HUEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SaturationbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array saturationFactor, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)saturationFactor,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SATURATIONBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array adjustmentValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)adjustmentValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_VignettebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array stdDev, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_VIGNETTEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array extractChannelNumber, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)extractChannelNumber,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pSrc3, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pSrc3,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array lutPtr, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)lutPtr,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LOOKUPTABLEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BOXFILTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SobelbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array sobelType, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)sobelType,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SOBELBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MEDIANFILTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernel, vx_array kernelWidth, vx_array kernelHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernel,
            (vx_reference)kernelWidth,
            (vx_reference)kernelHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array stdDev, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NONLINEARFILTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array stdDev, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LaplacianImagePyramid(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_scalar stdDev, vx_scalar kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CannyEdgeDetector(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array max, vx_array min, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)max,
            (vx_reference)min,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CANNYEDGEDETECTOR, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HarrisCornerDetector(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight,vx_image pDst, vx_array gaussianKernelSize, vx_array stdDev, vx_array kernelSize, vx_array kValue, vx_array threshold, vx_array nonMaxKernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)gaussianKernelSize,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)kValue,
            (vx_reference)threshold,
            (vx_reference)nonMaxKernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HARRISCORNERDETECTOR, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FastCornerDetector(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array noOfPixels, vx_array threshold, vx_array nonMaxKernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)noOfPixels,
            (vx_reference)threshold,
            (vx_reference)nonMaxKernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FASTCORNERDETECTOR, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_remap(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array rowRemap, vx_array colRemap, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)rowRemap,
            (vx_reference)colRemap,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_REMAP, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorAdd(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORADD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorSubtract(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORSUBTRACT, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorMultiply(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORMULTIPLY, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorMatrixMultiply(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_array tensorDimensionValues1, vx_array tensorDimensionValues2)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensionValues1,
            (vx_reference)tensorDimensionValues2,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORMATRIXMULTIPLY, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorLookup(vx_graph graph, vx_array pSrc, vx_array pDst, vx_array lutPtr, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)lutPtr,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORLOOKUP, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTwistbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array alpha, vx_array beta, vx_array hue, vx_array sat, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)alpha,
            (vx_reference)beta,
            (vx_reference)hue,
            (vx_reference)sat,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_COLORTWISTBATCHPD, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CropMirrorNormalizebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)mean,
            (vx_reference)std_dev,
            (vx_reference)flip,
            (vx_reference)chnShift,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD, params, 14);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CropPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CROPPD, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropMirrorPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_array mirror, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)x2,
            (vx_reference)y1,
            (vx_reference)y2,
            (vx_reference)mirror,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROPMIRRORPD, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeMirrorNormalizeTensor(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) srcImgWidth,
            (vx_reference) srcImgHeight,
            (vx_reference) pDst,
            (vx_reference) dstImgWidth,
            (vx_reference) dstImgHeight,
            (vx_reference) mean,
            (vx_reference) std_dev,
            (vx_reference) flip,
            (vx_reference) chnShift,
            (vx_reference) NBATCHSIZE,
            (vx_reference) DEV_TYPE
        };
        node = createNode(graph, VX_KERNEL_RPP_RESIZEMIRRORNORMALIZETENSOR, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CopybatchPD(vx_graph graph, vx_image pSrc, vx_image pDst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_COPYBATCHPD, params, 3);
    }
    return node;
}

//Creating node for Pixelate effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_NopbatchPD(vx_graph graph, vx_image pSrc, vx_image pDst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NOPBATCHPD, params, 3);
    }
    return node;
}

VX_API_CALL vx_node VX_API_CALL vxExtrppNode_SequenceRearrangebatchPD(vx_graph graph, vx_image pSrc, vx_image pDst, vx_array pNewOrder, vx_uint32 newSequenceLength, vx_uint32 sequenceLength, vx_uint32 sequenceCount)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_scalar newSequenceLengthScalar = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &newSequenceLength);
        vx_scalar sequenceLengthScalar = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &sequenceLength);
        vx_scalar sequenceCountScalar = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &sequenceCount);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) pNewOrder,
            (vx_reference) newSequenceLengthScalar,
            (vx_reference) sequenceLengthScalar,
            (vx_reference) sequenceCountScalar,
            (vx_reference) deviceType
        };
        node = createNode(graph, VX_KERNEL_RPP_SEQUENCEREARRANGEBATCHPD, params, 7);
    }
    return node;
}

//tensor

VX_API_ENTRY vx_node VX_API_CALL vxExtRppBrightness(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAlpha, vx_array pBeta, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAlpha,
            (vx_reference)pBeta,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESS, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppCopy(vx_graph graph, vx_tensor pSrc, vx_tensor pDst) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_COPY, params, 3);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppCropMirrorNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pMultiplier, vx_array pOffset, vx_array pMirror, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pMultiplier,
            (vx_reference)pOffset,
            (vx_reference)pMirror,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_CROPMIRRORNORMALIZE, params, 10);
    }
    return node;
}

VX_API_CALL vx_node VX_API_CALL vxExtRppNop(vx_graph graph, vx_tensor pSrc, vx_tensor pDst) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NOP, params, 3);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZE, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppBlend(vx_graph graph, vx_tensor pSrc1, vx_tensor pSrc2, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pShift, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pShift,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_BLEND, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppBlur(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pKernelSize, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pKernelSize,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_BLUR, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppColorTwist(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAlpha, vx_array pBeta, vx_array pHue, vx_array pSat, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAlpha,
            (vx_reference)pBeta,
            (vx_reference)pHue,
            (vx_reference)pSat,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_COLORTWIST, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppContrast(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pContrastFactor, vx_array pContrastCenter, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pContrastFactor,
            (vx_reference)pContrastCenter,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_CONTRAST, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppColorTemperature(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAdjustValue, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAdjustValue,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATURE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppCrop(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_CROP, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppExposure(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pExposureFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pExposureFactor,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_EXPOSURE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppFishEye(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_FISHEYE, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppFlip(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pHorizontalFlag, vx_array pVerticalFlag, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pHorizontalFlag,
            (vx_reference)pVerticalFlag,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_FLIP, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppFog(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pFogValue, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pFogValue,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_FOG, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppGammaCorrection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pGamma, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pGamma,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTION, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppGlitch(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,vx_array pXoffsetR, vx_array pYoffsetR, vx_array pXoffsetG, vx_array pYoffsetG, vx_array pXoffsetB, vx_array pYoffsetB, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pXoffsetR,
            (vx_reference)pYoffsetR,
            (vx_reference)pXoffsetG,
            (vx_reference)pYoffsetG,
            (vx_reference)pXoffsetB,
            (vx_reference)pYoffsetB,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_GLITCH, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppHue(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pHueShift, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pHueShift,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_HUE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppJitter(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pKernelSize, vx_scalar seed, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph); 
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pKernelSize,
            (vx_reference)seed,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_JITTER, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppLensCorrection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pStrength, vx_array pZoom, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pStrength,
            (vx_reference)pZoom,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTION, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppNoise(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pNoiseProb, vx_array pSaltProb, vx_array pSaltValue, vx_array pPepperValue, vx_scalar seed,vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pNoiseProb,
            (vx_reference)pSaltProb,
            (vx_reference)pSaltValue,
            (vx_reference)pPepperValue,
            (vx_reference)seed,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NOISE, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppPixelate(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_PIXELATE, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppRain(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pRainValue, vx_array pRainWidth, vx_array pRainHeight, vx_array pRainTransperancy, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pRainValue,
            (vx_reference)pRainWidth,
            (vx_reference)pRainHeight,
            (vx_reference)pRainTransperancy,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RAIN, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResizeCrop(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi,vx_tensor pCropTensor, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pCropTensor,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROP, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResizeCropMirror(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_array pMirror,vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)pMirror,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROPMIRROR, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResizeMirrorNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,vx_array pDstWidth,vx_array pDstHeight, vx_scalar interpolationType, vx_array pMean, vx_array pStdDev, vx_array pMirror, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)interpolationType,
            (vx_reference)pMean,
            (vx_reference)pStdDev,
            (vx_reference)pMirror,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZEMIRRORNORMALIZE, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppRotate(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAngle, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAngle,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_ROTATE, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSaturation(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pSaturationFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pSaturationFactor,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SATURATION, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSnow(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pSnowValue, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pSnowValue,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SNOW, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppVignette(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pStdDev, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pStdDev,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_VIGNETTE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppWarpAffine(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAffineArray, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAffineArray,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_WARPAFFINE, params, 9);
    }
    return node;
}

VX_API_CALL vx_node VX_API_CALL vxExtRppSequenceRearrange(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_array pNewOrder, vx_scalar layout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)pNewOrder,
            (vx_reference)layout,
            (vx_reference)deviceType
        };
        node = createNode(graph, VX_KERNEL_RPP_SEQUENCEREARRANGE, params, 5);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppPreemphasisFilter(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pPreemphCoeff, vx_scalar borderType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pPreemphCoeff,
            (vx_reference)borderType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_PREEMPHASISFILTER, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSpectrogram(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_array windowFunction, vx_scalar centerWindows, vx_scalar reflectPadding, vx_scalar spectrogramLayout,
                                                     vx_scalar power, vx_scalar nfft, vx_scalar windowLength, vx_scalar windowStep) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)windowFunction,
            (vx_reference)centerWindows,
            (vx_reference)reflectPadding,
            (vx_reference)spectrogramLayout,
            (vx_reference)power,
            (vx_reference)nfft,
            (vx_reference)windowLength,
            (vx_reference)windowStep,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SPECTROGRAM, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppNonSilentRegionDetection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pBegin, vx_tensor pLength, vx_scalar cutOffDB, vx_scalar referencePower, vx_scalar windowLength, vx_scalar resetInterval) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pBegin,
            (vx_reference)pLength,
            (vx_reference)cutOffDB,
            (vx_reference)referencePower,
            (vx_reference)windowLength,
            (vx_reference)resetInterval,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NONSILENTREGIONDETECTION, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSlice(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_tensor pAnchor, vx_tensor pShape,
                                               vx_array pFillValue, vx_scalar policy, vx_scalar inputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)pAnchor,
            (vx_reference)pShape,
            (vx_reference)pFillValue,
            (vx_reference)policy,
            (vx_reference)inputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SLICE, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppDownmix(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_tensor pSrcRoi) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)pSrcRoi,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_DOWNMIX, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppToDecibels(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar cutOffDB, vx_scalar multiplier, vx_scalar referenceMagnitude, vx_scalar inputLayout, vx_scalar outputLayout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)cutOffDB,
            (vx_reference)multiplier,
            (vx_reference)referenceMagnitude,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TODECIBELS, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResample(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_tensor pSrcRoi, vx_tensor pDstRoi,
                                                  vx_array pInRateTensor, vx_tensor pOutRateTensor, vx_scalar quality) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)pSrcRoi,
            (vx_reference)pDstRoi,
            (vx_reference)pOutRateTensor,
            (vx_reference)pInRateTensor,
            (vx_reference)quality,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESAMPLE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppTensorMulScalar(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_scalar scalarValue) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)scalarValue,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TENSORMULSCALAR, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppTensorAddTensor(vx_graph graph, vx_tensor pSrc1, vx_tensor pSrc2, vx_tensor pDst, vx_tensor pSrcRoi, vx_tensor pDstRoi) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)pSrcRoi,
            (vx_reference)pDstRoi,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TENSORADDTENSOR, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi,
                                                   vx_scalar axis_mask, vx_array pMean, vx_array pStddev, vx_scalar computeMeanAndStdDev,
                                                   vx_scalar scale, vx_scalar shift, vx_scalar inputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)axis_mask,
            (vx_reference)pMean,
            (vx_reference)pStddev,
            (vx_reference)computeMeanAndStdDev,
            (vx_reference)scale,
            (vx_reference)shift,
            (vx_reference)inputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NORMALIZE, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppMelFilterBank(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_scalar freqHigh, vx_scalar freqLow, vx_scalar melFormula,
                                                       vx_scalar nfilter, vx_scalar normalize, vx_scalar sampleRate, vx_scalar inputLayout, vx_scalar outputLayout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)freqHigh,
            (vx_reference)freqLow,
            (vx_reference)melFormula,
            (vx_reference)nfilter,
            (vx_reference)normalize,
            (vx_reference)sampleRate,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_MELFILTERBANK, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppTranspose(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,
                                                   vx_array pPerm, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pPerm,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TRANSPOSE, params, 8);
    }
    return node;
}

RpptDataType getRpptDataType(vx_enum vxDataType) {
    switch(vxDataType) {
        case vx_type_e::VX_TYPE_FLOAT32:
            return RpptDataType::F32;
        case vx_type_e::VX_TYPE_FLOAT16:
            return RpptDataType::F16;
        case vx_type_e::VX_TYPE_INT8:
            return RpptDataType::I8;
        default:
            return RpptDataType::U8;
    }
}

size_t getDataTypeSize(vx_enum vxDataType) {
    switch (vxDataType) {
#if defined(AMD_FP16_SUPPORT)
        case vx_type_e::VX_TYPE_FLOAT16:
            return sizeof(vx_float16);
#endif
        case vx_type_e::VX_TYPE_FLOAT32:
            return sizeof(vx_float32);
        case vx_type_e::VX_TYPE_INT8:
            return sizeof(vx_int8);
        case vx_type_e::VX_TYPE_INT16:
            return sizeof(vx_int16);
        case vx_type_e::VX_TYPE_INT32:
            return sizeof(vx_int32);
        case vx_type_e::VX_TYPE_UINT8:
            return sizeof(vx_uint8);
        case vx_type_e::VX_TYPE_UINT32:
            return sizeof(vx_uint32);
        default:
            throw std::runtime_error("Invalid datatype.");
    }
}

void fillDescriptionPtrfromDims(RpptDescPtr &descPtr, vxTensorLayout layout, size_t *tensorDims) {
    switch(layout) {
        case vxTensorLayout::VX_NHWC: {
            descPtr->n = tensorDims[0];
            descPtr->h = tensorDims[1];
            descPtr->w = tensorDims[2];
            descPtr->c = tensorDims[3];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->c * descPtr->w;
            descPtr->strides.wStride = descPtr->c;
            descPtr->strides.cStride = 1;
            descPtr->layout = RpptLayout::NHWC;
            break; 
        }
        case vxTensorLayout::VX_NCHW: {
            descPtr->n = tensorDims[0];
            descPtr->h = tensorDims[2];
            descPtr->w = tensorDims[3];
            descPtr->c = tensorDims[1];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.cStride = descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->w;
            descPtr->strides.wStride = 1;
            descPtr->layout = RpptLayout::NCHW;
            break;
        }
        case vxTensorLayout::VX_NFHWC: {
            descPtr->n = tensorDims[0] * tensorDims[1];
            descPtr->h = tensorDims[2];
            descPtr->w = tensorDims[3];
            descPtr->c = tensorDims[4];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->c * descPtr->w;
            descPtr->strides.wStride = descPtr->c;
            descPtr->strides.cStride = 1;
            descPtr->layout = RpptLayout::NHWC;
            break;
        }
        case vxTensorLayout::VX_NFCHW: {
            descPtr->n = tensorDims[0] * tensorDims[1];
            descPtr->h = tensorDims[3];
            descPtr->w = tensorDims[4];
            descPtr->c = tensorDims[2];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.cStride = descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->w;
            descPtr->strides.wStride = 1;
            descPtr->layout = RpptLayout::NCHW;
            break;
        }
        default: {
            throw std::runtime_error("Invalid layout value in fillDescriptionPtrfromDims.");
        }
    }
}

void fillAudioDescriptionPtrFromDims(RpptDescPtr &descPtr, size_t *maxTensorDims, vxTensorLayout layout) {
    descPtr->n = maxTensorDims[0];
    descPtr->h = maxTensorDims[1];
    descPtr->w = maxTensorDims[2];
    descPtr->c = 1;
    descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
    descPtr->strides.hStride = descPtr->c * descPtr->w;
    descPtr->strides.wStride = descPtr->c;
    descPtr->strides.cStride = 1;
    descPtr->numDims = 4;
    if(tensorLayoutMapping.find(layout) != tensorLayoutMapping.end()) {
        descPtr->layout = tensorLayoutMapping.at(layout);
    } else {
        throw std::runtime_error("Invalid layout");
    }
}

void fillGenericDescriptionPtrfromDims(RpptGenericDescPtr &genericDescPtr, vxTensorLayout layout, size_t *tensorDims) {
    if(tensorLayoutMapping.find(layout) != tensorLayoutMapping.end())
        genericDescPtr->layout = tensorLayoutMapping.at(layout);
    else
        throw std::runtime_error("Invalid layout value in fillGenericDescriptionPtrfromDims");
    switch(layout) {
        case vxTensorLayout::VX_NHWC:
        case vxTensorLayout::VX_NCHW: {
            genericDescPtr->numDims = 4;
            genericDescPtr->dims[0] = tensorDims[0];
            genericDescPtr->dims[1] = tensorDims[1];
            genericDescPtr->dims[2] = tensorDims[2];
            genericDescPtr->dims[3] = tensorDims[3];
            genericDescPtr->strides[0] = genericDescPtr->dims[1] * genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[1] = genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[2] = genericDescPtr->dims[3];
            genericDescPtr->strides[3] = 1;
            break;
        }
        case vxTensorLayout::VX_NHW:
        case vxTensorLayout::VX_NFT:
        case vxTensorLayout::VX_NTF: {
            genericDescPtr->dims[0] = tensorDims[0];
            genericDescPtr->dims[1] = tensorDims[1];
            genericDescPtr->dims[2] = tensorDims[2];
            genericDescPtr->dims[3] = 1;
            if(genericDescPtr->dims[2] == 1)
                genericDescPtr->numDims = 2;
            else
                genericDescPtr->numDims = 3;
            genericDescPtr->strides[0] = genericDescPtr->dims[1] * genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[1] = genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[2] = genericDescPtr->dims[3];
            break;
        }
        default: {
            throw std::runtime_error("Invalid layout value in fillGenericDescriptionPtrfromDims.");
        }
    }
}

// utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) != VX_SUCCESS)
    {
        return NULL;
    }
    vx_kernel kernel = vxGetKernelByEnum(context, kernelEnum);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++)
            {
                if (params[p])
                {
                    status = vxSetParameterByIndex(node, p, params[p]);
                    if (status != VX_SUCCESS)
                    {
                        char kernelName[VX_MAX_KERNEL_NAME];
                        vxQueryKernel(kernel, VX_KERNEL_NAME, kernelName, VX_MAX_KERNEL_NAME);
                        vxAddLogEntry((vx_reference)graph, status, "createNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
                        vxReleaseNode(&node);
                        node = 0;
                        break;
                    }
                }
            }
        }
        else
        {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to create node with kernel enum %d\n", kernelEnum);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else
    {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to retrieve kernel enum %d\n", kernelEnum);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

vx_status createRPPHandle(vx_node node, vxRppHandle **pHandle, Rpp32u batchSize, Rpp32u deviceType) {
    vxRppHandle *handle = NULL;
    STATUS_ERROR_CHECK(vxGetModuleHandle(node, OPENVX_KHR_RPP, (void **)&handle));
    vx_uint32 cpu_num_threads;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS, &cpu_num_threads, sizeof(cpu_num_threads)));

    if (handle) {
        handle->count++;
    } else {
        handle = new vxRppHandle;
        memset(handle, 0, sizeof(*handle));
        handle->count = 1;
        
        if (deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
            STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &handle->cmdq, sizeof(handle->cmdq)));
            rppCreate(&handle->rppHandle, batchSize, 0, handle->cmdq, RppBackend::RPP_OCL_BACKEND);
#elif ENABLE_HIP
            STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &handle->hipstream, sizeof(handle->hipstream)));
            rppCreate(&handle->rppHandle, batchSize, 0, handle->hipstream, RppBackend::RPP_HIP_BACKEND);
#endif
        } else if (deviceType == AGO_TARGET_AFFINITY_CPU) {
            rppCreate(&handle->rppHandle, batchSize, cpu_num_threads, NULL, RppBackend::RPP_HOST_BACKEND);
        }
        
        STATUS_ERROR_CHECK(vxSetModuleHandle(node, OPENVX_KHR_RPP, handle));
    }
    *pHandle = handle;
    return VX_SUCCESS;
}

vx_status releaseRPPHandle(vx_node node, vxRppHandle *handle, Rpp32u deviceType) {
    handle->count--;
    if (handle->count == 0) {
        if(deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
            rppDestroy(handle->rppHandle);
#endif   
        } else if (deviceType == AGO_TARGET_AFFINITY_CPU) {
            rppDestroy(handle->rppHandle);
        }

        delete handle;
        STATUS_ERROR_CHECK(vxSetModuleHandle(node, OPENVX_KHR_RPP, NULL));
    }
    return VX_SUCCESS;
}
