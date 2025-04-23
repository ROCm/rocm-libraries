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

#include "vx_ext_opencv.h"
#include "internal_publishKernels.h"

vx_node vxCreateNodeByStructure(vx_graph graph,
                                vx_enum kernelenum,
                                vx_reference params[],
                                vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, kernelenum);
    if (kernel)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++)
            {
                if (params[p])
                {
                    status = vxSetParameterByIndex(node,
                                                   p,
                                                   params[p]);
                    if (status != VX_SUCCESS)
                    {
                        vxAddLogEntry((vx_reference)graph, status, "Kernel %d Parameter %u is invalid.\n", kernelenum, p);
                        vxReleaseNode(&node);
                        node = 0;
                        break;
                    }
                }
            }
        }
        else
        {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "Failed to create node with kernel enum %d\n", kernelenum);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else
    {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "failed to retrieve kernel enum %d\n", kernelenum);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

/************************************************************************************************************
OpenCV Absdiff C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_absDiff(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ABSDIFF,
                                   params,
                                   dimof(params));
}

/************************************************************************************************************
OpenCV Add C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_add(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ADD,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Subtract C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_subtract(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SUBTRACT,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Bitwise_AND C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseAnd(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BITWISE_AND,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Bitwise_NOT C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseNot(vx_graph graph, vx_image input_1, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BITWISE_NOT,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Bitwise_OR C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseOr(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BITWISE_OR,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Bitwise_XOR C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bitwiseXor(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BITWISE_XOR,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Transpose C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_transpose(vx_graph graph, vx_image input, vx_image output)
{

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_TRANSPOSE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Compare C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_compare(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output, vx_int32 cmpop)
{

    vx_scalar CMPOP = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &cmpop);

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
        (vx_reference)CMPOP,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_COMPARE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV integral C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_integral(vx_graph graph, vx_image input, vx_image output, vx_int32 sdepth) {

    vx_scalar Sdepth = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &sdepth);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)Sdepth,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_INTEGRAL,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV NORM C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_norm(vx_graph graph, vx_image input, vx_float32 norm_value, vx_int32 norm_type) {

    vx_scalar Norm_value = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &norm_value);
    vx_scalar Norm_type = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &norm_type);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)Norm_value,
        (vx_reference)Norm_type,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_NORM,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV countNonZero C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_countNonZero(vx_graph graph, vx_image input, vx_int32 non_zero) {

    vx_scalar Non_zero = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &non_zero);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)Non_zero,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_COUNT_NON_ZERO,
                                   params,
                                   dimof(params));
}

/************************************************************************************************************
OpenCV flip C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_flip(vx_graph graph, vx_image input, vx_image output, vx_int32 FlipCode)
{

    vx_scalar FLIPCODE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &FlipCode);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)FLIPCODE,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_FLIP,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV MedianBlur C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_medianBlur(vx_graph graph, vx_image input, vx_image output, vx_uint32 ksize)
{

    vx_scalar KSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ksize);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)KSIZE,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_MEDIAN_BLUR,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Boxfilter C function.
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_boxFilter(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_uint32 kwidth, vx_uint32 kheight, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_bool Normalized, vx_int32 Bordertype)
{

    vx_scalar DDEPTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ddepth);
    vx_scalar KWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kwidth);
    vx_scalar KHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kheight);
    vx_scalar A_X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar A_Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar NORM = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &Normalized);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Bordertype);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)DDEPTH,
        (vx_reference)KWIDTH,
        (vx_reference)KHEIGHT,
        (vx_reference)A_X,
        (vx_reference)A_Y,
        (vx_reference)NORM,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BOXFILTER,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Gaussian C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_gaussianBlur(vx_graph graph, vx_image input, vx_image output, vx_uint32 kwidth, vx_uint32 kheight, vx_float32 sigmaX, vx_float32 sigmaY, vx_int32 border_mode)
{

    vx_scalar KWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kwidth);
    vx_scalar KHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kheight);
    vx_scalar SIGMA_X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sigmaX);
    vx_scalar SIGMA_Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sigmaY);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border_mode);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)KWIDTH,
        (vx_reference)KHEIGHT,
        (vx_reference)SIGMA_X,
        (vx_reference)SIGMA_Y,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_GAUSSIAN_BLUR,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Blur C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_blur(vx_graph graph, vx_image input, vx_image output, vx_uint32 kwidth, vx_uint32 kheight, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 Bordertype)
{

    vx_scalar KWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kwidth);
    vx_scalar KHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kheight);
    vx_scalar A_X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar A_Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Bordertype);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)KWIDTH,
        (vx_reference)KHEIGHT,
        (vx_reference)A_X,
        (vx_reference)A_Y,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BLUR,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Bilateral Filter C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_bilateralFilter(vx_graph graph, vx_image input, vx_image output, vx_uint32 d, vx_float32 Sigma_Color, vx_float32 Sigma_Space, vx_int32 border_mode)
{

    vx_scalar D = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &d);
    vx_scalar S_COLOR = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &Sigma_Color);
    vx_scalar S_SPACE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &Sigma_Space);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border_mode);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)D,
        (vx_reference)S_COLOR,
        (vx_reference)S_SPACE,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BILATERAL_FILTER,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
OpenCV Sobel C function.
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_sobel(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_int32  dx, vx_int32 dy, vx_int32 Ksize, vx_float32 scale, vx_float32 delta, vx_int32 bordertype)
{

    vx_scalar DDEPTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ddepth);
    vx_scalar DX = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dx);
    vx_scalar DY = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dy);
    vx_scalar KSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Ksize);
    vx_scalar SCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scale);
    vx_scalar DELTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &delta);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &bordertype);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)DDEPTH,
        (vx_reference)DX,
        (vx_reference)DY,
        (vx_reference)KSIZE,
        (vx_reference)SCALE,
        (vx_reference)DELTA,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SOBEL,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
 OpenCV convertScaleAbs C Function function.
 *************************************************************************************************************/

extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_convertScaleAbs(vx_graph graph, vx_image image_in, vx_image image_out, vx_float32 alpha, vx_float32 beta)
{

    vx_scalar ALPHA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &alpha);
    vx_scalar BETA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &beta);

    vx_reference params[] = {
        (vx_reference)image_in,
        (vx_reference)image_out,
        (vx_reference)ALPHA,
        (vx_reference)BETA,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_CONVERTSCALEABS,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
AddWeighted Node function.
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_addWeighted(vx_graph graph, vx_image imput_1, vx_float32 aplha, vx_image input_2, vx_float32 beta, vx_float32 gamma, vx_image output, vx_int32 dtype)
{

    vx_scalar ALPHA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &aplha);
    vx_scalar BETA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &beta);
    vx_scalar GAMMA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &gamma);
    vx_scalar DTYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dtype);

    vx_reference params[] = {
        (vx_reference)imput_1,
        (vx_reference)ALPHA,
        (vx_reference)input_2,
        (vx_reference)BETA,
        (vx_reference)GAMMA,
        (vx_reference)output,
        (vx_reference)DTYPE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ADDWEIGHTED,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Canny C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_canny(vx_graph graph, vx_image input, vx_image output, vx_float32 threshold1, vx_float32 threshold2, vx_int32 aperture_size, vx_bool L2_Gradient)
{

    vx_scalar THRESH1 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &threshold1);
    vx_scalar THRESH2 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &threshold2);
    vx_scalar A_SIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &aperture_size);
    vx_scalar L2_GRA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &L2_Gradient);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)THRESH1,
        (vx_reference)THRESH2,
        (vx_reference)A_SIZE,
        (vx_reference)L2_GRA,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_CANNY,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
cornerMinEigenVal C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_cornerMinEigenVal(vx_graph graph, vx_image input, vx_image output, vx_uint32 blockSize, vx_uint32 ksize, vx_int32 border) {

    vx_scalar BlockSize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &blockSize);
    vx_scalar KSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &ksize);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)BlockSize,
        (vx_reference)KSIZE,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_CORNER_MIN_EIGEN_VAL,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
cornerHarris C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_cornerHarris(vx_graph graph, vx_image input, vx_image output, vx_int32 blocksize, vx_int32 ksize, vx_float32 k, vx_int32 border) {

    vx_scalar Blocksize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &blocksize);
    vx_scalar KSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ksize);
    vx_scalar K = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &k);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)Blocksize,
        (vx_reference)KSIZE,
        (vx_reference)K,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_CORNERHARRIS,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Laplacian C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_laplacian(vx_graph graph, vx_image input, vx_image output, vx_uint32 ddepth, vx_uint32 ksize, vx_float32 scale, vx_float32 delta, vx_int32 border_mode)
{

    vx_scalar DDEPTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ddepth);
    vx_scalar KSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ksize);
    vx_scalar SCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scale);
    vx_scalar DELTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &delta);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border_mode);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)DDEPTH,
        (vx_reference)KSIZE,
        (vx_reference)SCALE,
        (vx_reference)DELTA,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_LAPLACIAN,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Scharr C function.
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_scharr(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_int32  dx, vx_int32 dy, vx_float32 scale, vx_float32 delta, vx_int32 bordertype)
{

    vx_scalar DDEPTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ddepth);
    vx_scalar DX = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dx);
    vx_scalar DY = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dy);
    vx_scalar SCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scale);
    vx_scalar DELTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &delta);
    vx_scalar BORDER = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &bordertype);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)DDEPTH,
        (vx_reference)DX,
        (vx_reference)DY,
        (vx_reference)SCALE,
        (vx_reference)DELTA,
        (vx_reference)BORDER,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SCHARR,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
SIFT Detector C function Call.
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_siftDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp,
        vx_int32 nfeatures, vx_int32 nOctaveLayers, vx_float32 contrastThreshold, vx_float32 edgeThreshold, vx_float32 sigma)
{

    vx_scalar NFEATURES = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nfeatures);
    vx_scalar N_O_LAYERS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nOctaveLayers);
    vx_scalar C_THRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &contrastThreshold);
    vx_scalar EDGETHRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &edgeThreshold);
    vx_scalar SIGMA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sigma);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)NFEATURES,
        (vx_reference)N_O_LAYERS,
        (vx_reference)C_THRESHOLD,
        (vx_reference)EDGETHRESHOLD,
        (vx_reference)SIGMA,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SIFT_DETECT,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
SURF function C call inside a graph
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_surfDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des,
        vx_float32 hessianThreshold, vx_int32 nOctaves, vx_int32 nOctaveLayers)
{

    vx_scalar H_THRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &hessianThreshold);
    vx_scalar NFEATURES = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nOctaves);
    vx_scalar N_O_LAYERS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nOctaveLayers);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)H_THRESHOLD,
        (vx_reference)NFEATURES,
        (vx_reference)N_O_LAYERS,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SURF_DETECT,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
SIFT_Compute  C call inside a graph
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_siftCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des,
        vx_int32 nfeatures, vx_int32 nOctaveLayers, vx_float32 contrastThreshold, vx_float32 edgeThreshold, vx_float32 sigma)
{

    vx_scalar NFEATURES = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nfeatures);
    vx_scalar N_O_LAYERS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nOctaveLayers);
    vx_scalar C_THRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &contrastThreshold);
    vx_scalar EDGETHRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &edgeThreshold);
    vx_scalar SIGMA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sigma);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)output_des,
        (vx_reference)NFEATURES,
        (vx_reference)N_O_LAYERS,
        (vx_reference)C_THRESHOLD,
        (vx_reference)EDGETHRESHOLD,
        (vx_reference)SIGMA,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SIFT_COMPUTE,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
SURF_Compute function C call inside a graph
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_surfCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des,
        vx_float32 hessianThreshold, vx_int32 nOctaves, vx_int32 nOctaveLayers, vx_bool extended, vx_bool upright)
{

    vx_scalar H_THRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &hessianThreshold);
    vx_scalar NFEATURES = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nOctaves);
    vx_scalar N_O_LAYERS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nOctaveLayers);
    vx_scalar EXTENDED = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &extended);
    vx_scalar UPRIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &upright);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)output_des,
        (vx_reference)H_THRESHOLD,
        (vx_reference)NFEATURES,
        (vx_reference)N_O_LAYERS,
        (vx_reference)EXTENDED,
        (vx_reference)UPRIGHT,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SURF_COMPUTE,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
FAST Feature Detector OpenVX C function Call
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_fast(vx_graph graph, vx_image input, vx_array output_kp, vx_int32 threshold, vx_bool nonmaxSuppression)
{

    vx_scalar THRESHOLD = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &threshold);
    vx_scalar NONMAX = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &nonmaxSuppression);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output_kp,
        (vx_reference)THRESHOLD,
        (vx_reference)NONMAX,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_FAST,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
Good Features To Track C function Call
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_goodFeaturesToTrack(vx_graph graph, vx_image input, vx_array output_kp,
        vx_int32 maxCorners, vx_float32 qualityLevel, vx_float32 minDistance, vx_image mask, vx_int32 blockSize, vx_bool useHarrisDetector, vx_float32 k)
{

    vx_scalar MAXCORNERS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &maxCorners);
    vx_scalar QUALITY_LEVEL = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &qualityLevel);
    vx_scalar MIN_DIS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &minDistance);
    vx_scalar BLOCK_SIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &blockSize);
    vx_scalar HARRIS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &useHarrisDetector);
    vx_scalar K = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &k);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output_kp,
        (vx_reference)MAXCORNERS,
        (vx_reference)QUALITY_LEVEL,
        (vx_reference)MIN_DIS,
        (vx_reference)mask,
        (vx_reference)BLOCK_SIZE,
        (vx_reference)HARRIS,
        (vx_reference)K,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_GOOD_FEATURE_TO_TRACK,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
Brisk Detector C function call.
**************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_briskDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp,
        vx_int32 thresh, vx_int32 octaves, vx_float32 patternScale)
{

    vx_scalar THRESH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &thresh);
    vx_scalar OCTAVES = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &octaves);
    vx_scalar PATTERNSCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &patternScale);

    vx_reference params[] = {

        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)THRESH,
        (vx_reference)OCTAVES,
        (vx_reference)PATTERNSCALE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BRISK_DETECT,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
MSER feature detector C function call
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_mserDetect(vx_graph graph, vx_image input, vx_array output_kp, vx_image mask,
        vx_int32 delta, vx_int32 min_area, vx_int32 max_area, vx_float32 max_variation, vx_float32 min_diversity, vx_int32 max_evolution, vx_float32 area_threshold, vx_float32 min_margin, vx_int32 edge_blur_size)
{

    vx_scalar DELTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &delta);
    vx_scalar MIN_AERA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &min_area);
    vx_scalar MAX_AREA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &max_area);
    vx_scalar MAX_EVO = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &max_evolution);
    vx_scalar EDGE_BLUR = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &edge_blur_size);

    vx_scalar MAX_VAR = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &max_variation);
    vx_scalar MIN_DIV = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &min_diversity);
    vx_scalar AREA_THRESH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &area_threshold);
    vx_scalar MIN_MAR = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &min_margin);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output_kp,
        (vx_reference)mask,
        (vx_reference)DELTA,
        (vx_reference)MIN_AERA,
        (vx_reference)MAX_AREA,
        (vx_reference)MAX_VAR,
        (vx_reference)MIN_DIV,
        (vx_reference)MAX_EVO,
        (vx_reference)AREA_THRESH,
        (vx_reference)MIN_MAR,
        (vx_reference)EDGE_BLUR,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_MSER_DETECT,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
ORB function C call inside a graph
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_orbDetect(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp,
        vx_int32 nfeatures, vx_float32 scaleFactor, vx_int32 nlevels, vx_int32 edgeThreshold, vx_int32 firstLevel, vx_int32 WTA_K, vx_int32 scoreType, vx_int32 patchSize)
{

    vx_scalar Nfeatures = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nfeatures);
    vx_scalar SCALE_FAC = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scaleFactor);
    vx_scalar NLEVELS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nlevels);
    vx_scalar EDGE_THRE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &edgeThreshold);
    vx_scalar FIRST_LEVEL = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &firstLevel);
    vx_scalar WTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &WTA_K);
    vx_scalar SCORE_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &scoreType);
    vx_scalar PATCH_SIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &patchSize);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)Nfeatures,
        (vx_reference)SCALE_FAC,
        (vx_reference)NLEVELS,
        (vx_reference)EDGE_THRE,
        (vx_reference)FIRST_LEVEL,
        (vx_reference)WTA,
        (vx_reference)SCORE_TYPE,
        (vx_reference)PATCH_SIZE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ORB_DETECT,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
Star feature detector C function call
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_starFeatureDetector(vx_graph graph, vx_image input, vx_array output_kp, vx_image mask, vx_int32 maxSize, vx_int32 responseThreshold, vx_int32 lineThresholdProjected, vx_int32 lineThresholdBinarized, vx_int32 suppressNonmaxSize) {

    vx_scalar MaxSize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &maxSize);
    vx_scalar ResponseThreshold = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &responseThreshold);
    vx_scalar LineThresholdProjected = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &lineThresholdProjected);
    vx_scalar LineThresholdBinarized = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &lineThresholdBinarized);
    vx_scalar SuppressNonmaxSize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &suppressNonmaxSize);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output_kp,
        (vx_reference)mask,
        (vx_reference)MaxSize,
        (vx_reference)ResponseThreshold,
        (vx_reference)LineThresholdProjected,
        (vx_reference)LineThresholdBinarized,
        (vx_reference)SuppressNonmaxSize,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_STAR_FEATURE_DETECT,
                                   params,
                                   dimof(params));
}

/*!***********************************************************************************************************
SIMPLE BLOB feature detector C function call
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_simpleBlobDetector(vx_graph graph, vx_image input, vx_array output_kp, vx_image mask)
{

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output_kp,
        (vx_reference)mask,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SIMPLE_BLOB_DETECT,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
Brisk Compute C function call.
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_briskCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des,
        vx_int32 thresh, vx_int32 octaves, vx_float32 patternScale)
{

    vx_scalar THRESH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &thresh);
    vx_scalar OCTAVES = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &octaves);
    vx_scalar PATTERN_SCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &patternScale);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,

        (vx_reference)output_kp,
        (vx_reference)output_des,

        (vx_reference)THRESH,
        (vx_reference)OCTAVES,
        (vx_reference)PATTERN_SCALE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BRISK_COMPUTE,
                                   params,
                                   dimof(params));

}

/*!***********************************************************************************************************
ORB function C call inside a graph
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_orbCompute(vx_graph graph, vx_image input, vx_image mask, vx_array output_kp, vx_array output_des,
        vx_int32 nfeatures, vx_float32 scaleFactor, vx_int32 nlevels, vx_int32 edgeThreshold, vx_int32 firstLevel, vx_int32 WTA_K, vx_int32 scoreType, vx_int32 patchSize)
{

    vx_scalar Nfeatures = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nfeatures);
    vx_scalar SCALE_FAC = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scaleFactor);
    vx_scalar NLEVELS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &nlevels);
    vx_scalar EDGE_THRE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &edgeThreshold);
    vx_scalar FIRST_LEVEL = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &firstLevel);
    vx_scalar WTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &WTA_K);
    vx_scalar SCORE_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &scoreType);
    vx_scalar PATCH_SIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &patchSize);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)mask,
        (vx_reference)output_kp,
        (vx_reference)output_des,

        (vx_reference)Nfeatures,
        (vx_reference)SCALE_FAC,
        (vx_reference)NLEVELS,
        (vx_reference)EDGE_THRE,
        (vx_reference)FIRST_LEVEL,
        (vx_reference)WTA,
        (vx_reference)SCORE_TYPE,
        (vx_reference)PATCH_SIZE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ORB_COMPUTE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
multiply C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_multiply(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output, vx_float32 scale, vx_int32 dtype)
{

    vx_scalar SCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scale);
    vx_scalar DTYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dtype);

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
        (vx_reference)SCALE,
        (vx_reference)DTYPE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_MULTIPLY,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Divide C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_divide(vx_graph graph, vx_image input_1, vx_image input_2, vx_image output, vx_float32 scale, vx_int32 dtype)
{

    vx_scalar SCALE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &scale);
    vx_scalar DTYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &dtype);

    vx_reference params[] = {
        (vx_reference)input_1,
        (vx_reference)input_2,
        (vx_reference)output,
        (vx_reference)SCALE,
        (vx_reference)DTYPE,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_DIVIDE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
adaptiveThreshold C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_adaptiveThreshold(vx_graph graph, vx_image input, vx_image output, vx_float32 maxValue, vx_int32 adaptiveMethod, vx_int32 thresholdType, vx_int32 blockSize, vx_float32 c)
{

    vx_scalar MAXVALUE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &maxValue);
    vx_scalar A_Method = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &adaptiveMethod);
    vx_scalar Thresh_type = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &thresholdType);
    vx_scalar BLOCK = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &blockSize);
    vx_scalar C = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &c);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)MAXVALUE,
        (vx_reference)A_Method,
        (vx_reference)Thresh_type,
        (vx_reference)BLOCK,
        (vx_reference)C,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ADAPTIVETHRESHOLD,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
distanceTransform C Function
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_distanceTransform(vx_graph graph, vx_image input, vx_image output)
{

    vx_reference params[] = {

        (vx_reference)input,
        (vx_reference)output,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_DISTANCETRANSFORM,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
cvtColor C Function
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_cvtColor(vx_graph graph, vx_image input, vx_image output, vx_uint32 CODE)
{

    vx_scalar code = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &CODE);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)code,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_CVTCOLOR,
                                   params,
                                   dimof(params));
}

/************************************************************************************************************
threshold C Function
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_threshold(vx_graph graph, vx_image input, vx_image output, vx_float32 thresh, vx_float32 maxVal, vx_int32 type)
{

    vx_scalar Thresh = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &thresh);
    vx_scalar MAXVAL = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &maxVal);
    vx_scalar TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &type);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)Thresh,
        (vx_reference)MAXVAL,
        (vx_reference)TYPE,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_THRESHOLD,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
fastNlMeansDenoising C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_fastNlMeansDenoising(vx_graph graph, vx_image input, vx_image output, vx_float32 h, vx_int32 template_ws, vx_int32 search_ws)
{

    vx_scalar H = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &h);
    vx_scalar T = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &template_ws);
    vx_scalar S = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &search_ws);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)H,
        (vx_reference)T,
        (vx_reference)S,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_FAST_NL_MEANS_DENOISING,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
fastNlMeansDenoisingColored C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_fastNlMeansDenoisingColored(vx_graph graph, vx_image input, vx_image output, vx_float32 h, vx_float32 h_color, vx_int32 template_ws, vx_int32 search_ws)
{

    vx_scalar H = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &h);
    vx_scalar H_C = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &h_color);
    vx_scalar T = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &template_ws);
    vx_scalar S = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &search_ws);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)H,
        (vx_reference)H_C,
        (vx_reference)T,
        (vx_reference)S,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_FAST_NL_MEANS_DENOISING_COLORED,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
pyrup C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_pyrUp(vx_graph graph, vx_image input, vx_image output, vx_uint32 Swidth, vx_uint32 Sheight, vx_int32 bordertype)
{

    vx_scalar W = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Swidth);
    vx_scalar H = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Sheight);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &bordertype);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)W,
        (vx_reference)H,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_PYRUP,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
pyrdown C Function
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_pyrDown(vx_graph graph, vx_image input, vx_image output, vx_uint32 Swidth, vx_uint32 Sheight, vx_int32 bordertype)
{

    vx_scalar W = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Swidth);
    vx_scalar H = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Sheight);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &bordertype);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)W,
        (vx_reference)H,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_PYRDOWN,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
filter2D C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_filter2D(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_float32 delta, vx_int32 border)
{

    vx_scalar D = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ddepth);
    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar DELTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &delta);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)D,
        (vx_reference)Kernel,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)DELTA,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_FILTER_2D,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
sepFilter2D C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_sepFilter2D(vx_graph graph, vx_image input, vx_image output, vx_int32 ddepth, vx_matrix KernelX, vx_matrix KernelY, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_float32 delta, vx_int32 border)
{

    vx_scalar D = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &ddepth);
    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar DELTA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &delta);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)D,
        (vx_reference)KernelX,
        (vx_reference)KernelY,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)DELTA,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_SEPFILTER_2D,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Dilate C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_dilate(vx_graph graph, vx_image input, vx_image output, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 iterations, vx_int32 border)
{

    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar I = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &iterations);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)Kernel,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)I,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_DILATE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Erode C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_erode(vx_graph graph, vx_image input, vx_image output, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 iterations, vx_int32 border)
{

    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar I = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &iterations);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)Kernel,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)I,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_ERODE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
WarpAffine C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_warpAffine(vx_graph graph, vx_image input, vx_image output, vx_matrix M, vx_int32 Size_X, vx_int32 Size_Y, vx_int32 flags, vx_int32 border)
{

    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Size_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Size_Y);
    vx_scalar F = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &flags);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)M,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)F,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_WARP_AFFINE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
WarpPerspective C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_warpPerspective(vx_graph graph, vx_image input, vx_image output, vx_matrix M, vx_int32 Size_X, vx_int32 Size_Y, vx_int32 flags, vx_int32 border)
{

    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Size_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Size_Y);
    vx_scalar F = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &flags);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)M,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)F,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_WARP_PERSPECTIVE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
Resize C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_resize(vx_graph graph, vx_image input, vx_image output, vx_int32 Size_X, vx_int32 Size_Y, vx_float32 FX, vx_float32 FY, vx_int32 interpolation)
{

    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Size_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Size_Y);
    vx_scalar Fx = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &FX);
    vx_scalar Fy = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &FY);
    vx_scalar I = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &interpolation);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)Fx,
        (vx_reference)Fy,
        (vx_reference)I,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_RESIZE,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
morphologyEX C function.
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_morphologyEX(vx_graph graph, vx_image input, vx_image output, vx_int32 OP, vx_matrix Kernel, vx_int32 Anchor_X, vx_int32 Anchor_Y, vx_int32 iterations, vx_int32 border)
{

    vx_scalar op = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &OP);
    vx_scalar X = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_X);
    vx_scalar Y = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Anchor_Y);
    vx_scalar I = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &iterations);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)op,
        (vx_reference)Kernel,
        (vx_reference)X,
        (vx_reference)Y,
        (vx_reference)I,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_MORPHOLOGYEX,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
buildPyramid C Function
************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_buildPyramid(vx_graph graph, vx_image input, vx_pyramid output, vx_uint32 maxLevel, vx_uint32 border)
{

    vx_scalar ML = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &maxLevel);
    vx_scalar B = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &border);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)ML,
        (vx_reference)B,
    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BUILD_PYRAMID,
                                   params,
                                   dimof(params));

}

/************************************************************************************************************
BuildOpticalFlow C Function
*************************************************************************************************************/
extern "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtCvNode_buildOpticalFlowPyramid(vx_graph graph, vx_image input, vx_pyramid output, vx_uint32 S_width, vx_uint32 S_height, vx_int32 WinSize, vx_bool WithDerivatives, vx_int32 Pyr_border, vx_int32 derviBorder, vx_bool tryReuse)
{

    vx_scalar W = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &S_width);
    vx_scalar H = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &S_height);
    vx_scalar Win_Size = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &WinSize);
    vx_scalar WITH_DERIVA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &WithDerivatives);
    vx_scalar P_border = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &Pyr_border);
    vx_scalar D_border = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &derviBorder);
    vx_scalar Reuse = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_BOOL, &tryReuse);

    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)W,
        (vx_reference)H,
        (vx_reference)Win_Size,
        (vx_reference)WITH_DERIVA,
        (vx_reference)P_border,
        (vx_reference)D_border,
        (vx_reference)Reuse,

    };

    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OPENCV_BUILD_OPTICAL_FLOW_PYRAMID,
                                   params,
                                   dimof(params));

}
