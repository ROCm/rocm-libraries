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

#ifndef _VX_EXT_AMD_H_
#define _VX_EXT_AMD_H_

#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#ifdef __cplusplus
#include <string>
#endif

/*!
 * \file
 * \brief The AMD OpenVX Extension Library.
 *
 * \defgroup group_amd Extension: AMD OpenVX Extension API
 * \brief AMD OpenVX Extension to enhance base, vision, and neural network feature sets.
 */

/*! \brief AMD target affinity enumerations for AgoTargetAffinityInfo.device_type
 * \ingroup group_amd
 */
#define AGO_TARGET_AFFINITY_CPU 0x0010 // CPU
/*! \brief AMD target affinity enumerations for AgoTargetAffinityInfo.device_type
 * \ingroup group_amd
 */
#define AGO_TARGET_AFFINITY_GPU 0x0020 // GPU
/*! \brief AMD target affinity enumerations for AgoTargetAffinityInfo.device_type
 * \ingroup group_amd
 */
#define AGO_TARGET_AFFINITY_APU 0x0030 // APU

/*! \brief AMD internal parameters. [TODO: This needs to be moved to ago_internal.h]
 * \ingroup group_amd
 */
#define AGO_MAX_PARAMS 32
#define AGO_MERGE_RULE_MAX_FIND 4
#define AGO_MERGE_RULE_MAX_REPLACE 4
#define AGO_MERGE_RULE_SOLITARY_FLAG 0x20
#define AGO_TARGET_AFFINITY_GPU_INFO_DEVICE_MASK 0x0F
#define AGO_TARGET_AFFINITY_GPU_INFO_SVM_MASK 0xF0
#define AGO_TARGET_AFFINITY_GPU_INFO_SVM_ENABLE 0x10
#define AGO_TARGET_AFFINITY_GPU_INFO_SVM_AS_CLMEM 0x20
#define AGO_TARGET_AFFINITY_GPU_INFO_SVM_NO_FGS 0x40

/*! \brief Maximum size of scalar string buffer. The local buffers used for accessing scalar strings
 * should be of size VX_MAX_STRING_BUFFER_SIZE_AMD and the maximum allowed string length is
 * VX_MAX_STRING_BUFFER_SIZE_AMD-1.
 * \ingroup group_scalar
 * \ingroup group_amd
 */
#define VX_MAX_STRING_BUFFER_SIZE_AMD 256

/*! \brief The Neural Network activation functions vx_nn_activation_function_e extension.
 * \ingroup group_amd
 * \ingroup group_amd_nn
 */
#define VX_NN_ACTIVATION_LEAKY_RELU (VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x9)

/*! \brief The attributes for vx_context
 * \ingroup group_amd
 */
#define VX_CONTEXT_ATTRIBUTE_NONLINEAR_MAX_DIMENSION VX_CONTEXT_NONLINEAR_MAX_DIMENSION
#define VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_POLICY VX_CONTEXT_IMMEDIATE_BORDER_POLICY

/*! \brief The attributes for vx_graph
 * \ingroup group_amd
 * \ingroup group_graph
 */
#define VX_GRAPH_ATTRIBUTE_STATE VX_GRAPH_STATE

/*! \brief The attributes for vx_object_array
 * \ingroup group_amd
 */
#define VX_OBJECT_ARRAY_ATTRIBUTE_ITEMTYPE VX_OBJECT_ARRAY_ITEMTYPE
#define VX_OBJECT_ARRAY_ATTRIBUTE_NUMITEMS VX_OBJECT_ARRAY_NUMITEMS

/*! \brief The type enumeration lists all the AMD specific types in OpenVX.
 * \ingroup group_amd
 */
enum ago_type_public_e
{
    /*! \brief AMD scalar data type for string
     */
    VX_TYPE_STRING_AMD = 0x011,
    /*! \brief AGO struct data type for keypoint XYS
     */
    AGO_TYPE_KEYPOINT_XYS = VX_TYPE_VENDOR_STRUCT_START,
    /*! \brief AGO data structure for AGO MeanStdDev kernels
     */
    AGO_TYPE_MEANSTDDEV_DATA = VX_TYPE_VENDOR_OBJECT_START,
    /*! \brief AGO data structure for AGO MinMaxLoc kernels
     */
    AGO_TYPE_MINMAXLOC_DATA,
    /*! \brief AGO data structure for AGO Canny kernels
     */
    AGO_TYPE_CANNY_STACK,
    /*! \brief AGO data structure for AGO Scale kernels
     */
    AGO_TYPE_SCALE_MATRIX,
};

/*! \brief The AMD context attributes list.
 * \ingroup group_amd
 */
enum vx_context_attribute_amd_e
{
    /*! \brief OpenCL context. Use a <tt>\ref cl_context</tt> parameter.*/
    VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x01,
    /*! \brief Context affinity. Use a <tt>\ref AgoTargetAffinityInfo</tt> parameter.*/
    VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x02,
    /*! \brief Set a text macro definition. Use a <tt>\ref AgoContextMacroInfo</tt> parameter.*/
    VX_CONTEXT_ATTRIBUTE_AMD_SET_TEXT_MACRO = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x03,
    /*! \brief Set a merge rule. Use a <tt>\ref AgoNodeMergeRule</tt> parameter.*/
    VX_CONTEXT_ATTRIBUTE_AMD_SET_MERGE_RULE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x04,
    /*! \brief Tensor Data max num of dimensions supported by HW. */
    VX_CONTEXT_MAX_TENSOR_DIMENSIONS = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x05,
    /*! \brief CL_QUEUE_PROPERTIES to be used for creating OpenCL command queue. Use a <tt>\ref cl_command_queue_properties</tt> parameter. */
    VX_CONTEXT_CL_QUEUE_PROPERTIES = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x06,
    /*! \brief HIP context. Use a <tt>\ref cl_context</tt> parameter.*/
    VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_CONTEXT) + 0x07,
};

/*! \brief The AMD kernel attributes list.
 * \ingroup group_amd
 */
enum vx_kernel_attribute_amd_e
{
    /*! \brief Kernel callback for query target support. Use a <tt>\ref amd_kernel_query_target_support_f</tt> parameter.*/
    VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_KERNEL) + 0x01,
    /*! \brief Kernel callback for OpenCL code generation. Use a <tt>\ref amd_kernel_opencl_codegen_callback_f</tt> parameter.*/
    VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_KERNEL) + 0x02,
    /*! \brief Kernel callback for node regeneration. Use a <tt>\ref amd_kernel_node_regen_callback_f</tt> parameter.*/
    VX_KERNEL_ATTRIBUTE_AMD_NODE_REGEN_CALLBACK = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_KERNEL) + 0x03,
    /*! \brief Kernel callback for OpenCL global work[]. Use a <tt>\ref amd_kernel_opencl_global_work_update_callback_f</tt> parameter.*/
    VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_KERNEL) + 0x04,
    /*! \brief Kernel flag to enable OpenCL buffer access (default OFF). Use a <tt>\ref vx_bool</tt> parameter.*/
    VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_KERNEL) + 0x05,
    /*! \brief Kernel callback for GPU buffer update. Use a <tt>\ref AgoKernelGpuBufferUpdateInfo</tt> parameter.*/
    VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_UPDATE_CALLBACK = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_KERNEL) + 0x06,
};

/*! \brief The AMD graph attributes list.
 * \ingroup group_amd
 */
enum vx_graph_attribute_amd_e
{
    /*! \brief Graph affinity. Use a <tt>\ref AgoNodeAffinityInfo</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_AFFINITY = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x01,
    /*! \brief Imports a graph from a text file. Use a <tt>\ref AgoGraphImportInfo</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x02,
    /*! \brief Export a graph into a text file. Use a <tt>\ref AgoGraphExportInfo</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x03,
    /*! \brief Graph optimizer flags. Use a <tt>\ref vx_uint32</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x04,
    /*! \brief Graph last performance (internal). Use a <tt>\ref AgoGraphPerfInternalInfo</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_LAST = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x05,
    /*! \brief Graph avg performance (internal). Use a <tt>\ref AgoGraphPerfInternalInfo</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_AVG = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x06,
    /*! \brief Graph internal performance profile. Use a char * fileName parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_PROFILE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x07,
    /*! \brief OpenCL command queue. Use a <tt>\ref cl_command_queue</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x08,
    /*! \brief CPU num_threads to be used in RPP. Use a <tt>\ref vx_uint32</tt> parameter.*/
    VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_GRAPH) + 0x09
};

/*! \brief The AMD node attributes list.
 * \ingroup group_amd
 */
enum vx_node_attribute_amd_e
{
    /*! \brief Node affinity. Use a <tt>\ref AgoTargetAffinityInfo</tt> parameter.*/
    VX_NODE_ATTRIBUTE_AMD_AFFINITY = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_NODE) + 0x01,
    /*! \brief OpenCL command queue. Use a <tt>\ref cl_command_queue</tt> parameter.*/
    VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_NODE) + 0x02,
    /*! \brief HIP stream. TBD.*/
    VX_NODE_ATTRIBUTE_AMD_HIP_STREAM = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_NODE) + 0x03,
    /*! \brief TBD.*/
    VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_NODE) + 0x04
};

/*! \brief The AMD image attributes list.
 * \ingroup group_amd
 */
enum vx_image_attribute_amd_e
{
    /*! \brief Sync with user specified OpenCL buffer. Use a <tt>\ref cl_mem</tt> parameter.*/
    VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_IMAGE) + 0x01,
    /*! \brief GPU buffer offset. Use a <tt>\ref cl_uint</tt> parameter.*/
    VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_IMAGE) + 0x02,
    /*! \brief Enable user kernel's own GPU buffer for virtual images. Supports only images with
     * single color plane and stride should match framework's internal alignment. image ROI not supported.
     * Use a <tt>\ref vx_bool</tt> parameter.*/
    VX_IMAGE_ATTRIBUTE_AMD_ENABLE_USER_BUFFER_GPU = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_IMAGE) + 0x03,
    /*! \brief GPU buffer stride. Use a <tt>\ref cl_uint</tt> parameter.*/
    VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_IMAGE) + 0x04,
    /*! \brief Sync with user specified host buffer. Use a <tt>\ref cl_mem</tt> parameter.*/
    VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_IMAGE) + 0x05,
    /*! \brief Sync with user specified hip memory. */
    VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_IMAGE) + 0x06,
};

/*! \brief The AMD tensor data attributes.
 * \ingroup group_amd
 * \ingroup group_tensor
 */
enum vx_tensor_attribute_amd_e
{
    /*! \brief GPU buffer strides (array of <tt>vx_size</tt>). */
    VX_TENSOR_STRIDE_GPU = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_TENSOR) + 0x5,
    /*! \brief GPU buffer offset. <tt>vx_size</tt>. */
    VX_TENSOR_OFFSET_GPU = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_TENSOR) + 0x6,
    /*! \brief OpenCL buffer. <tt>cl_mem</tt>. */
    VX_TENSOR_BUFFER_OPENCL = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_TENSOR) + 0x7,
    /*! \brief HIP buffer. <tt>cl_mem</tt>. */
    VX_TENSOR_BUFFER_HIP = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_TENSOR) + 0x8,
    /*! \brief Queries memory type if created using vxCreateTensorFromHandle. If vx_tensor was not created using
        vxCreateTensorFromHandle, VX_MEMORY_TYPE_NONE is returned. Use a <tt>\ref vx_memory_type_e</tt> parameter. */
    VX_TENSOR_MEMORY_TYPE = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_TENSOR) + 0x9,
    /*! \brief Sync with user specified host buffer.*/
    VX_TENSOR_BUFFER_HOST = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_TENSOR) + 0x10,
};

/*! \brief The AMD array data attributes.
 * \ingroup group_amd
 */
enum vx_array_attribute_amd_e
{
    /*! \brief OpenCL buffer. <tt>cl_mem</tt>. */
    VX_ARRAY_BUFFER_OPENCL = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_ARRAY) + 0x9,
    /*! \brief HIP buffer. <tt>hip_mem</tt>. */
    VX_ARRAY_BUFFER_HIP = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_ARRAY) + 0x10,
    /*! \brief TODO:. */
    VX_ARRAY_BUFFER = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_ARRAY) + 0x11,
    /*! \brief TODO:. */
    VX_ARRAY_OFFSET_GPU = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_ARRAY) + 0x12
};

/*! \brief These enumerations are given to the \c vxDirective API to enable/disable
 * platform optimizations and/or features. Directives are not optional and
 * usually are vendor-specific, by defining a vendor range of directives and
 * starting their enumeration from there.
 * \see <tt>vxDirective</tt>
 * \ingroup group_directive
 * \ingroup group_amd
 */
enum vx_directive_amd_e
{
    /*! \brief Data object is readonly after this directive is given. */
    VX_DIRECTIVE_AMD_READ_ONLY = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_DIRECTIVE) + 0x01,
    /*! \brief Data object copy to OpenCL. */
    VX_DIRECTIVE_AMD_COPY_TO_OPENCL = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_DIRECTIVE) + 0x02,
    /*! \brief Enable performance profile capture. */
    VX_DIRECTIVE_AMD_ENABLE_PROFILE_CAPTURE = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_DIRECTIVE) + 0x03,
    /*! \brief Disable performance profile capture. */
    VX_DIRECTIVE_AMD_DISABLE_PROFILE_CAPTURE = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_DIRECTIVE) + 0x04,
    /*! \brief Disable node level flush for a graph. */
    VX_DIRECTIVE_AMD_DISABLE_GPU_FLUSH = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_DIRECTIVE) + 0x05,
    /*! \brief Data object copy to HIP memory. */
    VX_DIRECTIVE_AMD_COPY_TO_HIPMEM = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_DIRECTIVE) + 0x02,
};

/*! \brief An enumeration of additional memory type imports.
 * \ingroup group_context
 * \ingroup group_amd
 */
enum vx_memory_type_amd_e
{
    /*! \brief The memory type to import from the OpenCL. */
    VX_MEMORY_TYPE_OPENCL = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_MEMORY_TYPE) + 0x2,
    /*! \brief The memory type to import from the HIP. */
    VX_MEMORY_TYPE_HIP = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_MEMORY_TYPE) + 0x3,
};

/*! \brief The image color space list used by the <tt>\ref VX_IMAGE_SPACE</tt> attribute of a <tt>\ref vx_image</tt>.
 * \ingroup group_image
 * \ingroup group_amd
 */
enum vx_color_space_amd_e
{
    /*! \brief Use to indicate that the BT.2020 coefficients are used for conversions. */
    VX_COLOR_SPACE_BT2020 = VX_ENUM_BASE(VX_ID_AMD, VX_ENUM_COLOR_SPACE) + 0x1,
};

/*! \brief Based on the VX_DF_IMAGE definition.
 * \ingroup group_amd
 * \note Use <tt>\ref vx_df_image</tt> to contain these values.
 */
enum vx_df_image_amd_e
{
    /*! \brief AMD image with 1-bit data. */
    VX_DF_IMAGE_U1_AMD = VX_DF_IMAGE('U', '0', '0', '1'),
    /*! \brief AMD image with 16-bit floating-point (half). */
    VX_DF_IMAGE_F16_AMD = VX_DF_IMAGE('F', '0', '1', '6'),
    /*! \brief AMD image with 32-bit floating-point (float). */
    VX_DF_IMAGE_F32_AMD = VX_DF_IMAGE('F', '0', '3', '2'),
    /*! \brief AMD image with 64-bit floating-point (double). */
    VX_DF_IMAGE_F64_AMD = VX_DF_IMAGE('F', '0', '6', '4'),
    /*! \brief AMD image with THREE 32-bit floating-point channels in one buffer. */
    VX_DF_IMAGE_F32x3_AMD = VX_DF_IMAGE('F', '3', '3', '2'),
};

/*! \brief scalar data attributes.
 * \ingroup group_scalar
 * \ingroup group_amd
 */
enum vx_scalar_attribute_amd_e
{
    /*! \brief scalar's buffer */
    VX_SCALAR_BUFFER = VX_ATTRIBUTE_BASE(VX_ID_AMD, VX_TYPE_SCALAR) + 0x1,
};

/*! \brief The multidimensional data object (Tensor).
 * \ingroup group_amd
 * \see vxCreateTensor
 * \ingroup group_tensor
 * \extends vx_reference
 */
typedef struct _vx_tensor_t *vx_tensor;

/*! \brief AMD data structure for image format information.
 * \ingroup group_amd
 */
typedef struct
{
    vx_size components;
    vx_size planes;
    vx_size pixelSizeInBitsNum;
    vx_color_space_e colorSpace;
    vx_channel_range_e channelRange;
    vx_size pixelSizeInBitsDenom;
} AgoImageFormatDescription;

/*! \brief AMD data structure to specify target affinity.
 * \ingroup group_amd
 */
typedef struct
{
    vx_uint32 device_type; // shall be AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU
    vx_uint32 device_info; // reserved -- shall be initialized to ZERO and shall not be modified
    vx_uint32 group; // reserved -- shall be initialized to ZERO and shall not be modified
    vx_uint32 reserved; // reserved -- shall be initialized to ZERO and shall not be modified
} AgoTargetAffinityInfo;

/*! \brief AMD data structure to set a text macro.
 * \ingroup group_amd
 */
typedef struct
{
    vx_char macroName[256];
    vx_char *text;
} AgoContextTextMacroInfo;

/*! \brief AMD data structure to import a graph from a text.
 * \ingroup group_amd
 **    text:
 **      "macro <macro-name>" to use a pre-defined macro
 **      "file <file-name>" to load from a file
 **      otherwise use the text as is
 */
typedef struct
{
    vx_char *text;
    vx_uint32 num_ref;
    vx_reference *ref;
    vx_int32 dumpToConsole;
    void(VX_CALLBACK *data_registry_callback_f)(void *obj, vx_reference ref, const char *name, const char *app_params);
    void *data_registry_callback_obj;
} AgoGraphImportInfo;

/*! \brief AMD data structure to export a graph to a text.
 * \ingroup group_amd
 */
typedef struct
{
    vx_char fileName[256];
    vx_uint32 num_ref;
    vx_reference *ref;
    vx_char comment[64];
} AgoGraphExportInfo;

/*! \brief AMD data structure to get internal performance data.
 * \ingroup group_amd
 */
typedef struct
{
    vx_uint64 kernel_enqueue;
    vx_uint64 kernel_wait;
    vx_uint64 buffer_read;
    vx_uint64 buffer_write;
} AgoGraphPerfInternalInfo;

/*! \brief AMD data structure to specify node merge rule.
 * \ingroup group_amd
 */
typedef struct AgoNodeMergeRule_t
{
    struct
    {
        vx_enum kernel_id;
        vx_uint32 arg_spec[AGO_MAX_PARAMS];
    } find[AGO_MERGE_RULE_MAX_FIND];
    struct
    {
        vx_enum kernel_id;
        vx_uint32 arg_spec[AGO_MAX_PARAMS];
    } replace[AGO_MERGE_RULE_MAX_REPLACE];
} AgoNodeMergeRule;

#ifdef __cplusplus
/*! \brief AMD usernode callback for target support check - supported_target_affinity shall contain bitfields AGO_TARGET_AFFINITY_CPU and AGO_TARGET_AFFINITY_GPU.
 *   @note When this callback is not available, the framework assumes that supported_target_affinity = AGO_TARGET_AFFINITY_CPU.
 * \ingroup group_amd
 */
typedef vx_status(VX_CALLBACK *amd_kernel_query_target_support_f)(vx_graph graph, vx_node node,
                                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
);

/*! \brief AMD usernode callback for OpenCL source code generation. The framework will pass
 * \ingroup group_amd
 *   OpenVX objects as parameters to OpenCL kernels in othe order they appear to OpenVX node.
 *   The mapping of OpenVX object to OpenCL kernel argument as shown below:
 *     vx_image:       uint width, uint height, __global <type> * buf, uint stride_in_bytes, uint offset
 *     vx_array:       __global <type> * buf, uint offset_in_bytes, uint numitems
 *     vx_scalar:      float value or uint value or int value
 *     vx_matrix:      float matrix[<ROWS>*<COLS>]
 *     vx_convolution: float convolution[<ROWS>*<COLS>]
 *     vx_threshold:   int value or int2 value
 *     vx_remap:       __global short2 * buf, uint stride_in_bytes
 *     vx_lut:         __global uchar * lut, uint count, uint offset
 */
typedef vx_status(VX_CALLBACK *amd_kernel_opencl_codegen_callback_f)(
    vx_node node,                                // [input] node
    const vx_reference parameters[],             // [input] parameters
    vx_uint32 num,                               // [input] number of parameters
    bool opencl_load_function,                   // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],        // [output] kernel_name for clCreateKernel()
    std::string &opencl_kernel_code,             // [output] string for clCreateProgramWithSource()
    std::string &opencl_build_options,           // [output] options for clBuildProgram()
    vx_uint32 &opencl_work_dim,                  // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                 // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32 &opencl_local_buffer_usage_mask,   // [output] reserved: must be ZERO
    vx_uint32 &opencl_local_buffer_size_in_bytes // [output] reserved: must be ZERO
);

/*! \brief AMD usernode callback for regenerating a node.
 * \ingroup group_amd
 */
typedef vx_status(VX_CALLBACK *amd_drama_add_node_f)(vx_node node, vx_enum kernel_id, vx_reference *paramList, vx_uint32 paramCount);
typedef vx_status(VX_CALLBACK *amd_kernel_node_regen_callback_f)(vx_node node, amd_drama_add_node_f add_node_f, vx_bool &replace_original);

/*! \brief AMD usernode callback for updating the OpenCL global_work[]. The framework will pass
 *   OpenVX objects as parameters to OpenCL kernels in othe order they appear to OpenVX node and
 *   previous values of local/global work. This function will get called before launching the kernel.
 * \ingroup group_amd
 */
typedef vx_status(VX_CALLBACK *amd_kernel_opencl_global_work_update_callback_f)(
    vx_node node,                     // [input] node
    const vx_reference parameters[],  // [input] parameters
    vx_uint32 num,                    // [input] number of parameters
    vx_uint32 opencl_work_dim,        // [input] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],     // [output] global_work[] for clEnqueueNDRangeKernel()
    const vx_size opencl_local_work[] // [input] local_work[] for clEnqueueNDRangeKernel()
);

/*! \brief AMD usernode callback for setting the GPU buffers. The framework will pass
 *   OpenVX objects as parameters to GPU kernels in othe order they appear to OpenVX node.
 *   This function will get called before executing the node.
 * \ingroup group_amd
 */
typedef vx_status(VX_CALLBACK *amd_kernel_gpu_buffer_update_callback_f)(
    vx_node node,                    // [input] node
    const vx_reference parameters[], // [input] parameters
    vx_uint32 num                    // [input] number of parameters
);

/*! \brief AMD data structure for use by VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_UPDATE_CALLBACK.
 * \ingroup group_amd
 */
typedef struct
{
    amd_kernel_gpu_buffer_update_callback_f gpu_buffer_update_callback_f;
    vx_uint32 gpu_buffer_update_param_index;
} AgoKernelGpuBufferUpdateInfo;

#if defined(AMD_FP16_SUPPORT)
/*! \brief A 16-bit float value.
 */
#include <half/half.hpp>
typedef half_float::half vx_float16;
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
     * \brief Retrieve the name of a reference
     * \ingroup vx_framework_reference
     * \ingroup group_amd
     *
     * This function is used to retrieve the name of a reference.
     *
     * \param [in] ref The reference.
     * \param [out] name Pointer to copy the name of the reference.
     * \param [in] size Size of the name buffer.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxGetReferenceName(vx_reference ref, vx_char name[], vx_size size);

    /*!
     * \brief Set module internal data.
     * \ingroup vx_framework_reference
     * \ingroup group_amd
     *
     * This function is used to set module specific internal data. This is for use by vxPublishKernels().
     *
     * \param [in] context The context.
     * \param [in] module The name of the module used in vxLoadKernels.
     * \param [in] ptr The module internal buffer.
     * \param [in] size Size of the module internal buffer.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxSetModuleInternalData(vx_context context, const vx_char *module, void *ptr, vx_size size);

    /*!
     * \brief Retrieve module internal data.
     * \ingroup vx_framework_reference
     * \ingroup group_amd
     *
     * This function is used to retrieve module specific internal data. This is for use by vxUnpublishKernels().
     *
     * \param [in] context The context.
     * \param [in] module The name of the module used in vxLoadKernels.
     * \param [out] ptr The module internal buffer.
     * \param [out] size Size of the module internal buffer.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxGetModuleInternalData(vx_context context, const vx_char *module, void **ptr, vx_size *size);

    /*!
     * \brief Set module handle.
     * \ingroup vx_framework_reference
     * \ingroup group_amd
     *
     * This function is used to set module specific of a graph from within a node.
     *
     * \param [in] node The node.
     * \param [in] module The name of the module used in vxLoadKernels.
     * \param [in] ptr The module internal buffer.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxSetModuleHandle(vx_node node, const vx_char *module, void *ptr);

    /*!
     * \brief Retrieve module handle.
     * \ingroup vx_framework_reference
     * \ingroup group_amd
     *
     * This function is used to retrieve module specific handle of a graph from within a node.
     *
     * \param [in] node The node.
     * \param [in] module The name of the module used in vxLoadKernels.
     * \param [out] ptr The module internal buffer.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxGetModuleHandle(vx_node node, const vx_char *module, void **ptr);

    /*!
     * \brief Set custom image format description.
     * \ingroup vx_framework_reference
     * \ingroup group_amd
     *
     * This function is used to support custom image formats with single-plane by ISVs. Should be called from vxPublishKernels().
     *
     * \param [in] context The context.
     * \param [in] format The image format.
     * \param [in] desc The image format description.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     * \retval VX_ERROR_INVALID_FORMAT if format is already in use.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxSetContextImageFormatDescription(vx_context context, vx_df_image format, const AgoImageFormatDescription *desc);

    /*!
     * \brief Get custom image format description.
     * \ingroup group_amd
     * \ingroup vx_framework_reference
     * \param [in] context The context.
     * \param [in] format The image format.
     * \param [out] desc The image format description.
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     * \retval VX_ERROR_INVALID_FORMAT if format is already in use.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxGetContextImageFormatDescription(vx_context context, vx_df_image format, AgoImageFormatDescription *desc);

    /*!
     * \brief vxAliasTensor.
     * \ingroup group_amd
     * \ingroup vx_framework_reference
     * @note - TBD
     * \return A \ref vx_status_e enumeration.
     * \retval VX_SUCCESS No errors.
     * \retval VX_ERROR_INVALID_REFERENCE if reference is not valid.
     * \retval VX_ERROR_INVALID_FORMAT if format is already in use.
     */
    VX_API_ENTRY vx_status VX_API_CALL vxAliasTensor(vx_tensor tensorMaster, vx_size offset, vx_tensor tensor);

    /*!
     * \brief vxIsTensorAliased.
     * \ingroup group_amd
     * \ingroup vx_framework_reference
     * @note - TBD
     */
    VX_API_ENTRY vx_bool VX_API_CALL vxIsTensorAliased(vx_tensor tensorMaster, vx_size offset, vx_tensor tensor);

#ifdef __cplusplus
}
#endif

#endif
