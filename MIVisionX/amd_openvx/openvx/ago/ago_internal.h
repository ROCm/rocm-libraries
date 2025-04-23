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


#ifndef __ago_internal_h__
#define __ago_internal_h__

#include "ago_platform.h"
#include "ago_kernels.h"
#include "ago_haf_cpu.h"
#include "vx_ext_amd.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// configuration flags and constants
//

// version
#define AGO_VERSION "1.3.0"

// debug configuration
#define ENABLE_DEBUG_MESSAGES                 0 // 0:disable 1:enable
#define SHOW_DEBUG_HIERARCHICAL_LEVELS        0 // 0:disable 1:enable debug hierarchical levels
#define ENABLE_LOG_MESSAGES_DEFAULT        true // default logging directive VX_DIRECTIVE_ENABLE_LOGGING

// fused OpenCL kernel workgroup size
#define AGO_OPENCL_WORKGROUP_SIZE_0          16 // workgroup_size[0]
#define AGO_OPENCL_WORKGROUP_SIZE_1          16 // workgroup_size[1]
#define AGO_OPENCL_WORKGROUP_SIZE_2           1 // workgroup_size[2]

// Flag to enable BMI2 instructions in the primitives
#define USE_BMI2 0

// Flag to enable AVX instructions (256 bit operations) in primitives
#define USE_AVX 0

// AGO configuration
#define USE_AGO_CANNY_SOBEL_SUPP_THRESHOLD    0// 0:seperate-sobel-and-nonmaxsupression 1:combine-sobel-and-nonmaxsupression
#define AGO_MEMORY_ALLOC_EXTRA_PADDING       64 // extra bytes to the left and right of buffer allocations
#define AGO_MAX_DEPTH_FROM_DELAY_OBJECT       4 // number of levels from delay object to low-level object

// AGO internal error codes for debug
#define AGO_SUCCESS                           0 // operation is successful
#define AGO_ERROR_FEATURE_NOT_IMPLEMENTED    -1 // TBD: this needs to be set to -ve number
#define AGO_ERROR_KERNEL_NOT_IMPLEMENTED     -1 // TBD: this needs to be set to -ve number
#define AGO_ERROR_HAFCPU_NOT_IMPLEMENTED     -1 // TBD: this needs to be set to -ve number

// AGO kernel flags that are part of kernel configuration
#define AGO_KERNEL_FLAG_GROUP_MASK       0x000f // kernel group mask
#define AGO_KERNEL_FLAG_GROUP_AMDLL      0x0000 // kernel group: AMD low-level kernels
#define AGO_KERNEL_FLAG_GROUP_OVX10      0x0001 // kernel group: OpenVX 1.0 built-in kernels
#define AGO_KERNEL_FLAG_GROUP_USER       0x0002 // kernel group: User kernels
#define AGO_KERNEL_FLAG_DEVICE_MASK      0x00f0 // kernel device mask
#define AGO_KERNEL_FLAG_DEVICE_CPU       0x0010 // kernel device: CPU (shall be same as AGO_TARGET_AFFINITY_CPU)
#define AGO_KERNEL_FLAG_DEVICE_GPU       0x0020 // kernel device: GPU (shall be same as AGO_TARGET_AFFINITY_GPU)
#define AGO_KERNEL_FLAG_GPU_INTEG_MASK   0x0f00 // kernel GPU integration type mask
#define AGO_KERNEL_FLAG_GPU_INTEG_NONE   0x0000 // kernel GPU integration: no integration needed
#define AGO_KERNEL_FLAG_GPU_INTEG_FULL   0x0100 // kernel GPU integration: full OpenCL kernel supplied
#define AGO_KERNEL_FLAG_GPU_INTEG_M2R    0x0200 // kernel GPU integration: need OpenCL kernel generation (MEM2REG)
#define AGO_KERNEL_FLAG_GPU_INTEG_R2R    0x0400 // kernel GPU integration: need OpenCL kernel generation (REG2REG)
#define AGO_KERNEL_FLAG_SUBGRAPH         0x1000 // kernel is a subgraph
#define AGO_KERNEL_FLAG_VALID_RECT_RESET 0x2000 // kernel valid_rect_reset is true

// AGO default target priority
#if (ENABLE_OPENCL||ENABLE_HIP)
#define AGO_KERNEL_TARGET_DEFAULT        AGO_KERNEL_FLAG_DEVICE_GPU // pick CPU or GPU
#else
#define AGO_KERNEL_TARGET_DEFAULT        AGO_KERNEL_FLAG_DEVICE_CPU // pick CPU or GPU
#endif

// AGO kernel argument flags
#define AGO_KERNEL_ARG_INPUT_FLAG          0x01 // argument is input
#define AGO_KERNEL_ARG_OUTPUT_FLAG         0x02 // argument is output
#define AGO_KERNEL_ARG_OPTIONAL_FLAG       0x04 // argument is optional

// AGO kernel operation type info
#define AGO_KERNEL_OP_TYPE_UNKNOWN            0 // unknown
#define AGO_KERNEL_OP_TYPE_ELEMENT_WISE       1 // element wise operation
#define AGO_KERNEL_OP_TYPE_FIXED_NEIGHBORS    2 // filtering operation with fixed neighborhood

// AGO magic code
#define AGO_MAGIC_VALID              0xC001C0DE // magic code: reference is valid
#define AGO_MAGIC_INVALID            0xC0FFC0DE // magic code: reference is invalid

// AGO limites
#define AGO_MAX_CONVOLUTION_DIM               9 // maximum size of convolution matrix
#define AGO_MAX_NONLINEAR_FILTER_DIM          9 // maximum size of nonlinear filter matrix the specification requires support for is 9x9
#define AGO_OPTICALFLOWPYRLK_MAX_DIM         15 // maximum size of opticalflow block size
#define AGO_MAX_TENSOR_DIMENSIONS             6 // maximum dimensions supported by tensor
#define AGO_MAX_OBJARR_REF 				   4096 // maximum number of references in a context for object array

// AGO remap data precision
#define AGO_REMAP_FRACTIONAL_BITS             3 // number of fractional bits in re-map locations
#define AGO_REMAP_CONSTANT_BORDER_VALUE  0xffff // corrdinate value indicating out of border for constant fills

// AGO buffer sync flags
#define AGO_BUFFER_SYNC_FLAG_DIRTY_MASK         0x0000000f // dirty bit mask
#define AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT    0x00000001 // buffer dirty by user
#define AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE      0x00000002 // buffer dirty by node
#define AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL   0x00000004 // OpenCL buffer dirty by node
#define AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED      0x00000008 // OpenCL buffer has been synced
#define AGO_BUFFER_SYNC_FLAG_DIRTY_BY_WRITE     0x00000010 // buffer dirty by write

// AGO graph optimizer
#define AGO_GRAPH_OPTIMIZER_FLAG_NO_DIVIDE                0x00000001 // don't run drama divide
#define AGO_GRAPH_OPTIMIZER_FLAG_NO_REMOVE_COPY_NODES     0x00000002 // don't remove unnecessary copy operations
#define AGO_GRAPH_OPTIMIZER_FLAG_NO_REMOVE_UNUSED_OUTPUTS 0x00000004 // don't remove nodes with unused outputs
#define AGO_GRAPH_OPTIMIZER_FLAG_NO_NODE_MERGE            0x00000008 // don't perform node merge
#define AGO_GRAPH_OPTIMIZER_FLAG_NO_CONVERT_8BIT_TO_1BIT  0x00000010 // don't convert 8-bit images to 1-bit images
#define AGO_GRAPH_OPTIMIZER_FLAG_NO_SUPERNODE_MERGE       0x00000020 // don't merge supernodes
#define AGO_GRAPH_OPTIMIZER_FLAGS_DEFAULT                 0x00000000 // default options

#if ENABLE_OPENCL
// bit-fields of opencl_type
#define NODE_OPENCL_TYPE_REG2REG              1 // register to register
#define NODE_OPENCL_TYPE_MEM2REG              2 // memory to register
#define NODE_OPENCL_TYPE_NEED_IMGSIZE         8 // need image size as argument for memory operation
#define NODE_OPENCL_TYPE_FULL_KERNEL         16 // node is a single kernel
// additional bit-fields for dataFlags[]
#define DATA_OPENCL_FLAG_BUFFER        (1 <<  8) // marks that the data is a buffer
#define DATA_OPENCL_FLAG_NEED_LOAD_R2R (1 <<  9) // marks that the data needs to load for REG2REG
#define DATA_OPENCL_FLAG_NEED_LOAD_M2R (1 << 10) // marks that the data needs to load for MEM2REG
#define DATA_OPENCL_FLAG_NEED_LOCAL    (1 << 11) // marks that the data needs to load into local buffer
#define DATA_OPENCL_FLAG_DISCARD_PARAM (1 << 12) // marks that the data needs to be discarded
#define DATA_OPENCL_FLAG_PASS_BY_VALUE (1 << 13) // marks that the data needs to be passed by value
// kernel name
#define NODE_OPENCL_KERNEL_NAME  "OpenVX_kernel"
// opencl related constants
#define DATA_GPU_ARRAY_OFFSET             16  // first 16 bytes of array buffer will be used for numitems
// opencl configuration flags
#define CONFIG_OPENCL_USE_1_2              0x0001  // use OpenCL 1.2
#if defined(CL_VERSION_2_0)
#define CONFIG_OPENCL_SVM_MASK             0x00F0  // OpenCL SVM flags mask
#define CONFIG_OPENCL_SVM_ENABLE           0x0010  // use OpenCL SVM
#define CONFIG_OPENCL_SVM_AS_FGS           0x0020  // use OpenCL SVM as fine grain system
#define CONFIG_OPENCL_SVM_AS_CLMEM         0x0040  // use OpenCL SVM as cl_mem
#endif
#elif ENABLE_HIP
#define NODE_HIP_TYPE_REG2REG              1 // register to register
#define NODE_HIP_TYPE_MEM2REG              2 // memory to register
#define NODE_HIP_TYPE_NEED_IMGSIZE         8 // need image size as argument for memory operation
#define NODE_HIP_TYPE_FULL_KERNEL         16 // node is a single kernel

// additional bit-fields for dataFlags[]
#define DATA_HIP_FLAG_BUFFER        (1 <<  8) // marks that the data is a buffer
#define DATA_HIP_FLAG_NEED_LOAD_R2R (1 <<  9) // marks that the data needs to load for REG2REG
#define DATA_HIP_FLAG_NEED_LOAD_M2R (1 << 10) // marks that the data needs to load for MEM2REG
#define DATA_HIP_FLAG_NEED_LOCAL    (1 << 11) // marks that the data needs to load into local buffer
#define DATA_HIP_FLAG_DISCARD_PARAM (1 << 12) // marks that the data needs to be discarded
#define DATA_HIP_FLAG_PASS_BY_VALUE (1 << 13) // marks that the data needs to be passed by value
#endif
// opencl related constants
#define DATA_GPU_ARRAY_OFFSET             16  // first 16 bytes of array buffer will be used for numitems
// GPU image fixed byte offset
#define GPU_IMAGE_FIXED_OFFSET             256

// thread scheduling configuration
#define CONFIG_THREAD_DEFAULT                 1  // 0:disable 1:enable separate threads for graph scheduling

// module specific
#define MAX_MODULE_NAME_SIZE 1024
#define MAX_MODULE_PATH_SIZE 2048
#define MAX_DESCRIPTION_DATA_SIZE 1024

// threshold default values
#define AGO_DEFAULT_THRESHOLD_FALSE_VALUE 0
#define AGO_DEFAULT_THRESHOLD_TRUE_VALUE  255

#define AGO_U1_THRESHOLD_FALSE_VALUE vx_false_e
#define AGO_U1_THRESHOLD_TRUE_VALUE vx_true_e
#define AGO_S16_THRESHOLD_FALSE_VALUE 0
#define AGO_S16_THRESHOLD_TRUE_VALUE  (-1)
#define AGO_U16_THRESHOLD_FALSE_VALUE 0
#define AGO_U16_THRESHOLD_TRUE_VALUE  0xFFFF
#define AGO_S32_THRESHOLD_FALSE_VALUE 0
#define AGO_S32_THRESHOLD_TRUE_VALUE  (-1)
#define AGO_U32_THRESHOLD_FALSE_VALUE 0
#define AGO_U32_THRESHOLD_TRUE_VALUE  0xFFFFFFFF

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// helpful macros
//
#define dimof(x)                    (sizeof(x)/sizeof(x[0]))
#define FORMAT_STR(fmt)             ((const char *)&(fmt))
#if ENABLE_DEBUG_MESSAGES
#define debug_printf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define debug_printf(fmt, ...)
#endif
//   ALIGN16 - aligns data to 16 multiple
//   ALIGN32 - aligns data to 32 multiple
//   ALIGN32PTR - aligns pointer to 32 multiple
#define ALIGN16(x)		((((size_t)(x))+15)&~15)
#define ALIGN32(x)		((((size_t)(x))+31)&~31)
#define ALIGN32PTR(x)	((((uintptr_t)(x))+31)&~31)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ago data types
//
#define AgoReference  _vx_reference
#define AgoContext    _vx_context
#define AgoGraph      _vx_graph
#define AgoKernel     _vx_kernel
#define AgoNode       _vx_node
#define AgoParameter  _vx_parameter
#define AgoMetaFormat _vx_meta_format
typedef enum {
    ago_kernel_cmd_execute                    =  0,
    ago_kernel_cmd_validate                   =  1,
    ago_kernel_cmd_get_image_plane_nonusage   =  2,
    ago_kernel_cmd_initialize                 =  3,
    ago_kernel_cmd_shutdown                   =  4,
    ago_kernel_cmd_query_target_support       =  5,
#if ENABLE_OPENCL
    ago_kernel_cmd_opencl_codegen             =  6,
#endif
    ago_kernel_cmd_valid_rect_callback        =  7,
#if ENABLE_HIP
    ago_kernel_cmd_hip_execute                =  8,
#endif
} AgoKernelCommand;
typedef enum {
    ago_profile_type_launch_begin,
    ago_profile_type_launch_end,
    ago_profile_type_wait_begin,
    ago_profile_type_wait_end,
    ago_profile_type_copy_begin,
    ago_profile_type_copy_end,
    ago_profile_type_exec_begin,
    ago_profile_type_exec_end,
} AgoProfileEntryType;
struct AgoProfileEntry {
    vx_uint32           id;
    AgoProfileEntryType type;
    vx_reference        ref;
    int64_t             time;
};
struct AgoNode;
struct AgoContext;
struct AgoData;
struct AgoReference {
    struct _vx_platform * platform; // platform handle to support Installable Client Driver (ICD) loader
    vx_uint32    magic;           // shall be always be AGO_MAGIC
    vx_enum      type;            // object type
    AgoContext * context;         // context
    AgoReference * scope;         // scope parent -- for virtual objects, this will be graph
    vx_uint32    external_count;  // user usage count -- can't be free when > 0, can't be access when == 0
    vx_uint32    internal_count;  // framework usage count -- can't be free when > 0
    vx_uint32    read_count;      // number of times object has been read
    vx_uint32    write_count;     // number of times object has been written
    bool         hint_serialize;  // serialize hint
    bool         enable_logging;  // enable logging
    bool         read_only;       // read only
    bool 		 enable_perf; 	  // enable performance counter
    vx_status    status;          // error status
public:
    AgoReference();
    ~AgoReference();
};
struct AgoConfigDelay {
    vx_enum type;
    vx_int32 age;
    vx_uint32 count;
};
struct AgoConfigArray {
    vx_enum itemtype;
    vx_size numitems;
    vx_size capacity;
    vx_size itemsize;
};
struct AgoConfigObjectArray {
    vx_enum itemtype;
    vx_size numitems;
    //vx_reference *items;
};
struct AgoConfigConvolution {
    vx_size rows;
    vx_size columns;
    vx_uint32 shift;
    bool is_separable;
};
struct AgoConfigDistribution {
    vx_size numbins;
    vx_int32 offset;
    vx_uint32 range;
    vx_uint32 window;
};
struct AgoConfigImage {
    vx_uint32 width;
    vx_uint32 height;
    vx_df_image format;
    vx_uint32 stride_in_bytes;
    vx_uint32 pixel_size_in_bits_num;
    vx_uint32 pixel_size_in_bits_denom;
    vx_size components;
    vx_size planes;
    vx_bool isVirtual;
    vx_bool isUniform;
    vx_size uniform[4];
    vx_bool isROI;
    vx_rectangle_t rect_roi;
    vx_rectangle_t rect_valid;
    AgoData * roiMasterImage;
    vx_bool hasMinMax;
    vx_int32 minValue;
    vx_int32 maxValue;
    vx_color_space_e color_space;
    vx_channel_range_e channel_range;
    vx_uint32 x_scale_factor_is_2; // will be 0 or 1
    vx_uint32 y_scale_factor_is_2; // will be 0 or 1
    vx_bool enableUserBufferGPU;
    vx_bool mem_handle;     // false by default ; true if image created from handle and the handle is NULL
};
struct AgoConfigLut {
    vx_enum type;
    vx_uint32 offset;
    vx_size count;
};
struct AgoConfigMatrix {
    vx_enum type;
    vx_size columns;
    vx_size rows;
    vx_size itemsize;
    vx_enum pattern;
    vx_coordinates2d_t origin;
};
struct AgoConfigPyramid {
    vx_uint32 width;
    vx_uint32 height;
    vx_df_image format;
    vx_float32 scale;
    vx_size levels;
    vx_bool isVirtual;
    vx_rectangle_t rect_valid;
};
struct AgoConfigRemap {
    vx_uint32 src_width;
    vx_uint32 src_height;
    vx_uint32 dst_width;
    vx_uint32 dst_height;
    vx_uint32 remap_fractional_bits;
};
struct AgoConfigScalar {
    vx_enum type;
    union {
        vx_enum e;
        vx_float32 f;
        vx_int32 i;
        vx_uint32 u;
        vx_df_image df;
        vx_size s;
        vx_int64 i64;
        vx_uint64 u64;
        vx_float64 f64;
    } u;
    vx_size itemsize;
};
struct AgoConfigThreshold {
    vx_enum thresh_type;
    vx_enum data_type;
    vx_df_image input_format;
    vx_df_image output_format;
    vx_pixel_value_t threshold_lower, threshold_upper, threshold_value;
    vx_pixel_value_t true_value, false_value;
};
struct AgoConfigTensor {
    vx_size num_dims;
    vx_size dims[AGO_MAX_TENSOR_DIMENSIONS];
    vx_enum data_type;
    vx_uint32 fixed_point_pos;
    vx_size stride[AGO_MAX_TENSOR_DIMENSIONS];
    vx_size offset;
    AgoData * roiMaster;
    vx_size start[AGO_MAX_TENSOR_DIMENSIONS];
    vx_size end[AGO_MAX_TENSOR_DIMENSIONS];
};
struct AgoConfigCannyStack {
    vx_uint32 count;
    vx_uint32 stackTop;
};
struct AgoConfigScaleMatrix {
    vx_float32 xscale;
    vx_float32 yscale;
    vx_float32 xoffset;
    vx_float32 yoffset;
};
struct AgoTargetAffinityInfo_ { // NOTE: make sure that this data structure is identical to AgoTargetAffinityInfo in vx_amd_ext.h
    vx_uint32 device_type;
    vx_uint32 device_info;
    vx_uint32 group;
    vx_uint32 reserved;
};
struct MappedData {
    vx_map_id map_id;
    void * ptr;
    vx_enum usage;
    bool used_external_ptr;
    vx_size stride;
    vx_uint32 plane;
};
struct AgoData {
    AgoReference ref;
    AgoData * next;
    std::string name;
    union {
        AgoConfigDelay delay;
        AgoConfigArray arr;
        AgoConfigObjectArray objarr;
        AgoConfigConvolution conv;
        AgoConfigDistribution dist;
        AgoConfigImage img;
        AgoConfigLut lut;
        AgoConfigMatrix mat;
        AgoConfigPyramid pyr;
        AgoConfigRemap remap;
        AgoConfigScalar scalar;
        AgoConfigThreshold thr;
        AgoConfigCannyStack cannystack;
        AgoConfigScaleMatrix scalemat;
        AgoConfigTensor tensor;
    } u;
    vx_size size;
    vx_enum import_type;
    vx_uint8 * buffer;
    vx_uint8 * buffer_allocated;
    vx_uint8 * reserved;
    vx_uint8 * reserved_allocated;
    vx_uint32  buffer_sync_flags;
#if ENABLE_OPENCL
    cl_mem     opencl_buffer;
    cl_mem     opencl_buffer_allocated;
#if defined(CL_VERSION_2_0)
    vx_uint8 * opencl_svm_buffer;
    vx_uint8 * opencl_svm_buffer_allocated;
#endif
#elif ENABLE_HIP
    vx_uint8*     hip_memory;
    vx_uint8*     hip_memory_allocated;
#endif
    vx_uint32  gpu_buffer_offset;
    vx_bool isVirtual;
    vx_bool isDelayed;
    vx_bool isNotFullyConfigured;
    vx_bool isInitialized;
    vx_int32 siblingIndex;
    vx_uint32 numChildren;
    AgoData ** children;
    AgoData * parent;
    vx_uint32 inputUsageCount, outputUsageCount, inoutUsageCount;
    std::list<MappedData> mapped;
    vx_map_id nextMapId;
    vx_uint32 hierarchical_level;
    struct AgoNode * ownerOfUserBufferGPU;
    std::list<AgoData *> roiDepList;
    vx_uint32 hierarchical_life_start;
    vx_uint32 hierarchical_life_end;
    vx_uint32 initialization_flags;
    vx_uint32 device_type_unused;
    AgoData * alias_data;
    vx_size   alias_offset;
public:
    AgoData();
    ~AgoData();
};
struct AgoDataList {
    vx_uint32 count;
    AgoData * head;
    AgoData * tail;
    AgoData * trash;
};
struct AgoMetaFormat {
    // TBD: this data struct needs some cleanup -- just keep only required fields
    AgoData data;
    vx_kernel_image_valid_rectangle_f set_valid_rectangle_callback;
public:
    AgoMetaFormat();
};
struct AgoParameter {
    AgoReference ref;
    AgoReference * scope;
    vx_uint32 index;
    vx_direction_e direction;
    vx_enum type;
    vx_parameter_state_e state;
public:
    AgoParameter();
    ~AgoParameter();
};
struct AgoKernel {
    AgoReference ref;
    AgoKernel * next;
    vx_enum id;
    vx_char name[VX_MAX_KERNEL_NAME];
    vx_uint32 flags;
    int(*func)(AgoNode * node, AgoKernelCommand cmd);
    vx_uint32 argCount;
    vx_uint8 argConfig[AGO_MAX_PARAMS];
    vx_enum argType[AGO_MAX_PARAMS];
    vx_uint8 kernOpType;
    vx_uint8 kernOpInfo;
    AgoParameter parameters[AGO_MAX_PARAMS];
    vx_size localDataSize;
    vx_uint8 * localDataPtr;
    bool external_kernel;
    bool finalized;
    vx_bool user_kernel;
    vx_kernel_f kernel_f;
    vx_kernel_validate_f validate_f;
    vx_kernel_input_validate_f input_validate_f;
    vx_kernel_output_validate_f output_validate_f;
    vx_kernel_initialize_f initialize_f;
    vx_kernel_deinitialize_f deinitialize_f;
    amd_kernel_query_target_support_f query_target_support_f;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f;
    amd_kernel_node_regen_callback_f regen_callback_f;
    amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f;
    amd_kernel_gpu_buffer_update_callback_f gpu_buffer_update_callback_f;
    vx_uint32 gpu_buffer_update_param_index;
    vx_bool opencl_buffer_access_enable;
    vx_uint32 importing_module_index_plus1;
public:
    AgoKernel();
    ~AgoKernel();
};
struct AgoSuperNodeDataInfo {
    vx_uint32 data_type_flags;
    bool needed_as_a_kernel_argument;
    vx_uint32 argument_usage[3]; // VX_INPUT, VX_OUTPUT, VX_BIDIRECTIONAL
    vx_uint32 local_buffer_size_in_bytes;
};
struct AgoSuperNode {
    AgoSuperNode * next;
    vx_uint32 group;
    vx_uint32 width;
    vx_uint32 height;
    std::vector<AgoNode *> nodeList;
    std::vector<AgoData *> dataList;
    std::vector<AgoData *> dataListForAgeDelay;
    std::vector<AgoSuperNodeDataInfo> dataInfo;
    std::string opencl_code;
    bool launched;
    bool isGpuOclSuperNode;
#if ENABLE_OPENCL
    cl_command_queue opencl_cmdq;
    cl_program opencl_program;
    cl_kernel opencl_kernel;
    cl_event opencl_event;
    size_t opencl_global_work[3];
    size_t opencl_local_work[3];
#elif ENABLE_HIP
    hipStream_t  hip_stream0;       // default stream for the graph
#endif
    vx_uint32 hierarchical_level_start;
    vx_uint32 hierarchical_level_end;
    vx_status status;
    vx_perf_t perf;
public:
    AgoSuperNode();
    ~AgoSuperNode();
};
struct AgoNode {
    AgoReference ref;
    AgoNode * next;
    AgoKernel * akernel;
    vx_uint32 flags;
    vx_border_mode_t attr_border_mode;
    vx_bool valid_rect_reset;
    AgoTargetAffinityInfo_ attr_affinity;
    vx_size localDataSize;
    vx_uint8 * localDataPtr;
    vx_uint8 * localDataPtr_allocated;
    vx_uint32 paramCount;
    AgoData * paramList[AGO_MAX_PARAMS];
    AgoData * paramListForAgeDelay[AGO_MAX_PARAMS];
    AgoParameter parameters[AGO_MAX_PARAMS];
    AgoMetaFormat metaList[AGO_MAX_PARAMS];
    vx_int32 funcExchange[AGO_MAX_PARAMS];
    vx_nodecomplete_f callback;
    AgoSuperNode * supernode;
    AgoNode * newchildnode;
    bool initialized;
    bool drama_divide_invoked;
    vx_uint32 valid_rect_num_inputs;
    vx_uint32 valid_rect_num_outputs;
    vx_rectangle_t ** valid_rect_inputs;
    vx_rectangle_t ** valid_rect_outputs;
    vx_uint32 target_support_flags;
    vx_uint32 hierarchical_level;
    vx_status status;
    vx_perf_t perf;
    vx_bool local_data_change_is_enabled;
    vx_bool local_data_set_by_implementation;
    struct { bool enable; int paramIndexScalar; int paramIndexArray; } gpu_scalar_array_output_sync;
#if ENABLE_OPENCL
    vx_uint32 opencl_type;
    char opencl_name[VX_MAX_KERNEL_NAME+8];
    std::string opencl_code;
    std::string opencl_build_options;
    vx_uint32 opencl_param_mem2reg_mask;
    vx_uint32 opencl_param_discard_mask;
    vx_uint32 opencl_param_as_value_mask;
    vx_uint32 opencl_param_atomic_mask;
    vx_uint32 opencl_local_buffer_usage_mask;
    vx_uint32 opencl_local_buffer_size_in_bytes;
    vx_uint32 opencl_work_dim;
    size_t opencl_global_work[3];
    size_t opencl_local_work[3];
    vx_uint32 opencl_compute_work_multiplier;
    vx_uint32 opencl_compute_work_param_index;
    vx_uint32 opencl_output_array_param_index_plus1;
    cl_program opencl_program;
    cl_kernel opencl_kernel;
    cl_event opencl_event;
#elif ENABLE_HIP
    hipStream_t hip_stream0;
#endif
public:
    AgoNode();
    ~AgoNode();
};
struct AgoUserStruct {
    vx_enum id;
    vx_size size;
    std::string name;
    vx_uint32 importing_module_index_plus1;
};
struct AgoKernelList {
    vx_uint32 count;
    AgoKernel * head;
    AgoKernel * tail;
};
struct AgoNodeList {
    vx_uint32 count;
    AgoNode * head;
    AgoNode * tail;
    AgoNode * trash;
};
struct AgoGraph {
    AgoReference ref;
    std::string name;
    AgoGraph * next;
    CRITICAL_SECTION cs;
    HANDLE hThread, hSemToThread, hSemFromThread;
    vx_int32 threadScheduleCount, threadExecuteCount, threadWaitCount, threadThreadTerminationState, threadThreadWaitState;
    AgoDataList dataList;
    AgoNodeList nodeList;
    vx_bool isReadyToExecute;
    bool detectedInvalidNode;
    vx_int32 status;
    vx_perf_t perf;
    vx_uint32 cpu_num_threads;
    vx_enum state;
    bool reverify;
    struct AgoGraphPerfInternalInfo_ { // shall be identical to AgoGraphPerfInternalInfo in amd_ext_amd.h
        vx_uint64 kernel_enqueue;
        vx_uint64 kernel_wait;
        vx_uint64 buffer_read;
        vx_uint64 buffer_write;
    };
    AgoGraphPerfInternalInfo_ gpu_perf, gpu_perf_total;
    vx_uint32 virtualDataGenerationCount;
    vx_uint32 optimizer_flags;
    bool verified;
    std::vector<vx_parameter> parameters;
    std::vector<AgoData *> autoAgeDelayList;
#if (ENABLE_OPENCL||ENABLE_HIP)
    std::vector<AgoNode *> gpu_nodeListQueued;
    AgoSuperNode * supernodeList;
    bool enable_node_level_gpu_flush;
#if ENABLE_OPENCL
    cl_command_queue opencl_cmdq;
    cl_device_id opencl_device;
#elif ENABLE_HIP
    hipStream_t  hip_stream0;       // default stream for the graph
#endif
#endif
    AgoTargetAffinityInfo_ attr_affinity;
    vx_uint32 execFrameCount;
    bool enable_performance_profiling;
    std::vector<AgoProfileEntry> performance_profile;
    std::map<std::string,void *> moduleHandle;
public:
    AgoGraph();
    ~AgoGraph();
};
struct AgoGraphList {
    vx_uint32 count;
    AgoGraph * head;
    AgoGraph * tail;
};
struct AgoImageFormatDescItem {
    vx_df_image               format;
    AgoImageFormatDescription desc;
};
struct ModuleData {
    char module_name[MAX_MODULE_NAME_SIZE];
    char module_path[MAX_MODULE_PATH_SIZE];
    ago_module hmodule;
    vx_uint8 * module_internal_data_ptr;
    vx_size module_internal_data_size;
};
struct MacroData {
    char name[256];
    char * text;
    char * text_allocated;
};
struct AgoContext {
    AgoReference ref;
    vx_uint64 perfNormFactor;
    CRITICAL_SECTION cs;
    AgoKernelList kernelList;
    AgoDataList dataList;
    AgoGraphList graphList;
    std::vector<AgoUserStruct> userStructList;
    vx_uint32 dataGenerationCount;
    vx_enum nextUserStructId;
    vx_uint32 nextUserKernelId;
    vx_uint32 nextUserLibraryId;
    vx_uint32 num_active_modules;
    vx_uint32 num_active_references;
    vx_border_mode_t immediate_border_mode;
    vx_border_mode_policy_e immediate_border_policy;
    vx_log_callback_f callback_log;
    vx_bool callback_reentrant;
    vx_uint32 thread_config;
    vx_char extensions[256];
    std::vector<ModuleData> modules;
    std::vector<MacroData> macros;
    std::vector<AgoNodeMergeRule> merge_rules;
    std::vector<AgoImageFormatDescItem> image_format_list;
    vx_uint32 importing_module_index_plus1;
    AgoData * graph_garbage_data;
    AgoNode * graph_garbage_node;
    AgoGraph * graph_garbage_list;
#if ENABLE_OPENCL
    bool opencl_context_imported;
    cl_context   opencl_context;
    cl_command_queue opencl_cmdq;
    vx_uint32 opencl_config_flags;
    char opencl_extensions[1024];
#if defined(CL_VERSION_2_0)
    cl_device_svm_capabilities opencl_svmcaps;
#endif
    cl_command_queue_properties opencl_cmdq_properties;
    cl_uint      opencl_num_devices;
    cl_device_id opencl_device_list[16];
    char opencl_build_options[256];
    bool isAmdMediaOpsSupported;
    vx_size opencl_mem_alloc_size;
    vx_size opencl_mem_alloc_count;
    vx_size opencl_mem_release_count;
#elif ENABLE_HIP
    bool hip_context_imported;
    hipDeviceProp_t hip_dev_prop;
    int hip_num_devices;
    int hip_device_id;
    hipDevice_t hip_device;
    vx_size hip_mem_alloc_size;
    vx_size hip_mem_alloc_count;
    vx_size hip_mem_release_count;
#endif
    AgoTargetAffinityInfo_ attr_affinity;
public:
    AgoContext();
    ~AgoContext();
};
struct AgoAllocInfo {
    void * allocated;
    vx_size requested_size;
    vx_int32 retain_count;
    vx_int32 allocate_id;
};

struct _vx_array { AgoData d; };
struct _vx_convolution { AgoData d; };
struct _vx_delay { AgoData d; };
struct _vx_distribution { AgoData d; };
struct _vx_image { AgoData d; };
struct _vx_lut { AgoData d; };
struct _vx_matrix { AgoData d; };
struct _vx_pyramid { AgoData d; };
struct _vx_remap { AgoData d; };
struct _vx_scalar { AgoData d; };
struct _vx_threshold { AgoData d; };
struct _vx_object_array { AgoData d; };

// framework
void * agoAllocMemory(vx_size size);
void agoRetainMemory(void * mem);
void agoReleaseMemory(void * mem);
int agoChannelEnum2Index(vx_enum channel);
const char * agoEnum2Name(vx_enum e);
size_t agoType2Size(vx_context context, vx_enum type);
vx_enum agoName2Enum(const char * name);
void agoResetReference(AgoReference * ref, vx_enum type, vx_context context, vx_reference scope);
void agoAddData(AgoDataList * dataList, AgoData * data);
void agoAddNode(AgoNodeList * nodeList, AgoNode * node);
void agoAddKernel(AgoKernelList * kernelList, AgoKernel * kernel);
void agoAddGraph(AgoGraphList * graphList, AgoGraph * graph);
vx_enum agoAddUserStruct(AgoContext * acontext, vx_size size, vx_char * name);
AgoGraph * agoRemoveGraph(AgoGraphList * list, AgoGraph * item);
int agoRemoveNode(AgoNodeList * nodeList, AgoNode * node, bool moveToTrash);
int agoShutdownNode(AgoNode * node);
int agoRemoveData(AgoDataList * list, AgoData * item, AgoData ** trash);
AgoKernel * agoRemoveKernel(AgoKernelList * list, AgoKernel * item);
void agoRemoveDataInGraph(AgoGraph * agraph, AgoData * data);
void agoReplaceDataInGraph(AgoGraph * agraph, AgoData * dataFind, AgoData * dataReplace);
void agoResetDataList(AgoDataList * dataList);
void agoResetNodeList(AgoNodeList * nodeList);
void agoResetKernelList(AgoKernelList * kernelList);
vx_size agoGetUserStructSize(AgoContext * acontext, vx_char * name);
vx_size agoGetUserStructSize(AgoContext * acontext, vx_enum id);
vx_enum agoGetUserStructType(AgoContext * acontext, vx_char * name);
const char * agoGetUserStructName(AgoContext * acontext, vx_enum id);
AgoKernel * agoFindKernelByEnum(AgoContext * acontext, vx_enum kernel_id);
AgoKernel * agoFindKernelByName(AgoContext * acontext, const vx_char * name);
AgoData * agoFindDataByName(AgoContext * acontext, AgoGraph * agraph, vx_char * name);
void agoMarkChildrenAsPartOfDelay(AgoData * adata);
bool agoIsPartOfDelay(AgoData * adata);
AgoData * agoGetSiblingTraceToDelayForInit(AgoData * data, int trace[], int& traceCount);
AgoData * agoGetSiblingTraceToDelayForUpdate(AgoData * data, int trace[], int& traceCount);
AgoData * agoGetDataFromTrace(AgoData * data, int trace[], int traceCount);
int agoUpdateDelaySlots(AgoNode * node);
void agoGetDescriptionFromData(AgoContext * acontext, char * desc, AgoData * data);
int agoGetDataFromDescription(AgoContext * acontext, AgoGraph * agraph, AgoData * data, const char * desc);
AgoData * agoCreateDataFromDescription(AgoContext * acontext, AgoGraph * agraph, const char * desc, bool isForExternalUse);
void agoGenerateDataName(AgoContext * acontext, const char * postfix, std::string& name);
void agoGenerateVirtualDataName(AgoGraph * agraph, const char * postfix, std::string& name);
int agoInitializeImageComponentsAndPlanes(AgoContext * acontext);
int agoSetImageComponentsAndPlanes(AgoContext * acontext, vx_df_image format, vx_size components, vx_size planes, vx_uint32 pixelSizeInBitsNum, vx_uint32 pixelSizeInBitsDenom, vx_color_space_e colorSpace, vx_channel_range_e channelRange);
int agoGetImageComponentsAndPlanes(AgoContext * acontext, vx_df_image format, vx_size * pComponents, vx_size * pPlanes, vx_uint32 * pPixelSizeInBitsNum, vx_uint32 * pPixelSizeInBitsDenom, vx_color_space_e * pColorSpace, vx_channel_range_e * pChannelRange);
int agoGetImagePlaneFormat(AgoContext * acontext, vx_df_image format, vx_uint32 width, vx_uint32 height, vx_uint32 plane, vx_df_image *pFormat, vx_uint32 * pWidth, vx_uint32 * pHeight);
void agoGetDataName(vx_char * name, AgoData * data);
int agoAllocData(AgoData * data);
void agoRetainData(AgoGraph * graph, AgoData * data, bool isForExternalUse);
int agoReleaseData(AgoData * data, bool isForExternalUse);
int agoReleaseKernel(AgoKernel * kernel, bool isForExternalUse);
AgoNode * agoCreateNode(AgoGraph * graph, AgoKernel * kernel);
AgoNode * agoCreateNode(AgoGraph * graph, vx_enum kernel_id);
int agoReleaseNode(AgoNode * node);
vx_status agoVerifyNode(AgoNode * node);
// sanity checks
int agoDataSanityCheckAndUpdate(AgoData * data);
bool agoIsValidReference(AgoReference * ref);
bool agoIsValidContext(AgoContext * context);
bool agoIsValidGraph(AgoGraph * graph);
bool agoIsValidKernel(AgoKernel * kernel);
bool agoIsValidNode(AgoNode * node);
bool agoIsValidParameter(AgoParameter * parameter);
bool agoIsValidData(AgoData * data, vx_enum type);
// kernels
int agoPublishKernels(AgoContext * acontext);
// drama
int agoOptimizeDrama(AgoGraph * agraph);
void agoOptimizeDramaMarkDataUsage(AgoGraph * agraph);
int agoOptimizeDramaComputeGraphHierarchy(AgoGraph * graph);
void agoOptimizeDramaSortGraphHierarchy(AgoGraph * graph);
int agoOptimizeDramaCheckArgs(AgoGraph * agraph);
int agoOptimizeDramaDivide(AgoGraph * agraph);
int agoOptimizeDramaRemove(AgoGraph * agraph);
int agoOptimizeDramaAnalyze(AgoGraph * agraph);
int agoOptimizeDramaMerge(AgoGraph * agraph);
int agoOptimizeDramaAlloc(AgoGraph * agraph);
// import
void agoImportKernelConfig(AgoKernel * kernel, vx_kernel vxkernel);
void agoImportNodeConfig(AgoNode * node, vx_node vxnode);
void agoImportDataConfig(AgoData * data, vx_reference vxref, AgoGraph * graph);
// string processing
void agoEvaluateIntegerExpression(char * expr);
// performance
void agoPerfProfileEntry(AgoGraph * graph, AgoProfileEntryType type, vx_reference ref);
void agoPerfCaptureReset(vx_perf_t * perf);
void agoPerfCaptureStart(vx_perf_t * perf);
void agoPerfCaptureStop(vx_perf_t * perf);
void agoPerfCopyNormalize(AgoContext * context, vx_perf_t * perfDst, vx_perf_t * perfSrc);
// log
void agoRegisterLogCallback(vx_context context, vx_log_callback_f callback, vx_bool reentrant);
void agoAddLogEntry(AgoReference * ref, vx_status status, const char *message, ...);

#if (ENABLE_OPENCL || ENABLE_HIP)
int agoGpuOclAllocBuffers(AgoGraph * graph);

#if ENABLE_OPENCL// OpenCL
int agoGpuOclDataSetBufferAsKernelArg(AgoData * data, cl_kernel opencl_kernel, vx_uint32 kernelArgIndex, vx_uint32 group);
int agoGpuOclReleaseContext(AgoContext * context);
int agoGpuOclReleaseGraph(AgoGraph * graph);
int agoGpuOclReleaseSuperNode(AgoSuperNode * supernode);
int agoGpuOclReleaseData(AgoData * data);
int agoGpuOclCreateContext(AgoContext * context, cl_context opencl_context);
int agoGpuOclAllocBuffer(AgoData * data);
int agoGpuOclSuperNodeMerge(AgoGraph * graph, AgoSuperNode * supernode, AgoNode * node);
int agoGpuOclSuperNodeUpdate(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuOclSuperNodeFinalize(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuOclSuperNodeLaunch(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuOclSuperNodeWait(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuOclSingleNodeFinalize(AgoGraph * graph, AgoNode * node);
int agoGpuOclSingleNodeLaunch(AgoGraph * graph, AgoNode * node);
int agoGpuOclSingleNodeWait(AgoGraph * graph, AgoNode * node);
#elif ENABLE_HIP
int agoGpuHipCreateContext(AgoContext * context, int deviceID = -1);
int agoGpuHipReleaseContext(AgoContext * context);
int agoGpuHipReleaseGraph(AgoGraph * graph);
int agoGpuHipReleaseSuperNode(AgoSuperNode * supernode);
int agoGpuHipReleaseData(AgoData * data);
int agoGpuHipAllocBuffer(AgoData * data);
int agoGpuHipSingleNodeWait(AgoGraph * graph, AgoNode * node);
int agoGpuHipSuperNodeMerge(AgoGraph * graph, AgoSuperNode * supernode, AgoNode * node);
int agoGpuHipSuperNodeUpdate(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuHipSuperNodeFinalize(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuHipSingleNodeLaunch(AgoGraph * graph, AgoNode * node);
int agoGpuHipSuperNodeLaunch(AgoGraph * graph, AgoSuperNode * supernode);
int agoGpuHipSuperNodeWait(AgoGraph * graph, AgoSuperNode * supernode);
#define HIP_MEM_KIND_BUFFER         0x10F0
#define HIP_MEM_KIND_IMAGE1D          0x10F4
#endif
#endif

///////////////////////////////////////////////////////////
// high-level functions
extern "C" typedef void (VX_CALLBACK * ago_data_registry_callback_f) (void * obj, vx_reference ref, const char * name, const char * app_params);
AgoContext * agoCreateContextFromPlatform(struct _vx_platform * platform);
AgoContext * agoCreateContext();
AgoGraph * agoCreateGraph(AgoContext * acontext);
int agoReleaseGraph(AgoGraph * agraph);
int agoReleaseContext(AgoContext * acontext);
int agoVerifyGraph(AgoGraph * agraph);
vx_status agoPrepareImageValidRectangleBuffers(AgoGraph * graph);
vx_status agoComputeImageValidRectangleOutputs(AgoGraph * graph);
int agoOptimizeGraph(AgoGraph * agraph);
int agoInitializeGraph(AgoGraph * agraph);
int agoShutdownGraph(AgoGraph * graph);
int agoExecuteGraph(AgoGraph * agraph);
int agoAgeDelay(AgoData * delay);
// scheduling
int agoProcessGraph(AgoGraph * agraph);
int agoScheduleGraph(AgoGraph * agraph);
int agoWaitGraph(AgoGraph * agraph);
int agoWriteGraph(AgoGraph * agraph, AgoReference * * ref, int num_ref, FILE * fp, const char * comment);
int agoReadGraph(AgoGraph * agraph, AgoReference * * ref, int num_ref, ago_data_registry_callback_f callback_f, void * callback_obj, FILE * fp, vx_int32 dumpToConsole);
int agoReadGraphFromString(AgoGraph * agraph, AgoReference * * ref, int num_ref, ago_data_registry_callback_f callback_f, void * callback_obj, char * str, vx_int32 dumpToConsole);
int agoLoadModule(AgoContext * context, const char * module);
int agoUnloadModule(AgoContext * context, const char * module);
vx_status agoGraphDumpPerformanceProfile(AgoGraph * graph, const char * fileName);
vx_status agoDirective(vx_reference reference, vx_enum directive);

///////////////////////////////////////////////////////////
// locks
void agoLockGlobalContext();
void agoUnlockGlobalContext();
class CAgoLockGlobalContext {
public:
    CAgoLockGlobalContext() { agoLockGlobalContext(); }
    ~CAgoLockGlobalContext() { agoUnlockGlobalContext(); }
};

class CAgoLock {
public:
    CAgoLock(CRITICAL_SECTION& cs) { m_cs = &cs; EnterCriticalSection(m_cs); }
    ~CAgoLock() { LeaveCriticalSection(m_cs); }
private:
    CRITICAL_SECTION * m_cs;
};

inline int leftmostbit(unsigned int n) {
    int pos = 31;
    while (pos >= 0 && !(n & (1 << pos)))
        pos--;
    return pos;
}

inline vx_uint32 ImageWidthInBytesFloor(vx_uint32 width, const AgoData * img)
{
    return ((width * img->u.img.pixel_size_in_bits_num + img->u.img.pixel_size_in_bits_denom - 1) / img->u.img.pixel_size_in_bits_denom) >> 3;
}

inline vx_uint32 ImageWidthInBytesCeil(vx_uint32 width, const AgoData * img)
{
    return ((width * img->u.img.pixel_size_in_bits_num + img->u.img.pixel_size_in_bits_denom - 1) / img->u.img.pixel_size_in_bits_denom + 7) >> 3;
}

#endif // __ago_internal_h__
