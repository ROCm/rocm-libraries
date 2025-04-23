/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "kernels.h"
#if ENABLE_OPENCL
#include <miopengemm/gemm.hpp>
#elif ENABLE_HIP
#include <rocblas/rocblas.h>
#endif
#include <algorithm>

struct LocalData {
    NeuralNetworkCommonHandle * handle;
    bool tA, tB, tI;
    size_t m, n, k;
    size_t a_offset, lda;
    size_t b_offset, ldb;
    size_t i_offset, ldi;
    size_t c_offset, ldc;
    int ID;
#if ENABLE_OPENCL
    cl_kernel copy_kernel;
    size_t copy_global[3];
    size_t copy_local[3];
#elif ENABLE_HIP
    vx_enum type;
    vx_size width, height;
    float alpha, beta;
    vx_uint8 *input1_mem, *input2_mem, *input3_mem, *output_mem;
    hipStream_t hip_stream;
    rocblas_handle rocBlasHandle;
#endif
};

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar type
    vx_enum type, out_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: matmul: #3 type=%d (must be MATMUL_PARAMS)\n", type);
    vx_tensor_matrix_multiply_params_t params = { 0 };
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &params, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // check tensor dimensions
    vx_size num_dims;
    vx_size input1_dims[4] = { 1, 1, 1, 1 };
    vx_size input2_dims[4] = { 1, 1, 1, 1 };
    vx_size input3_dims[4] = { 1, 1, 1, 1 };
    vx_size output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims < 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: #0 num_dims=%ld (must >= 2)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: matmul: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, num_dims*sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims < 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: #1 num_dims=%ld (must >= 2)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: matmul: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input2_dims, num_dims*sizeof(vx_size)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims < 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: #2 num_dims=%ld (must >= 2)\n", num_dims);
        if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: matmul: #2 type=%d (must be float)\n", type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input3_dims, num_dims*sizeof(vx_size)));
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims < 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: #4 num_dims=%ld (must >= 2)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: matmul: #4 type=%d (must be float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, num_dims*sizeof(vx_size)));

    // set output tensor configuration
    out_type = type;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // check that tensors are 2D
    if (((input1_dims[3] & input1_dims[2]) != 1) && ((input1_dims[1] & input1_dims[0]) != 1) ||
       ((input2_dims[3] & input2_dims[2]) != 1) && ((input2_dims[1] & input2_dims[0]) != 1) ||
       ((input3_dims[3] & input3_dims[2]) != 1) && ((input3_dims[1] & input3_dims[0]) != 1) ||
       ((output_dims[3] & output_dims[2]) != 1) && ((output_dims[1] & output_dims[0]) != 1))
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: dims input1[%ld,%ld,%ld,%ld] input2[%ld,%ld,%ld,%ld] input3[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3],
                    input3_dims[0], input3_dims[1], input3_dims[2], input2_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    // check the matrix dimensions for the multiply
    if(params.transpose_input1) {
        if (input1_dims[2]&input1_dims[3]) {
            std::swap(input1_dims[0], input1_dims[1]);
        }
        else if (input1_dims[0]&input1_dims[1]) {
            std::swap(input1_dims[2], input1_dims[3]);
        }
    }
    if(params.transpose_input2) {
        if (input2_dims[2]&input2_dims[3]) {
            std::swap(input2_dims[0], input2_dims[1]);
        }
        else if (input2_dims[0]&input2_dims[1]) {
            std::swap(input2_dims[2], input2_dims[3]);
        }
    }
    if(params.transpose_input3) {
        if (input3_dims[2]&input3_dims[3]) {
            std::swap(input3_dims[0], input3_dims[1]);
        }
        else if (input3_dims[0]&input3_dims[1]) {
            std::swap(input3_dims[2], input3_dims[3]);
        }
    }
    if (input1_dims[2]&input1_dims[3]) {
        if(input1_dims[0] != input2_dims[1] ||
       input1_dims[1] != output_dims[1] || input2_dims[0] != output_dims[0] ||
       (parameters[2] && (input3_dims[0] != output_dims[0] || input3_dims[1] != output_dims[1])))
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: transpose=[%d %d %d] dims input1[%ld,%ld,%ld,%ld] input2[%ld,%ld,%ld,%ld] input3[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    params.transpose_input1, params.transpose_input2, params.transpose_input3,
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3],
                    input3_dims[0], input3_dims[1], input3_dims[2], input2_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }
    else if(input1_dims[0]&input1_dims[1]) {
        if (input1_dims[2] != input2_dims[1] ||
       input1_dims[3] != output_dims[3] || input2_dims[0] != output_dims[2]) 
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: matmul: transpose=[%d %d %d] dims input1[%ld,%ld,%ld,%ld] input2[%ld,%ld,%ld,%ld] input3[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    params.transpose_input1, params.transpose_input2, params.transpose_input3,
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3],
                    input3_dims[0], input3_dims[1], input3_dims[2], input2_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    // get matrix multiply parameters
    vx_tensor_matrix_multiply_params_t params = { 0 };
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &params, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // get input and output dimensions
    vx_size num_dims;
    vx_size input1_dims[4] = { 1, 1, 1, 1 };
    vx_size input2_dims[4] = { 1, 1, 1, 1 };
    vx_size input3_dims[4] = { 1, 1, 1, 1 };
    vx_size output_dims[4] = { 1, 1, 1, 1 };
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims < 2) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, num_dims*sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims < 2) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input2_dims, num_dims*sizeof(vx_size)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims < 2) return VX_ERROR_INVALID_DIMENSION;
        if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, input3_dims, num_dims*sizeof(vx_size)));
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims < 2) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, num_dims*sizeof(vx_size)));

    // create and initialize local data
    LocalData * data = new LocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    // set flags to control matrix transpose and m, n, and k
    data->tA = params.transpose_input1 ? true : false;
    data->tB = params.transpose_input2 ? true : false;
    data->tI = params.transpose_input3 ? true : false;
    if (input1_dims[2] & input1_dims[3]) {
        data->k = input1_dims[params.transpose_input1 ? 1 : 0];
        data->m = input1_dims[params.transpose_input1 ? 0 : 1];
    }
    else if(input1_dims[0] & input1_dims[1]) {
        data->k = input1_dims[params.transpose_input1 ? 3 : 2];
        data->m = input1_dims[params.transpose_input1 ? 2 : 3];
    }
    if (input2_dims[2] & input2_dims[3]) {
        data->n = input2_dims[params.transpose_input2 ? 1 : 0];
    }
    else if (input2_dims[0] & input2_dims[1]) {
        data->n = input2_dims[params.transpose_input2 ? 3 : 2];
    }

    // get buffer offsets and stride
    vx_size a_stride[4], b_stride[4], c_stride[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_GPU, a_stride, sizeof(a_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_STRIDE_GPU, b_stride, sizeof(b_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_STRIDE_GPU, c_stride, sizeof(c_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_OFFSET_GPU, &data->a_offset, sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_OFFSET_GPU, &data->b_offset, sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_OFFSET_GPU, &data->c_offset, sizeof(vx_size)));

    int shift_value = (type == VX_TYPE_FLOAT32) ? 2 : 1; // elem_size for fp32 = 4, fp16 = 2
    data->a_offset >>= 2;
    data->b_offset >>= 2;
    data->c_offset >>= 2;
    if (input1_dims[2] & input1_dims[3]) {
        data->lda = a_stride[data->tA ? 2 : 1] >> shift_value;
    }
    else if(input1_dims[0] & input1_dims[1]) {   
        data->lda = a_stride[3] >> shift_value;
    }
    if (input2_dims[2] & input2_dims[3]) {
        data->ldb = b_stride[data->tB ? 2 : 1] >> shift_value;
    }
    else if(input1_dims[0] & input1_dims[1]) {
        data->ldb = b_stride[3] >> shift_value;
    }
    if (output_dims[2] == 1 && output_dims[3] == 1) {
        data->ldc = c_stride[1] >> shift_value;
    }
    else if (output_dims[0] == 1 && output_dims[1] == 1) {
        data->ldc = c_stride[3] >> shift_value;
    }
    if(parameters[2]) {
        vx_size i_stride[4];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_STRIDE_GPU, i_stride, sizeof(c_stride)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_OFFSET_GPU, &data->i_offset, sizeof(vx_size)));
        data->i_offset >>= 2;
        data->ldi = i_stride[data->tI ? 2 : 1] >> shift_value;
    }

#if ENABLE_OPENCL
    // input and output memory
    cl_mem input1_mem = nullptr, input2_mem = nullptr, input3_mem = nullptr, output_mem = nullptr;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &input1_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &input2_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &output_mem, sizeof(cl_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &input3_mem, sizeof(cl_mem)));
    }

    // if input3 is available, build OpenCL kernel for copy/transpose
    if(parameters[2]) {
        vx_size width = input3_dims[0], height = input3_dims[1];
        // generate OpenCL C code for copy/transpose
        std::string code;
        if(data->tI) {
            size_t BLKW = 16;
            data->copy_local[0] = BLKW;
            data->copy_local[1] = BLKW;
            data->copy_local[2] = 1;
            data->copy_global[0] = (width  + data->copy_local[0] - 1) & ~(data->copy_local[0] - 1);
            data->copy_global[1] = (height + data->copy_local[1] - 1) & ~(data->copy_local[1] - 1);
            data->copy_global[2] =  1;
            if (type == VX_TYPE_FLOAT32) {
            code =
                "#define BLKW " + std::to_string(BLKW) + "\n"
                "__kernel __attribute__((reqd_work_group_size(BLKW, BLKW, 1)))\n"
                "__kernel void copy(const __global float * __restrict inp, __global float * __restrict out)\n";
            }else
            {
                code =
                    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                    "#define BLKW " + std::to_string(BLKW) + "\n"
                    "__kernel __attribute__((reqd_work_group_size(BLKW, BLKW, 1)))\n"
                    "__kernel void copy(const __global half * __restrict inp, __global half * __restrict out)\n";
            }
            code +=
                "{\n"
                "   __local float lbuf[BLKW*BLKW];\n"
                "   uint gx = get_group_id(0);\n"
                "   uint gy = get_group_id(1);\n"
                "   uint lx = get_local_id(0);\n"
                "   uint ly = get_local_id(1);\n"
                "   uint ix = mad24(gx, (uint)BLKW, lx);\n"
                "   uint iy = mad24(gy, (uint)BLKW, ly);\n"
                "   if(ix < " + std::to_string(width) + " && iy < " + std::to_string(height) + ") {\n"
                "       uint iloc = iy * " + std::to_string(data->ldi) + " + ix + " + std::to_string(data->i_offset) + ";\n"
                "       lbuf[mad24(ly, (uint)(BLKW+1), lx)] = inp[iloc];\n"
                "   }\n"
                "   barrier(CLK_LOCAL_MEM_FENCE);\n"
                "   uint ox = mad24(gy, (uint)BLKW, lx);\n"
                "   uint oy = mad24(gx, (uint)BLKW, ly);\n"
                "   if(oy < " + std::to_string(width) + " && ox < " + std::to_string(height) + ") {\n"
                "       uint oloc = oy * " + std::to_string(data->ldc) + " + ox + " + std::to_string(data->c_offset) + ";\n"
                "       out[oloc] = lbuf[mad24(lx, (uint)(BLKW+1), ly)];\n"
                "   }\n"
                "}\n";
        }
        else {
            data->copy_local[0] = 64;
            data->copy_local[1] = 1;
            data->copy_local[2] = 1;
            data->copy_global[0] = (width + data->copy_local[0] - 1) & ~(data->copy_local[0] - 1);
            data->copy_global[1] =  height;
            data->copy_global[2] =  1;
            if (type == VX_TYPE_FLOAT32) {
            code =
                "__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n"
                "__kernel void copy(const __global float * __restrict inp, __global float * __restrict out)\n";
            }else
            {
                code =
                    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                    "__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n"
                    "__kernel void copy(const __global half * __restrict inp, __global half * __restrict out)\n";
            }
            code +=
                "{\n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   if(x < " + std::to_string(width) + " && y < " + std::to_string(height) + ") {\n"
                "       uint i = y * " + std::to_string(data->ldi) + " + x + " + std::to_string(data->i_offset) + ";\n"
                "       uint o = y * " + std::to_string(data->ldc) + " + x + " + std::to_string(data->c_offset) + ";\n"
                "       out[o] = inp[i];\n"
                "   }\n"
                "}\n";
        }
        // build OpenCL C code and save the kernel object
        cl_context opencl_context = nullptr;
        cl_device_id device_id = nullptr;
        ERROR_CHECK_STATUS(clGetCommandQueueInfo(data->handle->cmdq, CL_QUEUE_CONTEXT, sizeof(cl_context), &opencl_context, nullptr));
        ERROR_CHECK_STATUS(clGetCommandQueueInfo(data->handle->cmdq, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr));
        const char * program_src[] = { code.c_str() };
        cl_int err;
        cl_program program = clCreateProgramWithSource(opencl_context, 1, program_src, nullptr, &err);
        if(!program) {
            printf("ERROR: clCreateProgramWithSource failed with (%d) for below code:\n<<<\n%s\n>>>\n", err, code.c_str());
            return VX_FAILURE;
        }
        err = clBuildProgram(program, 1, &device_id, "", nullptr, nullptr);
        if(err) {
            printf("ERROR: clBuildProgram failed with (%d) for below code:\n<<<\n%s\n>>>\n", err, code.c_str());
            return VX_FAILURE;
        }
        data->copy_kernel = clCreateKernel(program, "copy", &err);
        if(!data->copy_kernel) {
            printf("ERROR: MatrixMultiply: clCreateKernel(*,copy,*) failed with (%d)\n", err);
            return VX_FAILURE;
        }
        ERROR_CHECK_STATUS(clReleaseProgram(program));
        // execute copy/transpose kernel first time
        ERROR_CHECK_STATUS(clSetKernelArg(data->copy_kernel, 0, sizeof(cl_mem), &input3_mem));
        ERROR_CHECK_STATUS(clSetKernelArg(data->copy_kernel, 1, sizeof(cl_mem), &output_mem));
        ERROR_CHECK_STATUS(clEnqueueNDRangeKernel(data->handle->cmdq, data->copy_kernel, 3, nullptr, data->copy_global, data->copy_local, 0, nullptr, nullptr));
        ERROR_CHECK_STATUS(clFinish(data->handle->cmdq));
    }
    // build and save ID
    MIOpenGEMM::GemmStatus status =
        MIOpenGEMM::xgemm<float>(false, data->tA, data->tB, data->m, data->n, data->k,
            1.0f,
            input1_mem, data->a_offset, data->lda,
            input2_mem, data->b_offset, data->ldb,
            parameters[2] ? 1.0f : 0.0f,
            output_mem, data->c_offset, data->ldc,
            nullptr, 0, 0, &data->handle->cmdq, 0, nullptr, nullptr, -1);
    if(!status.success) {
        delete data;
        printf("ERROR: MatrixMultiply: MIOpenGEMM::xgemm<float>() failed\n");
        return VX_FAILURE;
    }
    ERROR_CHECK_STATUS(clFinish(data->handle->cmdq));
    data->ID = status.ID;
#elif ENABLE_HIP
    rocblas_status rstatus = rocblas_create_handle(&data->rocBlasHandle);
    if (rstatus != rocblas_status_success) {
        printf("ERROR: rocblas_create_handle failed\n");
        return VX_FAILURE;
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->input3_mem, sizeof(data->input3_mem)));
    }
    data->type = type;
    data->width = input3_dims[0];
    data->height = input3_dims[1];
    // if input3 is available, launch HIP kernel for copy/transpose
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->hip_stream, sizeof(data->hip_stream)));
        if (HipExec_copy(data->hip_stream, data->type, data->input3_mem, data->output_mem, data->width, data->height, data->ldi, data->i_offset,
            data->ldc, data->c_offset, data->tI)) {
            return VX_FAILURE;
        }
        hipStreamSynchronize(data->hip_stream);
    }

    data->alpha = 1.0f;
    data->beta  = parameters[2] ? 1.0f : 0.0f;

    //rocBLAS array storage format is column major
    std::swap(data->input1_mem, data->input2_mem);
    std::swap(data->a_offset, data->b_offset);
    std::swap(data->tA, data->tB);
    std::swap(data->m, data->n);
    std::swap(data->lda, data->ldb);

    switch (data->type) {
        case VX_TYPE_FLOAT32:
            rstatus = rocblas_gemm_ex(
                        data->rocBlasHandle,
                        data->tA ? rocblas_operation_transpose : rocblas_operation_none,
                        data->tB ? rocblas_operation_transpose : rocblas_operation_none,
                        data->m,
                        data->n,
                        data->k,
                        &data->alpha,
                        reinterpret_cast<const float*>(data->input1_mem) + data->a_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->lda,
                        reinterpret_cast<const float*>(data->input2_mem) + data->b_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->ldb,
                        &data->beta,
                        reinterpret_cast<const float*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->ldc,
                        reinterpret_cast<float*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->ldc,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        0);
            if (rstatus != rocblas_status_success) {
                printf("ERROR: rocblas_gemm_ex initialization failed for fp32 with status %d\n", rstatus);
                return VX_FAILURE;
            }
            break;
        case VX_TYPE_FLOAT16:
            rstatus = rocblas_gemm_ex(
                        data->rocBlasHandle,
                        data->tA ? rocblas_operation_transpose : rocblas_operation_none,
                        data->tB ? rocblas_operation_transpose : rocblas_operation_none,
                        data->m,
                        data->n,
                        data->k,
                        &data->alpha,
                        reinterpret_cast<const rocblas_half*>(data->input1_mem) + data->a_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->lda,
                        reinterpret_cast<const rocblas_half*>(data->input2_mem) + data->b_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->ldb,
                        &data->beta,
                        reinterpret_cast<const rocblas_half*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->ldc,
                        reinterpret_cast<rocblas_half*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->ldc,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        0);
            if (rstatus != rocblas_status_success) {
                printf("ERROR: rocblas_gemm_ex initialization failed for fp16 with status %d\n", rstatus);
                return VX_FAILURE;
            }
        break;
        default:
            printf("ERROR: unsupported data type!\n");
            return VX_FAILURE;
    }
#endif

    // save local data ptr as node attribute
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK process(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    // get parameters and buffers
    LocalData * data = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if(!data) return VX_FAILURE;
#if ENABLE_OPENCL
    cl_mem input1_mem = nullptr, input2_mem = nullptr, input3_mem = nullptr, output_mem = nullptr;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &input1_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &input2_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &output_mem, sizeof(cl_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &input3_mem, sizeof(cl_mem)));
        // copy/transpose input3 to output
        ERROR_CHECK_STATUS(clSetKernelArg(data->copy_kernel, 0, sizeof(cl_mem), &input3_mem));
        ERROR_CHECK_STATUS(clSetKernelArg(data->copy_kernel, 1, sizeof(cl_mem), &output_mem));
        ERROR_CHECK_STATUS(clEnqueueNDRangeKernel(data->handle->cmdq, data->copy_kernel, 3, nullptr, data->copy_global, data->copy_local, 0, nullptr, nullptr));
    }
    // run GEMM
    MIOpenGEMM::GemmStatus status =
        MIOpenGEMM::xgemm<float>(false, data->tA, data->tB, data->m, data->n, data->k, 1.0f,
            input1_mem, data->a_offset, data->lda,
            input2_mem, data->b_offset, data->ldb,
            parameters[2] ? 1.0f : 0.0f,
            output_mem, data->c_offset, data->ldc,
            nullptr, 0, 0, &data->handle->cmdq, 0, nullptr, nullptr, data->ID);
    if(!status.success) {
        return VX_FAILURE;
    }
#elif ENABLE_HIP
    rocblas_status rstatus;
    if (parameters[2]) {
        // copy/transpose input3 to output
        if (HipExec_copy(data->hip_stream, data->type, data->input3_mem, data->output_mem, data->width, data->height, data->ldi, data->i_offset,
            data->ldc, data->c_offset, data->tI)) {
            return VX_FAILURE;
        }
    }

    switch (data->type) {
        case VX_TYPE_FLOAT32:
            rstatus = rocblas_gemm_ex(
                        data->rocBlasHandle,
                        data->tA ? rocblas_operation_transpose : rocblas_operation_none,
                        data->tB ? rocblas_operation_transpose : rocblas_operation_none,
                        data->m,
                        data->n,
                        data->k,
                        &data->alpha,
                        reinterpret_cast<const float*>(data->input1_mem) + data->a_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->lda,
                        reinterpret_cast<const float*>(data->input2_mem) + data->b_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->ldb,
                        &data->beta,
                        reinterpret_cast<const float*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->ldc,
                        reinterpret_cast<float*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        data->ldc,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        0);
            if (rstatus != rocblas_status_success) {
                printf("ERROR: rocblas_gemm_ex process failed for fp32 with status %d\n", rstatus);
                return VX_FAILURE;
            }
            break;
        case VX_TYPE_FLOAT16:
            rstatus = rocblas_gemm_ex(
                        data->rocBlasHandle,
                        data->tA ? rocblas_operation_transpose : rocblas_operation_none,
                        data->tB ? rocblas_operation_transpose : rocblas_operation_none,
                        data->m,
                        data->n,
                        data->k,
                        &data->alpha,
                        reinterpret_cast<const rocblas_half*>(data->input1_mem) + data->a_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->lda,
                        reinterpret_cast<const rocblas_half*>(data->input2_mem) + data->b_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->ldb,
                        &data->beta,
                        reinterpret_cast<const rocblas_half*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->ldc,
                        reinterpret_cast<rocblas_half*>(data->output_mem) + data->c_offset,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        data->ldc,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        0);
            if (rstatus != rocblas_status_success) {
                printf("ERROR: rocblas_gemm_ex process failed for fp16 with status %d\n", rstatus);
                return VX_FAILURE;
            }
        break;
        default:
            printf("ERROR: unsupported data type!\n");
            return VX_FAILURE;
    }

#endif

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    LocalData * data = nullptr;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
#if ENABLE_OPENCL
        if(data->copy_kernel) {
            clReleaseKernel(data->copy_kernel);
        }
#elif ENABLE_HIP
        if (data->rocBlasHandle) {
            rocblas_status rstatus = rocblas_destroy_handle(data->rocBlasHandle);
            if (rstatus != rocblas_status_success) {
                printf("ERROR: rocblas_destroy_handle failed\n");
                return VX_FAILURE;
            }
        }
#endif
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishTensorMatrixMultiply(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.openvx.tensor_matrix_multiply", VX_KERNEL_TENSOR_MATRIX_MULTIPLY, process, 5, validate, initialize, uninitialize);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorMatrixMultiplyNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor input3, const vx_tensor_matrix_multiply_params_t * params, vx_tensor output)
{
    vx_node node = nullptr;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_params = vxCreateScalarWithSize(context, VX_TYPE_TENSOR_MATRIX_MULTIPLY_PARAMS, params, sizeof(*params));
        if (vxGetStatus((vx_reference)s_params) == VX_SUCCESS) {
            vx_reference params[] = {
                (vx_reference)input1,
                (vx_reference)input2,
                (vx_reference)input3,
                (vx_reference)s_params,
                (vx_reference)output
            };
            node = createNode(graph, VX_KERNEL_TENSOR_MATRIX_MULTIPLY, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_params);
        }
    }
    return node;
}
