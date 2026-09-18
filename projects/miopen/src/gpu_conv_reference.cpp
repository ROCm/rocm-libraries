// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/gpu_conv_reference.hpp>
#include <miopen/datatype.hpp>
#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/tensor_layout.hpp>

#include <cassert>
#include <sstream>
#include <string>

namespace miopen {

using conv::ProblemDescription;
using solver::conv::conv_internal::GetGroupStrideIndex;
using solver::conv::conv_internal::MakeStrideArray;
using solver::conv::conv_internal::SplitStrideCtoGC;
using solver::conv::conv_internal::SplitWeiStrideKtoGK;

namespace {

constexpr size_t REF_BLOCK_SIZE = 256;

// Maximum grid size to avoid uint32_t overflow in hipExtModuleLaunchKernel.
// 2^32 / REF_BLOCK_SIZE = 16,777,216
constexpr size_t MAX_GRID_SIZE = static_cast<size_t>(1ULL << 32) / REF_BLOCK_SIZE;

// Build the kernel name for the GPU reference (always uses double accumulators).
std::string RefKernelName(const std::string& direction,
                          const std::string& layout,
                          miopenDataType_t in_type,
                          miopenDataType_t out_type)
{
    std::ostringstream name;
    name << "naive_conv_ab_nonpacked_" << direction << "_" << layout << "_";

    // Input data type
    switch(in_type)
    {
    case miopenFloat: name << "float_"; break;
    case miopenHalf: name << "half_"; break;
    case miopenBFloat16: name << "hip_bfloat16_"; break;
    case miopenInt8: name << "int8_t_"; break;
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64: MIOPEN_THROW("GpuConvReference: unsupported input data type");
    }

    // Accumulator type: always double for reference (int32 for int8)
    if(in_type == miopenInt8)
        name << "int32_t_";
    else
        name << "double_";

    // Output data type
    switch(out_type)
    {
    case miopenFloat: name << "float"; break;
    case miopenHalf: name << "half"; break;
    case miopenBFloat16: name << "hip_bfloat16"; break;
    case miopenInt8: name << "int8_t"; break;
    case miopenInt32: name << "int32_t"; break;
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64: MIOPEN_THROW("GpuConvReference: unsupported output data type");
    }

    // TF32 flag: always 0 for reference
    name << "_0";
    return name.str();
}

// Build compile options for the GPU reference kernel.
std::string RefCompileOptions(miopenDataType_t data_type)
{
    std::ostringstream ss;
    ss << GetDataTypeKBP(data_type).GenerateFor(kbp::HIP{});
    ss << " -DNAIVE_CONV_BLOCK_SIZE=" << REF_BLOCK_SIZE;
    return ss.str();
}

// Get layout string and spatial dims from a tensor descriptor.
void GetLayoutInfo(const TensorDescriptor& desc,
                   bool& is_default_layout,
                   bool& is_2d,
                   std::string& layout_str)
{
    const auto num_dims = desc.GetNumDims();
    is_2d               = (num_dims == 4);

    std::string layout_default = tensor_layout_get_default(num_dims);
    std::string actual_layout  = desc.GetLayout(layout_default);

    if(actual_layout == "NCHW" || actual_layout == "NCDHW")
    {
        is_default_layout = true;
        layout_str        = is_2d ? "nchw" : "ncdhw";
    }
    else if(actual_layout == "NHWC" || actual_layout == "NDHWC")
    {
        is_default_layout = false;
        layout_str        = is_2d ? "nhwc" : "ndhwc";
    }
    else
    {
        MIOPEN_THROW("GpuConvReference: unsupported layout " + actual_layout);
    }
}

// Compute grid size per batch for a 2D direction/layout combination.
size_t
ComputeGridSizePerBatch(bool is_fwd, bool is_default_layout, int k, int c, int n, int ho, int hi)
{
    if(is_fwd)
    {
        if(is_default_layout)
            return static_cast<size_t>(k);
        else
            return static_cast<size_t>(ho);
    }
    else // bwd
    {
        if(is_default_layout)
            return static_cast<size_t>(n) * c;
        else
            return static_cast<size_t>(n) * hi;
    }
}

// Compute grid size per batch for a 3D direction/layout combination.
size_t ComputeGridSizePerBatch3D(bool is_fwd,
                                 bool is_default_layout,
                                 int n,
                                 int do_,
                                 int di,
                                 int group,
                                 int k_per_group,
                                 int c_per_group)
{
    if(is_fwd)
    {
        if(is_default_layout)
            return static_cast<size_t>(group) * k_per_group;
        else
            return static_cast<size_t>(group) * do_;
    }
    else // bwd
    {
        if(is_default_layout)
            return static_cast<size_t>(group) * n * c_per_group;
        else
            return static_cast<size_t>(group) * n * di;
    }
}

// Build a ProblemDescription from raw descriptors. Needed for GetGroupStrideIndex.
ProblemDescription MakeProblem(const TensorDescriptor& xDesc,
                               const TensorDescriptor& wDesc,
                               const TensorDescriptor& yDesc,
                               const ConvolutionDescriptor& conv,
                               miopen::conv::Direction dir)
{
    return {xDesc, wDesc, yDesc, conv, dir};
}

} // anonymous namespace

void GpuConvReference::RunFwd(const Handle& handle,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              const ConvolutionDescriptor& conv,
                              double alpha,
                              double beta)
{
    auto problem = MakeProblem(xDesc, wDesc, yDesc, conv, miopen::conv::Direction::Forward);

    bool is_default_layout = true;
    bool is_2d             = true;
    std::string layout_str;
    GetLayoutInfo(xDesc, is_default_layout, is_2d, layout_str);

    const auto data_type = xDesc.GetType();

    int n           = xDesc.GetLengths()[0];
    int c           = xDesc.GetLengths()[1];
    int k           = wDesc.GetLengths()[0];
    int group       = conv.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;

    auto pads      = conv.GetConvPads();
    auto strides   = conv.GetConvStrides();
    auto dilations = conv.GetConvDilations();

    int G_stride_idx = GetGroupStrideIndex(problem);

    if(is_2d)
    {
        int hi = xDesc.GetLengths()[2];
        int wi = xDesc.GetLengths()[3];
        int ho = yDesc.GetLengths()[2];
        int wo = yDesc.GetLengths()[3];
        int fy = wDesc.GetLengths()[2];
        int fx = wDesc.GetLengths()[3];
        int py = pads[0], px = pads[1];
        int sy = strides[0], sx = strides[1];
        int dily = dilations[0], dilx = dilations[1];

        size_t grid_size_per_batch =
            ComputeGridSizePerBatch(true, is_default_layout, k, c, n, ho, hi);

        // Batch chunking
        int batch_chunk = static_cast<int>(MAX_GRID_SIZE / grid_size_per_batch);
        if(batch_chunk < 1)
            batch_chunk = 1;
        if(batch_chunk > n)
            batch_chunk = n;

        std::string kernel_name = RefKernelName("fwd", layout_str, data_type, yDesc.GetType());
        std::string kernel_file = "naive_conv_fwd.cpp";
        std::string comp_opts   = RefCompileOptions(data_type);

        auto in_strides =
            MakeStrideArray<5>(SplitStrideCtoGC(group, xDesc.GetStrides(), G_stride_idx));
        auto wei_strides = MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, wDesc.GetStrides()));
        auto out_strides =
            MakeStrideArray<5>(SplitStrideCtoGC(group, yDesc.GetStrides(), G_stride_idx));

        size_t in_batch_stride  = xDesc.GetStrides()[0];
        size_t out_batch_stride = yDesc.GetStrides()[0];
        size_t in_type_size     = GetTypeSize(data_type);
        size_t out_type_size    = GetTypeSize(yDesc.GetType());

        for(int batch_start = 0; batch_start < n; batch_start += batch_chunk)
        {
            int cur_n         = std::min(batch_chunk, n - batch_start);
            size_t grid_size  = grid_size_per_batch * cur_n;
            size_t in_offset  = static_cast<size_t>(batch_start) * in_batch_stride * in_type_size;
            size_t out_offset = static_cast<size_t>(batch_start) * out_batch_stride * out_type_size;
            const void* in_ptr = static_cast<const char*>(x) + in_offset;
            void* out_ptr      = static_cast<char*>(y) + out_offset;

            handle.AddKernel("gpu_ref_conv",
                             "",
                             kernel_file,
                             kernel_name,
                             {REF_BLOCK_SIZE, 1, 1},
                             {grid_size * REF_BLOCK_SIZE, 1, 1},
                             comp_opts)(in_ptr,
                                        w,
                                        alpha,
                                        beta,
                                        out_ptr,
                                        in_strides,
                                        wei_strides,
                                        out_strides,
                                        hi,
                                        wi,
                                        cur_n,
                                        k_per_group,
                                        c_per_group,
                                        ho,
                                        wo,
                                        sy,
                                        sx,
                                        dily,
                                        dilx,
                                        py,
                                        px,
                                        fy,
                                        fx,
                                        group);
        }
    }
    else
    {
        // 3D
        int di  = xDesc.GetLengths()[2];
        int hi  = xDesc.GetLengths()[3];
        int wi  = xDesc.GetLengths()[4];
        int do_ = yDesc.GetLengths()[2];
        int ho  = yDesc.GetLengths()[3];
        int wo  = yDesc.GetLengths()[4];
        int fz  = wDesc.GetLengths()[2];
        int fy  = wDesc.GetLengths()[3];
        int fx  = wDesc.GetLengths()[4];
        int pz = pads[0], py = pads[1], px = pads[2];
        int sz = strides[0], sy = strides[1], sx = strides[2];
        int dilz = dilations[0], dily = dilations[1], dilx = dilations[2];

        size_t grid_size_per_batch = ComputeGridSizePerBatch3D(
            true, is_default_layout, n, do_, di, group, k_per_group, c_per_group);

        int batch_chunk = static_cast<int>(MAX_GRID_SIZE / grid_size_per_batch);
        if(batch_chunk < 1)
            batch_chunk = 1;
        if(batch_chunk > n)
            batch_chunk = n;

        std::string kernel_name = RefKernelName("fwd", layout_str, data_type, yDesc.GetType());
        std::string kernel_file = "naive_conv_fwd.cpp";
        std::string comp_opts   = RefCompileOptions(data_type);

        auto in_strides =
            MakeStrideArray<6>(SplitStrideCtoGC(group, xDesc.GetStrides(), G_stride_idx));
        auto wei_strides = MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, wDesc.GetStrides()));
        auto out_strides =
            MakeStrideArray<6>(SplitStrideCtoGC(group, yDesc.GetStrides(), G_stride_idx));

        size_t in_batch_stride  = xDesc.GetStrides()[0];
        size_t out_batch_stride = yDesc.GetStrides()[0];
        size_t in_type_size     = GetTypeSize(data_type);
        size_t out_type_size    = GetTypeSize(yDesc.GetType());

        for(int batch_start = 0; batch_start < n; batch_start += batch_chunk)
        {
            int cur_n         = std::min(batch_chunk, n - batch_start);
            size_t grid_size  = grid_size_per_batch * cur_n;
            size_t in_offset  = static_cast<size_t>(batch_start) * in_batch_stride * in_type_size;
            size_t out_offset = static_cast<size_t>(batch_start) * out_batch_stride * out_type_size;
            const void* in_ptr = static_cast<const char*>(x) + in_offset;
            void* out_ptr      = static_cast<char*>(y) + out_offset;

            handle.AddKernel("gpu_ref_conv",
                             "",
                             kernel_file,
                             kernel_name,
                             {REF_BLOCK_SIZE, 1, 1},
                             {grid_size * REF_BLOCK_SIZE, 1, 1},
                             comp_opts)(in_ptr,
                                        w,
                                        alpha,
                                        beta,
                                        out_ptr,
                                        in_strides,
                                        wei_strides,
                                        out_strides,
                                        di,
                                        hi,
                                        wi,
                                        cur_n,
                                        k_per_group,
                                        c_per_group,
                                        do_,
                                        ho,
                                        wo,
                                        sz,
                                        sy,
                                        sx,
                                        dilz,
                                        dily,
                                        dilx,
                                        pz,
                                        py,
                                        px,
                                        fz,
                                        fy,
                                        fx,
                                        group);
        }
    }
}

void GpuConvReference::RunBwd(const Handle& handle,
                              const TensorDescriptor& dyDesc,
                              ConstData_t dy,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& dxDesc,
                              Data_t dx,
                              const ConvolutionDescriptor& conv,
                              double alpha,
                              double beta)
{
    // BWD: input gradient is dx (output of this function), output gradient is dy (input)
    auto problem = MakeProblem(dxDesc, wDesc, dyDesc, conv, miopen::conv::Direction::BackwardData);

    bool is_default_layout = true;
    bool is_2d             = true;
    std::string layout_str;
    GetLayoutInfo(dxDesc, is_default_layout, is_2d, layout_str);

    const auto data_type = dxDesc.GetType();

    int n           = dxDesc.GetLengths()[0];
    int c           = dxDesc.GetLengths()[1];
    int k           = wDesc.GetLengths()[0];
    int group       = conv.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;

    auto pads      = conv.GetConvPads();
    auto strides   = conv.GetConvStrides();
    auto dilations = conv.GetConvDilations();

    int G_stride_idx = GetGroupStrideIndex(problem);

    if(is_2d)
    {
        int hi = dxDesc.GetLengths()[2];
        int wi = dxDesc.GetLengths()[3];
        int ho = dyDesc.GetLengths()[2];
        int wo = dyDesc.GetLengths()[3];
        int fy = wDesc.GetLengths()[2];
        int fx = wDesc.GetLengths()[3];
        int py = pads[0], px = pads[1];
        int sy = strides[0], sx = strides[1];
        int dily = dilations[0], dilx = dilations[1];

        size_t grid_size_per_batch =
            ComputeGridSizePerBatch(false, is_default_layout, k, c, n, ho, hi);

        // For BWD, the kernel has no grid-stride loop — thread_length determines gridDim.y
        size_t thread_length = 1;
        if(is_default_layout)
            thread_length = static_cast<size_t>(hi) * wi;
        else
            thread_length = static_cast<size_t>(wi) * c;

        size_t num_spatial_tiles = (thread_length + REF_BLOCK_SIZE - 1) / REF_BLOCK_SIZE;

        std::string kernel_name = RefKernelName("bwd", layout_str, data_type, dxDesc.GetType());
        std::string kernel_file = "naive_conv_bwd.cpp";
        std::string comp_opts   = RefCompileOptions(data_type);

        // BWD strides: dy is "out", dx is "in" from the kernel's perspective
        auto out_strides =
            MakeStrideArray<5>(SplitStrideCtoGC(group, dyDesc.GetStrides(), G_stride_idx));
        auto wei_strides = MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, wDesc.GetStrides()));
        auto in_strides =
            MakeStrideArray<5>(SplitStrideCtoGC(group, dxDesc.GetStrides(), G_stride_idx));

        // The kernel's p_in is the result (dx) and p_out is the input (dy).
        // This matches the solver's reversed tensor convention
        // (see backward_tensors_reversed_why in conv_direct_naive_conv.cpp).
        handle.AddKernel("gpu_ref_conv",
                         "",
                         kernel_file,
                         kernel_name,
                         {REF_BLOCK_SIZE, 1, 1},
                         {grid_size_per_batch * REF_BLOCK_SIZE, num_spatial_tiles, 1},
                         comp_opts)(dx,
                                    w,
                                    alpha,
                                    beta,
                                    dy,
                                    in_strides,
                                    wei_strides,
                                    out_strides,
                                    hi,
                                    wi,
                                    n,
                                    k_per_group,
                                    c_per_group,
                                    ho,
                                    wo,
                                    sy,
                                    sx,
                                    dily,
                                    dilx,
                                    py,
                                    px,
                                    fy,
                                    fx,
                                    group);
    }
    else
    {
        // 3D BWD
        int di  = dxDesc.GetLengths()[2];
        int hi  = dxDesc.GetLengths()[3];
        int wi  = dxDesc.GetLengths()[4];
        int do_ = dyDesc.GetLengths()[2];
        int ho  = dyDesc.GetLengths()[3];
        int wo  = dyDesc.GetLengths()[4];
        int fz  = wDesc.GetLengths()[2];
        int fy  = wDesc.GetLengths()[3];
        int fx  = wDesc.GetLengths()[4];
        int pz = pads[0], py = pads[1], px = pads[2];
        int sz = strides[0], sy = strides[1], sx = strides[2];
        int dilz = dilations[0], dily = dilations[1], dilx = dilations[2];

        size_t grid_size_per_batch = ComputeGridSizePerBatch3D(
            false, is_default_layout, n, do_, di, group, k_per_group, c_per_group);

        size_t thread_length = 1;
        if(is_default_layout)
            thread_length = static_cast<size_t>(di) * hi * wi;
        else
            thread_length = static_cast<size_t>(hi) * wi * c_per_group;

        size_t num_spatial_tiles = (thread_length + REF_BLOCK_SIZE - 1) / REF_BLOCK_SIZE;

        std::string kernel_name = RefKernelName("bwd", layout_str, data_type, dxDesc.GetType());
        std::string kernel_file = "naive_conv_bwd.cpp";
        std::string comp_opts   = RefCompileOptions(data_type);

        auto out_strides =
            MakeStrideArray<6>(SplitStrideCtoGC(group, dyDesc.GetStrides(), G_stride_idx));
        auto wei_strides = MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, wDesc.GetStrides()));
        auto in_strides =
            MakeStrideArray<6>(SplitStrideCtoGC(group, dxDesc.GetStrides(), G_stride_idx));

        // Reversed tensor convention — see 2D BWD comment above.
        handle.AddKernel("gpu_ref_conv",
                         "",
                         kernel_file,
                         kernel_name,
                         {REF_BLOCK_SIZE, 1, 1},
                         {grid_size_per_batch * REF_BLOCK_SIZE, num_spatial_tiles, 1},
                         comp_opts)(dx,
                                    w,
                                    alpha,
                                    beta,
                                    dy,
                                    in_strides,
                                    wei_strides,
                                    out_strides,
                                    di,
                                    hi,
                                    wi,
                                    n,
                                    k_per_group,
                                    c_per_group,
                                    do_,
                                    ho,
                                    wo,
                                    sz,
                                    sy,
                                    sx,
                                    dilz,
                                    dily,
                                    dilx,
                                    pz,
                                    py,
                                    px,
                                    fz,
                                    fy,
                                    fx,
                                    group);
    }
}

void GpuConvReference::RunWrw(const Handle& handle,
                              const TensorDescriptor& dyDesc,
                              ConstData_t dy,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& dwDesc,
                              Data_t dw,
                              const ConvolutionDescriptor& conv,
                              double alpha,
                              double beta)
{
    auto problem =
        MakeProblem(xDesc, dwDesc, dyDesc, conv, miopen::conv::Direction::BackwardWeights);

    bool is_default_layout = true;
    bool is_2d             = true;
    std::string layout_str;
    GetLayoutInfo(xDesc, is_default_layout, is_2d, layout_str);

    const auto data_type = xDesc.GetType();

    int n           = xDesc.GetLengths()[0];
    int c           = xDesc.GetLengths()[1];
    int k           = dwDesc.GetLengths()[0];
    int group       = conv.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;

    auto pads      = conv.GetConvPads();
    auto strides   = conv.GetConvStrides();
    auto dilations = conv.GetConvDilations();

    int G_stride_idx = GetGroupStrideIndex(problem);

    if(is_2d)
    {
        int hi = xDesc.GetLengths()[2];
        int wi = xDesc.GetLengths()[3];
        int ho = dyDesc.GetLengths()[2];
        int wo = dyDesc.GetLengths()[3];
        int fy = dwDesc.GetLengths()[2];
        int fx = dwDesc.GetLengths()[3];
        int py = pads[0], px = pads[1];
        int sy = strides[0], sx = strides[1];
        int dily = dilations[0], dilx = dilations[1];

        // WRW reference: serial, no spatial tiling, no atomicAdd
        size_t grid_size = static_cast<size_t>(k);

        std::string kernel_name = RefKernelName("wrw", layout_str, data_type, dwDesc.GetType());
        std::string kernel_file = "naive_conv_wrw.cpp";
        std::string comp_opts   = RefCompileOptions(data_type);

        auto in_strides =
            MakeStrideArray<5>(SplitStrideCtoGC(group, xDesc.GetStrides(), G_stride_idx));
        auto wei_strides =
            MakeStrideArray<5>(SplitWeiStrideKtoGK(k_per_group, dwDesc.GetStrides()));
        auto out_strides =
            MakeStrideArray<5>(SplitStrideCtoGC(group, dyDesc.GetStrides(), G_stride_idx));

        handle.AddKernel("gpu_ref_conv",
                         "",
                         kernel_file,
                         kernel_name,
                         {REF_BLOCK_SIZE, 1, 1},
                         {grid_size * REF_BLOCK_SIZE, 1, 1},
                         comp_opts)(x,
                                    dw,
                                    alpha,
                                    beta,
                                    dy,
                                    in_strides,
                                    wei_strides,
                                    out_strides,
                                    hi,
                                    wi,
                                    n,
                                    k_per_group,
                                    c_per_group,
                                    ho,
                                    wo,
                                    sy,
                                    sx,
                                    dily,
                                    dilx,
                                    py,
                                    px,
                                    fy,
                                    fx,
                                    group);
    }
    else
    {
        // 3D WRW
        int di  = xDesc.GetLengths()[2];
        int hi  = xDesc.GetLengths()[3];
        int wi  = xDesc.GetLengths()[4];
        int do_ = dyDesc.GetLengths()[2];
        int ho  = dyDesc.GetLengths()[3];
        int wo  = dyDesc.GetLengths()[4];
        int fz  = dwDesc.GetLengths()[2];
        int fy  = dwDesc.GetLengths()[3];
        int fx  = dwDesc.GetLengths()[4];
        int pz = pads[0], py = pads[1], px = pads[2];
        int sz = strides[0], sy = strides[1], sx = strides[2];
        int dilz = dilations[0], dily = dilations[1], dilx = dilations[2];

        size_t grid_size = static_cast<size_t>(group) * k_per_group;

        std::string kernel_name = RefKernelName("wrw", layout_str, data_type, dwDesc.GetType());
        std::string kernel_file = "naive_conv_wrw.cpp";
        std::string comp_opts   = RefCompileOptions(data_type);

        auto in_strides =
            MakeStrideArray<6>(SplitStrideCtoGC(group, xDesc.GetStrides(), G_stride_idx));
        auto wei_strides =
            MakeStrideArray<6>(SplitWeiStrideKtoGK(k_per_group, dwDesc.GetStrides()));
        auto out_strides =
            MakeStrideArray<6>(SplitStrideCtoGC(group, dyDesc.GetStrides(), G_stride_idx));

        handle.AddKernel("gpu_ref_conv",
                         "",
                         kernel_file,
                         kernel_name,
                         {REF_BLOCK_SIZE, 1, 1},
                         {grid_size * REF_BLOCK_SIZE, 1, 1},
                         comp_opts)(x,
                                    dw,
                                    alpha,
                                    beta,
                                    dy,
                                    in_strides,
                                    wei_strides,
                                    out_strides,
                                    di,
                                    hi,
                                    wi,
                                    n,
                                    k_per_group,
                                    c_per_group,
                                    do_,
                                    ho,
                                    wo,
                                    sz,
                                    sy,
                                    sx,
                                    dilz,
                                    dily,
                                    dilx,
                                    pz,
                                    py,
                                    px,
                                    fz,
                                    fy,
                                    fx,
                                    group);
    }
}

} // namespace miopen
