/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "conv3d.hpp"
#include "get_handle.hpp"
#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>
#include <miopen/env.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/find_db.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/any_solver.hpp>

#include "test.hpp"
#include "verify.hpp"
#include "tensor_holder.hpp"
#include "cpu_conv.hpp"
#include "gpu_conv.hpp"

#include <half/half.hpp>
using half_float::half;

// Test specifically for ConvHipImplicitGemm3DChannelLastFwdWmmaops solver
template <class T>
struct conv3d_channel_last_wmmaops_driver : conv3d_driver<T>
{
    std::vector<std::string> disable_check;
    
    conv3d_channel_last_wmmaops_driver() : conv3d_driver<T>()
    {
        // Override layouts to specifically test channel-last (NDHWC)
        this->add(this->in_layout, "in_layout", this->generate_data({"NDHWC"}));
        this->add(this->fil_layout, "fil_layout", this->generate_data({"NDHWC"}));
        this->add(this->out_layout, "out_layout", this->generate_data({"NDHWC"}));
        
        // Add specific test configurations that should trigger your solver
        // MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1
        this->add(this->batch_size, "batch_size", this->generate_data({1}));
        this->add(this->input_channels, "input_channels", this->generate_data({16}));
        this->add(this->output_channels, "output_channels", this->generate_data({16}));
        this->add(this->spatial_dim_elements, "spatial_dim_elements", 
                  this->generate_data({{5, 104, 60}}));  // in_d=5, H=104, W=60
        this->add(this->filter_dims, "filter_dims", 
                  this->generate_data({{1, 1, 1}}));  // fil_d=1, y=1, x=1
        this->add(this->pads_strides_dilations, "pads_strides_dilations",
                  this->generate_data({{{0, 0, 0}, {1, 1, 1}, {1, 1, 1}}}));  // pad_d=0, p=0, q=0; conv_stride_d=1, u=1, v=1; dilation_d=1, l=1, j=1
        
        // Enable environment variable to force use of your solver
        this->add(disable_check, "disable_check", this->generate_data({"false"}));
    }
};

// Test to verify that the solver is applicable for specific configurations
void test_solver_applicability()
{
    using namespace miopen;
    using namespace miopen::solver;
    using namespace miopen::solver::conv;
    
    // Create a handle
    auto&& handle = get_handle();
    auto ctx = ExecutionContext{};
    ctx.SetStream(&handle);
    ctx.use_hip_kernels = true;
    
    // Create a 3D convolution problem with channel-last layout and FP16 data type
    std::vector<std::size_t> input_lens = {1, 16, 5, 104, 60};  // n=1, c=16, in_d=5, H=104, W=60
    std::vector<std::size_t> weight_lens = {16, 16, 1, 1, 1};   // k=16, c=16, fil_d=1, y=1, x=1
    
    // Create tensor descriptors with NDHWC layout
    auto input_tensor_desc = miopen::TensorDescriptor(miopenHalf, miopenTensorNDHWC, input_lens);
    auto weight_tensor_desc = miopen::TensorDescriptor(miopenHalf, miopenTensorNDHWC, weight_lens);
    
    // Create convolution descriptor using C API and then wrap it
    miopenConvolutionDescriptor_t conv_desc_raw;
    miopenCreateConvolutionDescriptor(&conv_desc_raw);
    
    // Set 3D convolution parameters
    int padA[3] = {0, 0, 0};      // pad_d=0, p=0, q=0
    int strideA[3] = {1, 1, 1};   // conv_stride_d=1, u=1, v=1
    int dilationA[3] = {1, 1, 1}; // dilation_d=1, l=1, j=1
    miopenInitConvolutionNdDescriptor(conv_desc_raw, 3, padA, strideA, dilationA, miopenConvolution);
    
    // Wrap the C API descriptor in C++ class
    auto conv_desc = miopen::deref(conv_desc_raw);
    
    // Create output tensor descriptor
    auto output_tensor_desc = conv_desc.GetForwardOutputTensor(input_tensor_desc, weight_tensor_desc);
    
    // Create problem description with NDHWC layout (channel-last) and FP16 data type
    auto problem = miopen::conv::ProblemDescription{
        input_tensor_desc, weight_tensor_desc, output_tensor_desc, conv_desc, miopen::conv::Direction::Forward};
    
    // Create solver instance
    ConvHipImplicitGemm3DChannelLastFwdWmmaops solver;
    
    // Test if the solver is applicable
    bool is_applicable = solver.IsApplicable(ctx, problem);
    
    // Print detailed diagnostics
    printf("Testing ConvHipImplicitGemm3DChannelLastFwdWmmaops solver applicability:\n");
    printf("  Problem: 3D Convolution with NDHWC layout, FP16 data type\n");
    printf("  Input shape: N=1, C=16, D=5, H=104, W=60\n");
    printf("  Weight shape: K=16, C=16, Z=1, Y=1, X=1\n");
    printf("  Is 3D: %s\n", problem.Is3d() ? "YES" : "NO");
    printf("  Is Forward: %s\n", problem.IsDirectionForward() ? "YES" : "NO");
    printf("  Is Layout NHWC: %s\n", problem.IsLayoutNHWC() ? "YES" : "NO");
    printf("  Is FP16: %s\n", problem.IsFp16() ? "YES" : "NO");
    printf("  Group Count: %d\n", problem.GetGroupCount());
    printf("  Tensors Casted: %s\n", problem.IsTensorsCasted() ? "YES" : "NO");
    printf("  All Tensors Dims Fit Into Int: %s\n", problem.AllTensorsDimsFitIntoInt() ? "YES" : "NO");
    printf("  Solver applicable: %s\n", is_applicable ? "YES" : "NO");
    
    // Additional debugging for environment variable
    const char* debug_env = std::getenv("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS");
    printf("  Environment variable MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS: %s\n", 
           debug_env ? debug_env : "NOT SET");
    
    // Additional debugging for HIP kernels
    printf("  HIP kernels enabled: %s\n", ctx.use_hip_kernels ? "YES" : "NO");
    
    if(is_applicable)
    {
        // Test getting default performance config
        auto config = solver.GetDefaultPerformanceConfig(ctx, problem);
        bool config_valid = config.IsValid(problem);
        printf("  Performance config valid: %s\n", config_valid ? "YES" : "NO");
        
        if(config_valid)
        {
            // Test solution generation
            auto solution = solver.GetSolution(ctx, problem, config);
            bool solution_valid = solution.construction_params.size() > 0;
            printf("  Solution generation: %s\n", solution_valid ? "SUCCESS" : "FAILED");
            
            if(solution_valid)
            {
                printf("  Generated %zu kernel(s)\n", solution.construction_params.size());
            }
        }
    }
    
    // Verification - the solver should be applicable for this configuration
    EXPECT_EQUAL(is_applicable, true);
    
    if(is_applicable)
    {
        // Test getting default performance config
        auto config = solver.GetDefaultPerformanceConfig(ctx, problem);
        EXPECT_EQUAL(config.IsValid(problem), true);
        
        // Test solution generation
        auto solution = solver.GetSolution(ctx, problem, config);
        // EXPECT_OP(solution.construction_params.size(), >, 0);
    }
}

// Test to verify that the solver works with grouped convolutions
void test_grouped_convolution()
{
    using namespace miopen;
    using namespace miopen::solver;
    using namespace miopen::solver::conv;
    
    // Create a handle
    auto&& handle = get_handle();
    auto ctx = ExecutionContext{};
    ctx.SetStream(&handle);
    ctx.use_hip_kernels = true;
    
    // Create a grouped 3D convolution problem with channel-last layout and FP16 data type
    const int groups = 4;
    const int batch = 1;
    const int input_channels_per_group = 16;
    const int output_channels_per_group = 32;
    const int depth = 8;
    const int height = 8;
    const int width = 8;
    
    // Create tensor descriptors with NDHWC layout directly
    std::vector<std::size_t> input_lens = {batch, input_channels_per_group * groups, depth, height, width};
    std::vector<std::size_t> weight_lens = {output_channels_per_group * groups, input_channels_per_group, 3, 3, 3};
    
    // Create tensor descriptors with NDHWC layout
    auto input_tensor_desc = miopen::TensorDescriptor(miopenHalf, miopenTensorNDHWC, input_lens);
    auto weight_tensor_desc = miopen::TensorDescriptor(miopenHalf, miopenTensorNDHWC, weight_lens);
    
    // Create convolution descriptor using C API and then wrap it
    miopenConvolutionDescriptor_t conv_desc_raw;
    miopenCreateConvolutionDescriptor(&conv_desc_raw);
    
    // Set 3D convolution parameters
    int padA[3] = {0, 0, 0};      // pad_d=0, p=0, q=0
    int strideA[3] = {1, 1, 1};   // conv_stride_d=1, u=1, v=1
    int dilationA[3] = {1, 1, 1}; // dilation_d=1, l=1, j=1
    miopenInitConvolutionNdDescriptor(conv_desc_raw, 3, padA, strideA, dilationA, miopenConvolution);
    
    // Set group count
    miopenSetConvolutionGroupCount(conv_desc_raw, groups);
    
    // Wrap the C API descriptor in C++ class
    auto conv_desc = miopen::deref(conv_desc_raw);
    
    // Create output tensor descriptor
    auto output_tensor_desc = conv_desc.GetForwardOutputTensor(input_tensor_desc, weight_tensor_desc);
    
    // Create problem description with NDHWC layout (channel-last) and FP16 data type
    auto problem = miopen::conv::ProblemDescription{
        input_tensor_desc, weight_tensor_desc, output_tensor_desc, conv_desc, miopen::conv::Direction::Forward};
    
    // Create solver instance
    ConvHipImplicitGemm3DChannelLastFwdWmmaops solver;
    
    // Test if the solver is applicable
    bool is_applicable = solver.IsApplicable(ctx, problem);
    
    // Print detailed diagnostics
    printf("\nTesting grouped convolution with ConvHipImplicitGemm3DChannelLastFwdWmmaops:\n");
    printf("  Groups: %d\n", groups);
    printf("  Input channels per group: %d\n", input_channels_per_group);
    printf("  Output channels per group: %d\n", output_channels_per_group);
    printf("  Is 3D: %s\n", problem.Is3d() ? "YES" : "NO");
    printf("  Is Forward: %s\n", problem.IsDirectionForward() ? "YES" : "NO");
    printf("  Is Layout NHWC: %s\n", problem.IsLayoutNHWC() ? "YES" : "NO");
    printf("  Is FP16: %s\n", problem.IsFp16() ? "YES" : "NO");
    printf("  Group Count: %d\n", problem.GetGroupCount());
    printf("  Tensors Casted: %s\n", problem.IsTensorsCasted() ? "YES" : "NO");
    printf("  All Tensors Dims Fit Into Int: %s\n", problem.AllTensorsDimsFitIntoInt() ? "YES" : "NO");
    printf("  Solver applicable: %s\n", is_applicable ? "YES" : "NO");
    
    // Verification
    EXPECT_EQUAL(is_applicable, true);
}

int main(int argc, char* argv[])
{
    printf("Running tests for ConvHipImplicitGemm3DChannelLastFwdWmmaops solver...\n");
    
    // Run applicability tests
    test_solver_applicability();
    test_grouped_convolution();
    
    printf("\nAll tests completed.\n");
    
    return 0;
}