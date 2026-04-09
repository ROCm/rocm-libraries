/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_LAYERNORM_DRIVER_HPP
#define GUARD_MIOPEN_LAYERNORM_DRIVER_HPP

#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>
#include <../test/cpu_layernorm.hpp>
#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/miopen.h"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/errors.hpp>
#include <miopen/tensor.hpp>
#include <vector>

template <typename T>
class LayerNormDriver : public Driver
{
public:
    LayerNormDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&weightDesc);
        miopenCreateTensorDescriptor(&biasDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&meanDesc);
        miopenCreateTensorDescriptor(&rstdDesc);
        miopenCreateTensorDescriptor(&dyDesc);
        miopenCreateTensorDescriptor(&dxDesc);
        miopenCreateTensorDescriptor(&dwDesc);
        miopenCreateTensorDescriptor(&dbDesc);

        data_type = miopen_type<T>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    void ValidateLayout();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    double GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~LayerNormDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(weightDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(meanDesc);
        miopenDestroyTensorDescriptor(rstdDesc);
        miopenDestroyTensorDescriptor(dyDesc);
        miopenDestroyTensorDescriptor(dxDesc);
        miopenDestroyTensorDescriptor(dwDesc);
        miopenDestroyTensorDescriptor(dbDesc);
    }

private:
    InputFlags inflags;

    int dim_size;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t weightDesc;
    miopenTensorDescriptor_t biasDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t meanDesc;
    miopenTensorDescriptor_t rstdDesc;
    miopenTensorDescriptor_t dyDesc;
    miopenTensorDescriptor_t dxDesc;
    miopenTensorDescriptor_t dwDesc;
    miopenTensorDescriptor_t dbDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> bias_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> mean_dev;
    std::unique_ptr<GPUMem> rstd_dev;
    std::unique_ptr<GPUMem> dy_dev;
    std::unique_ptr<GPUMem> dx_dev;
    std::unique_ptr<GPUMem> dw_dev;
    std::unique_ptr<GPUMem> db_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    tensor<T> in;
    tensor<T> weight;
    tensor<T> bias;
    tensor<T> out;
    tensor<T> mean;
    tensor<T> rstd;
    tensor<T> outhost;
    tensor<T> meanhost;
    tensor<T> rstdhost;
    tensor<T> dy;
    tensor<T> dx;
    tensor<T> dw;
    tensor<T> db;
    tensor<T> dxhost;
    tensor<T> dwhost;
    tensor<T> dbhost;

    size_t ws_sizeInBytes;

    float eps;
    int dim;
    miopenNormMode_t mode;

    bool use_multithread;
};

template <typename T>
int LayerNormDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    use_multithread = (inflags.GetValueInt("mt") != 0);

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::GetandSetData()
{
    auto in_len = GetInputTensorLengthsFromCmdLine();
    ValidateLayout();

    dim = inflags.GetValueInt("normalized_dim");

    MIOPEN_THROW_IF(dim < 0 || static_cast<size_t>(dim) >= in_len.size(),
                    "normalized_dim out of range");

    std::vector<int> inner_len;
    if(dim == in_len.size())
        inner_len = {1};
    else
        inner_len = {in_len.begin() + dim, in_len.end()};

    std::vector<int> outer_len;
    if(dim == 0)
        outer_len = {1};
    else
        outer_len = {in_len.begin(), in_len.end() - (in_len.size() - dim)};

    if(SetTensorNd(inputDesc, in_len, inflags.GetValueStr("layout"), data_type) !=
       miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor.");

    if(SetTensorNd(weightDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting weight tensor.");

    if(SetTensorNd(biasDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting bias tensor.");

    if(SetTensorNd(outputDesc, in_len, inflags.GetValueStr("layout"), data_type) !=
       miopenStatusSuccess)
        MIOPEN_THROW("Error setting output tensor.");

    if(SetTensorNd(meanDesc, outer_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting mean tensor.");

    if(SetTensorNd(rstdDesc, outer_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting rstd tensor.");

    if(SetTensorNd(dyDesc, in_len, inflags.GetValueStr("layout"), data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dy tensor.");

    if(SetTensorNd(dxDesc, in_len, inflags.GetValueStr("layout"), data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dx tensor.");

    if(SetTensorNd(dwDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting dw tensor.");

    if(SetTensorNd(dbDesc, inner_len, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error setting db tensor.");

    eps  = inflags.GetValueDouble("eps");
    mode = miopenNormMode_t(inflags.GetValueInt("mode"));

    return 0;
}

template <typename T>
int LayerNormDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only Forward LayerNorm (Default=0)", "int");
    inflags.AddInputFlag("layout",
                         'L',
                         "",
                         "Tensor layout: [NCHW, NHWC, NCDHW, NDHWC, NW] (Default=NCDHW/NCHW/NW)",
                         "string",
                         true);
    inflags.AddInputFlag("in_n", 'N', "100", "Input batch size (Default=100)", "int");
    inflags.AddInputFlag("in_c", 'C', "3", "Input channel number (Default=3)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input width (Default=32)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input depth (Default=0)", "int");

    inflags.AddInputFlag("eps", 'e', "0.00001", "Alpha (Default=0.00001)", "double");
    inflags.AddInputFlag(
        "normalized_dim",
        'o',
        "3",
        "Normalized Dim, given in NCDHW/NCHW/NW regardless of specified layout (Default=3)",
        "int");
    inflags.AddInputFlag(
        "mode", 'm', "0", "elemwise affine mode (0), weight and bias mode (1) (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("mt", 'u', "0", "Use multithreaded version (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::AllocateBuffersAndCopy()
{
    const T T0val      = static_cast<T>(0.0);
    const T T1val      = static_cast<T>(1.0);
    const T Tminus1val = static_cast<T>(-1.0);
    auto genT0val      = [&](auto...) { return T0val; };
    size_t in_sz       = GetTensorSize(inputDesc);
    size_t weight_sz   = GetTensorSize(weightDesc);
    size_t bias_sz     = GetTensorSize(biasDesc);
    size_t out_sz      = GetTensorSize(outputDesc);
    size_t mean_sz     = GetTensorSize(meanDesc);
    size_t rstd_sz     = GetTensorSize(rstdDesc);
    size_t dy_sz       = GetTensorSize(dyDesc);
    size_t dx_sz       = GetTensorSize(dxDesc);
    size_t dw_sz       = GetTensorSize(dwDesc);
    size_t db_sz       = GetTensorSize(dbDesc);

    auto status = miopenGetLayerNormBackwardWorkspaceSize(GetHandle(),
                                                          mode,
                                                          dyDesc,
                                                          inputDesc,
                                                          weightDesc,
                                                          meanDesc,
                                                          rstdDesc,
                                                          dim,
                                                          dxDesc,
                                                          dwDesc,
                                                          dbDesc,
                                                          &ws_sizeInBytes);
    if(status != miopenStatusSuccess)
        return miopenStatusAllocFailed;

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(T)));
    weight_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(T)));
    bias_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, bias_sz, sizeof(T)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(T)));
    mean_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, mean_sz, sizeof(T)));
    rstd_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, rstd_sz, sizeof(T)));
    dy_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dy_sz, sizeof(T)));
    dx_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dx_sz, sizeof(T)));
    dw_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, dw_sz, sizeof(T)));
    db_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, db_sz, sizeof(T)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in       = tensor<T>(miopen::deref(inputDesc)).generate(genT0val);
    weight   = tensor<T>(miopen::deref(weightDesc)).generate(genT0val);
    bias     = tensor<T>(miopen::deref(biasDesc)).generate(genT0val);
    out      = tensor<T>(miopen::deref(outputDesc)).generate(genT0val);
    mean     = tensor<T>(miopen::deref(meanDesc)).generate(genT0val);
    rstd     = tensor<T>(miopen::deref(rstdDesc)).generate(genT0val);
    dy       = tensor<T>(miopen::deref(dyDesc)).generate(genT0val);
    dx       = tensor<T>(miopen::deref(dxDesc)).generate(genT0val);
    dw       = tensor<T>(miopen::deref(dwDesc)).generate(genT0val);
    db       = tensor<T>(miopen::deref(dbDesc)).generate(genT0val);
    outhost  = tensor<T>(miopen::deref(outputDesc)).generate(genT0val);
    meanhost = tensor<T>(miopen::deref(meanDesc)).generate(genT0val);
    rstdhost = tensor<T>(miopen::deref(rstdDesc)).generate(genT0val);
    dxhost   = tensor<T>(miopen::deref(dxDesc)).generate(genT0val);
    dwhost   = tensor<T>(miopen::deref(dwDesc)).generate(genT0val);
    dbhost   = tensor<T>(miopen::deref(dbDesc)).generate(genT0val);

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<T>(T0val, T1val);
        dy[i] = prng::gen_A_to_B<T>(Tminus1val, T1val);
    }

    if(in_dev->ToGPU(GetStream(), in.data.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;
    if(dy_dev->ToGPU(GetStream(), dy.data.data()) != 0)
        std::cerr << "Error copying (dy) to GPU, size: " << dy_dev->GetSize() << std::endl;

    for(int i = 0; i < weight_sz; i++)
    {
        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
            weight[i] = static_cast<T>(1);
        else
            weight[i] = prng::gen_A_to_B<T>(T0val, T1val);
    }

    if(weight_dev->ToGPU(GetStream(), weight.data.data()) != 0)
        std::cerr << "Error copying (weight) to GPU, size: " << weight_dev->GetSize() << std::endl;

    for(int i = 0; i < bias_sz; i++)
    {
        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
            bias[i] = T0val;
        else
            bias[i] = prng::gen_A_to_B<T>(T0val, T1val);
    }
    if(bias_dev->ToGPU(GetStream(), bias.data.data()) != 0)
        std::cerr << "Error copying (bias) to GPU, size: " << bias_dev->GetSize() << std::endl;

    if(out_dev->ToGPU(GetStream(), out.data.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    if(mean_dev->ToGPU(GetStream(), mean.data.data()) != 0)
        std::cerr << "Error copying (mean) to GPU, size: " << mean_dev->GetSize() << std::endl;

    if(rstd_dev->ToGPU(GetStream(), rstd.data.data()) != 0)
        std::cerr << "Error copying (rstd) to GPU, size: " << rstd_dev->GetSize() << std::endl;

    if(dx_dev->ToGPU(GetStream(), dx.data.data()) != 0)
        std::cerr << "Error copying (dx) to GPU, size: " << dx_dev->GetSize() << std::endl;

    if(dw_dev->ToGPU(GetStream(), dw.data.data()) != 0)
        std::cerr << "Error copying (dw) to GPU, size: " << dw_dev->GetSize() << std::endl;

    if(db_dev->ToGPU(GetStream(), db.data.data()) != 0)
        std::cerr << "Error copying (db) to GPU, size: " << db_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLayerNormForward(GetHandle(),
                               mode,
                               inputDesc,
                               in_dev->GetMem(),
                               weightDesc,
                               weight_dev->GetMem(),
                               biasDesc,
                               bias_dev->GetMem(),
                               eps,
                               dim,
                               outputDesc,
                               out_dev->GetMem(),
                               meanDesc,
                               mean_dev->GetMem(),
                               rstdDesc,
                               rstd_dev->GetMem());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward LayerNorm Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    if(mean_dev->FromGPU(GetStream(), mean.data.data()) != 0)
        std::cerr << "Error copying (mean_dev) from GPU, size: " << mean_dev->GetSize()
                  << std::endl;

    if(rstd_dev->FromGPU(GetStream(), rstd.data.data()) != 0)
        std::cerr << "Error copying (rstd_dev) from GPU, size: " << rstd_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::RunForwardCPU()
{
    cpu_layernorm_forward<T>(
        in, weight, bias, outhost, meanhost, rstdhost, eps, dim, mode, use_multithread);

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenLayerNormBackward(GetHandle(),
                                mode,
                                workspace_dev->GetMem(),
                                ws_sizeInBytes,
                                dyDesc,
                                dy_dev->GetMem(),
                                inputDesc,
                                in_dev->GetMem(),
                                weightDesc,
                                weight_dev->GetMem(),
                                meanDesc,
                                mean_dev->GetMem(),
                                rstdDesc,
                                rstd_dev->GetMem(),
                                dim,
                                dxDesc,
                                dx_dev->GetMem(),
                                dwDesc,
                                dw_dev->GetMem(),
                                dbDesc,
                                db_dev->GetMem());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Backward LayerNorm Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward LayerNorm Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(dx_dev->FromGPU(GetStream(), dx.data.data()) != 0)
        std::cerr << "Error copying (dx_dev) from GPU, size: " << dx_dev->GetSize() << std::endl;

    if(dw_dev->FromGPU(GetStream(), dw.data.data()) != 0)
        std::cerr << "Error copying (dw_dev) from GPU, size: " << dw_dev->GetSize() << std::endl;

    if(db_dev->FromGPU(GetStream(), db.data.data()) != 0)
        std::cerr << "Error copying (db_dev) from GPU, size: " << db_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::RunBackwardCPU()
{
    cpu_layernorm_backward(dy, in, weight, meanhost, rstdhost, dxhost, dim, mode, use_multithread);

    cpu_layernorm_backward_weight_bias(
        dy, in, meanhost, rstdhost, dwhost, dbhost, dim, use_multithread);

    return miopenStatusSuccess;
}

template <typename T>
double LayerNormDriver<T>::GetTolerance()
{
    // Take the greater of twice a random walk of floating point errors in the accumulator type or
    // one floating point error in the buffer type
    auto mantissa_bits = std::is_same<T, float>::value              ? 23
                         : std::is_same<T, half_float::half>::value ? 10
                                                                    : 7;
    auto tolerance =
        std::max(2.0 * std::sqrt(in.GetSize()) / (1 << 23), 1.0 / (1 << mantissa_bits));
    return tolerance;
}

template <typename T>
int LayerNormDriver<T>::VerifyForward()
{
    RunForwardCPU();
    const double tolerance  = GetTolerance();
    auto error              = miopen::rms_range(outhost, out);
    std::string solver_type = use_multithread ? "multi-threaded" : "single-threaded";

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward LayerNorm FAILED against " << solver_type
                  << " CPU reference: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LayerNorm Verifies OK against " << solver_type << " CPU reference ("
                  << error << " < " << tolerance << ')' << std::endl;
    }

    auto meanerror = miopen::rms_range(meanhost, mean);
    if(!std::isfinite(meanerror) || meanerror > tolerance)
    {
        std::cout << "Forward LayerNorm mean FAILED against " << solver_type
                  << " CPU reference: " << meanerror << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LayerNorm mean Verifies OK against " << solver_type
                  << " CPU reference (" << meanerror << " < " << tolerance << ')' << std::endl;
    }

    auto rstderror = miopen::rms_range(rstdhost, rstd);
    if(!std::isfinite(rstderror) || rstderror > tolerance)
    {
        std::cout << "Forward LayerNorm rstd FAILED against " << solver_type
                  << " CPU reference: " << rstderror << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward LayerNorm rstd Verifies OK against " << solver_type
                  << " CPU reference (" << rstderror << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename T>
int LayerNormDriver<T>::VerifyBackward()
{
    RunBackwardCPU();
    const double tolerance  = GetTolerance();
    auto error              = miopen::rms_range(dxhost, dx);
    std::string solver_type = use_multithread ? "multi-threaded" : "single-threaded";

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward LayerNorm FAILED against " << solver_type
                  << " CPU reference: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward LayerNorm Verifies OK against " << solver_type << " CPU reference ("
                  << error << " < " << tolerance << ')' << std::endl;
    }

    auto dwerror = miopen::rms_range(dwhost, dw);
    if(!std::isfinite(dwerror) || dwerror > tolerance)
    {
        std::cout << "Backward LayerNorm dw FAILED against " << solver_type
                  << " CPU reference: " << dwerror << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward LayerNorm Verifies dw OK against " << solver_type
                  << " CPU reference (" << dwerror << " < " << tolerance << ')' << std::endl;
    }

    auto dberror = miopen::rms_range(dbhost, db);
    if(!std::isfinite(dberror) || dberror > tolerance)
    {
        std::cout << "Backward LayerNorm db FAILED against " << solver_type
                  << " CPU reference: " << dberror << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward LayerNorm Verifies db OK against " << solver_type
                  << " CPU reference (" << dberror << " < " << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename T>
std::vector<int> LayerNormDriver<T>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("in_n");
    int in_c = inflags.GetValueInt("in_c");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");
    int in_d = inflags.GetValueInt("in_d");

    if(in_d != 0)
    {
        return std::vector<int>{in_n, in_c, in_d, in_h, in_w};
    }
    else if(in_c != 0 && in_h != 0)
    {
        return std::vector<int>{in_n, in_c, in_h, in_w};
    }
    else if(in_n != 0 && in_c == 0 && in_h == 0 && in_w != 0)
    {
        return std::vector<int>{in_n, in_w};
    }
    else
    {
        MIOPEN_THROW("Invalid tensor sizes");
    }
}

template <typename T>
void LayerNormDriver<T>::ValidateLayout()
{
    auto layout_value = inflags.GetValueStr("layout");
    int in_d          = inflags.GetValueInt("in_d");
    int in_c          = inflags.GetValueInt("in_c");
    int in_h          = inflags.GetValueInt("in_h");
    if(layout_value.empty())
    {
        if(in_d != 0)
        {
            inflags.SetValue("layout", "NCDHW");
        }
        else if(in_c != 0 && in_h != 0)
        {
            inflags.SetValue("layout", "NCHW");
        }
        else
        {
            inflags.SetValue("layout", "NW");
        }
    }
    else if(layout_value != "NCHW" && layout_value != "NHWC" && layout_value != "NCDHW" &&
            layout_value != "NDHWC" && layout_value != "NW")
    {
        MIOPEN_THROW(miopenStatusBadParm, "Invalid layout parameter value: " + layout_value);
    }
    else if((in_d == 0 || in_c == 0 || in_h == 0) &&
            (layout_value == "NCDHW" || layout_value == "NDHWC"))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The input depth (in_d), channels (in_c) and height (in_h) must be greater "
                     "than zero for layouts NCDHW and NDHWC");
    }
    else if(in_d != 0 && (layout_value == "NCHW" || layout_value == "NHWC"))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The input depth (in_d) must be zero and the input channels (in_c) and height "
                     "(in_h) must be greater than zero for layouts NCHW and NHWC");
    }
    else if((in_d != 0 || in_c != 0 || in_h != 0) && layout_value == "NW")
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The input depth (in_d), channels (in_c) and height (in_h) must be zero for "
                     "layout NW");
    }

    int normalized_dim = inflags.GetValueInt("normalized_dim");
    if(normalized_dim >= 5 && (layout_value == "NCDHW" || layout_value == "NDHWC"))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The normalized dimension (normalized_dim) must be less than 5 for layouts "
                     "NCDHW and NDHWC");
    }
    else if(normalized_dim >= 4 && (layout_value == "NCHW" || layout_value == "NHWC"))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The normalized dimension (normalized_dim) must be less than 4 for layouts "
                     "NCHW and NHWC");
    }
    else if(normalized_dim >= 2 && layout_value == "NW")
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The normalized dimension (normalized_dim) must be less than 2 for layout NW");
    }
}

#endif // GUARD_MIOPEN_LAYERNORM_DRIVER_HPP
