// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ============================================================================
// Standalone driver: runs this branch's GPU SDPA reference kernel
// (hipdnn_gpu_ref::GpuFpReferenceSdpa::fprop) on Q/K/V (and optional additive
// mask) loaded from .npy files, then writes the fp32 output O (and optional
// fp32 LSE) to .npy files.
//
// A separate Python harness generates the inputs and compares this driver's
// output against the selected PyTorch MATH or AOTriton reference. This program
// is the C++ half only.
//
// Integration contract (must match the Python side):
//   - .npy v1.0, little-endian, C-contiguous.
//   - fp32 -> '<f4'; fp16 -> '<f2'; bf16 -> '<u2' (raw 16-bit bf16 bit patterns);
//     fp8 (e4m3/e5m2 and their fnuz variants) -> '|u1' (raw 8-bit fp8 bit patterns).
//   - Q = [B, Hq, Sq, D], K = [B, Hkv, Skv, D], V = [B, Hkv, Skv, Dv].
//   - Mask (optional) is always '<f4'.
//   - Output O = '<f4' [B, Hq, Sq, Dv]; LSE (optional) = '<f4' [B, Hq, Sq].
// ============================================================================

#include "NpyIO.hpp"

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <hipdnn_gpu_ref/GpuFpReferenceSdpa.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

enum class ElementDType
{
    Bf16,
    Fp16,
    Fp32,
    Fp8E4m3,
    Fp8E5m2,
    Fp8E4m3Fnuz,
    Fp8E5m2Fnuz
};

struct Options
{
    std::string qPath;
    std::string kPath;
    std::string vPath;
    std::string maskPath;
    std::string oPath;
    std::string lsePath;
    ElementDType dtype{ElementDType::Fp32};
    std::optional<float> scale;
    int64_t leftBound{-1};
    int64_t rightBound{-1};
    bool topLeftAlignment{true};
    bool haveMask{false};
    bool haveLse{false};
};

[[noreturn]] void usageError(const std::string& message)
{
    throw std::runtime_error(
        message
        + "\nUsage: sdpa_reference_driver --q Q.npy --k K.npy --v V.npy [--mask MASK.npy]"
          " --o O.npy [--lse LSE.npy]"
          " --dtype {bf16|fp16|fp32|fp8_e4m3|fp8_e5m2|fp8_e4m3_fnuz|fp8_e5m2_fnuz}"
          " [--scale FLOAT] [--left INT] [--right INT] [--top-left|--bottom-right]");
}

// Fetch the value that must follow a value-taking flag.
std::string nextValue(int argc, char** argv, int& i, const std::string& flag)
{
    if(i + 1 >= argc)
    {
        usageError("missing value for " + flag);
    }
    ++i;
    return std::string(argv[static_cast<size_t>(i)]);
}

Options parseArgs(int argc, char** argv)
{
    Options opts;
    bool dtypeSet = false;

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[static_cast<size_t>(i)]);
        if(arg == "--q")
        {
            opts.qPath = nextValue(argc, argv, i, arg);
        }
        else if(arg == "--k")
        {
            opts.kPath = nextValue(argc, argv, i, arg);
        }
        else if(arg == "--v")
        {
            opts.vPath = nextValue(argc, argv, i, arg);
        }
        else if(arg == "--mask")
        {
            opts.maskPath = nextValue(argc, argv, i, arg);
            opts.haveMask = true;
        }
        else if(arg == "--o")
        {
            opts.oPath = nextValue(argc, argv, i, arg);
        }
        else if(arg == "--lse")
        {
            opts.lsePath = nextValue(argc, argv, i, arg);
            opts.haveLse = true;
        }
        else if(arg == "--dtype")
        {
            const std::string value = nextValue(argc, argv, i, arg);
            if(value == "bf16")
            {
                opts.dtype = ElementDType::Bf16;
            }
            else if(value == "fp16")
            {
                opts.dtype = ElementDType::Fp16;
            }
            else if(value == "fp32")
            {
                opts.dtype = ElementDType::Fp32;
            }
            else if(value == "fp8_e4m3")
            {
                opts.dtype = ElementDType::Fp8E4m3;
            }
            else if(value == "fp8_e5m2")
            {
                opts.dtype = ElementDType::Fp8E5m2;
            }
            else if(value == "fp8_e4m3_fnuz")
            {
                opts.dtype = ElementDType::Fp8E4m3Fnuz;
            }
            else if(value == "fp8_e5m2_fnuz")
            {
                opts.dtype = ElementDType::Fp8E5m2Fnuz;
            }
            else
            {
                usageError("unknown --dtype '" + value
                           + "' (expected bf16|fp16|fp32|fp8_e4m3|fp8_e5m2|fp8_e4m3_fnuz|"
                             "fp8_e5m2_fnuz)");
            }
            dtypeSet = true;
        }
        else if(arg == "--scale")
        {
            opts.scale = std::stof(nextValue(argc, argv, i, arg));
        }
        else if(arg == "--left")
        {
            opts.leftBound = static_cast<int64_t>(std::stoll(nextValue(argc, argv, i, arg)));
        }
        else if(arg == "--right")
        {
            opts.rightBound = static_cast<int64_t>(std::stoll(nextValue(argc, argv, i, arg)));
        }
        else if(arg == "--top-left")
        {
            opts.topLeftAlignment = true;
        }
        else if(arg == "--bottom-right")
        {
            opts.topLeftAlignment = false;
        }
        else
        {
            usageError("unknown argument '" + arg + "'");
        }
    }

    if(opts.qPath.empty() || opts.kPath.empty() || opts.vPath.empty())
    {
        usageError("--q, --k and --v are all required");
    }
    if(opts.oPath.empty())
    {
        usageError("--o is required");
    }
    if(!dtypeSet)
    {
        usageError("--dtype is required");
    }

    return opts;
}

// On-disk dtype expected for the chosen element type.
sdpa_harness::npy::DType expectedInputDtype(ElementDType dtype)
{
    switch(dtype)
    {
    case ElementDType::Bf16:
        return sdpa_harness::npy::DType::U2;
    case ElementDType::Fp16:
        return sdpa_harness::npy::DType::F2;
    case ElementDType::Fp32:
        return sdpa_harness::npy::DType::F4;
    case ElementDType::Fp8E4m3:
    case ElementDType::Fp8E5m2:
    case ElementDType::Fp8E4m3Fnuz:
    case ElementDType::Fp8E5m2Fnuz:
        return sdpa_harness::npy::DType::U1;
    default:
        throw std::runtime_error("driver: unknown element dtype");
    }
}

const char* dtypeName(ElementDType dtype)
{
    switch(dtype)
    {
    case ElementDType::Bf16:
        return "bf16";
    case ElementDType::Fp16:
        return "fp16";
    case ElementDType::Fp32:
        return "fp32";
    case ElementDType::Fp8E4m3:
        return "fp8_e4m3";
    case ElementDType::Fp8E5m2:
        return "fp8_e5m2";
    case ElementDType::Fp8E4m3Fnuz:
        return "fp8_e4m3_fnuz";
    case ElementDType::Fp8E5m2Fnuz:
        return "fp8_e5m2_fnuz";
    default:
        return "?";
    }
}

const char* descrName(sdpa_harness::npy::DType dtype)
{
    switch(dtype)
    {
    case sdpa_harness::npy::DType::F4:
        return "<f4";
    case sdpa_harness::npy::DType::F2:
        return "<f2";
    case sdpa_harness::npy::DType::U2:
        return "<u2";
    case sdpa_harness::npy::DType::U1:
        return "|u1";
    default:
        return "?";
    }
}

void requireDtype(const sdpa_harness::npy::NpyArray& array,
                  sdpa_harness::npy::DType expected,
                  const std::string& which)
{
    if(array.dtype != expected)
    {
        throw std::runtime_error("driver: " + which + " has on-disk dtype '"
                                 + descrName(array.dtype) + "' but '" + descrName(expected)
                                 + "' was expected");
    }
}

void requireRank4(const std::vector<int64_t>& shape, const std::string& which)
{
    if(shape.size() != 4)
    {
        throw std::runtime_error("driver: " + which + " must be rank-4 [B, H, S, D] (got rank "
                                 + std::to_string(shape.size()) + ")");
    }
}

// Run the reference for a concrete input/output element type T.
template <typename T>
void runTyped(const Options& opts,
              const sdpa_harness::npy::NpyArray& qArr,
              const sdpa_harness::npy::NpyArray& kArr,
              const sdpa_harness::npy::NpyArray& vArr,
              const sdpa_harness::npy::NpyArray* maskArr)
{
    namespace util = hipdnn_data_sdk::utilities;

    const std::vector<int64_t>& dimsQ = qArr.shape;
    const std::vector<int64_t>& dimsK = kArr.shape;
    const std::vector<int64_t>& dimsV = vArr.shape;

    const int64_t batch = dimsQ[0];
    const int64_t numHeadsQ = dimsQ[1];
    const int64_t seqQ = dimsQ[2];
    const int64_t headDimV = dimsV[3];

    util::Tensor<T> q(dimsQ);
    util::Tensor<T> k(dimsK);
    util::Tensor<T> v(dimsV);
    q.fillWithData(qArr.data.data(), qArr.data.size());
    k.fillWithData(kArr.data.data(), kArr.data.size());
    v.fillWithData(vArr.data.data(), vArr.data.size());

    // Optional additive mask (always fp32).
    std::optional<util::Tensor<float>> mask;
    util::TensorBase<float>* maskPtr = nullptr;
    if(maskArr != nullptr)
    {
        mask.emplace(maskArr->shape);
        mask->fillWithData(maskArr->data.data(), maskArr->data.size());
        maskPtr = &mask.value();
    }

    util::Tensor<float> o({batch, numHeadsQ, seqQ, headDimV});

    std::optional<util::Tensor<float>> lse;
    util::TensorBase<float>* lsePtr = nullptr;
    if(opts.haveLse)
    {
        lse.emplace(std::vector<int64_t>{batch, numHeadsQ, seqQ});
        lsePtr = &lse.value();
    }

    hipdnn_gpu_ref::GpuFpReferenceSdpa::fprop<T, T, T, float, float>(q,
                                                                     k,
                                                                     v,
                                                                     o,
                                                                     opts.scale,
                                                                     maskPtr,
                                                                     opts.leftBound,
                                                                     opts.rightBound,
                                                                     opts.topLeftAlignment,
                                                                     lsePtr);

    // Non-const hostData() triggers the device->host migration that the kernel's
    // markDeviceModified() requires before the host buffer is readable.
    const float* oHost = o.memory().hostData();
    sdpa_harness::npy::writeF4(opts.oPath, oHost, o.dims());

    if(opts.haveLse)
    {
        const float* lseHost = lse->memory().hostData();
        sdpa_harness::npy::writeF4(opts.lsePath, lseHost, lse->dims());
    }
}

void run(const Options& opts)
{
    const sdpa_harness::npy::NpyArray qArr = sdpa_harness::npy::read(opts.qPath);
    const sdpa_harness::npy::NpyArray kArr = sdpa_harness::npy::read(opts.kPath);
    const sdpa_harness::npy::NpyArray vArr = sdpa_harness::npy::read(opts.vPath);

    const sdpa_harness::npy::DType expected = expectedInputDtype(opts.dtype);
    requireDtype(qArr, expected, "--q");
    requireDtype(kArr, expected, "--k");
    requireDtype(vArr, expected, "--v");

    requireRank4(qArr.shape, "--q");
    requireRank4(kArr.shape, "--k");
    requireRank4(vArr.shape, "--v");

    std::optional<sdpa_harness::npy::NpyArray> maskArr;
    if(opts.haveMask)
    {
        maskArr = sdpa_harness::npy::read(opts.maskPath);
        requireDtype(maskArr.value(), sdpa_harness::npy::DType::F4, "--mask");
    }
    const sdpa_harness::npy::NpyArray* maskPtr = opts.haveMask ? &maskArr.value() : nullptr;

    switch(opts.dtype)
    {
    case ElementDType::Bf16:
        runTyped<hipdnn_data_sdk::types::bfloat16>(opts, qArr, kArr, vArr, maskPtr);
        break;
    case ElementDType::Fp16:
        runTyped<hipdnn_data_sdk::types::half>(opts, qArr, kArr, vArr, maskPtr);
        break;
    case ElementDType::Fp32:
        runTyped<float>(opts, qArr, kArr, vArr, maskPtr);
        break;
    case ElementDType::Fp8E4m3:
        runTyped<hipdnn_data_sdk::types::fp8_e4m3>(opts, qArr, kArr, vArr, maskPtr);
        break;
    case ElementDType::Fp8E5m2:
        runTyped<hipdnn_data_sdk::types::fp8_e5m2>(opts, qArr, kArr, vArr, maskPtr);
        break;
    case ElementDType::Fp8E4m3Fnuz:
        runTyped<hipdnn_data_sdk::types::fp8_e4m3_fnuz>(opts, qArr, kArr, vArr, maskPtr);
        break;
    case ElementDType::Fp8E5m2Fnuz:
        runTyped<hipdnn_data_sdk::types::fp8_e5m2_fnuz>(opts, qArr, kArr, vArr, maskPtr);
        break;
    default:
        throw std::runtime_error("driver: unhandled dtype " + std::string(dtypeName(opts.dtype)));
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options opts = parseArgs(argc, argv);
        run(opts);
        std::cout << "OK " << opts.oPath << "\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "sdpa_reference_driver: error: " << e.what() << "\n";
        return 1;
    }
}
