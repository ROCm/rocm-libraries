// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormBwdPlan.hpp"
#include "HipKernelHandle.hpp"
#include "hip/HipKernel.hpp"
#include "hip/HipProgram.hpp"
#include "hip/HipUtils.hpp"

#include <algorithm>
#include <cmath>
#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hip_kernel_provider
{

// ============================================================================
// Variant selection heuristics (ported from MIOpen common_spatial.hpp)
// ============================================================================

namespace
{

struct SpatialSingleConfig
{
    int variant = 1;
    size_t vecSize = 1;
};

struct SpatialMultipleConfig
{
    size_t xlocalsize = 1;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;
    size_t nelements = 1;
    size_t vecSize = 4;
    int stashMethod = 0;
};

bool useMultiple(unsigned int n, unsigned int hw, bool isNHWC)
{
    unsigned int nhw = n * hw;
    if(!isNHWC)
    {
        if(!((nhw >= static_cast<unsigned int>(32 * 1024 * 1024) || hw <= 1024)
             && (nhw >= static_cast<unsigned int>(32 * 1024 * 1024) || hw <= 512) && hw > 512))
        {
            return false;
        }
    }
    return true;
}

SpatialSingleConfig
    defaultConfigSpatialSingle(int n, unsigned int hw, unsigned int nhw, bool isNHWC, bool isFpMix)
{
    SpatialSingleConfig cfg;
    cfg.variant = 1;
    cfg.vecSize = 1;

    if(!isNHWC)
    {
        if(hw < 200 && hw > 60 && isFpMix)
        {
            cfg.variant = 1;
            return cfg;
        }

        if(nhw < (32 * 1024 * 1024) && hw > 1024)
        {
            cfg.variant = 1;
            return cfg;
        }
        else if(nhw < (32 * 1024 * 1024) && hw > 512)
        {
            cfg.variant = (n >= 32) ? 1 : 3;
            return cfg;
        }
        else if(hw <= 512)
        {
            if(n > 64 && hw > 160)
            {
                cfg.variant = 3;
            }
            else
            {
                cfg.variant = 0;
            }
            return cfg;
        }
    }

    return cfg;
}

void getLocalConfigNHWC(
    size_t c, size_t hw, bool isFp32, size_t vecSize, size_t& xlocalsize, size_t& ylocalsize)
{
    size_t xlocalsize_limit = vecSize > 1 ? (isFp32 ? size_t{16} : size_t{32}) : size_t{64};
    size_t max_localsize = size_t{1024} / vecSize;

    size_t nworkgroups = 0;
    size_t minWGs = 1;

    while(nworkgroups < minWGs && max_localsize >= xlocalsize_limit && max_localsize > 64)
    {
        xlocalsize = std::min(static_cast<size_t>(1 << static_cast<int>(std::ceil(std::log2(
                                                      static_cast<double>(c / vecSize))))),
                              xlocalsize_limit);
        ylocalsize = max_localsize / xlocalsize;
        nworkgroups
            = ((c / vecSize + xlocalsize - 1) / xlocalsize) * ((hw + ylocalsize - 1) / ylocalsize);
        max_localsize >>= 1;
    }
}

void getSpatialMultipleConfig(size_t c,
                              size_t hw,
                              bool isNHWC,
                              bool isFp32,
                              size_t vecSize,
                              size_t& xlocalsize,
                              size_t& ylocalsize)
{
    xlocalsize = 1;
    ylocalsize = 1;

    if(isNHWC)
    {
        if(c % vecSize != 0)
        {
            return;
        }
        getLocalConfigNHWC(c, hw, isFp32, vecSize, xlocalsize, ylocalsize);
    }
    else
    {
        if(hw % vecSize != 0)
        {
            return;
        }
        ylocalsize = 1024;
        if(ylocalsize > hw / vecSize)
        {
            ylocalsize = std::max(size_t{64},
                                  static_cast<size_t>(1 << static_cast<int>(std::ceil(std::log2(
                                                          static_cast<double>(hw / vecSize))))));
        }
    }
}

bool isSpatialMultipleApplicable(size_t n,
                                 size_t c,
                                 size_t hw,
                                 bool isNHWC,
                                 bool isFp32,
                                 size_t vecSize,
                                 size_t stashValues,
                                 size_t ylocalsize,
                                 size_t zlocalsize,
                                 size_t nelements)
{
    if(isNHWC)
    {
        if(c % vecSize != 0)
            return false;

        size_t sv = stashValues * (isFp32 ? size_t{1} : size_t{2});
        size_t lastY = hw % ylocalsize == 0 ? ylocalsize : hw % ylocalsize;
        size_t lastZ = n % (zlocalsize * nelements) == 0 ? (zlocalsize * nelements)
                                                         : n % (zlocalsize * nelements);

        if((!isFp32 && (c % 2 != 0 && lastZ < sv)) || ((lastY < sv) && (lastZ < sv)))
            return false;
    }
    else
    {
        if(hw % vecSize != 0)
            return false;

        size_t sv = stashValues * (isFp32 ? size_t{1} : size_t{2});
        size_t lastY = hw % ylocalsize == 0 ? ylocalsize : hw % ylocalsize;
        size_t lastZ = n % (zlocalsize * nelements) == 0 ? (zlocalsize * nelements)
                                                         : n % (zlocalsize * nelements);
        if(lastY < sv && lastZ < sv)
            return false;
    }
    return true;
}

int getStashMethod(bool isNHWC,
                   bool isFp32,
                   size_t c,
                   size_t n,
                   size_t hw,
                   size_t stashValues,
                   size_t ylocalsize,
                   size_t zlocalsize,
                   size_t nelements)
{
    int method = 0;
    size_t sv = stashValues * (isFp32 ? size_t{1} : size_t{2});
    size_t lastY = hw % ylocalsize == 0 ? ylocalsize : hw % ylocalsize;
    size_t lastZ = n % (zlocalsize * nelements) == 0 ? (zlocalsize * nelements)
                                                     : n % (zlocalsize * nelements);
    if(lastY < sv && lastZ >= sv)
    {
        method = 1;
    }
    if(isNHWC && !isFp32 && (c % 2 != 0) && (lastZ >= sv))
    {
        method = 2;
    }
    return method;
}

void getHeuristicsConfigTuningNHWC(
    size_t n, size_t c, size_t hw, size_t& vecSize, size_t& xlocalsize)
{
    size_t c_next_pow2 = static_cast<size_t>(1)
                         << static_cast<int>(std::ceil(std::log2(static_cast<double>(c))));
    if(c != c_next_pow2)
    {
        size_t max_modulo = 0;
        for(size_t vs = 8; vs > 1; vs >>= 1)
        {
            for(size_t xl = 64; xl > 8; xl >>= 1)
            {
                size_t xl_pow2 = std::min(
                    static_cast<size_t>(
                        1 << static_cast<int>(std::ceil(std::log2(static_cast<double>(c / vs))))),
                    xl);
                size_t modulo = c % (xl_pow2 * vs);
                if(modulo == 0)
                {
                    vecSize = vs;
                    xlocalsize = xl_pow2;
                    break;
                }
                else
                {
                    if(modulo > max_modulo)
                    {
                        vecSize = vs;
                        xlocalsize = xl_pow2;
                        max_modulo = modulo;
                    }
                }
            }
        }
        return;
    }

    // Backward heuristics for power-of-2 C
    if(c <= 64)
    {
        vecSize = 2;
        xlocalsize = 32;
    }
    else if(c == 128)
    {
        vecSize = 2;
        xlocalsize = (hw >= 64) ? size_t{64} : size_t{32};
    }
    else if(c == 256)
    {
        vecSize = (n < 64) ? ((hw > 4096) ? size_t{8} : size_t{2})
                           : ((hw >= 1024) ? size_t{8} : size_t{2});
        xlocalsize = (n < 64) ? ((hw <= 4096) ? size_t{64} : size_t{32})
                              : ((hw < 1024) ? size_t{64} : size_t{32});
    }
    else if(c == 512)
    {
        vecSize = (n < 64) ? ((hw >= 4096) ? size_t{8} : size_t{2})
                           : ((hw >= 256) ? size_t{8} : size_t{2});
        xlocalsize = (n < 64) ? ((hw >= 4096) ? size_t{32} : size_t{64})
                              : ((hw > 256) ? size_t{32} : size_t{64});
    }
    else if(c == 1024)
    {
        vecSize = (n < 64) ? ((hw <= 1024) ? size_t{2} : size_t{8})
                           : ((hw <= 256) ? size_t{4} : size_t{8});
        xlocalsize = (n < 64) ? ((hw <= 1024) ? size_t{64} : size_t{32})
                              : ((hw <= 256) ? size_t{64} : size_t{32});
    }
    else
    {
        vecSize = (hw <= 64) ? size_t{4} : size_t{8};
        xlocalsize = 64;
    }
    xlocalsize = std::min(static_cast<size_t>(1 << static_cast<int>(std::ceil(
                                                  std::log2(static_cast<double>(c / vecSize))))),
                          xlocalsize);
}

} // anonymous namespace

// ============================================================================
// BatchnormBwdParams constructors
// ============================================================================

BatchnormBwdParams::BatchnormBwdParams(
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _dscale(tensorMap.at(attributes.dscale_tensor_uid()))
    , _dbias(tensorMap.at(attributes.dbias_tensor_uid()))
    , _savedMean(attributes.mean_tensor_uid().has_value()
                     ? tensorMap.at(attributes.mean_tensor_uid().value())
                     : nullptr)
    , _savedInvVariance(attributes.inv_variance_tensor_uid().has_value()
                            ? tensorMap.at(attributes.inv_variance_tensor_uid().value())
                            : nullptr)
    , _bias(nullptr)
{
}

BatchnormBwdParams::BatchnormBwdParams(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& backwardAttributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(backwardAttributes.x_tensor_uid()))
    , _dy(tensorMap.at(backwardAttributes.dy_tensor_uid()))
    , _dx(tensorMap.at(backwardAttributes.dx_tensor_uid()))
    , _scale(tensorMap.at(backwardAttributes.scale_tensor_uid()))
    , _dscale(tensorMap.at(backwardAttributes.dscale_tensor_uid()))
    , _dbias(tensorMap.at(backwardAttributes.dbias_tensor_uid()))
    , _savedMean(backwardAttributes.mean_tensor_uid().has_value()
                     ? tensorMap.at(backwardAttributes.mean_tensor_uid().value())
                     : nullptr)
    , _savedInvVariance(backwardAttributes.inv_variance_tensor_uid().has_value()
                            ? tensorMap.at(backwardAttributes.inv_variance_tensor_uid().value())
                            : nullptr)
    , _bias(tensorMap.at(inferenceAttributes.bias_tensor_uid()))
    , _optActivation(hip_kernel_utils::parseActivation(pointwiseAttributes))
{
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::x() const
{
    return _x;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dy() const
{
    return _dy;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dx() const
{
    return _dx;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::scale() const
{
    return _scale;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dscale() const
{
    return _dscale;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dbias() const
{
    return _dbias;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::savedMean() const
{
    return _savedMean;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::savedInvVariance() const
{
    return _savedInvVariance;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormBwdParams::bias() const
{
    return _bias;
}

const std::optional<hip_kernel_utils::ActivationParams>& BatchnormBwdParams::optActivation() const
{
    return _optActivation;
}

BatchnormBwdPlan::BatchnormBwdPlan(BatchnormBwdParams&& bwdParams, bool benchmarkingEnabled)
    : _bwdParams(std::move(bwdParams))
    , _benchmarkingEnabled(benchmarkingEnabled)
{
}

size_t BatchnormBwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    return 0;
}

bool BatchnormBwdPlan::isSpatialMode() const
{
    const auto* scaleDims = _bwdParams.scale()->dims();
    const auto* xDims = _bwdParams.x()->dims();

    if(scaleDims->size() != xDims->size())
    {
        return true;
    }

    for(flatbuffers::uoffset_t i = 2; i < scaleDims->size(); ++i)
    {
        if(scaleDims->Get(i) != 1)
        {
            return false;
        }
    }
    return true;
}

void BatchnormBwdPlan::execute(const HipKernelHandle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               [[maybe_unused]] void* workspace) const
{
    if(isSpatialMode())
    {
        executeSpatial(handle, deviceBuffers, numDeviceBuffers);
    }
    else
    {
        executePerActivation(handle, deviceBuffers, numDeviceBuffers);
    }
}

// ============================================================================
// Helper: build common compilation options
// ============================================================================

namespace
{
struct ProblemDims
{
    int n, c, h, w;
    unsigned int hw, chw, nhw, nchw;
    bool isLayoutNHWC;
    bool useFp32, useFp16Mix, useBfp16Mix;
    std::string archName;
    bool isGfx103X, isGfx110X, isGfx120X, isGfx115X;
    int nrnOpId;
    float actAlpha, actBeta;
};

ProblemDims extractProblemDims(const BatchnormBwdParams& params, const hipDeviceProp_t& props)
{
    ProblemDims d{};

    auto xDataType = params.x()->data_type();
    auto scaleDataType = params.scale()->data_type();

    d.useFp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::HALF
                    && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    d.useBfp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::BFLOAT16
                     && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    d.useFp32 = !d.useFp16Mix && !d.useBfp16Mix;

    const auto* xDims = params.x()->dims();
    const auto* xStrides = params.x()->strides();

    if(xDims->size() == 4)
    {
        d.n = static_cast<int>(xDims->Get(0));
        d.c = static_cast<int>(xDims->Get(1));
        d.h = static_cast<int>(xDims->Get(2));
        d.w = static_cast<int>(xDims->Get(3));
    }
    else if(xDims->size() == 5)
    {
        d.n = static_cast<int>(xDims->Get(0));
        d.c = static_cast<int>(xDims->Get(1));
        int dd = static_cast<int>(xDims->Get(2));
        d.h = static_cast<int>(xDims->Get(3));
        d.w = static_cast<int>(xDims->Get(4));
        d.h = dd * d.h;
    }
    else
    {
        throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(xDims->size()));
    }

    d.hw = static_cast<unsigned int>(d.h * d.w);
    d.chw = static_cast<unsigned int>(d.c) * d.hw;
    d.nhw = static_cast<unsigned int>(d.n) * d.hw;
    d.nchw = static_cast<unsigned int>(d.n) * d.chw;

    d.isLayoutNHWC = (xStrides->Get(1) == 1);

    d.archName = std::string(props.gcnArchName);
    d.isGfx103X = (d.archName.find("gfx103") == 0);
    d.isGfx110X = (d.archName.find("gfx110") == 0);
    d.isGfx120X = (d.archName.find("gfx120") == 0);
    d.isGfx115X = (d.archName.find("gfx115") == 0);

    d.nrnOpId = 0;
    d.actAlpha = 0.0f;
    d.actBeta = 0.0f;
    if(params.optActivation().has_value())
    {
        const auto& act = *params.optActivation();
        d.nrnOpId = static_cast<int>(act.mode);
        d.actAlpha = static_cast<float>(act.alpha);
        d.actBeta = static_cast<float>(act.beta);
    }

    return d;
}

std::vector<std::string> buildCommonOptions(const ProblemDims& d)
{
    std::vector<std::string> opts;
    opts.emplace_back("-I/opt/rocm/include");
    opts.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (d.useFp32 ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_USE_FP16=") + (d.useFp16Mix ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_USE_BFP16=") + (d.useBfp16Mix ? "1" : "0"));
    opts.emplace_back("-DHIP_PLUGIN_USE_RNE_BFLOAT16=1");
    opts.emplace_back(std::string("-DHIP_PLUGIN_USE_FPMIX=") + (d.useFp16Mix ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_USE_BFPMIX=") + (d.useBfp16Mix ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_LAYOUT_NHWC=") + (d.isLayoutNHWC ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX103X=") + (d.isGfx103X ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX110X=") + (d.isGfx110X ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX120X=") + (d.isGfx120X ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX115X=") + (d.isGfx115X ? "1" : "0"));
    opts.emplace_back(std::string("-DHIP_PLUGIN_NRN_OP_ID=") + std::to_string(d.nrnOpId));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_N=") + std::to_string(d.n));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_C=") + std::to_string(d.c));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_HW=") + std::to_string(d.hw));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_NHW=") + std::to_string(d.nhw));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_CHW=") + std::to_string(d.chw));
    opts.emplace_back(std::string("-DHIP_PLUGIN_BN_NCHW=") + std::to_string(d.nchw));
    opts.emplace_back(std::string("--offload-arch=") + d.archName);
    return opts;
}

} // anonymous namespace

// ============================================================================
// executeSpatial: variant selection + dispatch
// ============================================================================

void BatchnormBwdPlan::executeSpatial(const HipKernelHandle& handle,
                                      const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                      uint32_t numDeviceBuffers) const
{
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    auto d = extractProblemDims(_bwdParams, props);

    // Determine variant
    bool useMult = useMultiple(static_cast<unsigned int>(d.n), d.hw, d.isLayoutNHWC);

    auto xBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto dyBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto dxBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dx()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dscaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dscale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dbiasBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dbias()->uid(), deviceBuffers, numDeviceBuffers);
    auto savedMeanBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.savedMean()->uid(), deviceBuffers, numDeviceBuffers);
    auto savedInvVarianceBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.savedInvVariance()->uid(), deviceBuffers, numDeviceBuffers);

    void* biasPtr = nullptr;
    if(_bwdParams.bias() != nullptr)
    {
        auto biasBuffer = hip_kernel_utils::findDeviceBuffer(
            _bwdParams.bias()->uid(), deviceBuffers, numDeviceBuffers);
        biasPtr = biasBuffer.ptr;
    }

    float inhw = 1.0f / static_cast<float>(d.nhw);

    if(!useMult)
    {
        // --- Spatial Single: variants 0, 1, 3 ---
        auto singleCfg = defaultConfigSpatialSingle(d.n, d.hw, d.nhw, d.isLayoutNHWC, d.useFp16Mix);

        size_t xlocalsize = 1024;
        size_t ldsSize = xlocalsize;

        auto options = buildCommonOptions(d);
        options.emplace_back("-DHIP_PLUGIN_BN_GRP0=" + std::to_string(xlocalsize));
        options.emplace_back("-DHIP_PLUGIN_BN_LDS_SIZE=" + std::to_string(ldsSize));
        options.emplace_back("-DHIP_PLUGIN_BN_VARIANT=" + std::to_string(singleCfg.variant));
        options.emplace_back("-DHIP_PLUGIN_BN_VEC_SIZE=" + std::to_string(singleCfg.vecSize));
        options.emplace_back("-DHIP_PLUGIN_BN_MAXN=65");

        auto hipProgram = HipProgram("BatchNormBwdSpatial.cpp", options);
        auto hipKernel = HipKernel(hipProgram, "BatchNormBwdSpatialSaved");

        hipKernel.setBlockSize(static_cast<unsigned int>(xlocalsize), 1, 1);
        hipKernel.setGridSize(static_cast<unsigned int>(d.c), 1, 1);

        hipKernel.launch(handle.getStream(),
                         xBuffer.ptr,
                         dyBuffer.ptr,
                         dxBuffer.ptr,
                         scaleBuffer.ptr,
                         biasPtr,
                         dscaleBuffer.ptr,
                         dbiasBuffer.ptr,
                         savedMeanBuffer.ptr,
                         savedInvVarianceBuffer.ptr,
                         inhw,
                         d.actAlpha,
                         d.actBeta);
    }
    else
    {
        // --- Spatial Multiple: variant 2 ---
        // Stash values for backward (saved path): 2 (dscale + dbias)
        constexpr unsigned int stashValuesBwd = 2;

        size_t vecSize = 4;
        size_t xlocalsize = 1;
        size_t ylocalsize = 1;
        size_t zlocalsize = 1;
        size_t nelements = static_cast<size_t>(d.n);

        size_t sc = static_cast<size_t>(d.c);
        size_t sn = static_cast<size_t>(d.n);
        size_t shw = static_cast<size_t>(d.hw);

        if(d.isLayoutNHWC)
        {
            getHeuristicsConfigTuningNHWC(sn, sc, shw, vecSize, xlocalsize);
            while(sc % vecSize != 0)
            {
                vecSize >>= 1;
            }
            if(vecSize == 1)
            {
                xlocalsize = std::min(static_cast<size_t>(1 << static_cast<int>(std::ceil(std::log2(
                                                              static_cast<double>(sc / vecSize))))),
                                      size_t{64});
            }
        }

        getSpatialMultipleConfig(
            sc, shw, d.isLayoutNHWC, d.useFp32, vecSize, xlocalsize, ylocalsize);

        if(!isSpatialMultipleApplicable(sn,
                                        sc,
                                        shw,
                                        d.isLayoutNHWC,
                                        d.useFp32,
                                        vecSize,
                                        stashValuesBwd,
                                        ylocalsize,
                                        zlocalsize,
                                        nelements))
        {
            // Fallback: try with vecSize=1
            vecSize = 1;
            getSpatialMultipleConfig(
                sc, shw, d.isLayoutNHWC, d.useFp32, vecSize, xlocalsize, ylocalsize);

            if(!isSpatialMultipleApplicable(sn,
                                            sc,
                                            shw,
                                            d.isLayoutNHWC,
                                            d.useFp32,
                                            vecSize,
                                            stashValuesBwd,
                                            ylocalsize,
                                            zlocalsize,
                                            nelements))
            {
                // Fallback to single-kernel variant 1
                auto options = buildCommonOptions(d);
                size_t singleXLocal = 1024;
                options.emplace_back("-DHIP_PLUGIN_BN_GRP0=" + std::to_string(singleXLocal));
                options.emplace_back("-DHIP_PLUGIN_BN_LDS_SIZE=" + std::to_string(singleXLocal));
                options.emplace_back("-DHIP_PLUGIN_BN_VARIANT=1");
                options.emplace_back("-DHIP_PLUGIN_BN_VEC_SIZE=1");
                options.emplace_back("-DHIP_PLUGIN_BN_MAXN=65");

                auto hipProgram = HipProgram("BatchNormBwdSpatial.cpp", options);
                auto hipKernel = HipKernel(hipProgram, "BatchNormBwdSpatialSaved");

                hipKernel.setBlockSize(static_cast<unsigned int>(singleXLocal), 1, 1);
                hipKernel.setGridSize(static_cast<unsigned int>(d.c), 1, 1);

                hipKernel.launch(handle.getStream(),
                                 xBuffer.ptr,
                                 dyBuffer.ptr,
                                 dxBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasPtr,
                                 dscaleBuffer.ptr,
                                 dbiasBuffer.ptr,
                                 savedMeanBuffer.ptr,
                                 savedInvVarianceBuffer.ptr,
                                 inhw,
                                 d.actAlpha,
                                 d.actBeta);
                return;
            }
        }

        int stashMethod = getStashMethod(d.isLayoutNHWC,
                                         d.useFp32,
                                         sc,
                                         sn,
                                         shw,
                                         stashValuesBwd,
                                         ylocalsize,
                                         zlocalsize,
                                         nelements);

        // Compute grid sizes
        size_t xgridsize, ygridsize, zgridsize;
        if(d.isLayoutNHWC)
        {
            xgridsize = xlocalsize * ((sc / vecSize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((shw + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((sc + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((shw / vecSize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((sn / nelements + zlocalsize - 1) / zlocalsize);

        // Final kernel local/grid sizes
        size_t xlocalsize_final = xlocalsize;
        size_t ylocalsize_final = ylocalsize;
        size_t zlocalsize_final = zlocalsize;

        if(d.isLayoutNHWC && d.c % 2 == 0 && xlocalsize % 2 == 0)
        {
            xlocalsize_final = 2;
            zlocalsize_final = (zgridsize / zlocalsize) * zlocalsize;
            ylocalsize_final
                = (xlocalsize * ylocalsize * zlocalsize) / xlocalsize_final / zlocalsize_final;
            if(ylocalsize_final == 0)
                ylocalsize_final = 1;
        }

        size_t ldsSize = xlocalsize * ylocalsize * zlocalsize;

        auto options = buildCommonOptions(d);
        options.emplace_back("-DHIP_PLUGIN_BN_GRP0=" + std::to_string(xlocalsize));
        options.emplace_back("-DHIP_PLUGIN_BN_GRP1=" + std::to_string(ylocalsize));
        options.emplace_back("-DHIP_PLUGIN_BN_GRP2=" + std::to_string(zlocalsize));
        options.emplace_back("-DHIP_PLUGIN_BN_GRP0_FINAL=" + std::to_string(xlocalsize_final));
        options.emplace_back("-DHIP_PLUGIN_BN_GRP1_FINAL=" + std::to_string(ylocalsize_final));
        options.emplace_back("-DHIP_PLUGIN_BN_GRP2_FINAL=" + std::to_string(zlocalsize_final));
        options.emplace_back("-DHIP_PLUGIN_BN_NGRPS=" + std::to_string(ygridsize / ylocalsize));
        options.emplace_back("-DHIP_PLUGIN_BN_NGRPS2=" + std::to_string(zgridsize / zlocalsize));
        options.emplace_back("-DHIP_PLUGIN_BN_N_ELEMENTS=" + std::to_string(nelements));
        options.emplace_back("-DHIP_PLUGIN_BN_LDS_SIZE=" + std::to_string(ldsSize));
        options.emplace_back("-DHIP_PLUGIN_BN_VEC_SIZE=" + std::to_string(vecSize));
        options.emplace_back("-DHIP_PLUGIN_BN_STASH_METHOD=" + std::to_string(stashMethod));
        options.emplace_back("-DHIP_PLUGIN_BN_VARIANT=2");

        auto hipProgram = HipProgram("BatchNormBwdSpatialMultiple.cpp", options);

        // Kernel 1: DScaleDBias
        auto kernelDScaleDBias = HipKernel(hipProgram, "BatchNormBwdSpatialDScaleDBias");
        kernelDScaleDBias.setBlockSize(static_cast<unsigned int>(xlocalsize),
                                       static_cast<unsigned int>(ylocalsize),
                                       static_cast<unsigned int>(zlocalsize));
        kernelDScaleDBias.setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                      static_cast<unsigned int>(ygridsize / ylocalsize),
                                      static_cast<unsigned int>(zgridsize / zlocalsize));

        kernelDScaleDBias.launch(handle.getStream(),
                                 xBuffer.ptr,
                                 dyBuffer.ptr,
                                 dxBuffer.ptr, // used as stash workspace
                                 scaleBuffer.ptr,
                                 biasPtr,
                                 savedMeanBuffer.ptr,
                                 savedInvVarianceBuffer.ptr,
                                 d.actAlpha,
                                 d.actBeta);

        // Kernel 2: FinalDScaleDBias
        auto kernelFinal = HipKernel(hipProgram, "BatchNormBwdSpatialFinalDScaleDBias");
        kernelFinal.setBlockSize(static_cast<unsigned int>(xlocalsize_final),
                                 static_cast<unsigned int>(ylocalsize_final),
                                 static_cast<unsigned int>(zlocalsize_final));
        kernelFinal.setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize_final), 1, 1);

        kernelFinal.launch(handle.getStream(),
                           dxBuffer.ptr, // reads stash from dx
                           dscaleBuffer.ptr,
                           dbiasBuffer.ptr);

        // Kernel 3: DX
        auto kernelDX = HipKernel(hipProgram, "BatchNormBwdSpatialDX");
        kernelDX.setBlockSize(static_cast<unsigned int>(xlocalsize),
                              static_cast<unsigned int>(ylocalsize),
                              static_cast<unsigned int>(zlocalsize));
        kernelDX.setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                             static_cast<unsigned int>(ygridsize / ylocalsize),
                             static_cast<unsigned int>(zgridsize / zlocalsize));

        kernelDX.launch(handle.getStream(),
                        xBuffer.ptr,
                        dyBuffer.ptr,
                        dxBuffer.ptr,
                        scaleBuffer.ptr,
                        biasPtr,
                        dscaleBuffer.ptr,
                        dbiasBuffer.ptr,
                        savedMeanBuffer.ptr,
                        savedInvVarianceBuffer.ptr,
                        inhw,
                        d.actAlpha,
                        d.actBeta);
    }
}

void BatchnormBwdPlan::executePerActivation(const HipKernelHandle& handle,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers) const
{
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    auto xDataType = _bwdParams.x()->data_type();
    auto scaleDataType = _bwdParams.scale()->data_type();

    bool useFp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::HALF
                       && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    bool useBfp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::BFLOAT16
                        && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    bool useFp32 = !useFp16Mix && !useBfp16Mix;

    const auto* xDims = _bwdParams.x()->dims();
    const auto* xStrides = _bwdParams.x()->strides();

    int n, c, h, w;
    int nStride;

    if(xDims->size() == 4)
    {
        n = static_cast<int>(xDims->Get(0));
        c = static_cast<int>(xDims->Get(1));
        h = static_cast<int>(xDims->Get(2));
        w = static_cast<int>(xDims->Get(3));

        nStride = static_cast<int>(xStrides->Get(0));
    }
    else if(xDims->size() == 5)
    {
        n = static_cast<int>(xDims->Get(0));
        c = static_cast<int>(xDims->Get(1));
        int d = static_cast<int>(xDims->Get(2));
        h = static_cast<int>(xDims->Get(3));
        w = static_cast<int>(xDims->Get(4));
        h = d * h;

        nStride = static_cast<int>(xStrides->Get(0));
    }
    else
    {
        throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(xDims->size()));
    }

    unsigned int in_cstride = static_cast<unsigned int>(h * w);

    size_t xlocalsize = 1;
    size_t ylocalsize = (64 >= in_cstride) ? 64 : 256;
    size_t zlocalsize = 1;

    size_t xgridsize = static_cast<size_t>(c);
    size_t ygridsize = ((in_cstride + ylocalsize - 1) / ylocalsize) * ylocalsize;

    std::string archName(props.gcnArchName);
    bool isGfx103X = (archName.find("gfx103") == 0);
    bool isGfx110X = (archName.find("gfx110") == 0);
    bool isGfx120X = (archName.find("gfx120") == 0);
    bool isGfx115X = (archName.find("gfx115") == 0);

    std::vector<std::string> options;
    options.emplace_back("-I/opt/rocm/include");
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (useFp32 ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP16=") + (useFp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFP16=") + (useBfp16Mix ? "1" : "0"));
    options.emplace_back("-DHIP_PLUGIN_USE_RNE_BFLOAT16=1");
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FPMIX=") + (useFp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFPMIX=") + (useBfp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP0=") + std::to_string(xlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP1=") + std::to_string(ylocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP2=") + std::to_string(zlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_C=") + std::to_string(c));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_HW=") + std::to_string(in_cstride));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX103X=") + (isGfx103X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX110X=") + (isGfx110X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX120X=") + (isGfx120X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX115X=") + (isGfx115X ? "1" : "0"));
    options.emplace_back(std::string("--offload-arch=") + props.gcnArchName);

    auto hipProgram = HipProgram("BatchNormBwdPerAct.cpp", options);
    auto hipKernel = HipKernel(hipProgram, "BatchNormBwdPerActivationSaved");

    hipKernel.setBlockSize(static_cast<unsigned int>(xlocalsize),
                           static_cast<unsigned int>(ylocalsize),
                           static_cast<unsigned int>(zlocalsize));
    hipKernel.setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                          static_cast<unsigned int>(ygridsize / ylocalsize),
                          1);

    auto xBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto dyBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto dxBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dx()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dscaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dscale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dbiasBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.dbias()->uid(), deviceBuffers, numDeviceBuffers);
    auto savedMeanBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.savedMean()->uid(), deviceBuffers, numDeviceBuffers);
    auto savedInvVarianceBuffer = hip_kernel_utils::findDeviceBuffer(
        _bwdParams.savedInvVariance()->uid(), deviceBuffers, numDeviceBuffers);

    hipKernel.launch(handle.getStream(),
                     xBuffer.ptr,
                     dyBuffer.ptr,
                     static_cast<unsigned int>(n),
                     static_cast<unsigned int>(nStride),
                     in_cstride,
                     dxBuffer.ptr,
                     scaleBuffer.ptr,
                     dscaleBuffer.ptr,
                     dbiasBuffer.ptr,
                     savedMeanBuffer.ptr,
                     savedInvVarianceBuffer.ptr);
}

}
