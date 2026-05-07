// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_gpu_ref/ShallowGpuTensor.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_gpu_ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_test_sdk/utilities/ConvolutionValidation.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

template <typename XDataType, typename WDataType, typename YDataType, typename ComputeDataType>
inline std::vector<std::string> buildConvDefines(bool useTf32 = false)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DX_TYPE=") + HipRtcTypeName<XDataType>::VALUE);
    defines.emplace_back(std::string("-DW_TYPE=") + HipRtcTypeName<WDataType>::VALUE);
    defines.emplace_back(std::string("-DY_TYPE=") + HipRtcTypeName<YDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    if(useTf32)
    {
        defines.emplace_back("-DUSE_TF32");
    }
    return defines;
}

} // namespace detail

class GpuFpReferenceConvolution
{
public:
    // --- Forward convolution (fprop) ---

    // TensorBase overload — symmetric padding convenience wrapper.
    template <class XDataType,
              class WDataType = XDataType,
              class YDataType = XDataType,
              class ComputeDataType = double>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding,
                      double alpha = 1.0,
                      double beta = 0.0,
                      bool useTf32 = false)
    {
        fprop<XDataType, WDataType, YDataType, ComputeDataType>(
            x, w, y, convStrides, dilations, padding, padding, alpha, beta, useTf32);
    }

    // TensorBase overload — asymmetric padding.
    // Takes non-const references because deviceData() may trigger host→device sync.
    template <class XDataType,
              class WDataType = XDataType,
              class YDataType = XDataType,
              class ComputeDataType = double>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding,
                      double alpha = 1.0,
                      double beta = 0.0,
                      bool useTf32 = false)
    {
        fprop<XDataType, WDataType, YDataType, ComputeDataType>(x.memory().deviceData(),
                                                                w.memory().deviceData(),
                                                                y.memory().deviceData(),
                                                                x.dims(),
                                                                w.dims(),
                                                                y.dims(),
                                                                x.strides(),
                                                                w.strides(),
                                                                y.strides(),
                                                                convStrides,
                                                                dilations,
                                                                prePadding,
                                                                postPadding,
                                                                alpha,
                                                                beta,
                                                                useTf32);
        y.memory().markDeviceModified();
    }

    // --- Backward data gradient (dgrad) ---
    // gradX and w are non-const because MigratableMemory::deviceData() triggers
    // lazy host-to-device synchronization, which mutates internal state.

    // Overload for uniform padding
    template <class DxDataType,
              class WDataType = DxDataType,
              class DyDataType = DxDataType,
              class ComputeDataType = double>
    static void dgrad(hipdnn_data_sdk::utilities::TensorBase<DxDataType>& gradX,
                      hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding,
                      double alpha = 1.0,
                      double beta = 0.0)
    {
        dgrad<DxDataType, WDataType, DyDataType, ComputeDataType>(
            gradX, w, gradY, convStrides, dilations, padding, padding, alpha, beta);
    }

    template <class DxDataType,
              class WDataType = DxDataType,
              class DyDataType = DxDataType,
              class ComputeDataType = double>
    static void dgrad(hipdnn_data_sdk::utilities::TensorBase<DxDataType>& gradX,
                      hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding,
                      double alpha = 1.0,
                      double beta = 0.0)
    {
        validateInput(
            gradX.dims(), w.dims(), gradY.dims(), convStrides, dilations, prePadding, postPadding);

        const auto nDims = gradX.dims().size();
        auto defines
            = detail::buildConvDefines<DxDataType, WDataType, DyDataType, ComputeDataType>();

        auto* dxPtr = gradX.memory().deviceData();
        auto* wPtr = w.memory().deviceData();
        auto* dyPtr = gradY.memory().deviceData();

        // Only prePadding is passed to the kernel. Post-padding is implicitly
        // handled by the output tensor dimensions and the kernel's bounds checks.
        if(nDims == 3)
        {
            launchDgrad1d(dxPtr,
                          wPtr,
                          dyPtr,
                          gradX.dims(),
                          w.dims(),
                          gradY.dims(),
                          gradX.strides(),
                          w.strides(),
                          gradY.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 4)
        {
            launchDgrad2d(dxPtr,
                          wPtr,
                          dyPtr,
                          gradX.dims(),
                          w.dims(),
                          gradY.dims(),
                          gradX.strides(),
                          w.strides(),
                          gradY.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 5)
        {
            launchDgrad3d(dxPtr,
                          wPtr,
                          dyPtr,
                          gradX.dims(),
                          w.dims(),
                          gradY.dims(),
                          gradX.strides(),
                          w.strides(),
                          gradY.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else
        {
            throw std::invalid_argument(
                "GPU kernels support 1D/2D/3D convolutions (tensor rank 3/4/5), got rank "
                + std::to_string(nDims));
        }

        gradX.memory().markDeviceModified();
    }

    // --- Backward weight gradient (wgrad) ---
    // x and gradY are non-const because MigratableMemory::deviceData() triggers
    // lazy host-to-device synchronization, which mutates internal state.

    // Overload for uniform padding
    template <class XDataType,
              class DwDataType = XDataType,
              class DyDataType = XDataType,
              class ComputeDataType = double>
    static void wgrad(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<DwDataType>& gradW,
                      hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding,
                      double alpha = 1.0,
                      double beta = 0.0)
    {
        wgrad<XDataType, DwDataType, DyDataType, ComputeDataType>(
            x, gradW, gradY, convStrides, dilations, padding, padding, alpha, beta);
    }

    template <class XDataType,
              class DwDataType = XDataType,
              class DyDataType = XDataType,
              class ComputeDataType = double>
    static void wgrad(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<DwDataType>& gradW,
                      hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding,
                      double alpha = 1.0,
                      double beta = 0.0)
    {
        validateInput(
            x.dims(), gradW.dims(), gradY.dims(), convStrides, dilations, prePadding, postPadding);

        const auto nDims = x.dims().size();
        auto defines
            = detail::buildConvDefines<XDataType, DwDataType, DyDataType, ComputeDataType>();

        auto* xPtr = x.memory().deviceData();
        auto* dwPtr = gradW.memory().deviceData();
        auto* dyPtr = gradY.memory().deviceData();

        // Only prePadding is passed to the kernel. Post-padding is implicitly
        // handled by the output tensor dimensions and the kernel's bounds checks.
        if(nDims == 3)
        {
            launchWgrad1d(xPtr,
                          dwPtr,
                          dyPtr,
                          x.dims(),
                          gradW.dims(),
                          gradY.dims(),
                          x.strides(),
                          gradW.strides(),
                          gradY.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 4)
        {
            launchWgrad2d(xPtr,
                          dwPtr,
                          dyPtr,
                          x.dims(),
                          gradW.dims(),
                          gradY.dims(),
                          x.strides(),
                          gradW.strides(),
                          gradY.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 5)
        {
            launchWgrad3d(xPtr,
                          dwPtr,
                          dyPtr,
                          x.dims(),
                          gradW.dims(),
                          gradY.dims(),
                          x.strides(),
                          gradW.strides(),
                          gradY.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else
        {
            throw std::invalid_argument(
                "GPU kernels support 1D/2D/3D convolutions (tensor rank 3/4/5), got rank "
                + std::to_string(nDims));
        }

        gradW.memory().markDeviceModified();
    }

private:
    // --- Raw device-pointer fprop (implementation detail) ---

    template <class XDataType,
              class WDataType = XDataType,
              class YDataType = XDataType,
              class ComputeDataType = double>
    static void fprop(const void* xPtr,
                      const void* wPtr,
                      void* yPtr,
                      const std::vector<int64_t>& xDims,
                      const std::vector<int64_t>& wDims,
                      const std::vector<int64_t>& yDims,
                      const std::vector<int64_t>& xStrides,
                      const std::vector<int64_t>& wStrides,
                      const std::vector<int64_t>& yStrides,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding,
                      double alpha = 1.0,
                      double beta = 0.0,
                      bool useTf32 = false)
    {
        validateInput(xDims, wDims, yDims, convStrides, dilations, prePadding, postPadding);

        const auto nDims = xDims.size();
        auto defines
            = detail::buildConvDefines<XDataType, WDataType, YDataType, ComputeDataType>(useTf32);

        // Only prePadding is passed to the kernel. Post-padding is implicitly
        // handled by the output tensor dimensions and the kernel's bounds checks.
        if(nDims == 3)
        {
            launchFprop1d(xPtr,
                          wPtr,
                          yPtr,
                          xDims,
                          wDims,
                          yDims,
                          xStrides,
                          wStrides,
                          yStrides,
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 4)
        {
            launchFprop2d(xPtr,
                          wPtr,
                          yPtr,
                          xDims,
                          wDims,
                          yDims,
                          xStrides,
                          wStrides,
                          yStrides,
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 5)
        {
            launchFprop3d(xPtr,
                          wPtr,
                          yPtr,
                          xDims,
                          wDims,
                          yDims,
                          xStrides,
                          wStrides,
                          yStrides,
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else
        {
            throw std::invalid_argument(
                "GPU kernels support 1D/2D/3D convolutions (tensor rank 3/4/5), got rank "
                + std::to_string(nDims));
        }
    }

    // --- Validation ---

    static void validateInput(const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& yDims,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& prePadding,
                              const std::vector<int64_t>& postPadding)
    {
        const auto nDims = xDims.size();

        // GPU kernels only support 1D (3), 2D (4), and 3D (5) convolutions
        if(nDims != 3 && nDims != 4 && nDims != 5)
        {
            throw std::invalid_argument("Input tensor must have 3 dimensions (1D conv), "
                                        "4 dimensions (2D conv), or 5 dimensions (3D conv)");
        }

        hipdnn_test_sdk::utilities::validateConvolutionParams(
            xDims, wDims, yDims, strides, dilations, prePadding, postPadding);
    }

    // --- Kernel launchers (defined in GpuFpReferenceConvolution.cpp) ---

    static void launchFprop1d(const void* xPtr,
                              const void* wPtr,
                              void* yPtr,
                              const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& yDims,
                              const std::vector<int64_t>& xTensorStrides,
                              const std::vector<int64_t>& wTensorStrides,
                              const std::vector<int64_t>& yTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    static void launchFprop2d(const void* xPtr,
                              const void* wPtr,
                              void* yPtr,
                              const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& yDims,
                              const std::vector<int64_t>& xTensorStrides,
                              const std::vector<int64_t>& wTensorStrides,
                              const std::vector<int64_t>& yTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    static void launchFprop3d(const void* xPtr,
                              const void* wPtr,
                              void* yPtr,
                              const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& yDims,
                              const std::vector<int64_t>& xTensorStrides,
                              const std::vector<int64_t>& wTensorStrides,
                              const std::vector<int64_t>& yTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    // --- Dgrad kernel launchers (defined in GpuFpReferenceConvolution.cpp) ---

    static void launchDgrad1d(void* dxPtr,
                              const void* wPtr,
                              const void* dyPtr,
                              const std::vector<int64_t>& dxDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& dyDims,
                              const std::vector<int64_t>& dxTensorStrides,
                              const std::vector<int64_t>& wTensorStrides,
                              const std::vector<int64_t>& dyTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    static void launchDgrad2d(void* dxPtr,
                              const void* wPtr,
                              const void* dyPtr,
                              const std::vector<int64_t>& dxDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& dyDims,
                              const std::vector<int64_t>& dxTensorStrides,
                              const std::vector<int64_t>& wTensorStrides,
                              const std::vector<int64_t>& dyTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    static void launchDgrad3d(void* dxPtr,
                              const void* wPtr,
                              const void* dyPtr,
                              const std::vector<int64_t>& dxDims,
                              const std::vector<int64_t>& wDims,
                              const std::vector<int64_t>& dyDims,
                              const std::vector<int64_t>& dxTensorStrides,
                              const std::vector<int64_t>& wTensorStrides,
                              const std::vector<int64_t>& dyTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    // --- Wgrad kernel launchers (defined in GpuFpReferenceConvolution.cpp) ---

    static void launchWgrad1d(const void* xPtr,
                              void* dwPtr,
                              const void* dyPtr,
                              const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& dwDims,
                              const std::vector<int64_t>& dyDims,
                              const std::vector<int64_t>& xTensorStrides,
                              const std::vector<int64_t>& dwTensorStrides,
                              const std::vector<int64_t>& dyTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    static void launchWgrad2d(const void* xPtr,
                              void* dwPtr,
                              const void* dyPtr,
                              const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& dwDims,
                              const std::vector<int64_t>& dyDims,
                              const std::vector<int64_t>& xTensorStrides,
                              const std::vector<int64_t>& dwTensorStrides,
                              const std::vector<int64_t>& dyTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);

    static void launchWgrad3d(const void* xPtr,
                              void* dwPtr,
                              const void* dyPtr,
                              const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& dwDims,
                              const std::vector<int64_t>& dyDims,
                              const std::vector<int64_t>& xTensorStrides,
                              const std::vector<int64_t>& dwTensorStrides,
                              const std::vector<int64_t>& dyTensorStrides,
                              const std::vector<int64_t>& convStrides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding,
                              const std::vector<std::string>& defines,
                              double alpha,
                              double beta);
};

} // namespace hipdnn_gpu_ref
