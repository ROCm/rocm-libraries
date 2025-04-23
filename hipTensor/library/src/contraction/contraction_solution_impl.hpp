/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP

#include <algorithm>
#include <numeric>

#include "contraction_solution.hpp"
#include "hash.hpp"

namespace std
{
    template <>
    struct hash<hiptensor::ContractionSolution>
    {
        size_t operator()(hiptensor::ContractionSolution const& s) const noexcept
        {
            return hash<hiptensor::ContractionSolutionParams>{}(*s.params());
        }
    };
}

namespace hiptensor
{
    std::array<std::vector<std::size_t>, 8>
        normalizeTensorModes(std::vector<std::size_t> const& a_ms_ks_lengths,
                             std::vector<std::size_t> const& a_ms_ks_strides,
                             std::vector<int32_t> const&     a_ms_ks_modes,
                             std::vector<std::size_t> const& b_ns_ks_lengths,
                             std::vector<std::size_t> const& b_ns_ks_strides,
                             std::vector<int32_t> const&     b_ns_ks_modes,
                             std::vector<std::size_t> const& e_ms_ns_lengths,
                             std::vector<std::size_t> const& e_ms_ns_strides,
                             std::vector<int32_t> const&     e_ms_ns_modes);

    template <typename DeviceOp, typename Enabler = void>
    class ContractionSolutionImpl;

    template <typename DeviceOp>
    class ContractionSolutionImpl<
        DeviceOp,
        std::enable_if_t<(std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                         ck::tensor_operation::element_wise::Bilinear>)
                         || (std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                            ck::tensor_operation::element_wise::BilinearComplex>)>>
        : public ContractionSolution
    {
    public:
        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*              alpha,
                      void const*              A,
                      void const*              B,
                      void const*              beta,
                      void const*              D,
                      void*                    E,
                      std::vector<std::size_t> a_ms_ks_lengths,
                      std::vector<std::size_t> a_ms_ks_strides,
                      std::vector<int32_t>     a_ms_ks_modes,
                      std::vector<std::size_t> b_ns_ks_lengths,
                      std::vector<std::size_t> b_ns_ks_strides,
                      std::vector<int32_t>     b_ns_ks_modes,
                      std::vector<std::size_t> ds_ms_ns_lengths,
                      std::vector<std::size_t> ds_ms_ns_strides,
                      std::vector<int32_t>     ds_ms_ns_modes,
                      std::vector<std::size_t> e_ms_ns_lengths,
                      std::vector<std::size_t> e_ms_ns_strides,
                      std::vector<int32_t>     e_ms_ns_modes,
                      void*                    workspacePtr) override
        {
            using Base   = ContractionSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear out the previous arguments
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            ScalarData alphaF;
            ScalarData betaF;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<ScalarData>(
                    alpha, convertToComputeType(HipDataType_v<typename Traits::ComputeDataT>));
            }
            if(beta != nullptr)
            {
                betaF = hiptensor::readVal<ScalarData>(
                    beta, convertToComputeType(HipDataType_v<typename Traits::ComputeDataT>));
            }

            auto [normal_a_ms_ks_lengths,
                  normal_a_ms_ks_strides,
                  normal_b_ns_ks_lengths,
                  normal_b_ns_ks_strides,
                  normal_ds_ms_ns_lengths,
                  normal_ds_ms_ns_strides,
                  normal_e_ms_ns_lengths,
                  normal_e_ms_ns_strides]
                = normalizeTensorModes(a_ms_ks_lengths,
                                       a_ms_ks_strides,
                                       a_ms_ks_modes,
                                       b_ns_ks_lengths,
                                       b_ns_ks_strides,
                                       b_ns_ks_modes,
                                       e_ms_ns_lengths,
                                       e_ms_ns_strides,
                                       e_ms_ns_modes);

            // CK has its own format for indices...
            auto toCKVec = [](std::vector<size_t> const& v) {
                return std::vector<ck::index_t>(v.begin(), v.end());
            };

            // Initialize the argument pointer
            Base::mInvokerArgPtr = std::move(deviceOp->MakeArgumentPointer(
                A,
                B,
                std::array<const void*, 1>{D},
                E,
                toCKVec(normal_a_ms_ks_lengths),
                toCKVec(normal_a_ms_ks_strides),
                toCKVec(normal_b_ns_ks_lengths),
                toCKVec(normal_b_ns_ks_strides),
                std::array<std::vector<ck::index_t>, 1>{toCKVec(normal_ds_ms_ns_lengths)},
                std::array<std::vector<ck::index_t>, 1>{toCKVec(normal_ds_ms_ns_strides)},
                toCKVec(normal_e_ms_ns_lengths),
                toCKVec(normal_e_ms_ns_strides),
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp(alphaF, betaF)));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(Base::mInvokerArgPtr.get(), workspacePtr);

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(normal_a_ms_ks_lengths.begin(),
                                       normal_a_ms_ks_lengths.begin() + MaxNumDimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(normal_b_ns_ks_lengths.begin(),
                                       normal_b_ns_ks_lengths.begin() + MaxNumDimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(normal_a_ms_ks_lengths.begin() + MaxNumDimsM,
                                       normal_a_ms_ks_lengths.end(),
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::DDataT) * Base::mM * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mInvokerArgPtr.get());

            if(!Base::mValid)
            {
                resetArgs();
            }

            return Base::mValid;
        }
    };

    template <typename DeviceOp>
    class ContractionSolutionImpl<
        DeviceOp,
        std::enable_if_t<(std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                         ck::tensor_operation::element_wise::Scale>)
                         || (std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                            ck::tensor_operation::element_wise::ScaleComplex>)>>
        : public ContractionSolution
    {
    public:
        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*              alpha,
                      void const*              A,
                      void const*              B,
                      void const*              beta,
                      void const*              D,
                      void*                    E,
                      std::vector<std::size_t> a_ms_ks_lengths,
                      std::vector<std::size_t> a_ms_ks_strides,
                      std::vector<int32_t>     a_ms_ks_modes,
                      std::vector<std::size_t> b_ns_ks_lengths,
                      std::vector<std::size_t> b_ns_ks_strides,
                      std::vector<int32_t>     b_ns_ks_modes,
                      std::vector<std::size_t> ds_ms_ns_lengths,
                      std::vector<std::size_t> ds_ms_ns_strides,
                      std::vector<int32_t>     ds_ms_ns_modes,
                      std::vector<std::size_t> e_ms_ns_lengths,
                      std::vector<std::size_t> e_ms_ns_strides,
                      std::vector<int32_t>     e_ms_ns_modes,
                      void*                    workspacePtr) override
        {
            using Base   = ContractionSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear previous data
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            ScalarData alphaF;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<ScalarData>(
                    alpha, convertToComputeType(HipDataType_v<typename Traits::ComputeDataT>));
            }

            auto [normal_a_ms_ks_lengths,
                  normal_a_ms_ks_strides,
                  normal_b_ns_ks_lengths,
                  normal_b_ns_ks_strides,
                  _1,
                  _2,
                  normal_e_ms_ns_lengths,
                  normal_e_ms_ns_strides]
                = normalizeTensorModes(a_ms_ks_lengths,
                                       a_ms_ks_strides,
                                       a_ms_ks_modes,
                                       b_ns_ks_lengths,
                                       b_ns_ks_strides,
                                       b_ns_ks_modes,
                                       e_ms_ns_lengths,
                                       e_ms_ns_strides,
                                       e_ms_ns_modes);

            // CK has its own format for indices...
            auto toCKVec = [](std::vector<size_t> const& v) {
                return std::vector<ck::index_t>(v.begin(), v.end());
            };

            // Initialize the argument pointer
            Base::mInvokerArgPtr
                = std::move(deviceOp->MakeArgumentPointer(A,
                                                          B,
                                                          std::array<const void*, 0>{},
                                                          E,
                                                          toCKVec(normal_a_ms_ks_lengths),
                                                          toCKVec(normal_a_ms_ks_strides),
                                                          toCKVec(normal_b_ns_ks_lengths),
                                                          toCKVec(normal_b_ns_ks_strides),
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          toCKVec(normal_e_ms_ns_lengths),
                                                          toCKVec(normal_e_ms_ns_strides),
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp(alphaF)));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(Base::mInvokerArgPtr.get(), workspacePtr);

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(normal_a_ms_ks_lengths.begin(),
                                       normal_a_ms_ks_lengths.begin() + MaxNumDimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(normal_b_ns_ks_lengths.begin(),
                                       normal_b_ns_ks_lengths.begin() + MaxNumDimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(normal_a_ms_ks_lengths.begin() + MaxNumDimsM,
                                       normal_a_ms_ks_lengths.end(),
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mInvokerArgPtr.get());

            if(!Base::mValid)
            {
                resetArgs();
            }

            return Base::mValid;
        }
    };

    template <ck::index_t NumDimM,
              ck::index_t NumDimN,
              ck::index_t NumDimK,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation,
              typename ComputeDataType = ADataType>
    std::vector<std::unique_ptr<hiptensor::ContractionSolution>> enumerateContractionSolutions()
    {
        using ContractionOp
            = ck::tensor_operation::device::DeviceContractionMultipleD<NumDimM,
                                                                       NumDimN,
                                                                       NumDimK,
                                                                       ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation,
                                                                       ComputeDataType>;

        using Factory
            = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionOp>;

        std::vector<std::unique_ptr<ContractionSolution>> result;
        for(auto& opPtr : Factory::GetInstances())
        {
            result.push_back(
                std::make_unique<ContractionSolutionImpl<ContractionOp>>(std::move(opPtr)));
        }
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
