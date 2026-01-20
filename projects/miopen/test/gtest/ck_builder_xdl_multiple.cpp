// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/example_conversion.hpp>
#include <miopen/ck_builder/builder_conv_xdl.hpp>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

using InLayout                             = ck::tensor_layout::convolution::NGCHW;
using WeiLayout                            = ck::tensor_layout::convolution::GKCYX;
using OutLayout                            = ck::tensor_layout::convolution::NGKHW;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple                           = ck::Tuple<>;
static constexpr ck::index_t NumDimSpatial = 2;
template <typename DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;
template <typename DataType>
using DeviceOpGFwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdDefault<DataType>>;

template <typename T>
void test_instance(const std::unique_ptr<T>& builderKernelInstance)
{
    auto builderKernelInstanceString = builderKernelInstance->GetInstanceString();

    ASSERT_TRUE(builderKernelInstanceString.find(
                    "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3") == std::string::npos)
        << " builder returned wrong kind of instance";

    // These are the instances that MIOpen currently gets from CK's static library
    auto factoryInstances = DeviceOpGFwdDefaultPtrs<float>::GetInstances();

    ASSERT_GT(factoryInstances.size(), 0) << "Factory returned no instances";

    auto result =
        std::find_if(factoryInstances.begin(),
                     factoryInstances.end(),
                     [&builderKernelInstanceString](const auto& kernelPtr) {
                         return kernelPtr->GetInstanceString() == builderKernelInstanceString;
                     });

    ASSERT_TRUE(result != factoryInstances.end())
        << "Instance string " << builderKernelInstanceString
        << " not found in list of instances returned by factory.";

    print_closest_instance(builderKernelInstanceString, factoryInstances);
}

template <typename T, std::size_t N, typename F>
constexpr auto map_array(const std::array<T, N>& input, F&& func)
{
    using U = std::invoke_result_t<F, T>;
    std::array<U, N> result{};
    for(auto i = 0; i < N; i++)
    {
        result[i] = func(input[i]);
    }

    return result;
}

template <typename T1, std::size_t N1, typename T2, std::size_t N2, typename F>
constexpr auto
multiplex_array(const std::array<T1, N1>& input1, const std::array<T2, N2>& input2, F&& func)
{
    using U = std::invoke_result_t<F, T1, T2>;
    std::array<U, N1 * N2> result{};
    for(auto i1 = 0; i1 < N1; i1++)
    {
        auto arg1 = input1[i1];
        for(auto i2 = 0; i2 < N2; i2++)
        {
            auto arg2            = input2[i2];
            auto retval          = func(arg1, arg2);
            result[i1 * N2 + i2] = retval;
        }
    }

    return result;
}

template <auto KernelDescriptor>
constexpr void InstantiateKernel(std::vector<BaseOperatorPtr>& kernels)
{
    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<KernelDescriptor.signature, KernelDescriptor.algorithm>;
    do_builder_checks<Builder>();

    kernels.push_back(std::make_unique<typename Builder::Instance>());
}
template <typename T, T... values>
constexpr void build_kernels_helper(std::vector<BaseOperatorPtr>& kernels)
{
    std::array<BaseOperatorPtr, sizeof...(values)> result{};
    ((InstantiateKernel<values>(kernels)), ...);
}

template <typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
constexpr void build_kernels_impl(std::vector<BaseOperatorPtr>& kernels, std::index_sequence<I...>)
{
    build_kernels_helper<T, arr[I]...>(kernels);
}

template <typename T, std::size_t N, std::array<T, N> arr>
constexpr void build_kernels(std::vector<BaseOperatorPtr>& kernels)
{
    build_kernels_impl<T, N, arr>(kernels, std::make_index_sequence<N>{});
}

TEST(CKBuilderXdl, Multiple_Static_Instances)
{
    std::vector<BaseOperatorPtr> kernels{};
    constexpr auto instances = miopen::ck_builder::example_instances();

    build_kernels<XdlInstance, instances.size(), instances>(kernels);

    std::cout << "Instance count: " << kernels.size() << std::endl;

    for(auto&& k : kernels)
    {
        test_instance(k);
    }
}

template <typename DataType>
using DeviceOpGFwdBuilderPtrs = miopen::conv::ck_builder::instance::DeviceOperationInstanceFactory<
    DeviceOpGFwdDefault<DataType>>;

template <typename DataType>
void CompareInstanceLists()
{
    auto ckFactoryInstances      = DeviceOpGFwdDefaultPtrs<DataType>::GetInstances();
    auto builderFactoryInstances = DeviceOpGFwdBuilderPtrs<DataType>::GetInstances();

    ASSERT_EQ(ckFactoryInstances.size(), builderFactoryInstances.size());
}

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsFloat) { CompareInstanceLists<float>(); }

/*
TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsHalf) { CompareInstanceLists<ck::half_t>(); }

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsBHalf) { CompareInstanceLists<ck::bhalf_t>(); }

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsInt8) { CompareInstanceLists<int8_t>(); }
*/
