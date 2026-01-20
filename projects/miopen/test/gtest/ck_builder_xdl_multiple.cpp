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
