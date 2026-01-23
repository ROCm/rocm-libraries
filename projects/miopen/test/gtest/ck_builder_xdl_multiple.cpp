// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_builder_shared.hpp"

#include <miopen/ck_builder/factories/grouped_conv_2d_fwd_multiple_abd.hpp>

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

void print_instance_strings(std::vector<std::string>& instance_strings)
{
    for(auto&& s : instance_strings)
    {
        std::cout << "\t" << s << std::endl;
    }
}

template <typename DeviceOpA, typename DeviceOpB>
void compare_instance_vectors(std::vector<std::unique_ptr<DeviceOpA>>& instancesA,
                              std::vector<std::unique_ptr<DeviceOpB>>& instancesB)
{
    EXPECT_EQ(instancesA.size(), instancesB.size());

    // Convert instances to string lists
    std::vector<std::string> stringsA;
    std::vector<std::string> stringsB;

    for(const auto& instance : instancesA)
    {
        stringsA.push_back(instance->GetInstanceString());
    }

    for(const auto& instance : instancesB)
    {
        stringsB.push_back(instance->GetInstanceString());
    }

    // Sort for efficient set operations
    std::sort(stringsA.begin(), stringsA.end());
    std::sort(stringsB.begin(), stringsB.end());

    // Strings only in A
    std::vector<std::string> only_in_A;
    std::set_difference(stringsA.begin(),
                        stringsA.end(),
                        stringsB.begin(),
                        stringsB.end(),
                        std::back_inserter(only_in_A));

    EXPECT_EQ(only_in_A.size(), 0);

    // Strings only in B
    std::vector<std::string> only_in_B;
    std::set_difference(stringsB.begin(),
                        stringsB.end(),
                        stringsA.begin(),
                        stringsA.end(),
                        std::back_inserter(only_in_B));

    EXPECT_EQ(only_in_B.size(), 0);

    if(only_in_B.size() > 0)
    {
        std::cout << "There are " << only_in_B.size() << " kernels only in B: " << std::endl;
        print_instance_strings(only_in_B);
    }

    // Strings in both
    std::vector<std::string> in_both;
    std::set_intersection(stringsA.begin(),
                          stringsA.end(),
                          stringsB.begin(),
                          stringsB.end(),
                          std::back_inserter(in_both));

    if(in_both.size() > 0)
    {
        std::cout << "There are " << in_both.size() << " kernels in both: " << std::endl;
        print_instance_strings(in_both);
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

    compare_instance_vectors(ckFactoryInstances, builderFactoryInstances);
}

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsFloat) { CompareInstanceLists<float>(); }

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsHalf) { CompareInstanceLists<ck::half_t>(); }

/*
TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsBHalf) { CompareInstanceLists<ck::bhalf_t>(); }

TEST(CKBuilderGroupedFwdConv2D, CompareInstanceListsInt8) { CompareInstanceLists<int8_t>(); }
*/
