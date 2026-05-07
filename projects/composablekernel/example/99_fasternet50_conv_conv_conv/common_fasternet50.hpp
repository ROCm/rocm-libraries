// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "device_fasternet50_conv_conv_add_relu_conv_add_wcnn.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using half_t  = ck::half_t;
using bhalf_t = ck::bhalf_t;
using f8_t    = ck::f8_t;
using bf8_t   = ck::bf8_t;
using int4_t  = ck::int4_t;
using uint4_t = ck::uint4_t;

using BF16 = ck::bhalf_t;
using FP16 = ck::half_t;
using FP32 = float;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using I4 = ck::int4_t;
#endif
using I8  = std::int8_t;
using I32 = std::int32_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using MultiplyAdd = ck::tensor_operation::element_wise::MultiplyAdd;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

enum ShapeType
{
    Shape_4X2,
    Shape_4X4,
    Shape_8X4,
};

enum FilterType
{
    Filter_1X1,
    Filter_3X3,
    Filter_2X2,
};

template <typename InputLay, typename WeightLay, typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using OutputLayout = OutputLay;
};

template <ck::index_t NDimSpatial>
struct CommonLayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct CommonLayoutSettingSelector<1> final
    : CommonLayoutSetting<ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>
{
};

template <>
struct CommonLayoutSettingSelector<2> final
    : CommonLayoutSetting<ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>
{
};

template <>
struct CommonLayoutSettingSelector<3> final
    : CommonLayoutSetting<ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>
{
};

template <ck::index_t NDimSpatial>
using InputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::InputLayout;

template <ck::index_t NDimSpatial>
using WeightLayout = typename CommonLayoutSettingSelector<NDimSpatial>::WeightLayout;

template <ck::index_t NDimSpatial>
using OutputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::OutputLayout;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;
};

#define FasterNet50_DefaultConvParam_3x3_split                              \
    ck::utils::conv::ConvParam                                              \
    {                                                                       \
        2, 1, 1, 16, 16, {3, 3}, {96, 16}, {1, 1}, {1, 1}, {1, 1}, { 1, 1 } \
    }

#define FasterNet50_DefaultConvParam_3x3                                    \
    ck::utils::conv::ConvParam                                              \
    {                                                                       \
        2, 1, 1, 16, 32, {3, 3}, {96, 16}, {1, 1}, {1, 1}, {1, 1}, { 1, 1 } \
    }

#define FasterNet50_DefaultConvParam_1x1_ReluAdd                            \
    ck::utils::conv::ConvParam                                              \
    {                                                                       \
        2, 1, 1, 64, 32, {1, 1}, {96, 16}, {1, 1}, {1, 1}, {0, 0}, { 0, 0 } \
    }

#define FasterNet50_DefaultConvParam_1x1                                    \
    ck::utils::conv::ConvParam                                              \
    {                                                                       \
        2, 1, 1, 32, 64, {1, 1}, {96, 16}, {1, 1}, {1, 1}, {0, 0}, { 0, 0 } \
    }

template <typename Type>
const char* get_string()
{
    if constexpr(std::is_same<Type, half_t>::value)
    {
        return "half_t";
    }

    if constexpr(std::is_same<Type, float>::value)
    {
        return "float";
    }

    if constexpr(std::is_same<Type, bhalf_t>::value)
    {
        return "bhalf_t";
    }

    if constexpr(std::is_same<Type, f8_t>::value)
    {
        return "f8_t";
    }

    if constexpr(std::is_same<Type, bf8_t>::value)
    {
        return "bf8_t";
    }

    if constexpr(std::is_same<Type, int8_t>::value)
    {
        return "int8_t";
    }

    if constexpr(std::is_same<Type, int32_t>::value)
    {
        return "int32_t";
    }

    if constexpr(std::is_same<Type, uint8_t>::value)
    {
        return "uint8_t";
    }

    if constexpr(std::is_same<Type, uint32_t>::value)
    {
        return "uint32_t";
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    if constexpr(std::is_same<Type, ck::int4_t>::value)
    {
        return "int4_t";
    }

    if constexpr(std::is_same<Type, ck::uint4_t>::value)
    {
        return "uint4_t";
    }
#endif
}

inline const char* get_string(ShapeType type)
{
    switch(type)
    {
    case Shape_4X2: return "Shape_4x2";
    case Shape_4X4: return "Shape_4x4";
    case Shape_8X4: return "Shape_8x4";
    }
}

inline const char* get_string(FilterType filter)
{
    switch(filter)
    {
    case Filter_1X1: return "Filter_1X1";
    case Filter_3X3: return "Filter_3X3";
    case Filter_2X2: return "Filter_2X2";
    }
}

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config,
                           ck::utils::conv::ConvParam& conv_param)
{
    constexpr int num_execution_config_args =
        3; // arguments for do_verification, init_method, time_kernel
    constexpr int num_conv_param_leading_args = 5; // arguments for num_dim_spatial_, G_, N_, K_, C_

    constexpr int threshold_to_catch_partial_args = 1 + num_execution_config_args;
    constexpr int threshold_to_catch_all_args =
        threshold_to_catch_partial_args + num_conv_param_leading_args;

    if(argc == 1)
    {
        // use default
    }
    // catch only ExecutionConfig arguments
    else if(argc == threshold_to_catch_partial_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    // catch both ExecutionConfig & ConvParam arguments
    else if(threshold_to_catch_all_args < argc && ((argc - threshold_to_catch_all_args) % 3 == 0))
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        conv_param                        = ck::utils::conv::parse_conv_param(
            num_dim_spatial, threshold_to_catch_partial_args + 1, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

inline HostTensorDescriptor make_input_descriptor(const ck::utils::conv::ConvParam& conv_param)
{
    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.C_, conv_param.input_spatial_lengths_[0]},
            {
                conv_param.C_,                                                        // g
                conv_param.input_spatial_lengths_[0] * conv_param.G_ * conv_param.C_, // n
                1,                                                                    // c
                conv_param.G_ * conv_param.C_                                         // wi
            });

    case 2:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.C_,
             conv_param.input_spatial_lengths_[0],
             conv_param.input_spatial_lengths_[1]},
            {
                conv_param.C_, // g
                conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1] *
                    conv_param.G_ * conv_param.C_,                                    // n
                1,                                                                    // c
                conv_param.input_spatial_lengths_[1] * conv_param.G_ * conv_param.C_, // hi
                conv_param.G_ * conv_param.C_                                         // wi
            });

    case 3:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.C_,
             conv_param.input_spatial_lengths_[0],
             conv_param.input_spatial_lengths_[1],
             conv_param.input_spatial_lengths_[2]},
            {
                conv_param.C_, // g
                conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1] *
                    conv_param.input_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // n
                1,                                                                        // c
                conv_param.input_spatial_lengths_[1] * conv_param.input_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.C_,                                    // di
                conv_param.input_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // hi
                conv_param.G_ * conv_param.C_                                         // wi
            });
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

inline HostTensorDescriptor make_weight_descriptor(const ck::utils::conv::ConvParam& conv_param)
{
    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.K_, conv_param.C_, conv_param.filter_spatial_lengths_[0]},
            {
                conv_param.K_ * conv_param.filter_spatial_lengths_[0] * conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.C_,                 // k
                1,                                                                     // c
                conv_param.C_                                                          // x
            });
    case 2:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.K_,
             conv_param.C_,
             conv_param.filter_spatial_lengths_[0],
             conv_param.filter_spatial_lengths_[1]},
            {
                conv_param.K_ * conv_param.filter_spatial_lengths_[0] *
                    conv_param.filter_spatial_lengths_[1] * conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1] *
                    conv_param.C_,                                     // k
                1,                                                     // c
                conv_param.filter_spatial_lengths_[1] * conv_param.C_, // y
                conv_param.C_                                          // x
            });
    case 3:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.K_,
             conv_param.C_,
             conv_param.filter_spatial_lengths_[0],
             conv_param.filter_spatial_lengths_[1],
             conv_param.filter_spatial_lengths_[2]},
            {
                conv_param.K_ * conv_param.filter_spatial_lengths_[0] *
                    conv_param.filter_spatial_lengths_[1] * conv_param.filter_spatial_lengths_[2] *
                    conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1] *
                    conv_param.filter_spatial_lengths_[2] * conv_param.C_, // k
                1,                                                         // c
                conv_param.filter_spatial_lengths_[1] * conv_param.filter_spatial_lengths_[2] *
                    conv_param.C_,                                     // z
                conv_param.filter_spatial_lengths_[2] * conv_param.C_, // y
                conv_param.C_                                          // x
            });
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

inline HostTensorDescriptor make_bias_descriptor(const ck::utils::conv::ConvParam& conv_param)
{
    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.K_, conv_param.output_spatial_lengths_[0]},
            {
                conv_param.K_, // g
                0,             // k
                1,             // c
                0              // x
            });
    case 2:
        return HostTensorDescriptor({conv_param.G_,
                                     conv_param.N_,
                                     conv_param.K_,
                                     conv_param.output_spatial_lengths_[0],
                                     conv_param.output_spatial_lengths_[1]},
                                    {
                                        conv_param.K_, // g
                                        0,             // n
                                        1,             // k
                                        0,             // ho
                                        0              // wo
                                    });
    case 3:
        return HostTensorDescriptor({conv_param.G_,
                                     conv_param.N_,
                                     conv_param.K_,
                                     conv_param.output_spatial_lengths_[0],
                                     conv_param.output_spatial_lengths_[1],
                                     conv_param.output_spatial_lengths_[2]},
                                    {
                                        conv_param.K_, // g
                                        0,             // n
                                        1,             // k
                                        0,             // z
                                        0,             // y
                                        0              // x
                                    });
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

inline HostTensorDescriptor make_output_descriptor(const ck::utils::conv::ConvParam& conv_param)
{

    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.K_, conv_param.output_spatial_lengths_[0]},
            {
                conv_param.K_,                                                         // g
                conv_param.output_spatial_lengths_[0] * conv_param.G_ * conv_param.K_, // n
                1,                                                                     // k
                conv_param.G_ * conv_param.K_                                          // wo
            });
    case 2:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.K_,
             conv_param.output_spatial_lengths_[0],
             conv_param.output_spatial_lengths_[1]},
            {
                conv_param.K_, // g
                conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1] *
                    conv_param.G_ * conv_param.K_,                                     // n
                1,                                                                     // k
                conv_param.output_spatial_lengths_[1] * conv_param.G_ * conv_param.K_, // ho
                conv_param.G_ * conv_param.K_                                          // wo
            });

    case 3:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.K_,
             conv_param.output_spatial_lengths_[0],
             conv_param.output_spatial_lengths_[1],
             conv_param.output_spatial_lengths_[2]},
            {
                conv_param.K_, // g
                conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1] *
                    conv_param.output_spatial_lengths_[2] * conv_param.G_ * conv_param.K_, // n
                1,                                                                         // k
                conv_param.output_spatial_lengths_[1] * conv_param.output_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.K_,                                     // do
                conv_param.output_spatial_lengths_[2] * conv_param.G_ * conv_param.K_, // ho
                conv_param.G_ * conv_param.K_                                          // wo
            });
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

template <typename DataType>
void dump_tensor(const Tensor<DataType>& tensor, const char* str)
{
    assert(tensor.GetNumOfDimension() >= 4 && tensor.GetNumOfDimension() <= 6);
    auto lengths = tensor.GetLengths();

    std::cout << str << "  [ " << std::endl;

    auto dump_data = [&](size_t len, std::vector<std::size_t>& idx) {
        if(len > 1)
        {
            std::cout << "  [";
        }
        for(uint32_t i = 0; i < len; i++)
        {
            idx.back() = i;
            std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
        }
        if(len > 1)
        {
            std::cout << "]";
            if(len > 3)
            {
                std::cout << std::endl;
            }
        }
    };

    auto dim = tensor.GetNumOfDimension() - 3;
    for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
    {
        if(lengths[1] > 1)
        {
            std::cout << "  [";
        }
        for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
        {
            if(lengths[2] > 1)
            {
                std::cout << "  [";
            }
            for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
            {
                if(dim == 1)
                {
                    std::vector<std::size_t> idx({i0, i1, i2, 0});
                    dump_data(lengths[3], idx);
                }
                else
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(dim == 2)
                        {
                            std::vector<std::size_t> idx({i0, i1, i2, i3, 0});
                            dump_data(lengths[4], idx);
                        }
                        else
                        {
                            if(lengths[4] > 1)
                            {
                                std::cout << "  [";
                            }
                            for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                            {
                                std::vector<std::size_t> idx({i0, i1, i2, i3, i4, 0});
                                dump_data(lengths[5], idx);
                            }
                            if(lengths[4] > 1)
                            {
                                std::cout << "]" << std::endl;
                            }
                        }
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
                    }
                }
            }
            if(lengths[2] > 1)
            {
                std::cout << "]" << std::endl;
            }
        }
        if(lengths[1] > 1)
        {
            std::cout << "]" << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

template <typename DataType>
void partition_tensor(const Tensor<DataType>& in_tensor, Tensor<DataType>& out_tensor)
{
    assert(out_tensor.GetNumOfDimension() >= 4 && out_tensor.GetNumOfDimension() <= 6);
    auto lengths = out_tensor.GetLengths();

    for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
    {
        for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
        {
            for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
            {
                for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                {
                    for(uint32_t i4 = 0; i4 < lengths[4]; i4++) // Partial in input Channel
                    {
                        std::vector<std::size_t> idx({i0, i1, i2, i3, i4, 0});
                        out_tensor(idx) = in_tensor(idx);
                    }
                }
            }
        }
    }
}

template <typename DataType>
void merge_tensor(const Tensor<DataType>& in_0_tensor,
                  const Tensor<DataType>& in_1_tensor,
                  Tensor<DataType>& out_tensor)
{
    assert(out_tensor.GetNumOfDimension() >= 4 && out_tensor.GetNumOfDimension() <= 6);
    auto out_lengths  = out_tensor.GetLengths();
    auto in_0_lengths = in_0_tensor.GetLengths();

    for(uint32_t i0 = 0; i0 < out_lengths[0]; i0++)
    {
        for(uint32_t i1 = 0; i1 < out_lengths[1]; i1++)
        {
            for(uint32_t i3 = 0; i3 < out_lengths[3]; i3++)
            {
                for(uint32_t i4 = 0; i4 < out_lengths[4]; i4++) // Merge in_0 input Channel
                {
                    for(uint32_t i2 = 0; i2 < out_lengths[2]; i2++)
                    {
                        std::vector<std::size_t> out_idx({i0, i1, i2, i3, i4, 0});
                        if(i2 < 16)
                        {
                            // IN_0:[0~15]-->OUT:[0~15]
                            std::vector<std::size_t> in_0_idx({i0, i1, i2, i3, i4, 0});
                            out_tensor(out_idx) = in_0_tensor(in_0_idx);
                        }
                        else
                        {
                            // IN_0:[16~31]-->OUT:[16~31]
                            std::vector<std::size_t> in_1_idx({i0, i1, i2, i3, i4, 0});
                            out_tensor(out_idx) = in_1_tensor(in_1_idx);
                        }
                    }
                }
            }
        }
    }
}
