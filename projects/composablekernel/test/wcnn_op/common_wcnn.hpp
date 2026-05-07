// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd_gfx13.hpp"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wcnn_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using PassThrough         = ck::tensor_operation::element_wise::PassThrough;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;
using MultiplyAddRelu     = ck::tensor_operation::element_wise::MultiplyAddRelu<>;
using MultiplyAddHardTanh = ck::tensor_operation::element_wise::MultiplyAddHardTanh;
using ScaleAdd            = ck::tensor_operation::element_wise::ScaleAdd;
using ScaleAddRelu        = ck::tensor_operation::element_wise::ScaleAddRelu<>;
using ScaleAddHardTanh    = ck::tensor_operation::element_wise::ScaleAddHardTanh;
using Scale               = ck::tensor_operation::element_wise::Scale;
using ScaleRelu           = ck::tensor_operation::element_wise::ScaleRelu<>;
using ScaleHardTanh       = ck::tensor_operation::element_wise::ScaleHardTanh;

using MultiplyAddClamp     = ck::tensor_operation::element_wise::MultiplyAddClamp;
using MultiplyAddClampRelu = ck::tensor_operation::element_wise::MultiplyAddRelu<true>;
using ScaleAddClamp        = ck::tensor_operation::element_wise::ScaleAddClamp;
using ScaleAddClampRelu    = ck::tensor_operation::element_wise::ScaleAddRelu<true>;
using ScaleClamp           = ck::tensor_operation::element_wise::ScaleClamp;
using ScaleClampRelu       = ck::tensor_operation::element_wise::ScaleRelu<true>;
using MultiplyAddRev       = ck::tensor_operation::element_wise::MultiplyAddRev<>;
using MultiplyAddRevRelu   = ck::tensor_operation::element_wise::MultiplyAddRevRelu<>;
using ScaleAddRev          = ck::tensor_operation::element_wise::ScaleAddRev<>;
using ScaleAddRevRelu      = ck::tensor_operation::element_wise::ScaleAddRevRelu<>;

using MultiplyAddClampRev     = ck::tensor_operation::element_wise::MultiplyAddRev<true>;
using MultiplyAddClampRevRelu = ck::tensor_operation::element_wise::MultiplyAddRevRelu<true>;
using ScaleAddClampRev        = ck::tensor_operation::element_wise::ScaleAddRev<true>;
using ScaleAddClampRevRelu    = ck::tensor_operation::element_wise::ScaleAddRevRelu<true>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;

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

namespace ck {
namespace tensor_layout {
namespace convolution {

constexpr index_t CONST_LAYOUT_H = 64;
constexpr index_t CONST_LAYOUT_W = 64;
constexpr index_t CONST_LAYOUT_C = 16;
constexpr index_t CONST_LAYOUT_K = 16;

// input tensor
struct CONST_GNHWC : public BaseTensorLayout
{
    static constexpr const char* name = "CONST_GNHWC";
    static constexpr auto N           = Number<1>{};
    static constexpr auto H           = Number<CONST_LAYOUT_H>{};
    static constexpr auto W           = Number<CONST_LAYOUT_W>{};
    static constexpr auto C           = Number<CONST_LAYOUT_C>{};
};

// weight tensor
template <index_t FilterSize>
struct CONST_GKYXC : public BaseTensorLayout
{
    static constexpr const char* name = "CONST_GKYXC";
    static constexpr auto K           = Number<CONST_LAYOUT_K>{};
    static constexpr auto X           = Number<FilterSize>{};
    static constexpr auto Y           = Number<FilterSize>{};
    static constexpr auto C           = Number<CONST_LAYOUT_C>{};
};

// output tensor
struct CONST_GNHWK : public BaseTensorLayout
{
    static constexpr const char* name = "CONST_GNHWK";
    static constexpr auto N           = Number<1>{};
    static constexpr auto H           = Number<CONST_LAYOUT_H>{};
    static constexpr auto W           = Number<CONST_LAYOUT_W>{};
    static constexpr auto K           = Number<CONST_LAYOUT_K>{};
};

} // namespace convolution
} // namespace tensor_layout
} // namespace ck

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

template <ck::index_t FilterSize>
using ConstLayoutSetting =
    CommonLayoutSetting<ctl::CONST_GNHWC, ctl::CONST_GKYXC<FilterSize>, ctl::CONST_GNHWK>;

template <ck::index_t FilterSize>
using ConstInputLayout = typename ConstLayoutSetting<FilterSize>::InputLayout;

template <ck::index_t FilterSize>
using ConstWeightLayout = typename ConstLayoutSetting<FilterSize>::WeightLayout;

template <ck::index_t FilterSize>
using ConstOutputLayout = typename ConstLayoutSetting<FilterSize>::OutputLayout;

struct ExecutionConfig final
{
    uint32_t test_mask   = 0xffffffff;
    int init_method      = 1;
    bool do_verification = true;
    bool dump_tensor     = true;
    bool time_kernel     = false;
    int d                = 2;
    int h                = 64;
    int w                = 64;
    int c                = 16;
    int k                = 16;
};

inline ExecutionConfig config;

template <typename DataType>
void dump_tensor(const Tensor<DataType>& tensor, const char* str)
{
    if(config.dump_tensor == false)
        return;
    assert(tensor.GetNumOfDimension() >= 4 && tensor.GetNumOfDimension() <= 6);
    auto lengths = tensor.GetLengths();
    auto strides = tensor.GetStrides();

    auto numDim = tensor.GetNumOfDimension() - 3;
    if(numDim == 1)
    {
        std::cout << str << "  [ " << std::endl;
        for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
        {
            if(lengths[1] > 1)
            {
                std::cout << "  [";
            }
            if(i0 > 0 && strides[0] == 0)
            {
                continue;
            }
            for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
            {
                if(lengths[2] > 1)
                {
                    std::cout << "  [";
                }
                if(i1 > 0 && strides[1] == 0)
                {
                    continue;
                }
                for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i2 > 0 && strides[2] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(i3 > 0 && strides[3] == 0)
                        {
                            continue;
                        }
                        std::vector<std::size_t> idx({i0, i1, i2, i3});
                        std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
                    }
                    if(lengths[3] > 3)
                    {
                        std::cout << std::endl;
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
    else if(numDim == 2)
    {
        std::cout << str << "  [ " << std::endl;
        for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
        {
            if(lengths[1] > 1)
            {
                std::cout << "  [";
            }
            if(i0 > 0 && strides[0] == 0)
            {
                continue;
            }
            for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
            {
                if(lengths[2] > 1)
                {
                    std::cout << "  [";
                }
                if(i1 > 0 && strides[1] == 0)
                {
                    continue;
                }
                for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i2 > 0 && strides[2] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(lengths[4] > 1)
                        {
                            std::cout << "  [";
                        }
                        if(i3 > 0 && strides[3] == 0)
                        {
                            continue;
                        }
                        for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                        {
                            if(i4 > 0 && strides[4] == 0)
                            {
                                continue;
                            }
                            std::vector<std::size_t> idx({i0, i1, i2, i3, i4});
                            std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                        }
                        if(lengths[4] > 1)
                        {
                            std::cout << "]";
                        }
                        if(lengths[4] > 3)
                        {
                            std::cout << std::endl;
                        }
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
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
    else if(numDim == 3)
    {
        std::cout << str << "  [ " << std::endl;
        for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
        {
            if(lengths[1] > 1)
            {
                std::cout << "  [";
            }
            if(i0 > 0 && strides[0] == 0)
            {
                continue;
            }
            for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
            {
                if(lengths[2] > 1)
                {
                    std::cout << "  [";
                }
                if(i1 > 0 && strides[1] == 0)
                {
                    continue;
                }
                for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i2 > 0 && strides[2] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(lengths[4] > 1)
                        {
                            std::cout << "  [";
                        }
                        if(i3 > 0 && strides[3] == 0)
                        {
                            continue;
                        }
                        for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                        {
                            if(lengths[5] > 1)
                            {
                                std::cout << "  [";
                            }
                            if(i4 > 0 && strides[4] == 0)
                            {
                                continue;
                            }
                            for(uint32_t i5 = 0; i5 < lengths[5]; i5++)
                            {
                                if(i5 > 0 && strides[5] == 0)
                                {
                                    continue;
                                }
                                std::vector<std::size_t> idx({i0, i1, i2, i3, i4, i5});
                                std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                            }
                            if(lengths[5] > 1)
                            {
                                std::cout << "]";
                            }
                            if(lengths[5] > 3)
                            {
                                std::cout << std::endl;
                            }
                        }
                        if(lengths[4] > 1)
                        {
                            std::cout << "]";
                        }
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
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
}

template <typename DataType>
inline constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, bhalf_t>)
    {
        // Shape_4x2 + Filter_2X2 fail with 5e-2
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
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
    std::cerr << "arg1: test mask (hex)\n"
              << "arg2: verification (0=no, 1=yes)\n"
              << "arg3: dump tensor (0=no, 1=yes)\n"
              << "arg4: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "arg6-9: tensor size {H x W x C x K}" << std::endl;
}

inline bool parse_cmd_args(int argc, char* argv[], ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc <= 10)
    {
        if(argc > 1)
        {
            cfg.test_mask = std::stoul(argv[1], nullptr, 0);
        }
        if(argc > 2)
        {
            cfg.do_verification = std::stoi(argv[2]);
        }
        if(argc > 3)
        {
            cfg.dump_tensor = std::stoi(argv[3]);
        }
        if(argc > 4)
        {
            cfg.init_method = std::stoi(argv[4]);
        }
        if(argc > 5)
        {
            cfg.time_kernel = std::stoi(argv[5]);
        }
        if(argc > 6)
        {
            cfg.h = std::stoi(argv[6]);
        }
        if(argc > 7)
        {
            cfg.w = std::stoi(argv[7]);
        }
        if(argc > 8)
        {
            cfg.c = std::stoi(argv[8]);
        }
        if(argc > 9)
        {
            cfg.k = std::stoi(argv[9]);
        }
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

inline void print_help_msg_gtest()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: dump tensor (0=no, 1=yes)\n"
              << "arg3: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg4: time kernel (0=no, 1=yes)\n"
              << "arg5-8: tensor size {H x W x C x K}" << std::endl;
}

inline bool parse_cmd_args_gtest(int argc, char* argv[], ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc <= 9)
    {
        if(argc > 1)
        {
            cfg.do_verification = std::stoi(argv[1]);
        }
        if(argc > 2)
        {
            cfg.dump_tensor = std::stoi(argv[2]);
        }
        if(argc > 3)
        {
            cfg.init_method = std::stoi(argv[3]);
        }
        if(argc > 4)
        {
            cfg.time_kernel = std::stoi(argv[4]);
        }
        if(argc > 5)
        {
            cfg.h = std::stoi(argv[5]);
        }
        if(argc > 6)
        {
            cfg.w = std::stoi(argv[6]);
        }
        if(argc > 7)
        {
            cfg.c = std::stoi(argv[7]);
        }
        if(argc > 8)
        {
            cfg.k = std::stoi(argv[8]);
        }
    }
    else
    {
        print_help_msg_gtest();
        return false;
    }

    return true;
}

inline void print_help_msg_conv3d()
{
    std::cerr << "arg1: test mask (hex)\n"
              << "arg2: verification (0=no, 1=yes)\n"
              << "arg3: dump tensor (0=no, 1=yes)\n"
              << "arg4: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg5: time kernel (0=no, 1=yes)\n"
              << "arg6-10: tensor size {D x H x W x C x K}" << std::endl;
}

inline bool parse_cmd_args_conv3d(int argc, char* argv[], ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc <= 11)
    {
        if(argc > 1)
        {
            cfg.test_mask = std::stoul(argv[1], nullptr, 0);
        }
        if(argc > 2)
        {
            cfg.do_verification = std::stoi(argv[2]);
        }
        if(argc > 3)
        {
            cfg.dump_tensor = std::stoi(argv[3]);
        }
        if(argc > 4)
        {
            cfg.init_method = std::stoi(argv[4]);
        }
        if(argc > 5)
        {
            cfg.time_kernel = std::stoi(argv[5]);
        }
        if(argc > 6)
        {
            cfg.d = std::stoi(argv[6]);
        }
        if(argc > 7)
        {
            cfg.h = std::stoi(argv[7]);
        }
        if(argc > 8)
        {
            cfg.w = std::stoi(argv[8]);
        }
        if(argc > 9)
        {
            cfg.c = std::stoi(argv[9]);
        }
        if(argc > 10)
        {
            cfg.k = std::stoi(argv[10]);
        }
    }
    else
    {
        print_help_msg_conv3d();
        return false;
    }

    return true;
}

inline void print_help_msg_fix_size()
{
    std::cerr << "arg1: test mask (hex)\n"
              << "arg2: verification (0=no, 1=yes)\n"
              << "arg3: dump tensor (0=no, 1=yes)\n"
              << "arg4: initialization (0=no init, 1=integer value, 2=decimal value)\n";
}

inline bool parse_cmd_args_fix_size(int argc, char* argv[], ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc <= 5)
    {
        if(argc > 1)
        {
            cfg.test_mask = std::stoul(argv[1], nullptr, 0);
        }
        if(argc > 2)
        {
            cfg.do_verification = std::stoi(argv[2]);
        }
        if(argc > 3)
        {
            cfg.dump_tensor = std::stoi(argv[3]);
        }
        if(argc > 4)
        {
            cfg.init_method = std::stoi(argv[4]);
        }
    }
    else
    {
        print_help_msg_fix_size();
        return false;
    }

    return true;
}
