// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>

#include <hip/hip_runtime.h>

namespace origami
{
    enum class data_type_t : int
    {
        Float,
        Double,
        ComplexFloat,
        ComplexDouble,
        Half,
        Int8x4,
        Int32,
        BFloat16,
        Int8,
        Int64,
        XFloat32,
        Float8_fnuz,
        BFloat8_fnuz,
        Float8BFloat8_fnuz,
        BFloat8Float8_fnuz,
        Float8,
        BFloat8,
        Float8BFloat8,
        BFloat8Float8,
        Float6,
        BFloat6,
        Float4,
        Count,
        None = Count
    };

    inline data_type_t int_to_data_type(int dt)
    {
        return (data_type_t)dt;
    }

    inline int data_type_to_bits(data_type_t type)
    {
        switch(type)
        {
        case data_type_t::Float:
            return 32;
        case data_type_t::Double:
            return 64;
        case data_type_t::ComplexFloat:
            return 64;
        case data_type_t::ComplexDouble:
            return 128;
        case data_type_t::Half:
            return 16;
        case data_type_t::Int8x4:
            return 32;
        case data_type_t::Int32:
            return 32;
        case data_type_t::BFloat16:
            return 16;
        case data_type_t::Int8:
            return 8;
        case data_type_t::Int64:
            return 64;
        case data_type_t::XFloat32:
            return 32;
        case data_type_t::Float8_fnuz:
            return 8;
        case data_type_t::BFloat8_fnuz:
            return 8;
        case data_type_t::Float8BFloat8_fnuz:
            return 8;
        case data_type_t::BFloat8Float8_fnuz:
            return 8;
        case data_type_t::Float8:
            return 8;
        case data_type_t::BFloat8:
            return 8;
        case data_type_t::Float8BFloat8:
            return 8;
        case data_type_t::BFloat8Float8:
            return 8;
        case data_type_t::Float6:
            return 6;
        case data_type_t::BFloat6:
            return 6;
        case data_type_t::Float4:
            return 4;
        default:
            return -1; // Invalid type
        }
    }

    inline std::string to_string(data_type_t type)
    {
        switch(type)
        {
        case data_type_t::Float:
            return "Float";
        case data_type_t::Double:
            return "Double";
        case data_type_t::ComplexFloat:
            return "ComplexFloat";
        case data_type_t::ComplexDouble:
            return "ComplexDouble";
        case data_type_t::Half:
            return "Half";
        case data_type_t::Int8x4:
            return "Int8x4";
        case data_type_t::Int32:
            return "Int32";
        case data_type_t::BFloat16:
            return "BFloat16";
        case data_type_t::Int8:
            return "Int8";
        case data_type_t::Int64:
            return "Int64";
        case data_type_t::XFloat32:
            return "XFloat32";
        case data_type_t::Float8_fnuz:
            return "Float8_fnuz";
        case data_type_t::BFloat8_fnuz:
            return "BFloat8_fnuz";
        case data_type_t::Float8BFloat8_fnuz:
            return "Float8BFloat8_fnuz";
        case data_type_t::BFloat8Float8_fnuz:
            return "BFloat8Float8_fnuz";
        case data_type_t::Float8:
            return "Float8";
        case data_type_t::BFloat8:
            return "BFloat8";
        case data_type_t::Float8BFloat8:
            return "Float8BFloat8";
        case data_type_t::BFloat8Float8:
            return "BFloat8Float8";
        case data_type_t::Float6:
            return "Float6";
        case data_type_t::BFloat6:
            return "BFloat6";
        case data_type_t::Float4:
            return "Float4";
        default:
            return "Invalid";
        }
        return "Invalid";
    }

    inline data_type_t string_to_data_type(std::string s)
    {
        if (s == "f32")
            return data_type_t::Float;
        if (s == "c32")
            return data_type_t::ComplexFloat;
        if (s == "c64")
            return data_type_t::ComplexDouble;
        if (s == "f64")
            return data_type_t::Double;
        if (s == "f16")
            return data_type_t::Half;
        if (s == "i32")
            return data_type_t::Int32;
        if (s == "bf16")
            return data_type_t::BFloat16;
        if (s == "i8")
            return data_type_t::Int8;
        if (s == "xf32")
            return data_type_t::XFloat32;
        if (s == "f8")
            return data_type_t::Float8;
        if (s == "bf8")
            return data_type_t::BFloat8;
        if (s == "f6")
            return data_type_t::Float6;
        if (s == "bf6")
            return data_type_t::BFloat6;
        if (s == "f4")
            return data_type_t::Float4;
        return data_type_t::None;
    }

    struct matrix_instruction
    {
        size_t MI_M;
        size_t MI_N;
        size_t MI_K;
        data_type_t mi_input_type;

        matrix_instruction()
            : MI_M(0)
            , MI_N(0)
            , MI_K(0)
            , mi_input_type(data_type_t::Float)
        {
        }

        matrix_instruction(size_t m, size_t n, size_t k, data_type_t mi_input_type)
            : MI_M(m)
            , MI_N(n)
            , MI_K(k)
            , mi_input_type(mi_input_type)
        {
        }

        matrix_instruction(const matrix_instruction& other)
            : MI_M(other.MI_M)
            , MI_N(other.MI_N)
            , MI_K(other.MI_K)
            , mi_input_type(other.mi_input_type)
        {
        }

        bool operator<(const matrix_instruction& other) const
        {
            return std::tie(MI_M, MI_N, MI_K, mi_input_type)
                    < std::tie(other.MI_M, other.MI_N, other.MI_K, other.mi_input_type);
        }

        bool operator==(const matrix_instruction& other) const
        {
            return MI_M == other.MI_M && MI_N == other.MI_N && MI_K == other.MI_K
                    && mi_input_type == other.mi_input_type;
        }

        std::size_t hash() const
        {
            return std::hash<size_t>()(MI_M) ^ std::hash<size_t>()(MI_N)
                    ^ std::hash<size_t>()(MI_K) ^ std::hash<data_type_t>()(mi_input_type);
        }
    };
}

// Specialize std::hash for the matrix_instruction struct to use it as an unordered_map key.
namespace std
{
    template <>
    struct hash<origami::matrix_instruction>
    {
        std::size_t operator()(const origami::matrix_instruction& k) const
        {
            return k.hash();
        }
    };
}

namespace origami
{
    class hardware_t
    {
    public:
        enum class architecture_t
        {
            gfx90a,
            gfx942,
            gfx950,
            Count
        };

        static architecture_t arch_name_to_enum(const std::string& str)
        {
            static const std::unordered_map<std::string, architecture_t> str_to_enum_map
                = {{"gfx90a", architecture_t::gfx90a},
                    {"gfx942", architecture_t::gfx942},
                    {"gfx950", architecture_t::gfx950}};

            auto it = str_to_enum_map.find(str);
            if(it != str_to_enum_map.end())
            {
                return it->second;
            }
            else
            {
                return architecture_t::Count;
            }
        }

        struct architecture_constants
        {
            size_t num_xcds;
            double mem1_perf_ratio;
            double mem2_perf_ratio;
            double mem3_perf_ratio;
            size_t parallel_mi_cu;
            std::tuple<double, double, double> mem_bw_per_wg_coefficients;
            double mem_clock_ratio;

            architecture_constants(size_t num_xcds,
                                    double mem1_perf_ratio,
                                    double mem2_perf_ratio,
                                    double mem3_perf_ratio,
                                    size_t parallel_mi_cu,
                                    std::tuple<double, double, double> mem_bw_per_wg_coefficients,
                                    double mem_clock_ratio) //Obtained through microbenchmarking
                : num_xcds(num_xcds)
                , mem1_perf_ratio(mem1_perf_ratio)
                , mem2_perf_ratio(mem2_perf_ratio)
                , mem3_perf_ratio(mem3_perf_ratio)
                , parallel_mi_cu(parallel_mi_cu)
                , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
                , mem_clock_ratio(mem_clock_ratio)
            {
            }
        };

        inline static const std::unordered_map<architecture_t, architecture_constants> ARCH_CONSTANT_MAP
            = {{hardware_t::architecture_t::gfx90a,
                hardware_t::architecture_constants(
                    1, 5.5, 1.21875121875121875122 * 1.2, 1.2, 4, std::make_tuple(0, 0.03, 0), 1.5)},
               {hardware_t::architecture_t::gfx942,
                hardware_t::architecture_constants(
                    8, 17, 1.21875121875121875122 * 6, 4, 4, std::make_tuple(0, 0.015, 0), 1.5)},
               {hardware_t::architecture_t::gfx950,
                // hardware_t::architecture_constants(
                //     8, 17, 1.21875121875121875122 * 7, 6, 4, std::make_tuple(-0.000013, 0.007070, 0.027355), 1.5)}};
                hardware_t::architecture_constants(
                    8, 17, 1.21875121875121875122 * 7, 6, 4, std::make_tuple(0, 0.008, 0), 1.5)}};

        inline static const std::unordered_map<architecture_t,
                                        std::unordered_map<matrix_instruction, size_t>> INSTRUCTION_MAP
            = {{hardware_t::architecture_t::gfx90a,
                {
                    // F32
                    {matrix_instruction(32, 32, 2, data_type_t::Float), 64}, // v_mfma_f32_32x32x2_f32
                    {matrix_instruction(32, 32, 1, data_type_t::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {matrix_instruction(16, 16, 4, data_type_t::Float), 32}, // v_mfma_f32_16x16x4_f32
                    {matrix_instruction(16, 16, 1, data_type_t::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {matrix_instruction(4, 4, 1, data_type_t::Float), 8}, // v_mfma_f32_4x4x1_16b_f32
                    // F64
                    {matrix_instruction(16, 16, 4, data_type_t::Double), 32}, // v_mfma_f64_16x16x4_f64
                    {matrix_instruction(4, 4, 4, data_type_t::Double), 16}, // v_mfma_f64_4x4x4_4b_f64
                    // TODO ComplexFloat
                    // TODO ComplexDouble
                    // F16
                    {matrix_instruction(32, 32, 4, data_type_t::Half), 64}, // v_mfma_f32_32x32x4_2b_f16
                    {matrix_instruction(32, 32, 8, data_type_t::Half), 64}, // v_mfma_f32_32x32x8_f16
                    {matrix_instruction(16, 16, 4, data_type_t::Half), 32}, // v_mfma_f32_16x16x4_4b_f16
                    {matrix_instruction(16, 16, 16, data_type_t::Half), 32}, // v_mfma_f32_16x16x16_f16
                    {matrix_instruction(4, 4, 4, data_type_t::Half), 8}, // v_mfma_f32_4x4x4_16b_f16
                    // BF16
                    {matrix_instruction(32, 32, 4, data_type_t::BFloat16), 64}, // v_mfma_f32_32x32x4_2b_bf16
                    {matrix_instruction(32, 32, 8, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
                    {matrix_instruction(16, 16, 4, data_type_t::BFloat16), 32}, // v_mfma_f32_16x16x4_4b_bf16
                    {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    {matrix_instruction(4, 4, 4, data_type_t::BFloat16), 8}, // v_mfma_f32_4x4x4_16b_bf16
                    // I8
                    {matrix_instruction(32, 32, 8, data_type_t::Int8), 64}, // v_mfma_f32_32x32x16_f8
                    {matrix_instruction(32, 32, 4, data_type_t::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
                    {matrix_instruction(16, 16, 16, data_type_t::Int8), 32}, // v_mfma_f32_16x16x32_i8
                    {matrix_instruction(16, 16, 4, data_type_t::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {matrix_instruction(4, 4, 4, data_type_t::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    // XF32
                    {matrix_instruction(32, 32, 8, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x8_bf16 * 3
                    {matrix_instruction(32, 32, 16, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x16_bf16 * 3
                    {matrix_instruction(16, 16, 16, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
                    {matrix_instruction(16, 16, 32, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
                }},
               {hardware_t::architecture_t::gfx942,
                {
                    // F32
                    {matrix_instruction(32, 32, 2, data_type_t::Float), 64}, // v_mfma_f32_32x32x2_f32
                    {matrix_instruction(32, 32, 1, data_type_t::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {matrix_instruction(16, 16, 4, data_type_t::Float), 32}, // v_mfma_f32_16x16x4_f32
                    {matrix_instruction(16, 16, 1, data_type_t::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {matrix_instruction(4, 4, 1, data_type_t::Float), 8}, // v_mfma_f32_4x4x1_16b_f32
                    // F64
                    {matrix_instruction(16, 16, 4, data_type_t::Double), 32}, // v_mfma_f64_16x16x4_f64
                    {matrix_instruction(4, 4, 4, data_type_t::Double), 16}, // v_mfma_f64_4x4x4_4b_f64
                    // TODO ComplexFloat
                    // TODO ComplexDouble
                    // F16
                    {matrix_instruction(32, 32, 4, data_type_t::Half), 64}, // v_mfma_f32_32x32x4_2b_f16
                    {matrix_instruction(32, 32, 8, data_type_t::Half), 32}, // v_mfma_f32_32x32x8_f16
                    {matrix_instruction(16, 16, 4, data_type_t::Half), 32}, // v_mfma_f32_16x16x4_4b_f16
                    {matrix_instruction(16, 16, 16, data_type_t::Half), 16}, // v_mfma_f32_16x16x16_f16
                    {matrix_instruction(4, 4, 4, data_type_t::Half), 8}, // v_mfma_f32_4x4x4_16b_f16
                    // BF16
                    {matrix_instruction(32, 32, 4, data_type_t::BFloat16), 64}, // v_mfma_f32_32x32x4_2b_bf16
                    {matrix_instruction(32, 32, 8, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
                    {matrix_instruction(16, 16, 4, data_type_t::BFloat16), 32}, // v_mfma_f32_16x16x4_4b_bf16
                    {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    {matrix_instruction(4, 4, 4, data_type_t::BFloat16), 8}, // v_mfma_f32_4x4x4_16b_bf16
                    // F8
                    {matrix_instruction(32, 32, 16, data_type_t::Float8_fnuz), 32}, // v_mfma_f32_32x32x16_f8
                    {matrix_instruction(16, 16, 32, data_type_t::Float8_fnuz), 16}, // v_mfma_f32_16x16x32_f8
                    // BF8
                    {matrix_instruction(32, 32, 16, data_type_t::BFloat8_fnuz), 32}, // v_mfma_f32_32x32x16_bf8
                    {matrix_instruction(16, 16, 32, data_type_t::BFloat8_fnuz), 16}, // v_mfma_f32_16x16x32_bf8
                    // F8B8
                    {matrix_instruction(32, 32, 16, data_type_t::Float8BFloat8_fnuz), 32}, // v_mfma_f32_32x32x16_f8_bf8
                    {matrix_instruction(16, 16, 32, data_type_t::Float8BFloat8_fnuz), 16}, // v_mfma_f32_16x16x32_f8_bf8
                    // B8F8
                    {matrix_instruction(32, 32, 16, data_type_t::BFloat8Float8_fnuz), 32}, // v_mfma_f32_32x32x16_bf8_f8
                    {matrix_instruction(16, 16, 32, data_type_t::BFloat8Float8_fnuz), 16}, // v_mfma_f32_16x16x32_bf8_f8
                    // I8
                    {matrix_instruction(32, 32, 16, data_type_t::Int8), 32}, // v_mfma_f32_32x32x16_f8
                    {matrix_instruction(32, 32, 4, data_type_t::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
                    {matrix_instruction(16, 16, 32, data_type_t::Int8), 16}, // v_mfma_f32_16x16x32_i8
                    {matrix_instruction(16, 16, 4, data_type_t::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {matrix_instruction(4, 4, 4, data_type_t::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    // XF32
                    {matrix_instruction(32, 32, 4, data_type_t::XFloat32), 32}, // v_mfma_f32_32x32x4_xf32
                    {matrix_instruction(16, 16, 32, data_type_t::XFloat32), 16}, // v_mfma_f32_16x16x8_xf32
                }},
               {hardware_t::architecture_t::gfx950,
                {
                    // F32
                    {matrix_instruction(32, 32, 2, data_type_t::Float), 64}, // v_mfma_f32_32x32x2_f32
                    {matrix_instruction(32, 32, 1, data_type_t::Float), 64}, // v_mfma_f32_32x32x1_2b_f32
                    {matrix_instruction(16, 16, 4, data_type_t::Float), 32}, // v_mfma_f32_16x16x4_f32
                    {matrix_instruction(16, 16, 1, data_type_t::Float), 32}, // v_mfma_f32_16x16x1_4b_f32
                    {matrix_instruction(4, 4, 1, data_type_t::Float), 8}, // v_mfma_f32_4x4x1_16b_f32
                    // F64
                    {matrix_instruction(16, 16, 4, data_type_t::Double), 64}, // v_mfma_f64_16x16x4_f64
                    {matrix_instruction(4, 4, 4, data_type_t::Double), 16}, // v_mfma_f64_4x4x4_4b_f64
                    // TODO ComplexFloat
                    // TODO ComplexDouble
                    // F16
                    {matrix_instruction(32, 32, 8, data_type_t::Half), 32}, // v_mfma_f32_32x32x8_f16
                    {matrix_instruction(32, 32, 16, data_type_t::Half), 32}, // v_mfma_f32_32x32x16_f16
                    {matrix_instruction(16, 16, 16, data_type_t::Half), 16}, // v_mfma_f32_16x16x16_f16
                    {matrix_instruction(16, 16, 32, data_type_t::Half), 16}, // v_mfma_f32_16x16x32_f16
                    // BF16
                    {matrix_instruction(32, 32, 8, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x8_bf16
                    {matrix_instruction(32, 32, 16, data_type_t::BFloat16), 32}, // v_mfma_f32_32x32x16_bf16
                    {matrix_instruction(16, 16, 16, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    {matrix_instruction(16, 16, 32, data_type_t::BFloat16), 16}, // v_mfma_f32_16x16x16_bf16
                    // F8
                    {matrix_instruction(32, 32, 64, data_type_t::Float8), 64}, // v_mfma_f32_32x32x64_f8
                    {matrix_instruction(32, 32, 16, data_type_t::Float8), 32}, // v_mfma_f32_32x32x16_f8
                    {matrix_instruction(16, 16, 128, data_type_t::Float8), 32}, // v_mfma_f32_16x16x128_f8
                    {matrix_instruction(16, 16, 32, data_type_t::Float8), 16}, // v_mfma_f32_16x16x32_f8
                    // BF8
                    {matrix_instruction(32, 32, 64, data_type_t::BFloat8), 64}, // v_mfma_f32_32x32x64_bf8
                    {matrix_instruction(32, 32, 16, data_type_t::BFloat8), 32}, // v_mfma_f32_32x32x16_bf8
                    {matrix_instruction(16, 16, 128, data_type_t::BFloat8), 32}, // v_mfma_f32_16x16x128_bf8
                    {matrix_instruction(16, 16, 32, data_type_t::BFloat8), 16}, // v_mfma_f32_16x16x32_bf8
                    // F8B8
                    {matrix_instruction(32, 32, 64, data_type_t::Float8BFloat8), 64}, // v_mfma_f32_32x32x64_f8_bf8
                    {matrix_instruction(32, 32, 16, data_type_t::Float8BFloat8), 32}, // v_mfma_f32_32x32x16_f8_bf8
                    {matrix_instruction(16, 16, 128, data_type_t::Float8BFloat8), 32}, // v_mfma_f32_16x16x128_f8_bf8
                    {matrix_instruction(16, 16, 32, data_type_t::Float8BFloat8), 16}, // v_mfma_f32_16x16x32_f8_bf8
                    // B8F8
                    {matrix_instruction(32, 32, 64, data_type_t::BFloat8Float8), 64}, // v_mfma_f32_32x32x64_bf8_f8
                    {matrix_instruction(32, 32, 16, data_type_t::BFloat8Float8), 32}, // v_mfma_f32_32x32x16_bf8_f8
                    {matrix_instruction(16, 16, 128, data_type_t::BFloat8Float8), 32}, // v_mfma_f32_16x16x128_bf8_f8
                    {matrix_instruction(16, 16, 32, data_type_t::BFloat8Float8), 16}, // v_mfma_f32_16x16x32_bf8_f8
                    // I8
                    {matrix_instruction(32, 32, 16, data_type_t::Int8), 32}, // v_mfma_f32_32x32x16_f8
                    {matrix_instruction(32, 32, 4, data_type_t::Int8), 64}, // v_mfma_i32_32x32x4_2b_i8
                    {matrix_instruction(16, 16, 32, data_type_t::Int8), 16}, // v_mfma_f32_16x16x32_i8
                    {matrix_instruction(16, 16, 4, data_type_t::Int8), 32}, // v_mfma_i32_16x16x4_4b_i8
                    {matrix_instruction(4, 4, 4, data_type_t::Int8), 8}, // v_mfma_i32_4x4x4_16b_i8
                    // XF32
                    {matrix_instruction(32, 32, 8, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x8_bf16 * 3
                    {matrix_instruction(32, 32, 16, data_type_t::XFloat32), 96}, // v_mfma_f32_32x32x16_bf16 * 3
                    {matrix_instruction(16, 16, 16, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
                    {matrix_instruction(16, 16, 32, data_type_t::XFloat32), 48}, // v_mfma_f32_16x16x16_bf16 * 3
                    // F6
                    {matrix_instruction(32, 32, 64, data_type_t::Float6), 32}, // v_mfma_f32_32x32x64_f6
                    {matrix_instruction(16, 16, 128, data_type_t::Float6), 16}, // v_mfma_f32_16x16x128_f6
                    // BF6
                    {matrix_instruction(32, 32, 64, data_type_t::BFloat6), 32}, // v_mfma_f32_32x32x64_bf6
                    {matrix_instruction(16, 16, 128, data_type_t::BFloat6), 16}, // v_mfma_f32_16x16x128_bf6
                    // F4
                    {matrix_instruction(32, 32, 64, data_type_t::Float4), 32}, // v_mfma_f32_32x32x64_f4
                    {matrix_instruction(16, 16, 128, data_type_t::Float4), 16}, // v_mfma_f32_16x16x128_f4
                    // DOT2
                    {matrix_instruction( 1,  1,  64, data_type_t::Half), 16}, // V_DOT2_F32_F16
                    {matrix_instruction( 1,  1,  64, data_type_t::BFloat16), 16}, // V_DOT2_F32_BF16
                }}};

        architecture_t                      arch;
        size_t                              N_CU; // Number of Compute Units
        size_t                              lds_capacity; // Capacity of LDS
        double                              mem1_perf_ratio;
        double                              mem2_perf_ratio;
        double                              mem3_perf_ratio;
        size_t                              L2_capacity; // Capacity of L2 in bytes
        size_t                              CU_per_L2; // Number of compute units per L2 domain
        double                              compute_clock_ghz;
        size_t                              parallel_mi_cu; // The number of parallel MI in a CU
        std::tuple<double, double, double>  mem_bw_per_wg_coefficients;
        size_t                              NUM_XCD;

        hardware_t(architecture_t                       arch,
                    size_t                              N_CU,
                    size_t                              lds_capacity,
                    size_t                              NUM_XCD,
                    double                              mem1_perf_ratio,
                    double                              mem2_perf_ratio,
                    double                              mem3_perf_ratio,
                    size_t                              L2_capacity,
                    double                              compute_clock_ghz,
                    size_t                              parallel_mi_cu,
                    std::tuple<double, double, double>  mem_bw_per_wg_coefficients)
            : arch(arch)
            , N_CU(N_CU)
            , lds_capacity(lds_capacity)
            , mem1_perf_ratio(mem1_perf_ratio)
            , mem2_perf_ratio(mem2_perf_ratio)
            , mem3_perf_ratio(mem3_perf_ratio)
            , L2_capacity(L2_capacity)
            , CU_per_L2(N_CU / NUM_XCD)
            , compute_clock_ghz(compute_clock_ghz)
            , parallel_mi_cu(parallel_mi_cu)
            , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
            , NUM_XCD(NUM_XCD)
        {
        }

        hardware_t(hipDeviceProp_t properties)
            : hardware_t(get_hardware_for_properties(properties))
        {
        }

        hardware_t(const hardware_t& other)
            : arch(other.arch)
            , N_CU(other.N_CU)
            , lds_capacity(other.lds_capacity)
            , mem1_perf_ratio(other.mem1_perf_ratio)
            , mem2_perf_ratio(other.mem2_perf_ratio)
            , mem3_perf_ratio(other.mem3_perf_ratio)
            , L2_capacity(other.L2_capacity)
            , CU_per_L2(other.CU_per_L2)
            , compute_clock_ghz(other.compute_clock_ghz)
            , parallel_mi_cu(other.parallel_mi_cu)
            , mem_bw_per_wg_coefficients(other.mem_bw_per_wg_coefficients)
            , NUM_XCD(other.NUM_XCD)
        {
        }

        static hardware_t get_hardware_for_properties(hipDeviceProp_t properties)
        {
            auto arch_name = get_before_first_colon(properties.gcnArchName);
            auto arch_enum = arch_name_to_enum(arch_name);
            auto it       = ARCH_CONSTANT_MAP.find(arch_enum);
            if(it == ARCH_CONSTANT_MAP.end())
            {
                throw std::runtime_error(
                    "Attempting to retrieve hardware constants for unsupported architecture: "
                    + arch_name); // Could also return default values here.
            }
            auto constants = it->second;
            return hardware_t(arch_enum,
                            properties.multiProcessorCount,
                            properties.sharedMemPerBlock,
                            constants.num_xcds,
                            1e9 * constants.mem1_perf_ratio / properties.clockRate,
                            1e9 * constants.mem2_perf_ratio
                                / (properties.memoryClockRate * constants.mem_clock_ratio),
                            1e9 * constants.mem3_perf_ratio / properties.memoryClockRate,
                            properties.l2CacheSize,
                            properties.clockRate / 1e6,
                            constants.parallel_mi_cu,
                            constants.mem_bw_per_wg_coefficients);
        }

        static hardware_t get_hardware_for_device(int deviceId)
        {
            hipDeviceProp_t prop;
            hipError_t      e = hipGetDeviceProperties(&prop, deviceId);
            if(e)
            {
                throw std::runtime_error(hipGetErrorString(e));
            }
            return get_hardware_for_properties(prop);
        }

        static bool is_hardware_supported(hipDeviceProp_t properties)
        {
            auto arch_name = get_before_first_colon(properties.gcnArchName);
            auto arch_enum = arch_name_to_enum(arch_name);
            auto it       = ARCH_CONSTANT_MAP.find(arch_enum);
            return it != ARCH_CONSTANT_MAP.end();
        }

        // Function to print hardware details
        void print() const
        {
            std::cout << "================== Hardware Configuration ==================\n";
            std::cout << "Number of CUs (N_CU)      : " << N_CU << "\n";
            std::cout << "LDS capacity              : " << lds_capacity << " bytes\n";
            std::cout << "mem1_perf_ratio           : " << mem1_perf_ratio << "\n";
            std::cout << "mem2_perf_ratio           : " << mem2_perf_ratio << "\n";
            std::cout << "mem3_perf_ratio           : " << mem3_perf_ratio << "\n";
            std::cout << "L2 Cache capacity         : " << L2_capacity << " bytes\n";
            std::cout << "CUs per L2 domain         : " << CU_per_L2 << "\n";
            std::cout << "Compute clock (GHz)       : " << compute_clock_ghz << "\n";
            std::cout << "Parallel MI/CU            : " << parallel_mi_cu << "\n";
            std::cout << "Number of XCDs (NUM_XCD)  : " << NUM_XCD << "\n";
            std::cout << "mem_bw_per_wg_coefficients: " << std::get<0>(mem_bw_per_wg_coefficients) << ", "
                                                        << std::get<1>(mem_bw_per_wg_coefficients) << ", "
                                                        << std::get<2>(mem_bw_per_wg_coefficients) << "\n\n";

            std::cout << "------------------ Instruction Map -------------------------\n";
            // Loop over the instruction_map and print each entry
            for(const auto& kv : INSTRUCTION_MAP.at(arch))
            {
                const auto& key  = kv.first;
                const auto& L_MI = kv.second;

                std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N
                            << ", MI_K=" << key.MI_K << ", mi_input_type=" << to_string(key.mi_input_type)
                            << " bytes\n"
                            << "  -> Latency (L_MI): " << L_MI << "\n";
            }
            std::cout << "===========================================================\n";
        }
        // Debug tracking info
        mutable std::unordered_map<std::string, std::string> debug_info;

        static bool is_debug_enabled()
        {
            static bool debugEnvVar = read_debug_env_var(); //Used to cache the read.
            return debugEnvVar;
        }

        static bool is_heuristics_enabled()
        {
            static bool heuristicsEnvVar = read_heuristics_env_var(); //Used to cache the read.
            return heuristicsEnvVar;
        }

        void log_debug(const std::string& key, const std::string& value) const
        {
            debug_info[key] = value;
        }

        void log_debug(const std::string& key, double value) const
        {
            debug_info[key] = std::to_string(value);
        }

        void clear_debug() const
        {
            debug_info.clear();
        }

        void print_debug_info() const
        {
            std::cout << "=== Hardware Debug Info ===\n";
            for(const auto& [key, val] : debug_info)
            {
                std::cout << key << ": " << val << "\n";
            }
            std::cout << "===========================\n";
        }

        size_t get_mi_latency(size_t MI_M, size_t MI_N, size_t MI_K, data_type_t mi_input_type) const
        {
            const auto& instruction_map = INSTRUCTION_MAP.at(arch);
            auto        key             = matrix_instruction(MI_M, MI_N, MI_K, mi_input_type);

            auto it = instruction_map.find(key);
            if(it != instruction_map.end())
            {
                return it->second / parallel_mi_cu;
            }
            else
            {
                std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                            << ", MI_K=" << MI_K << ", mi_input_type=" << to_string(mi_input_type)
                            << ". Returning latency value of 32 (really slow).\n";
                return 32 / parallel_mi_cu; // Default latency if instruction is not found
            }
        }

    private:
        static std::string get_before_first_colon(const std::string& input)
        {
            size_t pos = input.find(':');
            if(pos != std::string::npos)
            {
                return input.substr(0, pos);
            }
            return input; // Return the whole string if ':' is not found
        }

        // Helper function to read the debug environment variable
        static bool read_debug_env_var()
        {
            const char* env = std::getenv("ANALYTICAL_GEMM_DEBUG");
            return env && std::string(env) == "1";
        }

        // Helper function to read the heuristics environment variable
        static bool read_heuristics_env_var()
        {
            const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS");
            return !(env && std::string(env) == "0");
        }
    };
} // namespace origami
