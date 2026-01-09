/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <origami/origami.hpp>
#include <origami/formocast_predict.hpp>
#include <rocisa/include/enum.hpp>
#include <formocast_simulator.hpp>

namespace TensileLite
{

    inline origami::data_type_t datatypeToAnalyticalDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return origami::data_type_t::Float;
        case rocisa::DataType::Double:
            return origami::data_type_t::Double;
        case rocisa::DataType::ComplexFloat:
            return origami::data_type_t::ComplexFloat;
        case rocisa::DataType::ComplexDouble:
            return origami::data_type_t::ComplexDouble;
        case rocisa::DataType::Half:
            return origami::data_type_t::Half;
        case rocisa::DataType::Int8x4:
            return origami::data_type_t::Int8x4;
        case rocisa::DataType::Int32:
            return origami::data_type_t::Int32;
        case rocisa::DataType::BFloat16:
            return origami::data_type_t::BFloat16;
        case rocisa::DataType::Int8:
            return origami::data_type_t::Int8;
        case rocisa::DataType::Int64:
            return origami::data_type_t::Int64;
        case rocisa::DataType::XFloat32:
            return origami::data_type_t::XFloat32;
        case rocisa::DataType::Float8_fnuz:
            return origami::data_type_t::Float8_fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return origami::data_type_t::BFloat8_fnuz;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return origami::data_type_t::Float8BFloat8_fnuz;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return origami::data_type_t::BFloat8Float8_fnuz;
        case rocisa::DataType::Float8:
            return origami::data_type_t::Float8;
        case rocisa::DataType::BFloat8:
            return origami::data_type_t::BFloat8;
        case rocisa::DataType::Float8BFloat8:
            return origami::data_type_t::Float8BFloat8;
        case rocisa::DataType::BFloat8Float8:
            return origami::data_type_t::BFloat8Float8;

        default:
            return origami::data_type_t::None;
        }
    }

    // Convert rocisa::DataType to Tensilelite::DataType for Formocast
    inline Tensilelite::DataType datatypeToFormocastDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return Tensilelite::DataType::Float;
        case rocisa::DataType::Double:
            return Tensilelite::DataType::Double;
        case rocisa::DataType::Half:
            return Tensilelite::DataType::Half;
        case rocisa::DataType::Int32:
            return Tensilelite::DataType::Int32;
        case rocisa::DataType::BFloat16:
            return Tensilelite::DataType::BFloat16;
        case rocisa::DataType::Int8:
            return Tensilelite::DataType::Int8;
        case rocisa::DataType::XFloat32:
            return Tensilelite::DataType::TF32;

        default:
            return Tensilelite::DataType::Unknown;
        }
    }
    
    // Convert origami::hardware_t::architecture_t to Tensilelite::HardwareArchitecture
    inline Tensilelite::HardwareArchitecture origamiArchToFormocastArch(origami::hardware_t::architecture_t arch)
    {
        switch(arch)
        {
        case origami::hardware_t::architecture_t::gfx950:
            return Tensilelite::HardwareArchitecture::gfx950;
        case origami::hardware_t::architecture_t::gfx942:
            return Tensilelite::HardwareArchitecture::gfx942;
        case origami::hardware_t::architecture_t::gfx1201:
            return Tensilelite::HardwareArchitecture::gfx1201;
        default:
            return Tensilelite::HardwareArchitecture::Unknown;
        }
    }

    // Convert origami::data_type_t to Tensilelite::DataType
    inline Tensilelite::DataType origamiDatatypeToFormocastDatatype(origami::data_type_t type)
    {
        switch(type)
        {
        case origami::data_type_t::Float:
            return Tensilelite::DataType::Float;
        case origami::data_type_t::Double:
            return Tensilelite::DataType::Double;
        case origami::data_type_t::Half:
            return Tensilelite::DataType::Half;
        case origami::data_type_t::Int32:
            return Tensilelite::DataType::Int32;
        case origami::data_type_t::BFloat16:
            return Tensilelite::DataType::BFloat16;
        case origami::data_type_t::Int8:
            return Tensilelite::DataType::Int8;
        case origami::data_type_t::XFloat32:
            return Tensilelite::DataType::TF32;
        default:
            return Tensilelite::DataType::Unknown;
        }
    }

    // Convert Tensilelite::DataType to origami::data_type_t
    inline origami::data_type_t formocastDatatypeToOrigamiDatatype(Tensilelite::DataType type)
    {
        switch(type)
        {
        case Tensilelite::DataType::Float:
            return origami::data_type_t::Float;
        case Tensilelite::DataType::Double:
            return origami::data_type_t::Double;
        case Tensilelite::DataType::Half:
            return origami::data_type_t::Half;
        case Tensilelite::DataType::Int32:
            return origami::data_type_t::Int32;
        case Tensilelite::DataType::BFloat16:
            return origami::data_type_t::BFloat16;
        case Tensilelite::DataType::Int8:
            return origami::data_type_t::Int8;
        case Tensilelite::DataType::TF32:
            return origami::data_type_t::XFloat32;
        default:
            return origami::data_type_t::None;
        }
    }

    // Convert Formocast::IntermediatePerformanceMetrics to origami::IntermediatePerformanceMetrics
    inline origami::IntermediatePerformanceMetrics convertIntermediateMetrics(
        const Tensilelite::Formocast::IntermediatePerformanceMetrics& src)
    {
        origami::IntermediatePerformanceMetrics dst;
        // Cache hit rates
        dst.tile0_L1_hit = src.cache_hits.A_L1_hit;
        dst.tile1_L1_hit = src.cache_hits.B_L1_hit;
        dst.tile0_L2_hit = src.cache_hits.A_L2_hit;
        dst.tile1_L2_hit = src.cache_hits.B_L2_hit;
        dst.tile0_L3_hit = src.cache_hits.A_L3_hit;
        dst.tile1_L3_hit = src.cache_hits.B_L3_hit;
        dst.totalL2HitRate = src.cache_hits.totalL2HitRate;
        dst.totalL3HitRate = src.cache_hits.totalL3HitRate;
        // Other metrics
        dst.output_write_cost = src.output_write_cost;
        dst.output_write_cost_edge = src.output_write_cost_edge;
        dst.split_accumulation_overhead = src.split_accumulation_overhead;
        dst.compute_cycles = src.compute_cycles;
        dst.local_split_overhead = src.local_split_overhead;
        dst.tile0_l1_request = src.tile0_l1_request;
        dst.tile0_l2_request = src.tile0_l2_request;
        dst.tile0_l3_request = src.tile0_l3_request;
        dst.tile0_mem_request = src.tile0_mem_request;
        dst.tile1_l1_request = src.tile1_l1_request;
        dst.tile1_l2_request = src.tile1_l2_request;
        dst.tile1_l3_request = src.tile1_l3_request;
        dst.tile1_mem_request = src.tile1_mem_request;
        dst.prefetch_cost = src.prefetch_cost;
        dst.startup_cost = src.startup_cost;
        return dst;
    }

    // Convert Formocast::ProblemInfo to origami::ProblemInfo
    inline origami::ProblemInfo convertProblemInfo(const Tensilelite::Formocast::ProblemInfo& src)
    {
        origami::ProblemInfo dst;
        dst.M = src.M;
        dst.N = src.N;
        dst.NumBatches = src.NumBatches;
        dst.K = src.K;
        dst.bpeA = src.bpeA;
        dst.bpeB = src.bpeB;
        dst.bpeD = src.bpeD;
        dst.bpeCompute = src.bpeCompute;
        dst.transA = src.transA;
        dst.transB = src.transB;
        dst.swizzleTensorA = src.swizzleTensorA;
        dst.swizzleTensorB = src.swizzleTensorB;
        dst.dataType = formocastDatatypeToOrigamiDatatype(src.dataType);
        return dst;
    }

    // Convert Formocast::SizeMapping to origami::ConfigMapping
    inline origami::ConfigMapping convertSizeMapping(const Tensilelite::Formocast::SizeMapping& src)
    {
        origami::ConfigMapping dst;
        dst.macroTile.x = src.macroTile.x;
        dst.macroTile.y = src.macroTile.y;
        dst.workGroupMapping = src.workGroupMapping;
        dst.CUOccupancy = src.CUOccupancy;
        dst.depthU = src.depthU;
        dst.globalSplitU = src.globalSplitU;
        dst.gwvwD = src.gwvwD;
        dst.PrefetchGlobalRead = src.PrefetchGlobalRead;
        return dst;
    }

} // namespace TensileLite
