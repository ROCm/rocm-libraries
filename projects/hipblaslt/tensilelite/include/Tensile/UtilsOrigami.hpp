/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <origami/origami.hpp>
#include <rocisa/include/enum.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>

#include <Tensile/Debug.hpp>

#include <tensilelitehost/export.h>

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
        case rocisa::DataType::Float6:
            return origami::data_type_t::Float6;
        case rocisa::DataType::BFloat6:
            return origami::data_type_t::BFloat6;   
        case rocisa::DataType::Float4:
            return origami::data_type_t::Float4;

        default:
            throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(type)));
        }
    }

    // Build an Origami analytical config from a solution's size mapping. The
    // resulting config carries `index` so callers can map an origami ranking
    // result back to the originating solution. Kept templated (rather than
    // taking ContractionSolution directly) so UtilsOrigami.hpp stays free of
    // heavy Tensile headers and avoids include cycles.
    template <typename Solution>
    inline origami::config_t makeOrigamiConfig(const Solution& solution, int index)
    {
        origami::dim3_t origami_mi;
        if(solution.sizeMapping.matrixInstruction[0] == 0
           && solution.sizeMapping.matrixInstruction[1] == 0
           && solution.sizeMapping.matrixInstruction[2] == 0)
        {
            // Override dot2 instruction with vector lane widths
            origami_mi = {1, 1, 64};
        }
        else
        {
            origami_mi = {static_cast<size_t>(solution.sizeMapping.matrixInstruction[0]),
                          static_cast<size_t>(solution.sizeMapping.matrixInstruction[1]),
                          static_cast<size_t>(solution.sizeMapping.matrixInstruction[2])};
        }

        if(Debug::Instance().printPropertyEvaluation() && solution.sizeMapping.CUOccupancy <= 0)
        {
            std::cerr << "TensileLite::DEBUG: sizeMapping.CUOccupancy="
                      << solution.sizeMapping.CUOccupancy << " (<=0) for solution '"
                      << solution.kernelName << "'; clamping to 1 in origami config.\n";
        }

        origami::config_t origami_config = {
            .mt = {solution.sizeMapping.macroTile.x,
                   solution.sizeMapping.macroTile.y,
                   solution.sizeMapping.depthU},
            .mi                       = origami_mi,
            .hand_optimized_main_loop = (solution.sizeMapping.customMainLoopScheduling > 0) ? true
                                                                                            : false,
            .subtile                  = solution.sizeMapping.useSubtileImpl,
            .occupancy = std::max(solution.sizeMapping.CUOccupancy, static_cast<int>(1)),
            .workgroup_mapping         = solution.sizeMapping.workGroupMapping,
            .cache_hints_a             = solution.sizeMapping.nonTemporalA,
            .cache_hints_b             = solution.sizeMapping.nonTemporalB,
            .workspace_size            = std::numeric_limits<size_t>::max(),
            .workspace_size_per_elem_c = std::numeric_limits<size_t>::max(),
            .index                     = static_cast<size_t>(index),
        };

        return origami_config;
    }

    // Build an Origami analytical problem descriptor from a Tensile problem.
    template <typename Problem>
    inline origami::problem_t makeOrigamiProblem(const Problem& problem)
    {
        size_t m     = 1;
        size_t n     = 1;
        size_t k     = 1;
        size_t batch = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            m *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            n *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            k *= problem.boundSize(i);
        for(size_t i = 0; i < problem.batchIndices().size(); ++i)
            batch *= problem.batchSize(i);

        auto miDataType = datatypeToAnalyticalDatatype(problem.computeInputTypeA());
        if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32) // Check F32 compute type
            miDataType = origami::data_type_t::XFloat32;

        origami::problem_t origami_problem = {
            .size  = {m, n, k},
            .batch = batch,
            // CU budget hint; 0 = use all CUs.
            .num_cus     = static_cast<size_t>(problem.getParams().smCountTarget()),
            .a_transpose = problem.transA() ? origami::transpose_t::T : origami::transpose_t::N,
            .b_transpose = problem.transB() ? origami::transpose_t::T : origami::transpose_t::N,
            .a_dtype     = datatypeToAnalyticalDatatype(problem.a().dataType()),
            .b_dtype     = datatypeToAnalyticalDatatype(problem.b().dataType()),
            .c_dtype     = datatypeToAnalyticalDatatype(problem.c().dataType()),
            .d_dtype     = datatypeToAnalyticalDatatype(problem.d().dataType()),
            .mi_dtype    = miDataType,
            .a_mx_block_size = 0, // MX Data types come from rocroller
            .b_mx_block_size = 0, // MX Data types come from rocroller
        };

        return origami_problem;
    }

    // Assemble the analytical hardware descriptor used for ranking. This mirrors
    // the prediction library, which ranks against the full analytical hardware.
    // (A StreamK CU cap is layered on separately by the skMaxCUs feature and is
    // intentionally not applied here so this path stays decoupled from it.)
    template <typename GPU>
    inline origami::hardware_t makeRankingHardware(const GPU& gpu)
    {
        return *(gpu.analyticalHardware);
    }
} // namespace TensileLite

