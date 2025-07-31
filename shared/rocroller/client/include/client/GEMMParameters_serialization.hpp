/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include "GEMMParameters.hpp"

#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/GPUArchitecture.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace rocRoller::Serialization
{

    template <typename T, typename IO>
    void flatMap(IO& io, T& object)
    {
        MappingTraits<T, IO, EmptyContext>::mapping(io, object);
    }

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::TypeParameters, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::TypeParameters& params)
        {
            iot::mapRequired(io, "type_A", params.typeA);
            iot::mapRequired(io, "type_B", params.typeB);
            iot::mapRequired(io, "type_C", params.typeC);
            iot::mapRequired(io, "type_D", params.typeD);
            iot::mapRequired(io, "type_acc", params.typeAcc);

            iot::mapRequired(io, "trans_A", params.transA);
            iot::mapRequired(io, "trans_B", params.transB);

            iot::mapRequired(io, "scale_A", params.scaleA);
            iot::mapRequired(io, "scaleType_A", params.scaleTypeA);
            iot::mapRequired(io, "scale_B", params.scaleB);
            iot::mapRequired(io, "scaleType_B", params.scaleTypeB);

            iot::mapRequired(io, "scaleBlockSize", params.scaleBlockSize);
            iot::mapRequired(io, "scaleSkipPermlane", params.scaleSkipPermlane);
        }

        static void mapping(IO& io, Client::GEMMClient::TypeParameters& params, EmptyContext& ctx)
        {
            mapping(io, params);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::ProblemParameters, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::ProblemParameters& params)
        {
            iot::mapRequired(io, "M", params.m);
            iot::mapRequired(io, "N", params.n);
            iot::mapRequired(io, "K", params.k);
            iot::mapRequired(io, "alpha", params.alpha);
            iot::mapRequired(io, "beta", params.beta);
            iot::mapRequired(io, "types", params.types);
            iot::mapRequired(io, "scaleValue_A", params.scaleValueA);
            iot::mapRequired(io, "scaleValue_B", params.scaleValueB);
            iot::mapRequired(io, "workgroupMapping", params.workgroupMapping);
        }

        static void
            mapping(IO& io, Client::GEMMClient::ProblemParameters& params, EmptyContext& ctx)
        {
            mapping(io, params);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::Result, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::Result& result)
        {
            iot::mapRequired(io, "resultType", result.benchmarkResults.resultType);
            iot::mapRequired(io, "device", result.benchmarkResults.runParams.device);

            flatMap(io, result.problemParams);
            flatMap(io, result.solutionParams);

            iot::mapRequired(io, "numWGs", result.benchmarkResults.runParams.numWGs);

            iot::mapRequired(io, "numWarmUp", result.benchmarkResults.runParams.numWarmUp);
            iot::mapRequired(io, "numOuter", result.benchmarkResults.runParams.numOuter);
            iot::mapRequired(io, "numInner", result.benchmarkResults.runParams.numInner);

            iot::mapRequired(io, "kernelGenerate", result.benchmarkResults.kernelGenerate);
            iot::mapRequired(io, "kernelAssemble", result.benchmarkResults.kernelAssemble);
            iot::mapRequired(io, "kernelExecute", result.benchmarkResults.kernelExecute);

            iot::mapRequired(io, "checked", result.benchmarkResults.checked);
            iot::mapRequired(io, "correct", result.benchmarkResults.correct);
            iot::mapRequired(io, "rnorm", result.benchmarkResults.rnorm);
        }

        static void mapping(IO& io, Client::GEMMClient::Result& result, EmptyContext& ctx)
        {
            mapping(io, result);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::SolutionParameters, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::SolutionParameters& params)
        {
            iot::mapRequired(io, "architecture", params.architecture);

            iot::mapRequired(io, "mac_m", params.macM);
            iot::mapRequired(io, "mac_n", params.macN);
            iot::mapRequired(io, "mac_k", params.macK);
            iot::mapRequired(io, "wave_m", params.waveM);
            iot::mapRequired(io, "wave_n", params.waveN);
            iot::mapRequired(io, "wave_k", params.waveK);
            iot::mapRequired(io, "wave_b", params.waveB);
            iot::mapRequired(io, "workgroup_size_x", params.workgroupSizeX);
            iot::mapRequired(io, "workgroup_size_y", params.workgroupSizeY);
            iot::mapRequired(io, "workgroupMapping", params.workgroupMapping);
            iot::mapRequired(io, "workgroupRemapXCC", params.workgroupRemapXCC);
            iot::mapRequired(io, "workgroupRemapXCCValue", params.workgroupRemapXCCValue);
            iot::mapRequired(io, "unroll_x", params.unrollX);
            iot::mapRequired(io, "unroll_y", params.unrollY);
            iot::mapRequired(io, "loadLDS_A", params.loadLDSA);
            iot::mapRequired(io, "loadLDS_B", params.loadLDSB);
            iot::mapRequired(io, "storeLDS_D", params.storeLDSD);
            iot::mapRequired(io, "direct2LDS_A", params.direct2LDSA);
            iot::mapRequired(io, "direct2LDS_B", params.direct2LDSB);
            iot::mapRequired(io, "prefetch", params.prefetch);
            iot::mapRequired(io, "prefetchInFlight", params.prefetchInFlight);
            iot::mapRequired(io, "prefetchLDSFactor", params.prefetchLDSFactor);
            iot::mapRequired(io, "prefetchMixMemOps", params.prefetchMixMemOps);
            iot::mapRequired(io, "betaInFma", params.betaInFma);
            iot::mapRequired(io, "scheduler", params.scheduler);
            iot::mapRequired(io, "matchMemoryAccess", params.matchMemoryAccess);

            iot::mapRequired(io, "types", params.types);

            iot::mapRequired(io, "loadLDSScale_A", params.loadLDSScaleA);
            iot::mapRequired(io, "loadLDSScale_B", params.loadLDSScaleB);
            iot::mapRequired(io, "swizzleScale", params.swizzleScale);
            iot::mapRequired(io, "prefetchScale", params.prefetchScale);

            iot::mapRequired(io, "streamK", params.streamK);
            iot::mapRequired(io, "streamKTwoTile", params.streamKTwoTile);

            iot::mapOptional(io, "version", params.version);
        }

        static void
            mapping(IO& io, Client::GEMMClient::SolutionParameters& params, EmptyContext& ctx)
        {
            mapping(io, params);
        }
    };
}
