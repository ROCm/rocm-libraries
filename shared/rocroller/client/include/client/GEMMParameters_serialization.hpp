// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
    struct MappingTraits<Client::GEMMClient::XYTuple, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::XYTuple& x)
        {
            iot::mapRequired(io, "x", x.x);
            iot::mapRequired(io, "y", x.y);
        }

        static void mapping(IO& io, Client::GEMMClient::XYTuple& x, EmptyContext& ctx)
        {
            mapping(io, x);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::MNKTuple, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::MNKTuple& x)
        {
            iot::mapRequired(io, "m", x.m);
            iot::mapRequired(io, "n", x.n);
            iot::mapRequired(io, "k", x.k);
        }

        static void mapping(IO& io, Client::GEMMClient::MNKTuple& x, EmptyContext& ctx)
        {
            mapping(io, x);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::MNKBTuple, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::MNKBTuple& x)
        {
            iot::mapRequired(io, "m", x.m);
            iot::mapRequired(io, "n", x.n);
            iot::mapRequired(io, "k", x.k);
            iot::mapRequired(io, "b", x.b);
        }

        static void mapping(IO& io, Client::GEMMClient::MNKBTuple& x, EmptyContext& ctx)
        {
            mapping(io, x);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::GEMMClient::MKNLTuple, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::GEMMClient::MKNLTuple& x)
        {
            iot::mapRequired(io, "m", x.m);
            iot::mapRequired(io, "k", x.k);
            iot::mapRequired(io, "n", x.n);
            iot::mapRequired(io, "l", x.l);
        }

        static void mapping(IO& io, Client::GEMMClient::MKNLTuple& x, EmptyContext& ctx)
        {
            mapping(io, x);
        }
    };

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

            iot::mapRequired(io, "scaleShuffleTileA", params.scaleShuffleTileA);
            iot::mapRequired(io, "scaleShuffleTileB", params.scaleShuffleTileB);

            iot::mapRequired(io, "scalePreTileA", params.scalePretileA);
            iot::mapRequired(io, "scalePreTileB", params.scalePretileB);

            iot::mapRequired(io, "pretileA", params.pretileA);
            iot::mapRequired(io, "pretileB", params.pretileB);
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
            iot::mapRequired(io, "initMode_A", DGen::toString(params.initModeA));
            iot::mapRequired(io, "initMode_B", DGen::toString(params.initModeB));
            iot::mapRequired(io, "initMode_C", DGen::toString(params.initModeC));
            iot::mapRequired(io, "workgroupMappingDim", params.workgroupMappingDim);
        }

        static void
            mapping(IO& io, Client::GEMMClient::ProblemParameters& params, EmptyContext& ctx)
        {
            mapping(io, params);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::BenchmarkResults, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::BenchmarkResults& result)
        {
            iot::mapRequired(io, "device", result.benchmarkParams.device);
            iot::mapRequired(io, "workgroupMappingValue", result.runParams.workgroupMappingValue);
            iot::mapRequired(io, "numWGs", result.runParams.numWGs);
            iot::mapRequired(io, "numWarmUp", result.benchmarkParams.numWarmUp);
            iot::mapRequired(io, "numOuter", result.benchmarkParams.numOuter);
            iot::mapRequired(io, "numInner", result.benchmarkParams.numInner);
            iot::mapRequired(io, "kernelGenerate", result.kernelGenerate);
            iot::mapRequired(io, "kernelAssemble", result.kernelAssemble);
            iot::mapRequired(io, "kernelExecute", result.kernelExecute);
            iot::mapRequired(io, "checked", result.checked);
            iot::mapRequired(io, "correct", result.correct);
            iot::mapRequired(io, "rnorm", result.rnorm);
            iot::mapRequired(io, "sgprCount", result.sgprCount);
            iot::mapRequired(io, "vgprCount", result.vgprCount);
            iot::mapRequired(io, "agprCount", result.agprCount);
            iot::mapRequired(io, "ldsBytes", result.ldsBytes);
        }

        static void mapping(IO& io, Client::BenchmarkResults& result, EmptyContext& ctx)
        {
            mapping(io, result);
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
            iot::mapRequired(io, "problem", result.problemParams);
            iot::mapRequired(io, "solution", result.solutionParams);
            iot::mapRequired(io, "benchmark", result.benchmarkResults);
        }

        static void mapping(IO& io, Client::GEMMClient::Result& result, EmptyContext& ctx)
        {
            mapping(io, result);
        }
    };

    template <typename IO>
    struct MappingTraits<Client::RunParameters, IO, EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Client::RunParameters& params)
        {
            iot::mapRequired(io, "workgroupMappingValue", params.workgroupMappingValue);
            iot::mapRequired(io, "numWGs", params.numWGs);
        }

        static void mapping(IO& io, Client::RunParameters& params, EmptyContext& ctx)
        {
            mapping(io, params);
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
            iot::mapRequired(io, "workgroupMappingDim", params.workgroupMappingDim);
            iot::mapRequired(io, "workgroupRemapXCC", params.workgroupRemapXCC);
            iot::mapRequired(io, "workgroupRemapXCCValue", params.workgroupRemapXCCValue);
            iot::mapRequired(io, "workgroup_cluster_size_x", params.workgroupClusterSizeX);
            iot::mapRequired(io, "workgroup_cluster_size_y", params.workgroupClusterSizeY);
            iot::mapRequired(io, "workgroup_cluster_size_z", params.workgroupClusterSizeZ);
            iot::mapRequired(io, "load_A", params.loadPathA);
            iot::mapRequired(io, "load_B", params.loadPathB);
            iot::mapRequired(io, "padLDS_A", params.padLDSA);
            iot::mapRequired(io, "padLDS_B", params.padLDSB);
            iot::mapRequired(io, "store", params.storePath);
            iot::mapRequired(io, "prefetch", params.prefetch);
            iot::mapRequired(io, "prefetchInFlight", params.prefetchInFlight);
            iot::mapRequired(io, "prefetchLDSFactor", params.prefetchLDSFactor);
            iot::mapRequired(io, "prefetchMixMemOps", params.prefetchMixMemOps);
            iot::mapRequired(io, "betaInFma", params.betaInFma);
            iot::mapRequired(io, "scheduler", params.scheduler);
            iot::mapRequired(io, "schedulerCost", params.schedulerCost);

            iot::mapRequired(io, "tailLoops", params.tailLoops);

            iot::mapRequired(io, "types", params.types);

            iot::mapRequired(io, "loadScale_A", params.loadPathAScale);
            iot::mapRequired(io, "loadScale_B", params.loadPathBScale);
            iot::mapRequired(io, "swizzleScale", params.swizzleScale);
            iot::mapRequired(io, "swizzleTileSize", params.swizzleTileSize);
            iot::mapRequired(io, "prefetchScale", params.prefetchScale);

            iot::mapRequired(io, "streamK", params.streamK);

            iot::mapOptional(io, "version", params.version);
        }

        static void
            mapping(IO& io, Client::GEMMClient::SolutionParameters& params, EmptyContext& ctx)
        {
            mapping(io, params);
        }
    };
}
