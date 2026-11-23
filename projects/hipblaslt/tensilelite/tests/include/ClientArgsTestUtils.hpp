// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Helpers that build the minimal command-line options a ClientProblemFactory needs, for tests that
// create factories (and the objects built from them) without parsing a command line.

#pragma once

#include "DataInitialization.hpp"
#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/FusedA2A.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>

#include <string>
#include <utility>
#include <vector>

namespace TensileLite::TestUtils
{
    template <typename T>
    void setOption(TensileLite::Client::po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

    inline TensileLite::Client::po::variables_map makeMinimalClientArgs(int mxScaleFormat)
    {
        using namespace TensileLite::Client;

        po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Ailk_Bljk_Cijk_Dijk"));
        setOption(args, "problem-size", std::vector<std::vector<size_t>>{{128, 128, 1, 512}});
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", false);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::Float8);
        setOption(args, "a-type", rocisa::DataType::Float8);
        setOption(args, "b-type", rocisa::DataType::Float8);
        setOption(args, "c-type", rocisa::DataType::Float);
        setOption(args, "d-type", rocisa::DataType::Float);
        setOption(args, "alpha-type", rocisa::DataType::Float);
        setOption(args, "beta-type", rocisa::DataType::Float);
        setOption(args, "compute-input-type-A", rocisa::DataType::Float8);
        setOption(args, "compute-input-type-B", rocisa::DataType::Float8);
        setOption(args, "f32-xdl-math-op", rocisa::DataType::Float);
        setOption(args, "activation-compute-type", rocisa::DataType::Float);
        // Deliberately 0 here (not 32): these only control whether
        // ClientProblemFactory calls setMXScaleA/B on the *internal dummy*
        // problem it builds for sizing purposes (see ClientProblemFactory.cpp),
        // which is unrelated to the `problem` object this test passes directly
        // into referenceNeedsPerSolutionRecompute(). When non-zero here, the
        // DataInitialization ctor's storage-geometry computation for the
        // swizzled MX-scale tensor indexes problem.freeIndicesA()[0] /
        // freeIndicesB()[0] on that dummy problem without a bounds check; for
        // the minimal synthetic problem-size/identifier used here those
        // indices can be empty, which segfaults (only reachable on real
        // hardware, so it stays hidden behind GTEST_SKIP() on any other arch).
        // m_mxScaleFormat (checked by referenceNeedsPerSolutionRecompute) is
        // read directly from "mx-scale-format" below, independent of these.
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "mx-scale-format", mxScaleFormat);
        setOption(args, "fused-a2a", TensileLite::FusedA2A::Disabled);
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensileLite::TensorOps{});
        setOption(args, "b-ops", TensileLite::TensorOps{});
        setOption(args, "c-ops", TensileLite::TensorOps{});
        setOption(args, "d-ops", TensileLite::TensorOps{});
        setOption(args, "use-gradient", false);
        setOption(args, "output-amaxD", false);
        setOption(args, "use-scaleAB", std::string());
        setOption(args, "use-scaleCD", false);
        setOption(args, "use-bias", 0);
        setOption(args, "bias-source", static_cast<int>(ContractionProblemGemm::TENSOR::D));
        setOption(args, "use-scaleAlphaVec", 0);
        setOption(args, "device-idx", 0);
        setOption(args, "num-elements-to-validate", 0);
        setOption(args, "pristine-on-gpu", true);
        setOption(args, "prune-mode", TensileLite::Client::PruneSparseMode::PruneRandom);
        setOption(args, "rotating-buffer-size", 0);
        setOption(args, "rotating-buffer-mode", 0);
        setOption(args, "bounds-check", TensileLite::Client::BoundsCheckMode::Disable);
        return args;
    }

    // Options for a float GEMM, D = A * B with A: i x l, B: l x j, C and D: i x j, of problem size
    // (M, N, batch, K), split across `splitDevices` devices along `splitDim` ("M" or "N"). A split
    // of 0 leaves the --split-devices options out.
    inline TensileLite::Client::po::variables_map
        makeSplitClientArgs(std::vector<size_t> size, int splitDevices, std::string const& splitDim)
    {
        auto args = makeMinimalClientArgs(0);
        for(const char* name : {"type",
                                "a-type",
                                "b-type",
                                "c-type",
                                "d-type",
                                "alpha-type",
                                "beta-type",
                                "compute-input-type-A",
                                "compute-input-type-B"})
            setOption(args, name, rocisa::DataType::Float);
        setOption(args, "problem-size", std::vector<std::vector<size_t>>{std::move(size)});
        if(splitDevices > 0)
        {
            setOption(args, "split-devices", splitDevices);
            setOption(args, "split-dim", splitDim);
        }
        return args;
    }

} // namespace TensileLite::TestUtils
