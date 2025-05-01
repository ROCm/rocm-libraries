/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Tensile/Tensile.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    struct PerfModel
    {
        double clock            = std::numeric_limits<double>::quiet_NaN();
        double memClock         = std::numeric_limits<double>::quiet_NaN();
        double peakGFlops       = std::numeric_limits<double>::quiet_NaN();
        double memBandwidthMBps = std::numeric_limits<double>::quiet_NaN();
        double l2ReadBwMul      = std::numeric_limits<double>::quiet_NaN();
        double gFlops           = std::numeric_limits<double>::quiet_NaN();
        double readEff          = 0.0;
        double l2ReadHitRate    = 0.0;
        double l2WriteHitRate   = 0.0;
        int    CUs              = 0;
    };

    extern PerfModel perf;

    struct BufferLoadCheckPacket
    {
        size_t shiftPtrElemA;
        size_t shiftPtrElemB;
        size_t depthUorMT0;
        size_t depthUorMT1;
    };

    struct SizeMapping
    {
        dim3 workGroupSize;
        dim3 threadTile;
        dim3 macroTile;

        size_t staggerU           = 0;
        size_t depthU             = 0;
        size_t globalSplitU       = 0;
        size_t staggerStrideShift = 0;
        int    workGroupMapping   = 0;

        size_t packBatchDims              = 0;
        int    packSummationDims          = 0;
        int    magicDivAlg                = 1;
        int    streamK                    = 0;
        int    streamKAtomic              = 0;
        int    persistentKernel           = 0;
        bool   persistentKernelAlongBatch = false;

        bool   sourceKernel          = false;
        int    preloadKernargs       = 0;
        int    globalAccumulation    = 0;
        size_t workspaceSizePerElemC = 0;
    };

    std::ostream& operator<<(std::ostream& stream, const SizeMapping& sizeMapping);

    /**
 * Represents a single kernel or set of kernels that can perform a single
 * tensor contraction.
 *
 * Can generate `KernelInvocation` objects to solve a particular problem
 * given a set of `ContractionInputs`.
 */
    class ContractionSolution : public Solution
    {
    public:
        using Problem = ContractionProblem;
        using Inputs  = ContractionInputs;

        static std::string Type()
        {
            return "Contraction";
        }
        virtual std::string type() const
        {
            return Type();
        }

        virtual std::string KernelName() const
        {
            return kernelName;
        }

        virtual std::string name() const
        {
            return kernelName;
        }
        virtual std::string description() const
        {
            return kernelName;
        }

        bool getMatrixInstructionFromKernelName(vector4<std::uint32_t>& miInst) const;
        bool getGSUAlgorithmFromKernelName(std::string& gsuAlg) const;

        bool isSourceKernel() const;

        //! Estimates based on problem size, solution tile, and  machine hardware
        //! charz:
        struct StaticPerformanceModel
        {
            size_t memReadBytesA   = 0.0; //! Estimated memory reads A
            size_t memReadBytesB   = 0.0; //! Estimated memory reads B
            size_t memReadBytesC   = 0.0; //! Estimated memory reads C
            size_t memWriteBytesD  = 0.0; //! Estimated memory writes D
            size_t memReadBytes    = 0.0;
            size_t memGlobalReads  = 0;
            size_t memGlobalWrites = 0;
        };

        struct Granularities
        {
            double numTiles0  = 0.0; //! number of tiles in 0 dimension
            double numTiles1  = 0.0; //! number of tiles in 1 dimension
            double totalTiles = 0.0;
            double tilesPerCu = 0.0;

            //! Granularity is measured 0..1 with 1.0 meaning no granularity loss
            double tile0Granularity          = 0.0; // loss due to tile0
            double tile1Granularity          = 0.0;
            double cuGranularity             = 0.0;
            double waveGranularity           = 0.0;
            double totalGranularity          = 0.0;
            double totalTileAwareGranularity = 0.0;
            double natCuGranularity          = 0.0;
            double natTilesPerCu             = 0.0;
            double suTilesPerCu              = 0.0;
            double suCuGranularity           = 0.0;
            double waves                     = 0.0;
            double suWavesPerSimdx2          = 0.0;
            double suWaveGranularity         = 0.0;

            int CUs = 0;

            double MT0;
            double MT1;
            double GSU;
            double LSU;
        };

        struct ProjectedPerformance
        {
            Granularities granularities;

            double speedGFlops = 0.0; //! final gflops projection
            int    CUs         = 0;

            StaticPerformanceModel staticModel;
        };

        struct TAMetricProblemScore
        {
            Granularities granularites;

            int CUs = 0;

            double summationPerformance = 0.0;

            double M;
            double N;
            double K;
        };

        /**
        * Calculate required workspace size.
        */

        size_t requiredWorkspaceSize(Problem const& problem, Hardware const& hardware) const;
        int    predictNumFullTiles(Problem const&  problem,
                                   Hardware const& hardware,
                                   size_t          tiles,
                                   size_t          skGrid) const;
        size_t getSKGrid(Problem const& problem, Hardware const& hardware, size_t tiles) const;
        size_t partialTileSize(size_t skGrid) const;
        static float computeGranularity(float x);

        Granularities computeGranularities(
            Hardware const& hardware, double M, double N, double K, double NumBatches) const;

        StaticPerformanceModel staticPerformanceModel(double M,
                                                      double N,
                                                      double K,
                                                      double NumBatches,
                                                      double MT0,
                                                      double MT1,
                                                      double NumCUs,
                                                      double totalGranularity,
                                                      int    globalSplitU) const;

        TAMetricProblemScore computeProblemScore(
            Hardware const& hardware, double M, double N, double K, double NumBatches) const;

        double computeTileAwareMetric(TAMetricProblemScore pp,
                                      TAMetricProblemScore ppReference) const;

        double computeTAMScore(Problem const&  problem,
                               Hardware const& hardware,
                               double          model_M,
                               double          model_N,
                               double          model_K,
                               double          model_NumBatches) const;

        /**
   * Calculate the projected performance based on granularity loss.
   */
        ProjectedPerformance projectedPerformance(Problem const&  problem,
                                                  Hardware const& hardware) const;

        /**
   * Generate a set of kernel calls to solve a particular problem.
   */
        virtual std::vector<KernelInvocation>
            solve(Problem const& problem, Inputs const& inputs, Hardware const& hardware) const;

        template <typename TypedInputs>
        std::vector<KernelInvocation> solveTyped(Problem const&     problem,
                                                 TypedInputs const& inputs,
                                                 Hardware const&    hardware) const;

        template <typename TypedInputs, bool T_Debug>
        KernelInvocation generateSingleCall(Problem const&     problem,
                                            TypedInputs const& inputs,
                                            Hardware const&    hardware) const;

        template <typename TypedInputs, bool T_Debug>
        KernelInvocation generateStreamKInitCall(Problem const&     problem,
                                                 TypedInputs const& inputs,
                                                 Hardware const&    hardware) const;

        template <typename TypedInputs>
        std::string streamKInitKernelName(Problem const&     problem,
                                          TypedInputs const& inputs,
                                          Hardware const&    hardware) const;

        template <typename TypedInputs, bool T_Debug>
        KernelInvocation generateBetaOnlyCall(Problem const&     problem,
                                              TypedInputs const& inputs,
                                              Hardware const&    hardware) const;

        template <typename TypedInputs>
        std::string betaOnlyKernelName(Problem const&     problem,
                                       TypedInputs const& inputs,
                                       Hardware const&    hardware) const;

        template <typename TypedInputs, bool T_Debug>
        KernelInvocation generateOutputConversionCall(Problem const&     problem,
                                                      TypedInputs const& inputs,
                                                      Hardware const&    hardware) const;

        template <typename TypedInputs>
        std::string outputConversionKernelName(Problem const&     problem,
                                               TypedInputs const& inputs,
                                               int                gsu,
                                               int                vw,
                                               int                reduction,
                                               int                gsuUnrollUnit,
                                               Hardware const&    hardware) const;

        bool canSolve(Problem const& problem, Hardware const& hardware) const;

        bool matchesProblemType(Problem const& problem, Hardware const& hardware) const;

        struct ProblemType
        {
            std::string operationIdentifier;
            DataType    aType                   = DataType::Float;
            DataType    bType                   = DataType::Float;
            DataType    cType                   = DataType::Float;
            DataType    dType                   = DataType::Float;
            bool        highPrecisionAccumulate = false;
            bool        useBeta                 = true;
            bool        useInitialStridesAB     = false;
            bool        useInitialStridesCD     = false;
            bool        stridedBatched          = true;
            bool        fp16AltImpl             = false;
            bool        fp16AltImplRound        = false;
            bool        stochasticRounding      = false;
            DataType    f32XdlMathOp            = DataType::Float;
        };

        struct LinearModel
        {
            double slope     = 1.0;
            double intercept = 0.0;
            double max       = 1000.0;
        };

        int                          index = 0;
        std::string                  kernelName;
        ThreadSafeValue<std::string> codeObjectFilename;
        bool                         debugKernel   = false;
        bool                         kernelArgsLog = false;

        std::shared_ptr<Predicates::Predicate<Task>> taskPredicate
            = std::make_shared<Predicates::True<Task>>();
        std::shared_ptr<Predicates::Predicate<Problem>> problemPredicate
            = std::make_shared<Predicates::True<Problem>>();
        std::shared_ptr<Predicates::Predicate<Hardware>> hardwarePredicate
            = std::make_shared<Predicates::True<Hardware>>();

        SizeMapping sizeMapping;

        ProblemType problemType;

        /// Debugging purposes.  Shouldn't contain any vital information that isn't
        /// somewhere else.
        int32_t               libraryLogicIndex = -1;
        std::map<int, double> ideals;
        LinearModel           linearModel;

        int32_t staggerUIter(Problem const&  problem,
                             Inputs const&   inputs,
                             Hardware const& hardware) const;

        uint32_t magicNumberAlg1(uint32_t x, uint32_t* magicShift) const;
        uint32_t magicNumberAlg2(uint32_t x, uint32_t* magicShift) const;
        uint32_t magicNumber(int magicDivAlg, uint32_t x, uint32_t* magicShift) const;
        uint32_t smallMagicNumber(uint32_t x) const;
    };

    std::ostream& operator<<(std::ostream&                                      stream,
                             ContractionSolution::StaticPerformanceModel const& spm);
    std::ostream& operator<<(std::ostream&                                    stream,
                             ContractionSolution::ProjectedPerformance const& spm);
    std::ostream& operator<<(std::ostream& stream, BufferLoadCheckPacket const& st);
} // namespace Tensile
