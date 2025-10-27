/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Activation.hpp>
#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Utils.hpp>

#include <origami/streamk.hpp>

#define TENSILE_COMMON_KERNEL_ARGS_SIZE 16

namespace TensileLite
{
    template <typename TAct>
    struct DeviceUserArguments
    {
        uint32_t m;
        uint32_t n;
        uint32_t batch;
        uint32_t k;
        void*    d;
        void*    c;
        void*    a;
        void*    b;
        uint32_t strideD1;
        uint32_t strideD2;
        uint32_t strideC1;
        uint32_t strideC2;
        uint32_t strideA1;
        uint32_t strideA2;
        uint32_t strideB1;
        uint32_t strideB2;
        int8_t   alpha[16];
        int8_t   beta[16];
        void*    scaleA;
        void*    scaleB;
        void*    scaleC;
        void*    scaleD;
        void*    scaleAlphaVec;
        void*    bias;
        int      biasType;
        uint32_t reserved;
        void*    e;
        uint32_t strideE1;
        uint32_t strideE2;
        TAct     act0;
        TAct     act1;
        int      activationType;
    } __attribute__((packed));

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
        size_t waveNum;

        dim3 workGroupSize;
        dim3 threadTile;
        dim3 macroTile;

        std::array<int, 4> matrixInstruction;
        size_t             grvwA = 1;
        size_t             grvwB = 1;
        size_t             gwvwC = 1;
        size_t             gwvwD = 1;

        size_t  staggerU           = 0;
        size_t  staggerUMapping    = 0;
        size_t  depthU             = 0;
        size_t  globalSplitUPGR    = 0;
        int16_t globalSplitU       = 0;
        size_t  staggerStrideShift = 0;
        int     workGroupMapping   = 0;

        size_t packBatchDims              = 0;
        int    packSummationDims          = 0;
        int    magicDivAlg                = 1;
        int    streamK                    = 0;
        int    streamKAtomic              = 0;
        int    persistentKernel           = 0;
        bool   persistentKernelAlongBatch = false;

        bool sourceKernel = false;

        int    globalAccumulation       = 0;
        size_t workspaceSizePerElemC    = 0;
        size_t workspaceSizePerElemBias = 0;

        bool activationFused = true;

        std::string customKernelName;

        int  workGroupMappingXCC                    = 0;
        int  workGroupMappingXCCGroup               = 0;
        bool globalSplitUCoalesced                  = false;
        bool globalSplitUWorkGroupMappingRoundRobin = false;

        int CUOccupancy            = 0;
        int PrefetchGlobalRead     = 2;
        int MathClocksUnrolledLoop = 0;

        size_t synchronizerSizePerWG = 0;

        int nonTemporalA = 0;
        int nonTemporalB = 0;
    };

    struct StreamKSettings
    {
        origami::streamk::reduction_type reduction = origami::streamk::reduction_type::Tree;
        size_t grid = 0;
    };

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
        using Problem       = ContractionProblemGemm;
        using Inputs        = ContractionInputs;
        using GroupedInputs = ContractionGroupedInputs;

        /**
         * Indicate a solution is equally or estimatedly matched.
         */
        enum class MatchingTag
        {
            Equal,
            Estimated
        };

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
            return solutionName;
        }
        virtual std::string description() const
        {
            return kernelName;
        }
        virtual bool isFallbackForHW(Hardware const&) const;

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

        bool checkInternalArgumentsSupport(ContractionProblem const& problem,
                                           std::ostream&             stream,
                                           bool                      debug = false) const;

        /**
         * Calculate required workspace size.
         */
        size_t requiredWorkspaceSize(Problem const& problem, Hardware const& hardware) const;
        size_t requiredWorkspaceSizeGroupedGemm(std::vector<Problem> const& problems,
                                                Hardware const&             hardware) const;
        size_t requiredHostSizeGroupedGemmSingle(Problem const&  problem,
                                                 Hardware const& hardware) const;

        size_t requiredSynchronizerSize(Problem const& problem, Hardware const& hardware) const;

        origami::streamk::reduction_type getSKReduction(Problem const& problem, Hardware const& hardware) const;
        size_t getSKGrid(Problem const& problem, Hardware const& hardware, size_t tiles, origami::streamk::reduction_type reductionStrat) const;
        size_t partialTileSize(size_t skGrid) const;

        static float computeGranularity(float x);

        Granularities computeGranularities(
            Hardware const& hardware, double M, double N, double K, double NumBatches, uint32_t autoGsuVal) const;

        StaticPerformanceModel staticPerformanceModel(double M,
                                                      double N,
                                                      double K,
                                                      double NumBatches,
                                                      double MT0,
                                                      double MT1,
                                                      double NumCUs,
                                                      double totalGranularity,
                                                      int    globalSplitU) const;

        /**
         * Calculate the projected performance based on granularity loss.
         */
        ProjectedPerformance projectedPerformance(Problem const&  problem,
                                                  Hardware const& hardware) const;

        /**
         * Generate a set of kernel calls to solve a particular problem.
         */
        virtual std::vector<KernelInvocation> solve(ContractionProblem const& problem,
                                                    ProblemInputs const&      inputs,
                                                    Hardware const&           hardware,
                                                    void*                     hipHostMemory,
                                                    size_t                    hipHostMemorySize,
                                                    hipStream_t               stream) const;

        virtual std::vector<KernelInvocation>
            solve(Problem const& problem, Inputs const& inputs, Hardware const& hardware) const;

        virtual std::vector<KernelInvocation> solveGroupedGemm(std::vector<Problem> const& problems,
                                                               GroupedInputs const&        inputs,
                                                               Hardware const&             hardware,
                                                               void*       hipHostMemory,
                                                               size_t      hipHostMemorySize,
                                                               hipStream_t stream) const;

        // The problems and inputs are passed by device memory
        virtual std::vector<KernelInvocation>
            solveGroupedGemmGPU(std::vector<Problem> const& problems,
                                GroupedInputs const&        inputs,
                                Hardware const&             hardware,
                                const void*                 dUA,
                                const void*                 workspace,
                                hipStream_t                 stream) const;

        // For Tensile debugging, will allocate and initialize DeviceUserArguments with the problems and inputs.
        virtual std::vector<KernelInvocation> solveTensileGPU(ContractionProblem const& problem,
                                                              ProblemInputs const&      inputs,
                                                              Hardware const&           hardware,
                                                              void**                    dUA,
                                                              void**                    dUAHost,
                                                              void*       hipHostMemory,
                                                              size_t      hipHostMemorySize,
                                                              hipStream_t stream) const;

        // For Tensile debugging, will allocate and initialize DeviceUserArguments with the problems and inputs.
        virtual std::vector<KernelInvocation>
            solveTensileGroupedGemmGPU(std::vector<Problem> const& problems,
                                       GroupedInputs const&        inputs,
                                       Hardware const&             hardware,
                                       void**                      dUA,
                                       void**                      dUAHost,
                                       void*                       hipHostMemory,
                                       size_t                      hipHostMemorySize,
                                       hipStream_t                 stream) const;

        virtual void relaseDeviceUserArgs(void* dUA, void* dUAHost);

        template <bool T_Debug, bool insertKernelArgs, typename KA>
        void singleCallArgs(Problem const&           problem,
                            ContractionInputs const& inputs,
                            uint32_t const&          workspaceOffsetInByte,
                            Hardware const*          hardware,
                            dim3 const&              problemNumGroupTiles,
                            dim3 const&              numWorkGroups,
                            KA&                      args,
                            StreamKSettings const&   sk) const;

        // Common kernel related arguments (e.g. gemm_count, arg type, MT, GSU...)
        template <bool T_Debug, bool Legacy, typename KA>
        void kernelArgs(uint32_t                            gemmCount,
                        uint32_t                            argType,
                        KA&                                 args,
                        uint32_t                            numWorkGroups,
                        Hardware const*                     hardware,
                        const ContractionProblemParameters& param,
                        int32_t                             defaultWGM,
                        uint32_t                            defaultWGMXCC,
                        uint32_t                            autoGsuVal) const;

        template <typename KA>
        inline void calculateSingleCallWorkGroupItems(std::vector<Problem> const& problems,
                                                      const TensileLite::dim3&    workGroupSize,
                                                      TensileLite::dim3&          numWorkGroups,
                                                      TensileLite::dim3&          numWorkItems,
                                                      KA&                         h_args,
                                                      uint32_t                    autoGsuVal) const;

        template <bool T_Debug>
        KernelInvocation generateSingleCall(Problem const&           problem,
                                            ContractionInputs const& inputs,
                                            Hardware const&          hardware,
                                            StreamKSettings const&   sk) const;

        template <bool T_Debug, typename KA>
        KernelInvocation generateSingleCallGroupedGemm(std::vector<Problem> const& problems,
                                                       GroupedInputs const&        inputs,
                                                       Hardware const&             hardware,
                                                       KA&                         h_args,
                                                       void const* userArgs = nullptr) const;

        template <bool T_Debug>
        KernelInvocation generateBetaOnlyCall(Problem const&           problem,
                                              ContractionInputs const& inputs) const;

        template <bool T_Debug>
        KernelInvocation generateBetaOnlyCallGroupedGemm(std::vector<Problem> const& problems,
                                                         GroupedInputs const&        inputs) const;

        std::string betaOnlyKernelName(Problem const& problem) const;

        template <bool T_Debug, typename KA>
        void outputConversionCallArgs(Problem const&           problem,
                                      ContractionInputs const& inputs,
                                      uint32_t const&          workspaceOffsetInByte,
                                      KA&                      args,
                                      StreamKSettings const&   sk,
                                      uint32_t                 autoGsuVal) const;

        template <typename KA>
        inline void calculateConversionCallWorkGroupItems(
            std::vector<ContractionSolution::Problem> const& problems,
            size_t&                                          vw,
            const TensileLite::dim3&                         workGroupSize,
            TensileLite::dim3&                               numWorkGroups,
            TensileLite::dim3&                               numWorkItems,
            KA&                                              args) const;

        template <bool T_Debug>
        KernelInvocation generateOutputConversionCall(Problem const&           problem,
                                                      ContractionInputs const& inputs,
                                                      StreamKSettings const&   sk,
                                                      uint32_t                 autoGsuVal) const;

        template <bool T_Debug, typename KA>
        KernelInvocation
            generateOutputConversionCallGroupedGemm(std::vector<Problem> const& problems,
                                                    GroupedInputs const&        inputs,
                                                    Hardware const&             hardware,
                                                    KA&                         h_args) const;

        template <bool T_Debug>
        KernelInvocation updateUserArgsOutputConversionCallGroupedGemm(
            std::vector<ContractionSolution::Problem> const& problems,
            const void*                                      userArgs,
            const void*                                      workspace) const;

        std::string outputConversionKernelName(Problem const&           problem,
                                               ContractionInputs const& inputs,
                                               size_t                   vw,
                                               size_t                   gsu) const;

        template <bool T_Debug>
        KernelInvocation generateReductionCall(Problem const&           problem,
                                               ContractionInputs const& inputs) const;

        std::string outputReductionKernelName(Problem const&           problem,
                                              ContractionInputs const& inputs,
                                              size_t                   mt0,
                                              size_t                   mt1,
                                              size_t                   vw) const;

        struct InternalArgsSupport
        {
            int  version          = 0;
            bool gsu              = true;
            bool wgm              = true;
            bool staggerU         = true;
            bool useUniversalArgs = true;
            bool useSFC           = false;
        };

        struct ProblemType
        {
            std::string      operationIdentifier;
            bool             transA                    = false;
            bool             transB                    = false;
            rocisa::DataType aType                     = rocisa::DataType::Float;
            rocisa::DataType bType                     = rocisa::DataType::Float;
            rocisa::DataType cType                     = rocisa::DataType::Float;
            rocisa::DataType dType                     = rocisa::DataType::Float;
            rocisa::DataType eType                     = rocisa::DataType::Float;
            rocisa::DataType computeInputType          = rocisa::DataType::Float;
            rocisa::DataType computeType               = rocisa::DataType::Float;
            rocisa::DataType f32XdlMathOp              = rocisa::DataType::Float;
            rocisa::DataType activationComputeDataType = rocisa::DataType::Float;
            bool             highPrecisionAccumulate   = false;
            bool             useBeta                   = true;
            bool             useGradient               = false;
            int              useBias                   = 0;
            bool             useE                      = false;
            std::string      useScaleAB                = "";
            bool             useScaleCD                = false;
            int              useScaleAlphaVec          = 0;
            bool             useInitialStridesAB       = false;
            bool             useInitialStridesCD       = false;
            bool             stridedBatched            = true;
            bool             outputAmaxD               = false;
            bool             groupedGemm               = false;
            ActivationType   activationType            = ActivationType::None;
            int              activationArgLength       = 0;
            bool             activationNoGuard         = false;

            std::vector<int>              biasSrcWhiteList;
            std::vector<rocisa::DataType> biasDataTypeWhiteList;

            int  sparse                     = 0;
            bool stochasticRounding         = false;
            bool supportDeviceUserArguments = false;
            bool swizzleTensorA             = false;
            bool swizzleTensorB             = false;
        };

        struct LinearModel
        {
            double slope     = 1.0;
            double intercept = 0.0;
            double max       = 1000.0;
        };

        int                          index = 0;
        std::string                  kernelName;
        std::string                  solutionName;
        ThreadSafeValue<std::string> codeObjectFilename;
        bool                         debugKernel     = false;
        bool                         kernelArgsLog   = false;
        mutable int                  isFallbackCUSol = -1; // -1:unset, 0:false, 1:true

        std::shared_ptr<Predicates::Predicate<Task>> taskPredicate
            = std::make_shared<Predicates::True<Task>>();
        std::shared_ptr<Predicates::Predicate<Problem>> problemPredicate
            = std::make_shared<Predicates::True<Problem>>();
        std::shared_ptr<Predicates::Predicate<Hardware>> hardwarePredicate
            = std::make_shared<Predicates::True<Hardware>>();

        SizeMapping sizeMapping;

        InternalArgsSupport internalArgsSupport;

        ProblemType problemType;

        // This will be calculated when getSolution is called
        size_t requiredHostWorkspaceSizePerProblem = static_cast<size_t>(-1);

        /// Debugging purposes.  Shouldn't contain any vital information that isn't
        /// somewhere else.
        int32_t               libraryLogicIndex = -1;
        std::map<int, double> ideals;
        LinearModel           linearModel;
        MatchingTag           tag{MatchingTag::Estimated};

        uint32_t magicNumberAlg1(uint32_t x, uint32_t* magicShift) const;
        uint32_t magicNumberAlg2(uint32_t x, uint32_t* magicShift) const;
        uint32_t magicNumber(int magicDivAlg, uint32_t x, uint32_t* magicShift) const;
        uint32_t smallMagicNumber(uint32_t x) const;

        std::pair<int32_t, uint32_t> calculateAutoWGM(Problem const& problem, Hardware const* hardware) const;
        uint32_t calculateAutoGSU(Problem const& problem, Hardware const* hardware) const;
    };

    template <typename TAct>
    void setDeviceUserArgs(std::vector<ContractionSolution::Problem> const& problems,
                           ContractionSolution::GroupedInputs const&        inputs,
                           DeviceUserArguments<TAct>*                       args);

    std::ostream& operator<<(std::ostream&                                      stream,
                             ContractionSolution::StaticPerformanceModel const& spm);
    std::ostream& operator<<(std::ostream&                                    stream,
                             ContractionSolution::ProjectedPerformance const& spm);
    std::ostream& operator<<(std::ostream& stream, BufferLoadCheckPacket const& st);
} // namespace TensileLite
