// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @brief Common graphs used for unit tests.
 */

#pragma once

#include <map>
#include <vector>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Operations/Scratch_fwd.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include <common/GEMMProblem.hpp>
#include <common/Utilities.hpp>

namespace rocRollerTest
{
    namespace Graphs
    {
        using CommandLaunchParameters    = rocRoller::CommandLaunchParameters;
        using CommandLaunchParametersPtr = rocRoller::CommandLaunchParametersPtr;
        using CommandParameters          = rocRoller::CommandParameters;
        using CommandParametersPtr       = rocRoller::CommandParametersPtr;
        using CommandArguments           = rocRoller::CommandArguments;
        using CommandPtr                 = rocRoller::CommandPtr;
        using ContextPtr                 = rocRoller::ContextPtr;
        using KernelArguments            = rocRoller::KernelArguments;
        using KernelGraph                = rocRoller::KernelGraph::KernelGraph;
        using DataType                   = rocRoller::DataType;

        /**
         * @brief Graph for linear: alpha x + beta y.
         *
         * Creates a command graph that is essentially:
         * - LoadScalar(alpha)
         * - LoadScalar(beta)
         * - LoadLinear(x)
         * - LoadLinear(y)
         * - Assign(z = alpha x + beta y)
         * - StoreLinear(z)
         */
        template <typename T>
        class VectorAdd
        {
        public:
            VectorAdd();
            VectorAdd(bool useBeta);

            CommandPtr      getCommand();
            KernelGraph     getKernelGraph();
            KernelArguments getRuntimeArguments(size_t nx, T* alpha, T beta, T* x, T* y, T* rv);
            KernelArguments getRuntimeArguments(size_t nx, T* alpha, T* x, T* y, T* rv);
            std::vector<T>
                referenceSolution(T alpha, std::vector<T> const& x, std::vector<T> const& y);
            std::vector<T> referenceSolution(T                     alpha,
                                             T                     beta,
                                             std::vector<T> const& x,
                                             std::vector<T> const& y);

        private:
            void createCommand();

            bool       m_useBeta;
            CommandPtr m_command;
        };

        /**
         * @brief Graph for linear: - (x + y) * (x + y).
         *
         * Creates a command graph that is essentially:
         * - LoadLinear(x)
         * - LoadLinear(y)
         * - Assign(w = x + y)
         * - Assign(z = - w * w)
         * - StoreLinear(z)
         */
        template <typename T>
        class VectorAddNegSquare
        {
        public:
            VectorAddNegSquare();
            VectorAddNegSquare(bool useScalarLoads);

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            rocRoller::Operations::OperationTag aTag, bTag;
            rocRoller::Operations::OperationTag resultTag;

            CommandParametersPtr getCommandParameters() const;

        private:
            void createCommand();

            bool       m_useScalarLoads;
            CommandPtr m_command;
        };

        /**
         * @brief Graph for tiled: A B.
         *
         * Creates a command graph that is essentially:
         * - LoadTiled(A)
         * - LoadTiled(B)
         * - Assign(D = TensorContraction(A, B))
         * - StoreTiled(D)
         */
        class MatrixMultiply
        {
        public:
            MatrixMultiply(rocRoller::DataType              aType,
                           rocRoller::DataType              bType  = rocRoller::DataType::None,
                           rocRoller::DataType              cdType = rocRoller::DataType::None,
                           rocRoller::Operations::ScaleMode aMode
                           = rocRoller::Operations::ScaleMode::None,
                           rocRoller::Operations::ScaleMode bMode
                           = rocRoller::Operations::ScaleMode::None);

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n, int k);
            void setMFMA(int m, int n, int k, int b);
            void setUseLDS(bool a, bool b, bool d);

            std::shared_ptr<CommandParameters> getCommandParameters() const;

        private:
            void createCommand();

            rocRoller::DataType m_aType;
            rocRoller::DataType m_bType;
            rocRoller::DataType m_cdType;

            rocRoller::Operations::ScaleMode m_aMode;
            rocRoller::Operations::ScaleMode m_bMode;

            int                      m_macM, m_macN, m_macK;
            int                      m_waveM, m_waveN, m_waveK, m_waveB;
            SolutionParams::LoadPath m_loadPathA = SolutionParams::LoadPath::BufferToVGPR;
            SolutionParams::LoadPath m_loadPathB = SolutionParams::LoadPath::BufferToVGPR;
            bool                     m_useLDSD   = false;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagD;
            rocRoller::Operations::OperationTag m_tagScaleA, m_tagScaleB;

            CommandPtr m_command;
        };

        /**
         * @brief Graph for GEMM: alpha A B + beta C
         *
         * Creates a command graph that is essentially:
         * - LoadScalar(alpha)
         * - LoadScalar(beta)
         * - LoadTiled(A)
         * - LoadTiled(B)
         * - LoadTiled(C)
         * - Assign(AB = TensorContraction(A, B))
         * - Assign(D = alpha * AB + beta * C)
         * - StoreTiled(D)
         */
        class GEMM
        {
        public:
            GEMM(DataType ta);
            GEMM(DataType ta, DataType tb);
            GEMM(DataType ta, DataType tb, DataType tc);
            GEMM(DataType ta, DataType tb, DataType tc, DataType td);

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n, int k);
            void setMFMA(int m, int n, int k, int b);
            void setUseLDS(bool a, bool b, bool d);
            void setUnroll(unsigned int unrollK);
            void setStreamK(rocRoller::StreamKMode streamKMode);
            void setPrefetch(bool prefetch,
                             int  prefetchInFlight,
                             int  prefetchLDSFactor,
                             bool prefetchMixMemOps);
            void setScaling(rocRoller::Operations::ScaleMode aMode,
                            rocRoller::Operations::ScaleMode bMode,
                            DataType                         scaleTypeA,
                            DataType                         scaleTypeB,
                            int                              scaleBlockSize);
            void setScaleLoadPaths(SolutionParams::LoadPath scalePathA,
                                   SolutionParams::LoadPath scalePathB);
            void setSwizzle(int m, int n, int k, int b, bool prefetch);
            void setTranspose(std::string const& transA, std::string const& transB);
            void setPad(decltype(GEMMProblem::padA) padA, decltype(GEMMProblem::padB) padB);

            GEMMProblem const& getProblem() const
            {
                return m_problem;
            };
            void setProblem(GEMMProblem const& problem);

            int getFlattenedWorkgroupSize() const;

            CommandParametersPtr getCommandParameters() const;

            template <typename TA,
                      typename TB  = TA,
                      typename TC  = TA,
                      typename TD  = TC,
                      typename ACC = float>
            auto getCommandArguments() const
            {
                using namespace rocRoller;

                auto dataTypeA   = TypeInfo<TA>::Var.dataType;
                auto dataTypeB   = TypeInfo<TB>::Var.dataType;
                auto dataTypeC   = TypeInfo<TC>::Var.dataType;
                auto dataTypeD   = TypeInfo<TD>::Var.dataType;
                auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

                // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
                int   M     = m_problem.m;
                int   N     = m_problem.n;
                int   K     = m_problem.k;
                float alpha = m_problem.alpha;
                float beta  = m_problem.beta;

                // Host data
                using PackedTypeA = typename PackedTypeOf<TA>::type;
                using PackedTypeB = typename PackedTypeOf<TB>::type;
                std::vector<PackedTypeA> hostA;
                std::vector<PackedTypeB> hostB;
                std::vector<TC>          hostC;

                std::vector<uint8_t> hostScaleA, hostScaleB;

                TensorDescriptor descA(dataTypeA, {size_t(M), size_t(K)}, m_problem.transA);
                TensorDescriptor descB(dataTypeB, {size_t(K), size_t(N)}, m_problem.transB);
                TensorDescriptor descC(dataTypeD, {size_t(M), size_t(N)}, "N");
                TensorDescriptor descD(dataTypeD, {size_t(M), size_t(N)}, "N");

                auto seed = 31415u;
                if(m_problem.scaleAMode == Operations::ScaleMode::Separate
                   || m_problem.scaleBMode == Operations::ScaleMode::Separate)
                {
                    auto scaleBlockSize = m_problem.scaleBlockSize;
                    AssertFatal(scaleBlockSize > 0, "scaleBlockSize must be set to scale A or B.");
                    AssertFatal(m_problem.k % scaleBlockSize == 0,
                                fmt::format("K: {} must be a multiple of the scale block size: {}",
                                            m_problem.k,
                                            scaleBlockSize));
                    DGenInput(seed,
                              hostA,
                              descA,
                              hostB,
                              descB,
                              hostC,
                              descC,
                              hostScaleA,
                              hostScaleB,
                              m_problem.scaleAMode == Operations::ScaleMode::Separate,
                              m_problem.scaleBMode == Operations::ScaleMode::Separate,
                              -1.f,
                              1.f,
                              static_cast<uint>(scaleBlockSize));
                }
                else
                {
                    DGenInput(seed, hostA, descA, hostB, descB, hostC, descC);
                }

                auto deviceA = make_shared_device<TA>(hostA);
                auto deviceB = make_shared_device<TB>(hostB);

                std::shared_ptr<TC> deviceC = make_shared_device(hostC);
                std::shared_ptr<TD> deviceD = make_shared_device<TD>(M * N, TD{});

                CommandArguments commandArgs = m_command->createArguments();

                setCommandTensorArg(commandArgs, m_tagTensorA, descA, deviceA.get());
                setCommandTensorArg(commandArgs, m_tagTensorB, descB, deviceB.get());
                setCommandTensorArg(commandArgs, m_tagTensorC, descC, deviceC.get());
                setCommandTensorArg(commandArgs, m_tagTensorD, descD, deviceD.get());

                commandArgs.setArgument(m_tagScalarAlpha, ArgumentType::Value, alpha);
                commandArgs.setArgument(m_tagScalarBeta, ArgumentType::Value, beta);

                return std::make_tuple(commandArgs, deviceA, deviceB, deviceC, deviceD);
            }

            std::tuple<rocRoller::Operations::OperationTag,
                       rocRoller::Operations::OperationTag,
                       rocRoller::Operations::OperationTag,
                       rocRoller::Operations::OperationTag>
                getOperationTags() const;

            std::pair<std::optional<rocRoller::Operations::OperationTag>,
                      std::optional<rocRoller::Operations::OperationTag>>
                getABScaleTags() const;

        private:
            void createCommand();

            DataType m_ta, m_tb, m_tc, m_td;

            GEMMProblem m_problem;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagC, m_tagD;
            rocRoller::Operations::OperationTag m_tagTensorA, m_tagTensorB, m_tagTensorC,
                m_tagTensorD;
            rocRoller::Operations::OperationTag m_tagScaleA, m_tagScaleB;
            rocRoller::Operations::OperationTag m_tagScalarAlpha, m_tagScalarBeta;
            rocRoller::Operations::OperationTag m_tagNumWGs;

            std::map<rocRoller::Operations::ScratchPolicy, rocRoller::Operations::OperationTag>
                m_scratchTags;

            CommandPtr m_command;
        };

        /**
         * @brief Graph for tiled: (x + x) + (y + y).
         *
         * Creates a command graph that is essentially:
         * - LoadTiled(x)
         * - LoadTiled(y)
         * - Assign(z = (x + x) + (y + y))
         * - StoreTiled(z)
         */
        template <typename T>
        class TileDoubleAdd
        {
        public:
            TileDoubleAdd();

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n);
            void setSubTileSize(int m, int n);

            CommandParametersPtr       getCommandParameters(size_t nx, size_t ny) const;
            CommandLaunchParametersPtr getCommandLaunchParameters(size_t nx, size_t ny) const;

            KernelArguments getRuntimeArguments(size_t nx, size_t ny, T* x, T* y, T* rv);

            std::vector<T> referenceSolution(std::vector<T> const& x, std::vector<T> const& y);

        private:
            void createCommand();

            int m_macM, m_macN;
            int m_thrM, m_thrN;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagD;

            CommandPtr m_command;
        };

        /**
         * @brief Graph for tiled: x.
         *
         * Creates a command graph that is essentially:
         * - LoadTiled(x)
         * - StoreTiled(x)
         */
        template <typename T>
        class TileCopy
        {
        public:
            TileCopy();

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n);
            void setSubTileSize(int m, int n);
            void setLiteralStrides(std::vector<size_t> const& literalStrides);

            CommandParametersPtr       getCommandParameters(size_t nx, size_t ny) const;
            CommandLaunchParametersPtr getCommandLaunchParameters(size_t nx, size_t ny) const;

            KernelArguments getRuntimeArguments(size_t nx, size_t ny, T* x, T* rv);

            std::vector<T> referenceSolution(std::vector<T> const& x);

        private:
            void createCommand();

            rocRoller::Operations::OperationTag m_tag;
            int                                 m_macM, m_macN;
            int                                 m_thrM, m_thrN;

            std::vector<size_t> m_literalStrides;
            CommandPtr          m_command;
        };
    }
}
#include "CommonGraphs_impl.hpp"
