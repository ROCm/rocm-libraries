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

/**
 * @brief Common graphs used for unit tests.
 */

#pragma once

#include <vector>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>

#include <common/GEMMProblem.hpp>

namespace rocRollerTest
{
    namespace Graphs
    {
        using CommandLaunchParameters    = rocRoller::CommandLaunchParameters;
        using CommandLaunchParametersPtr = rocRoller::CommandLaunchParametersPtr;
        using CommandParameters          = rocRoller::CommandParameters;
        using CommandParametersPtr       = rocRoller::CommandParametersPtr;
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

            int  m_macM, m_macN, m_macK;
            int  m_waveM, m_waveN, m_waveK, m_waveB;
            bool m_useLDSA = false, m_useLDSB = false, m_useLDSD = false;

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
            void setUnroll(unsigned int unrollX, unsigned int unrollY);
            void setPrefetch(bool prefetch,
                             int  prefetchInFlight,
                             int  prefetchLDSFactor,
                             bool prefetchMixMemOps);

            GEMMProblem const& getProblem() const
            {
                return m_problem;
            };
            void setProblem(GEMMProblem const& problem);

            CommandParametersPtr getCommandParameters() const;

        private:
            void createCommand();

            DataType m_ta, m_tb, m_tc, m_td;

            GEMMProblem m_problem;

            // int  m_macM, m_macN, m_macK;
            // int  m_waveM, m_waveN, m_waveK, m_waveB;
            // bool m_useLDSA = false, m_useLDSB = false, m_useLDSD = false;

            rocRoller::Operations::OperationTag m_tagA, m_tagB, m_tagC, m_tagD;
            rocRoller::Operations::OperationTag m_tagNumWGs;

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

    template <typename Transform, typename... Args>
    rocRoller::KernelGraph::KernelGraph transform(rocRoller::KernelGraph::KernelGraph& graph,
                                                  Args... args)
    {
        auto xform = std::make_shared<Transform>(std::forward<Args>(args)...);
        return graph.transform(xform);
    }
}
#include "CommonGraphs_impl.hpp"
