/**
 * @brief Common graphs used for unit tests.
 */

#pragma once

#include <vector>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>

namespace rocRollerTest
{
    namespace Graphs
    {
        using ContextPtr        = rocRoller::ContextPtr;
        using CommandPtr        = rocRoller::CommandPtr;
        using CommandParameters = rocRoller::CommandParameters;
        using KernelGraph       = rocRoller::KernelGraph::KernelGraph;
        using KernelArguments   = rocRoller::KernelArguments;

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
        template <typename T>
        class MatrixMultiply
        {
        public:
            MatrixMultiply();

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

        private:
            void createCommand();

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
        template <typename T>
        class GEMM
        {
        public:
            GEMM();

            CommandPtr  getCommand();
            KernelGraph getKernelGraph();

            void setTileSize(int m, int n, int k);
            void setMFMA(int m, int n, int k, int b);
            void setUseLDS(bool a, bool b, bool d);

            std::shared_ptr<CommandParameters> getCommandParameters() const;

        private:
            void createCommand();

            int  m_macM, m_macN, m_macK;
            int  m_waveM, m_waveN, m_waveK, m_waveB;
            bool m_useLDSA = false, m_useLDSB = false, m_useLDSD = false;

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

            std::shared_ptr<CommandParameters> getCommandParameters(size_t nx, size_t ny) const;

            KernelArguments getRuntimeArguments(size_t nx, size_t ny, T* x, T* y, T* rv);

            std::vector<T> referenceSolution(std::vector<T> const& x, std::vector<T> const& y);

        private:
            void createCommand();

            int m_macM, m_macN;
            int m_thrM, m_thrN;

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

            std::shared_ptr<CommandParameters> getCommandParameters(size_t nx, size_t ny) const;

            KernelArguments getRuntimeArguments(size_t nx, size_t ny, T* x, T* rv);

            std::vector<T> referenceSolution(std::vector<T> const& x);

        private:
            void createCommand();

            int m_macM, m_macN;
            int m_thrM, m_thrN;

            std::vector<size_t> m_literalStrides;
            CommandPtr          m_command;
        };

    }
}

#include "CommonGraphs_impl.hpp"
