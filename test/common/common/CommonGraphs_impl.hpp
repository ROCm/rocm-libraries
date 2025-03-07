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

#include "CommonGraphs.hpp"
#include <rocRoller/DataTypes/DataTypes.hpp>

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/Operations/Command.hpp>

namespace rocRollerTest::Graphs
{
    using namespace rocRoller;

    /*
     * VectorAdd
     */

    template <typename T>
    VectorAdd<T>::VectorAdd()
        : VectorAdd(false)
    {
    }

    template <typename T>
    VectorAdd<T>::VectorAdd(bool useBeta)
        : m_useBeta(useBeta)
    {
        createCommand();
    }

    template <typename T>
    void VectorAdd<T>::createCommand()
    {
        m_command = std::make_shared<Command>();

        auto dataType = TypeInfo<T>::Var.dataType;

        auto xTensorTag = m_command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        auto xLoadTag   = m_command->addOperation(rocRoller::Operations::T_Load_Linear(xTensorTag));

        auto yTensorTag = m_command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        auto yLoadTag   = m_command->addOperation(rocRoller::Operations::T_Load_Linear(yTensorTag));

        auto alphaScalarTag = m_command->addOperation(
            rocRoller::Operations::Scalar({dataType, PointerType::PointerGlobal}));
        auto alphaLoadTag
            = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(alphaScalarTag));

        rocRoller::Operations::OperationTag betaLoadTag;
        if(m_useBeta)
        {
            auto betaScalarTag = m_command->addOperation(rocRoller::Operations::Scalar(dataType));
            betaLoadTag
                = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(betaScalarTag));
        }

        auto execute   = rocRoller::Operations::T_Execute(m_command->getNextTag());
        auto alphaXTag = execute.addXOp(rocRoller::Operations::E_Mul(xLoadTag, alphaLoadTag));
        auto betaYTag  = yLoadTag;
        if(m_useBeta)
        {
            betaYTag = execute.addXOp(rocRoller::Operations::E_Mul(yLoadTag, betaLoadTag));
        }
        auto sumTag = execute.addXOp(rocRoller::Operations::E_Add(alphaXTag, betaYTag));
        m_command->addOperation(std::move(execute));

        auto sumTensorTag = m_command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        m_command->addOperation(rocRoller::Operations::T_Store_Linear(sumTag, sumTensorTag));
    }

    template <typename T>
    CommandPtr VectorAdd<T>::getCommand()
    {
        return m_command;
    }

    template <typename T>
    KernelGraph VectorAdd<T>::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(m_command);
    }

    template <typename T>
    KernelArguments VectorAdd<T>::getRuntimeArguments(size_t nx, T* alpha, T* x, T* y, T* rv)
    {
        KernelArguments runtimeArgs;

        AssertFatal(!m_useBeta);

        runtimeArgs.append("user0", x);
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);

        runtimeArgs.append("user1", y);
        runtimeArgs.append("d_b_limit", nx);
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        runtimeArgs.append("user2", alpha);

        runtimeArgs.append("user6", rv);
        runtimeArgs.append("d_c_limit", nx);
        runtimeArgs.append("d_c_size", nx);
        runtimeArgs.append("d_c_stride", (size_t)1);

        return runtimeArgs;
    }

    template <typename T>
    KernelArguments
        VectorAdd<T>::getRuntimeArguments(size_t nx, T* alpha, T beta, T* x, T* y, T* rv)
    {
        KernelArguments runtimeArgs;

        AssertFatal(m_useBeta);

        runtimeArgs.append("user0", x);
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);

        runtimeArgs.append("user1", y);
        runtimeArgs.append("d_b_limit", nx);
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        runtimeArgs.append("user2", alpha);

        runtimeArgs.append("user3", beta);

        runtimeArgs.append("user6", rv);
        runtimeArgs.append("d_c_limit", nx);
        runtimeArgs.append("d_c_size", nx);
        runtimeArgs.append("d_c_stride", (size_t)1);

        return runtimeArgs;
    }

    template <typename T>
    std::vector<T>
        VectorAdd<T>::referenceSolution(T alpha, std::vector<T> const& x, std::vector<T> const& y)
    {
        AssertFatal(!m_useBeta);

        std::vector<T> rv(x.size());
        for(size_t i = 0; i < x.size(); ++i)
            rv[i] = alpha * x[i] + y[i];
        return rv;
    }

    template <typename T>
    std::vector<T> VectorAdd<T>::referenceSolution(T                     alpha,
                                                   T                     beta,
                                                   std::vector<T> const& x,
                                                   std::vector<T> const& y)
    {
        AssertFatal(m_useBeta);

        std::vector<T> rv(x.size());
        for(size_t i = 0; i < x.size(); ++i)
            rv[i] = alpha * x[i] + beta * y[i];
        return rv;
    }

    /*
     * VectorAddNegSquare
     */

    template <typename T>
    VectorAddNegSquare<T>::VectorAddNegSquare()
        : VectorAddNegSquare(false)
    {
    }

    template <typename T>
    VectorAddNegSquare<T>::VectorAddNegSquare(bool useScalarLoads)
        : m_useScalarLoads(useScalarLoads)
    {
        createCommand();
    }

    template <typename T>
    void VectorAddNegSquare<T>::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        auto dataType = TypeInfo<T>::Var.dataType;

        rocRoller::Operations::OperationTag aLoadTag;
        rocRoller::Operations::OperationTag bLoadTag;

        if(m_useScalarLoads)
        {
            auto aScalarTag = m_command->addOperation(rocRoller::Operations::Scalar(dataType));
            aLoadTag = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(aScalarTag));
            auto bScalarTag = m_command->addOperation(rocRoller::Operations::Scalar(dataType));
            bLoadTag = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(bScalarTag));
        }
        else
        {
            auto aTensorTag = m_command->addOperation(rocRoller::Operations::Tensor(1, dataType));
            aLoadTag = m_command->addOperation(rocRoller::Operations::T_Load_Linear(aTensorTag));
            auto bTensorTag = m_command->addOperation(rocRoller::Operations::Tensor(1, dataType));
            bLoadTag = m_command->addOperation(rocRoller::Operations::T_Load_Linear(bTensorTag));
        }

        Operations::T_Execute execute(m_command->getNextTag());
        auto                  aPlusB    = execute.addXOp(Operations::E_Add(aLoadTag, bLoadTag));
        auto                  negAPlusB = execute.addXOp(Operations::E_Neg(aPlusB));
        auto                  result    = execute.addXOp(Operations::E_Mul(aPlusB, negAPlusB));
        m_command->addOperation(std::move(execute));

        if(!m_useScalarLoads)
        {
            auto resultTensorTag
                = m_command->addOperation(rocRoller::Operations::Tensor(1, dataType));
            m_command->addOperation(Operations::T_Store_Linear(result, resultTensorTag));
        }
    }

    template <typename T>
    CommandPtr VectorAddNegSquare<T>::getCommand()
    {
        return m_command;
    }

    template <typename T>
    KernelGraph VectorAddNegSquare<T>::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(m_command);
    }

    /*
     * GEMM
     */

    template <typename T>
    GEMM<T>::GEMM()
    {
        createCommand();
    }

    template <typename T>
    void GEMM<T>::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        auto dataType = TypeInfo<T>::Var.dataType;

        auto tagTensorA = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
        m_tagA          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
        m_tagB          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagTensorC = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // C
        m_tagC          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

        auto tagScalarAlpha
            = m_command->addOperation(rocRoller::Operations::Scalar(dataType)); // alpha
        auto tagLoadAlpha
            = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

        auto tagScalarBeta
            = m_command->addOperation(rocRoller::Operations::Scalar(dataType)); // beta
        auto tagLoadBeta
            = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta)); // beta

        auto tagAB = m_command->addOperation(rocRoller::Operations::T_Mul(m_tagA, m_tagB)); // A * B

        rocRoller::Operations::T_Execute execute(m_command->getNextTag());
        auto                             tagAlphaAB
            = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)
        auto tagBetaC
            = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, m_tagC)); // beta * C
        m_tagD = execute.addXOp(rocRoller::Operations::E_Add(tagAlphaAB, tagBetaC));
        // alpha * (A * B) + beta * C
        m_command->addOperation(std::move(execute));

        auto tagTensorD = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // D
        m_command->addOperation(rocRoller::Operations::T_Store_Tiled(m_tagD, tagTensorD)); // D
    }

    template <typename T>
    CommandPtr GEMM<T>::getCommand()
    {
        return m_command;
    }

    template <typename T>
    KernelGraph GEMM<T>::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(m_command);
    }

    template <typename T>
    void GEMM<T>::setTileSize(int m, int n, int k)
    {
        m_macM = m;
        m_macN = n;
        m_macK = k;
    }

    template <typename T>
    void GEMM<T>::setMFMA(int m, int n, int k, int b)
    {
        m_waveM = m;
        m_waveN = n;
        m_waveK = k;
        m_waveB = b;
    }

    template <typename T>
    void GEMM<T>::setUseLDS(bool a, bool b, bool d)
    {
        m_useLDSA = a;
        m_useLDSB = b;
        m_useLDSD = d;
    }

    template <typename T>
    CommandParametersPtr GEMM<T>::getCommandParameters() const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        auto macTileA = MacroTile({m_macM, m_macK},
                                  LayoutType::MATRIX_A,
                                  {m_waveM, m_waveN, m_waveK, m_waveB},
                                  m_useLDSA ? MemoryType::LDS : MemoryType::WAVE);
        auto macTileB = MacroTile({m_macK, m_macN},
                                  LayoutType::MATRIX_B,
                                  {m_waveM, m_waveN, m_waveK, m_waveB},
                                  m_useLDSB ? MemoryType::LDS : MemoryType::WAVE);
        auto macTileC = MacroTile(
            {m_macM, m_macN}, LayoutType::MATRIX_ACCUMULATOR, {m_waveM, m_waveN, m_waveK, m_waveB});

        params->setDimensionInfo(m_tagA, macTileA);
        params->setDimensionInfo(m_tagB, macTileB);
        params->setDimensionInfo(m_tagC, macTileC);

        // Workgroup size
        uint wavefrontSize  = 64;
        uint workgroupSizeX = 2 * wavefrontSize;
        uint workgroupSizeY = 4;

        uint jammedM = wavefrontSize * m_macM / m_waveM / workgroupSizeX;
        uint jammedN = m_macN / m_waveN / workgroupSizeY;

        Log::debug("GEMM workgroup sizes {} {} {}", workgroupSizeX, workgroupSizeY, 1);
        Log::debug("GEMM jamming {} {}", jammedM, jammedN);

        params->setWaveTilesPerWavefront(jammedM, jammedN);

        return params;
    }

    /*
     * TileDoubleAdd
     */

    template <typename T>
    TileDoubleAdd<T>::TileDoubleAdd()
    {
        createCommand();
    }

    template <typename T>
    void TileDoubleAdd<T>::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        auto dataType = TypeInfo<T>::Var.dataType;

        auto tagTensorA = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
        m_tagA          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
        m_tagB          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto execute = rocRoller::Operations::T_Execute(m_command->getNextTag());
        auto tag2A   = execute.addXOp(rocRoller::Operations::E_Add(m_tagA, m_tagA)); // A + A
        auto tag2B   = execute.addXOp(rocRoller::Operations::E_Add(m_tagB, m_tagB)); // B + B
        m_tagD       = execute.addXOp(rocRoller::Operations::E_Add(tag2A, tag2B)); // C = 2A + 2B
        m_command->addOperation(std::move(execute));

        auto tagTensorC = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType));
        m_command->addOperation(rocRoller::Operations::T_Store_Tiled(m_tagD, tagTensorC));
    }

    template <typename T>
    CommandPtr TileDoubleAdd<T>::getCommand()
    {
        return m_command;
    }

    template <typename T>
    KernelGraph TileDoubleAdd<T>::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(m_command);
    }

    template <typename T>
    void TileDoubleAdd<T>::setTileSize(int m, int n)
    {
        m_macM = m;
        m_macN = n;
    }

    template <typename T>
    void TileDoubleAdd<T>::setSubTileSize(int m, int n)
    {
        m_thrM = m;
        m_thrN = n;
    }

    template <typename T>
    CommandParametersPtr TileDoubleAdd<T>::getCommandParameters(size_t nx, size_t ny) const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        auto macTileLDS  = MacroTile({m_macM, m_macN}, MemoryType::LDS, {m_thrM, m_thrN});
        auto macTileVGPR = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});

        params->setDimensionInfo(m_tagA, macTileLDS);
        params->setDimensionInfo(m_tagB, macTileVGPR);
        // TODO Fix MemoryType promotion (LDS)
        params->setDimensionInfo(m_tagD, macTileVGPR);

        uint workgroupSizeX = m_macM / m_thrM;
        uint workgroupSizeY = m_macN / m_thrN;

        AssertFatal(m_macM > 0 && m_macN > 0 && m_thrM > 0 && m_thrN > 0
                        && (size_t)m_macM * m_macN
                               == m_thrM * m_thrN * workgroupSizeX * workgroupSizeY,
                    "MacroTile size mismatch");

        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});

        return params;
    }

    template <typename T>
    CommandLaunchParametersPtr TileDoubleAdd<T>::getCommandLaunchParameters(size_t nx,
                                                                            size_t ny) const
    {
        auto NX = std::make_shared<Expression::Expression>(nx / m_thrM);
        auto NY = std::make_shared<Expression::Expression>(ny / m_thrN);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        auto launch = std::make_shared<CommandLaunchParameters>();
        launch->setManualWorkitemCount({NX, NY, NZ});

        return launch;
    }

    template <typename T>
    KernelArguments TileDoubleAdd<T>::getRuntimeArguments(size_t nx, size_t ny, T* x, T* y, T* rv)
    {
        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", x);
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("user1", y);
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_size_0", (size_t)nx);
        runtimeArgs.append("d_b_size_1", (size_t)ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        runtimeArgs.append("user2", rv);
        runtimeArgs.append("d_c_limit", (size_t)nx * ny);
        runtimeArgs.append("d_c_size_0", (size_t)nx);
        runtimeArgs.append("d_c_size_1", (size_t)ny);
        runtimeArgs.append("d_c_stride_0", (size_t)ny);
        runtimeArgs.append("d_c_stride_1", (size_t)1);

        return runtimeArgs;
    }

    template <typename T>
    std::vector<T> TileDoubleAdd<T>::referenceSolution(std::vector<T> const& x,
                                                       std::vector<T> const& y)
    {
        std::vector<T> rv(x.size());
        for(size_t i = 0; i < x.size(); ++i)
            rv[i] = 2 * x[i] + 2 * y[i];
        return rv;
    }

    /*
     * TileCopy
     */

    template <typename T>
    TileCopy<T>::TileCopy()
    {
        createCommand();
    }

    template <typename T>
    void TileCopy<T>::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        auto dataType = TypeInfo<T>::Var.dataType;

        rocRoller::Operations::OperationTag tagInputTensor;
        rocRoller::Operations::OperationTag tagOutputTensor;

        if(!m_literalStrides.empty())
        {
            tagInputTensor = m_command->addOperation(
                rocRoller::Operations::Tensor(2, dataType, m_literalStrides));
            tagOutputTensor = m_command->addOperation(
                rocRoller::Operations::Tensor(2, dataType, m_literalStrides));
        }
        else
        {
            tagInputTensor  = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType));
            tagOutputTensor = m_command->addOperation(rocRoller::Operations::Tensor(2, dataType));
        }

        m_tag = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagInputTensor));
        m_command->addOperation(rocRoller::Operations::T_Store_Tiled(m_tag, tagOutputTensor));
    }

    template <typename T>
    CommandPtr TileCopy<T>::getCommand()
    {
        return m_command;
    }

    template <typename T>
    KernelGraph TileCopy<T>::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(m_command);
    }

    template <typename T>
    void TileCopy<T>::setTileSize(int m, int n)
    {
        m_macM = m;
        m_macN = n;
    }

    template <typename T>
    void TileCopy<T>::setSubTileSize(int m, int n)
    {
        m_thrM = m;
        m_thrN = n;
    }

    template <typename T>
    void TileCopy<T>::setLiteralStrides(std::vector<size_t> const& literalStrides)
    {
        m_literalStrides = literalStrides;
        createCommand();
    }

    template <typename T>
    CommandParametersPtr TileCopy<T>::getCommandParameters(size_t nx, size_t ny) const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        auto macTile = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});
        params->setDimensionInfo(m_tag, macTile);

        uint workgroupSizeX = m_macM / m_thrM;
        uint workgroupSizeY = m_macN / m_thrN;

        AssertFatal(m_macM > 0 && m_macN > 0 && m_thrM > 0 && m_thrN > 0
                        && (size_t)m_macM * m_macN
                               == m_thrM * m_thrN * workgroupSizeX * workgroupSizeY,
                    "MacroTile size mismatch");

        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});

        params->transposeMemoryAccess[LayoutType::None] = true;

        return params;
    }

    template <typename T>
    CommandLaunchParametersPtr TileCopy<T>::getCommandLaunchParameters(size_t nx, size_t ny) const
    {
        auto NX = std::make_shared<Expression::Expression>(nx / m_thrM);
        auto NY = std::make_shared<Expression::Expression>(ny / m_thrN);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        auto launch = std::make_shared<CommandLaunchParameters>();
        launch->setManualWorkitemCount({NX, NY, NZ});
        return launch;
    }

    template <typename T>
    KernelArguments TileCopy<T>::getRuntimeArguments(size_t nx, size_t ny, T* x, T* rv)
    {
        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", x);
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("user2", rv);
        runtimeArgs.append("d_c_limit", (size_t)nx * ny);
        runtimeArgs.append("d_c_size_0", (size_t)nx);
        runtimeArgs.append("d_c_size_1", (size_t)ny);
        runtimeArgs.append("d_c_stride_0", (size_t)ny);
        runtimeArgs.append("d_c_stride_1", (size_t)1);

        return runtimeArgs;
    }

    template <typename T>
    std::vector<T> TileCopy<T>::referenceSolution(std::vector<T> const& x)
    {
        std::vector<T> rv(x.size());
        for(size_t i = 0; i < x.size(); ++i)
            rv[i] = x[i];
        return rv;
    }
}
