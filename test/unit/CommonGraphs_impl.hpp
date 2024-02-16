
#pragma once

#include "CommonGraphs.hpp"
#include "DataTypes/DataTypes.hpp"

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

        int tag = 0;

        auto xTag = tag++;
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(dataType, 1, xTag)));

        auto yTag = tag++;
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(dataType, 1, yTag)));

        auto alphaTag = tag++;
        m_command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Scalar(
                {dataType, PointerType::PointerGlobal}, alphaTag)));

        auto betaTag = m_useBeta ? (tag++) : -1;
        if(m_useBeta)
        {
            m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
                rocRoller::Operations::T_Load_Scalar(dataType, betaTag)));
        }

        auto execute = rocRoller::Operations::T_Execute();

        auto alphaXTag = tag++;
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(alphaXTag, xTag, alphaTag)));

        auto betaYTag = m_useBeta ? (tag++) : yTag;
        if(m_useBeta)
        {
            execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
                rocRoller::Operations::E_Mul(betaYTag, yTag, betaTag)));
        }

        auto sumTag = tag++;
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(sumTag, alphaXTag, betaYTag)));

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Linear(1, sumTag)));
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

        if(m_useScalarLoads)
        {
            Operations::T_Load_Scalar load_A(dataType, 0);
            m_command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

            Operations::T_Load_Scalar load_B(dataType, 1);
            m_command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));
        }
        else
        {
            Operations::T_Load_Linear load_A(dataType, 1, 0);
            m_command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

            Operations::T_Load_Linear load_B(dataType, 1, 1);
            m_command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));
        }

        Operations::T_Execute execute;
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(2, 1, 0)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(3, 2)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(4, 2, 3)));

        m_command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

        if(!m_useScalarLoads)
        {
            Operations::T_Store_Linear store_C(1, 4);
            m_command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));
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
     * MatrixMultiply
     */

    template <typename T>
    MatrixMultiply<T>::MatrixMultiply()
    {
        createCommand();
    }

    template <typename T>
    void MatrixMultiply<T>::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        auto dataType = TypeInfo<T>::Var.dataType;

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // A
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // B

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1))); // D = A * B

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 2))); // D
    }

    template <typename T>
    CommandPtr MatrixMultiply<T>::getCommand()
    {
        return m_command;
    }

    template <typename T>
    KernelGraph MatrixMultiply<T>::getKernelGraph()
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

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // A
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // B
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 2))); // C
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(dataType, 3))); // alpha
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(dataType, 4))); // beta

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(5, 0, 1))); // A * B

        rocRoller::Operations::T_Execute execute;
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(6, 3, 5))); // alpha * (A * B)
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(7, 4, 2))); // beta * C
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(8, 6, 7))); // alpha * (A * B) + beta * C
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 8))); // D
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
    std::shared_ptr<CommandParameters> GEMM<T>::getCommandParameters() const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        auto macTileA = MacroTile({m_macM, m_macK},
                                  LayoutType::MATRIX_A,
                                  {m_waveM, m_waveN, m_waveK, m_waveB},
                                  m_useLDSA ? MemoryType::WAVE_LDS : MemoryType::WAVE);
        auto macTileB = MacroTile({m_macK, m_macN},
                                  LayoutType::MATRIX_B,
                                  {m_waveM, m_waveN, m_waveK, m_waveB},
                                  m_useLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);
        auto macTileC = MacroTile(
            {m_macM, m_macN}, LayoutType::MATRIX_ACCUMULATOR, {m_waveM, m_waveN, m_waveK, m_waveB});

        auto macTileD = MacroTile({m_macM, m_macN},
                                  LayoutType::MATRIX_ACCUMULATOR,
                                  {m_waveM, m_waveN, m_waveK, m_waveB},
                                  m_useLDSD ? MemoryType::WAVE_LDS : MemoryType::WAVE);

        params->setDimensionInfo(4, macTileA);
        params->setDimensionInfo(11, macTileB);
        params->setDimensionInfo(18, macTileC);
        if(m_useLDSD)
            params->setDimensionInfo(28, macTileC);
        params->setDimensionInfo(30, macTileC);
        params->setDimensionInfo(32, macTileC);
        params->setDimensionInfo(34, macTileD);

        // Workgroup size
        uint wavefrontSize  = 64;
        uint workgroupSizeX = 2 * wavefrontSize;
        uint workgroupSizeY = 4;

        uint jammedM = wavefrontSize * m_macM / m_waveM / workgroupSizeX;
        uint jammedN = m_macN / m_waveN / workgroupSizeY;

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

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // a
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // b

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(2, 0, 0))); // a + a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 1, 1))); // b + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(4, 3, 2))); // 2a + 2b

        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 4))); // c
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
    std::shared_ptr<CommandParameters> TileDoubleAdd<T>::getCommandParameters(size_t nx,
                                                                              size_t ny) const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        auto macTile0 = MacroTile({m_macM, m_macN}, MemoryType::LDS, {m_thrM, m_thrN});
        auto macTile1 = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});
        auto macTile2 = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});
        auto macTile3 = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});
        auto macTile4 = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});

        params->setDimensionInfo(4, macTile0);
        params->setDimensionInfo(11, macTile1);
        params->setDimensionInfo(15, macTile2);
        params->setDimensionInfo(17, macTile3);
        params->setDimensionInfo(19, macTile4);

        uint workgroupSizeX = m_macM / m_thrM;
        uint workgroupSizeY = m_macN / m_thrN;

        AssertFatal(m_macM > 0 && m_macN > 0 && m_thrM > 0 && m_thrN > 0
                        && (size_t)m_macM * m_macN
                               == m_thrM * m_thrN * workgroupSizeX * workgroupSizeY,
                    "MacroTile size mismatch");

        auto NX = std::make_shared<Expression::Expression>(nx / m_thrM);
        auto NY = std::make_shared<Expression::Expression>(ny / m_thrN);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        return params;
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

        if(!m_literalStrides.empty())
        {
            m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
                rocRoller::Operations::T_Load_Tiled(dataType, 2, 0, m_literalStrides)));
        }
        else
        {
            m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
                rocRoller::Operations::T_Load_Tiled(dataType, 2, 0)));
        }
        m_command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 0)));
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
    std::shared_ptr<CommandParameters> TileCopy<T>::getCommandParameters(size_t nx, size_t ny) const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        auto macTile = MacroTile({m_macM, m_macN}, MemoryType::VGPR, {m_thrM, m_thrN});

        params->setDimensionInfo(4, macTile);

        if(!m_literalStrides.empty())
        {
            auto storeColStrideOverride   = SubDimension(1);
            storeColStrideOverride.stride = Expression::literal(1u);
            params->setDimensionInfo(9, storeColStrideOverride);
        }

        uint workgroupSizeX = m_macM / m_thrM;
        uint workgroupSizeY = m_macN / m_thrN;

        AssertFatal(m_macM > 0 && m_macN > 0 && m_thrM > 0 && m_thrN > 0
                        && (size_t)m_macM * m_macN
                               == m_thrM * m_thrN * workgroupSizeX * workgroupSizeY,
                    "MacroTile size mismatch");

        auto NX = std::make_shared<Expression::Expression>(nx / m_thrM);
        auto NY = std::make_shared<Expression::Expression>(ny / m_thrN);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        return params;
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
