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

#include <common/CommonGraphs.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/Operations/Command.hpp>

namespace rocRollerTest::Graphs
{
    using namespace rocRoller;

    /*
     * MatrixMultiply
     */

    MatrixMultiply::MatrixMultiply(DataType              aType,
                                   DataType              bType,
                                   DataType              cdType,
                                   Operations::ScaleMode aMode,
                                   Operations::ScaleMode bMode)
        : m_aType(aType)
        , m_bType(bType)
        , m_cdType(cdType)
        , m_aMode(aMode)
        , m_bMode(bMode)
    {
        AssertFatal(m_aMode == Operations::ScaleMode::None
                        || m_aMode == Operations::ScaleMode::Separate,
                    "Only Separate scale mode supported.",
                    ShowValue(m_aMode));
        AssertFatal(m_bMode == Operations::ScaleMode::None
                        || m_bMode == Operations::ScaleMode::Separate,
                    "Only Separate scale mode supported.",
                    ShowValue(m_bMode));

        if(m_bType == DataType::None)
            m_bType = m_aType;

        if(m_cdType == DataType::None)
            m_cdType = m_bType;

        createCommand();
    }

    void MatrixMultiply::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        auto tagTensorA = m_command->addOperation(rocRoller::Operations::Tensor(2, m_aType)); // A
        m_tagA          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));
        auto tagA       = m_tagA;

        if(m_aMode == Operations::ScaleMode::Separate)
        {
            auto scaleA
                = m_command->addOperation(rocRoller::Operations::Tensor(2, DataType::UInt8));
            m_tagScaleA = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(scaleA));

            tagA = m_command->addOperation(
                rocRoller::Operations::BlockScale(tagA, 2, m_tagScaleA, {1, 32}));
        }

        auto tagTensorB = m_command->addOperation(rocRoller::Operations::Tensor(2, m_bType)); // B
        m_tagB          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));
        auto tagB       = m_tagB;

        if(m_bMode == Operations::ScaleMode::Separate)
        {
            auto scaleB
                = m_command->addOperation(rocRoller::Operations::Tensor(2, DataType::UInt8));
            m_tagScaleB = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(scaleB));

            tagB = m_command->addOperation(
                rocRoller::Operations::BlockScale(tagB, 2, m_tagScaleB, {32, 1}));
        }

        m_tagD = m_command->addOperation(rocRoller::Operations::T_Mul(tagA, tagB)); // D = A * B

        auto tagTensorD = m_command->addOperation(rocRoller::Operations::Tensor(2, m_cdType)); // D
        m_command->addOperation(rocRoller::Operations::T_Store_Tiled(m_tagD, tagTensorD));
    }

    CommandPtr MatrixMultiply::getCommand()
    {
        return m_command;
    }

    KernelGraph MatrixMultiply::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(m_command);
    }

    void MatrixMultiply::setTileSize(int m, int n, int k)
    {
        m_macM = m;
        m_macN = n;
        m_macK = k;
    }

    void MatrixMultiply::setMFMA(int m, int n, int k, int b)
    {
        m_waveM = m;
        m_waveN = n;
        m_waveK = k;
        m_waveB = b;
    }

    void MatrixMultiply::setUseLDS(bool a, bool b, bool d)
    {
        m_useLDSA = a;
        m_useLDSB = b;
        m_useLDSD = d;
    }

    std::shared_ptr<CommandParameters> MatrixMultiply::getCommandParameters() const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        {
            auto macTileA = MacroTile({m_macM, m_macK},
                                      LayoutType::MATRIX_A,
                                      {m_waveM, m_waveN, m_waveK, m_waveB},
                                      m_useLDSA ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            params->setDimensionInfo(m_tagA, macTileA);
        }
        if(m_aMode == Operations::ScaleMode::Separate)
        {
            auto macTileScaleA = MacroTile({m_macM, m_macK / 32},
                                           LayoutType::MATRIX_A,
                                           {m_waveM, m_waveN, m_waveK / 32, m_waveB},
                                           m_useLDSA ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            params->setDimensionInfo(m_tagScaleA, macTileScaleA);
        }
        {
            auto macTileB = MacroTile({m_macK, m_macN},
                                      LayoutType::MATRIX_B,
                                      {m_waveM, m_waveN, m_waveK, m_waveB},
                                      m_useLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            params->setDimensionInfo(m_tagB, macTileB);
        }
        if(m_bMode == Operations::ScaleMode::Separate)
        {
            auto macTileScaleB = MacroTile({m_macK, m_macN / 32},
                                           LayoutType::MATRIX_B,
                                           {m_waveM, m_waveN, m_waveK / 32, m_waveB},
                                           m_useLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);
            params->setDimensionInfo(m_tagScaleB, macTileScaleB);
        }
        {
            auto macTileD = MacroTile({m_macM, m_macN},
                                      LayoutType::MATRIX_ACCUMULATOR,
                                      {m_waveM, m_waveN, m_waveK, m_waveB});
            params->setDimensionInfo(m_tagD, macTileD);
        }

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
     * GEMM
     */

    GEMM::GEMM(DataType ta)
        : GEMM(ta, ta)
    {
    }
    GEMM::GEMM(DataType ta, DataType tb)
        : GEMM(ta, tb, tb)
    {
    }
    GEMM::GEMM(DataType ta, DataType tb, DataType tc)
        : GEMM(ta, tb, tc, tc)
    {
    }
    GEMM::GEMM(DataType ta, DataType tb, DataType tc, DataType td)
        : m_ta(ta)
        , m_tb(tb)
        , m_tc(tc)
        , m_td(td)
    {
    }

    void GEMM::createCommand()
    {
        m_command = std::make_shared<rocRoller::Command>();

        std::vector<size_t> oneStridesN
            = m_problem.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

        std::vector<size_t> oneStridesT = m_problem.literalStrides
                                              ? std::vector<size_t>({(size_t)0, (size_t)1})
                                              : std::vector<size_t>({});

        auto tagTensorA = m_command->addOperation(rocRoller::Operations::Tensor(
            2, m_ta, m_problem.transA == "N" ? oneStridesN : oneStridesT)); // A
        m_tagA          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = m_command->addOperation(rocRoller::Operations::Tensor(
            2, m_tb, m_problem.transB == "N" ? oneStridesN : oneStridesT)); // B
        m_tagB          = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagTensorC
            = m_command->addOperation(rocRoller::Operations::Tensor(2, m_tc, oneStridesN)); // C
        m_tagC = m_command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

        auto tagScalarAlpha
            = m_command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
        auto tagLoadAlpha
            = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

        auto tagScalarBeta = m_command->addOperation(rocRoller::Operations::Scalar(m_tc)); // beta
        auto tagLoadBeta
            = m_command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta)); // beta

        auto tagAB = m_command->addOperation(rocRoller::Operations::T_Mul(m_tagA, m_tagB)); // A * B

        rocRoller::Operations::T_Execute execute(m_command->getNextTag());

        auto tagBetaC
            = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, m_tagC)); // beta * C

        auto tagAlphaAB
            = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)

        if(m_problem.betaInFma)
        {
            m_tagD = execute.addXOp(rocRoller::Operations::E_Add(tagBetaC, tagAlphaAB));
            // alpha * (A * B) + beta * C
        }
        else
        {
            m_tagD = execute.addXOp(rocRoller::Operations::E_Add(tagAlphaAB, tagBetaC));
            // alpha * (A * B) + beta * C
        }
        m_command->addOperation(std::move(execute));

        auto tagTensorD
            = m_command->addOperation(rocRoller::Operations::Tensor(2, m_td, oneStridesN)); // D
        m_command->addOperation(rocRoller::Operations::T_Store_Tiled(m_tagD, tagTensorD)); // D

        if(m_problem.streamK)
        {
            m_tagNumWGs    = m_command->allocateTag();
            auto numWGsArg = m_command->allocateArgument(DataType::UInt32,
                                                         m_tagNumWGs,
                                                         ArgumentType::Value,
                                                         DataDirection::ReadOnly,
                                                         rocRoller::NUMWGS);
        }

        auto tagScratch = m_command->allocateTag();
        m_command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                    tagScratch,
                                    ArgumentType::Value,
                                    DataDirection::ReadWrite,
                                    rocRoller::SCRATCH);
    }

    CommandPtr GEMM::getCommand()
    {
        if(!m_command)
            createCommand();

        return m_command;
    }

    KernelGraph GEMM::getKernelGraph()
    {
        return rocRoller::KernelGraph::translate(getCommand());
    }

    void GEMM::setTileSize(int m, int n, int k)
    {
        m_problem.macM = m;
        m_problem.macN = n;
        m_problem.macK = k;
    }

    void GEMM::setMFMA(int m, int n, int k, int b)
    {
        m_problem.waveM = m;
        m_problem.waveN = n;
        m_problem.waveK = k;
        m_problem.waveB = b;
    }

    void GEMM::setUseLDS(bool a, bool b, bool d)
    {
        m_problem.loadLDSA  = a;
        m_problem.loadLDSB  = b;
        m_problem.storeLDSD = d;
    }

    void GEMM::setPrefetch(bool prefetch,
                           int  prefetchInFlight,
                           int  prefetchLDSFactor,
                           bool prefetchMixMemOps)
    {
        m_problem.prefetch          = prefetch;
        m_problem.prefetchInFlight  = prefetchInFlight;
        m_problem.prefetchLDSFactor = prefetchLDSFactor;
        m_problem.prefetchMixMemOps = prefetchMixMemOps;

        m_problem.unrollK = prefetchInFlight;
    }

    void GEMM::setUnroll(unsigned int unrollX, unsigned int unrollY)
    {
        m_problem.unrollX = unrollX;
        m_problem.unrollY = unrollY;
    }

    void GEMM::setProblem(GEMMProblem const& problem)
    {
        m_problem = problem;
    }

    CommandParametersPtr GEMM::getCommandParameters() const
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto params = std::make_shared<CommandParameters>();

        params->setManualKernelDimension(2);

        AssertFatal(m_problem.workgroupSizeX % m_problem.wavefrontSize == 0,
                    "Workgroup Size X must be multiply of wave front size");

        uint wavetilePerWavefrontM
            = m_problem.wavefrontSize * m_problem.macM / m_problem.waveM / m_problem.workgroupSizeX;
        uint wavetilePerWavefrontN = m_problem.macN / m_problem.waveN / m_problem.workgroupSizeY;

        AssertFatal(m_problem.macM % (m_problem.waveM * wavetilePerWavefrontM) == 0,
                    "WaveTile size mismatch (M)");
        AssertFatal(m_problem.macN % (m_problem.waveN * wavetilePerWavefrontN) == 0,
                    "WaveTile size mismatch (N)");

        uint workgroupSizeX = m_problem.workgroupSizeX * m_problem.workgroupSizeY;
        uint workgroupSizeY = 1;
        params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});

        auto macTileA
            = MacroTile({m_problem.macM, m_problem.macK},
                        LayoutType::MATRIX_A,
                        {m_problem.waveM, m_problem.waveN, m_problem.waveK, m_problem.waveB},
                        m_problem.loadLDSA ? MemoryType::WAVE_LDS : MemoryType::WAVE);
        auto macTileB
            = MacroTile({m_problem.macK, m_problem.macN},
                        LayoutType::MATRIX_B,
                        {m_problem.waveM, m_problem.waveN, m_problem.waveK, m_problem.waveB},
                        m_problem.loadLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);
        auto macTileC
            = MacroTile({m_problem.macM, m_problem.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {m_problem.waveM, m_problem.waveN, m_problem.waveK, m_problem.waveB});
        auto macTileD
            = MacroTile({m_problem.macM, m_problem.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {m_problem.waveM, m_problem.waveN, m_problem.waveK, m_problem.waveB},
                        m_problem.storeLDSD ? MemoryType::WAVE_LDS : MemoryType::WAVE);

        params->setDimensionInfo(m_tagA, macTileA);
        params->setDimensionInfo(m_tagB, macTileB);
        params->setDimensionInfo(m_tagC, macTileC);
        params->setDimensionInfo(m_tagD, macTileD);

        // Workgroup size
        // uint wavefrontSize  = 64;
        // uint workgroupSizeX = 2 * wavefrontSize;
        // uint workgroupSizeY = 4;

        Log::debug("GEMM workgroup sizes {} {} {}", workgroupSizeX, workgroupSizeY, 1);
        Log::debug("GEMM jamming {} {}", wavetilePerWavefrontM, wavetilePerWavefrontN);

        params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);

        // params->setManualWavefrontCount({2, 2});
        params->setManualWavefrontCount(
            {static_cast<uint>(m_problem.macM / m_problem.waveM / wavetilePerWavefrontM),
             static_cast<uint>(m_problem.macN / m_problem.waveN / wavetilePerWavefrontN)});

        params->fuseLoops                     = m_problem.fuseLoops;
        params->tailLoops                     = m_problem.tailLoops;
        params->allowAmbiguousMemoryNodes     = m_problem.allowAmbiguousMemoryNodes;
        params->unrollX                       = m_problem.unrollX;
        params->unrollY                       = m_problem.unrollY;
        params->unrollK                       = m_problem.unrollK;
        params->packMultipleElementsInto1VGPR = m_problem.packMultipleElementsInto1VGPR;
        params->prefetch                      = m_problem.prefetch;
        params->prefetchInFlight              = m_problem.prefetchInFlight;
        params->prefetchLDSFactor             = m_problem.prefetchLDSFactor;
        params->prefetchMixMemOps             = m_problem.prefetchMixMemOps;
        // params->prefetchMixMemOps             = true;
        params->transposeMemoryAccess[LayoutType::MATRIX_A] = m_problem.transA == "T";
        params->transposeMemoryAccess[LayoutType::MATRIX_B] = m_problem.transB == "T";
        // params->transposeMemoryAccess[LayoutType::None]     = false;

        if(m_problem.streamK)
        {
            params->loopOverOutputTilesDimensions = {0, 1};
            params->streamK                       = true;
            params->streamKTwoTile                = m_problem.streamKTwoTile;
        }

        return params;
    }
}
