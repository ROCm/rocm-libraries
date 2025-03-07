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

}
