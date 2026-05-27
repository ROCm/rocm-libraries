// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "emit_mfma.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/mfma.hpp"

namespace tl_emit
{
    std::shared_ptr<rocisa::Module> emitMfmaInstruction(rocisa::InstType mxInstType,
                                                        int              miK,
                                                        bool             sourceSwap,
                                                        bool             miArchVgpr,
                                                        int              vgprAStart,
                                                        int              opASize,
                                                        int              vgprBStart,
                                                        int              opBSize,
                                                        int              vgprCStart,
                                                        int              opCSize,
                                                        bool             cIsAccvgpr,
                                                        int              vgprDStart,
                                                        int              opDSize,
                                                        bool             dIsAccvgpr,
                                                        int              scaleAVgpr,
                                                        int              scaleBVgpr,
                                                        int              scaleAsel,
                                                        int              scaleBsel,
                                                        int              tmpScaleVgpr,
                                                        const std::string& comment)
    {
        using namespace rocisa;

        auto module = std::make_shared<Module>();

        auto dAcc = (miArchVgpr || !dIsAccvgpr) ? vgpr(vgprDStart, opDSize)
                                                 : accvgpr(vgprDStart, opDSize);
        auto cAcc = (miArchVgpr || !cIsAccvgpr) ? vgpr(vgprCStart, opCSize)
                                                 : accvgpr(vgprCStart, opCSize);

        auto aOperand = sourceSwap ? vgpr(vgprBStart, opBSize) : vgpr(vgprAStart, opASize);
        auto bOperand = sourceSwap ? vgpr(vgprAStart, opASize) : vgpr(vgprBStart, opBSize);

        if(miK == 128)
        {
            std::vector<int> variant = {16, 16, miK, 1};

            if(scaleAVgpr >= 0 && scaleBVgpr >= 0)
            {
                VOP3PModifiers vop3({scaleAsel % 2, scaleBsel % 2},
                                    {(scaleAsel >> 1) % 2, (scaleBsel >> 1) % 2});
                module->addT<MXMFMAInstruction>(mxInstType,
                                                InstType::INST_F32,
                                                variant,
                                                dAcc,
                                                aOperand,
                                                bOperand,
                                                cAcc,
                                                vgpr(scaleAVgpr),
                                                vgpr(scaleBVgpr),
                                                vop3,
                                                InstType::INST_F32,
                                                InstType::INST_F32,
                                                0,
                                                comment);
            }
            else
            {
                module->addT<VMovB32>(vgpr(tmpScaleVgpr),
                                      static_cast<int>(0x7f7f7f7f),
                                      std::nullopt,
                                      "hardcoded scale 0x7f (E8M0)");
                module->addT<MXMFMAInstruction>(mxInstType,
                                                InstType::INST_F32,
                                                variant,
                                                dAcc,
                                                aOperand,
                                                bOperand,
                                                cAcc,
                                                vgpr(tmpScaleVgpr),
                                                vgpr(tmpScaleVgpr),
                                                std::nullopt,
                                                InstType::INST_F32,
                                                InstType::INST_F32,
                                                0,
                                                comment);
            }
        }
        else
        {
            module->addT<MFMAInstruction>(InstType::INST_BF16,
                                          InstType::INST_F32,
                                          std::vector<int>{16, 16, miK, 1},
                                          false,
                                          dAcc,
                                          aOperand,
                                          bOperand,
                                          std::optional<InstructionInput>(cAcc),
                                          false,
                                          comment);
        }

        return module;
    }
} // namespace tl_emit
