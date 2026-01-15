/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/// WaitCntLegalizationPass for gfx1250 (CDNA5) architecture
///
/// This pass lowers deprecated s_waitcnt and s_barrier instructions into
/// architecture-specific wait and barrier instructions for gfx1250.
///
/// On gfx1250, instead of s_waitcnt, separate wait instructions are provided:
/// - s_wait_loadcnt: Wait for VMEM loads
/// - s_wait_storecnt: Wait for VMEM stores
/// - s_wait_dscnt: Wait for LDS/DS operations
/// - s_wait_kmcnt: Wait for scalar memory/constant fetch
/// - s_wait_asynccnt: Wait for async operations
///
/// Combined instructions (for efficiency):
/// - s_wait_loadcnt_dscnt: Wait for both VMEM loads and DS operations
/// - s_wait_storecnt_dscnt: Wait for both VMEM stores and DS operations
///
/// On gfx1250, s_barrier is also deprecated and replaced with:
/// - s_barrier_signal <barrier_id>: Signal a barrier
/// - s_barrier_wait <barrier_id>: Wait on a barrier
///
/// Example transformations:
///   s_waitcnt vmcnt(2) lgkmcnt(0)  // Before (deprecated)
///   →
///   s_wait_loadcnt_dscnt 0x0200    // After (gfx1250)
///
///   s_barrier                       // Before (deprecated)
///   →
///   s_barrier_signal -1             // After (gfx1250)
///   s_barrier_wait -1
///
/// The pass:
/// - Detects s_waitcnt and s_barrier instructions in the IR
/// - For s_waitcnt: extracts SWaitCntData modifier and generates optimal wait sequence
/// - For s_barrier: replaces with s_barrier_signal/s_barrier_wait pair
/// - Uses combined instructions when multiple counters are active
/// - Removes the original deprecated instructions
///
/// Architecture gating: Only runs for gfx1250
///
/// Usage: Should be added at the END of the optimization pipeline, after all
/// scheduling and optimization passes, but before final code emission.

#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyModifiers.hpp"
#include "isa/ArchHelper.hpp"
#include "support/Casting.hpp"

#include <optional>
#include <vector>

namespace
{
    using namespace stinkytofu;

    /// Implementation of the WaitCntLegalizationPass
    class WaitCntLegalizationPassImpl : public Pass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "WaitCntLegalizationPass";
        }

        PassID getPassID() const override
        {
            return &WaitCntLegalizationPassImpl::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            // Only run for gfx1250 where s_waitcnt is deprecated
            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);

            if(arch != GfxArchID::Gfx1250)
            {
                // Not gfx1250, skip legalization
                return;
            }

            // Process each BasicBlock
            for(BasicBlock& bb : func)
                processBasicBlock(bb, func, passCtx, arch);
        }

    private:
        int processBasicBlock(BasicBlock& bb, Function& func, PassContext& passCtx, GfxArchID arch)
        {
            IRList& irlist = bb.getIR();

            // Collect s_waitcnt instructions to replace
            struct WaitCntToReplace
            {
                IRList::iterator           iter;
                const SWaitCntData         waitData;
                std::optional<std::string> comment;
            };
            std::vector<WaitCntToReplace> waitCntsToReplace;

            // Collect s_barrier instructions to replace
            std::vector<IRList::iterator> barriersToReplace;

            // Find all s_waitcnt and s_barrier instructions
            for(auto it = irlist.begin(); it != irlist.end(); ++it)
            {
                if(it->getType() != IRBase::IRType::StinkyTofu)
                    continue;

                StinkyInstruction* inst = cast<StinkyInstruction>(&*it);
                if(inst->getUnifiedOpcode() == GFX::s_waitcnt)
                {
                    const SWaitCntData* waitData = inst->getModifier<SWaitCntData>();
                    if(waitData)
                    {
                        std::optional<std::string> comment;
                        if(const CommentData* commentData = inst->getModifier<CommentData>())
                        {
                            comment = commentData->comment;
                        }
                        waitCntsToReplace.push_back({it, *waitData, comment});
                    }
                }
                else if(inst->getUnifiedOpcode() == GFX::s_barrier)
                {
                    barriersToReplace.push_back(it);
                }
            }

            if(waitCntsToReplace.empty() && barriersToReplace.empty())
                return 0;

            // Create IRBuilder
            auto irBuilder = passCtx.getIRBuilder<StinkyInstIRBuilder>(irlist, arch);

            // Replace each s_waitcnt with new wait instructions
            for(const auto& entry : waitCntsToReplace)
            {
                legalizeWaitCnt(entry.iter, entry.waitData, entry.comment, irBuilder, irlist, arch);
            }

            // Replace each s_barrier with signal/wait pair
            for(const auto& barrierIter : barriersToReplace)
            {
                legalizeBarrier(barrierIter, irBuilder, irlist, arch);
            }

            return waitCntsToReplace.size() + barriersToReplace.size();
        }

        void legalizeWaitCnt(IRList::iterator                  waitCntIter,
                             const SWaitCntData&               waitData,
                             const std::optional<std::string>& comment,
                             StinkyInstIRBuilder&              irBuilder,
                             IRList&                           irlist,
                             GfxArchID                         arch)
        {
            // The wait count values in SWaitCntData:
            // - vlcnt: VMEM load count
            // - vscnt: VMEM store count
            // - dlcnt: DS/LDS load count (deprecated, combined with dscnt on newer architectures)
            // - dscnt: DS/LDS store count
            // - kmcnt: Scalar memory/constant fetch count
            //
            // Value -1 means "don't wait" (ignore this counter)
            // Value >= 0 means "wait until outstanding count <= value"

            bool hasVlcnt = (waitData.vlcnt != -1);
            bool hasVscnt = (waitData.vscnt != -1);
            bool hasDlcnt = (waitData.dlcnt != -1);
            bool hasDscnt = (waitData.dscnt != -1);
            bool hasKmcnt = (waitData.kmcnt != -1);

            // Combine dlcnt and dscnt into a single dscnt for gfx1250
            // On gfx1250, there's a single DS counter (not separate load/store)
            int8_t combinedDscnt = -1;
            if(hasDlcnt && hasDscnt)
            {
                // Both specified: use the minimum (most restrictive)
                combinedDscnt = std::min(waitData.dlcnt, waitData.dscnt);
            }
            else if(hasDlcnt)
            {
                combinedDscnt = waitData.dlcnt;
            }
            else if(hasDscnt)
            {
                combinedDscnt = waitData.dscnt;
            }

            bool hasCombinedDs = (combinedDscnt != -1);

            // Strategy: Use combined instructions when possible to minimize instruction count
            // Priority:
            // 1. s_wait_loadcnt_dscnt if both vlcnt and ds are needed
            // 2. s_wait_storecnt_dscnt if both vscnt and ds are needed
            // 3. Separate s_wait_loadcnt, s_wait_storecnt, s_wait_dscnt, s_wait_kmcnt as needed

            // Insert new wait instructions before the old s_waitcnt
            auto insertPoint = waitCntIter;

            // Case 1: Both VMEM load and DS operations
            if(hasVlcnt && hasCombinedDs)
            {
                createCombinedWaitInst(GFX::s_wait_loadcnt_dscnt,
                                       waitData.vlcnt,
                                       combinedDscnt,
                                       comment,
                                       irBuilder,
                                       insertPoint,
                                       arch);
                hasVlcnt      = false; // Already handled
                hasCombinedDs = false;
            }
            // Case 2: Both VMEM store and DS operations
            else if(hasVscnt && hasCombinedDs)
            {
                createCombinedWaitInst(GFX::s_wait_storecnt_dscnt,
                                       waitData.vscnt,
                                       combinedDscnt,
                                       comment,
                                       irBuilder,
                                       insertPoint,
                                       arch);
                hasVscnt      = false; // Already handled
                hasCombinedDs = false;
            }

            // Case 3: Separate wait instructions for remaining counters
            if(hasVlcnt)
            {
                createSingleWaitInst(
                    GFX::s_wait_loadcnt, waitData.vlcnt, comment, irBuilder, insertPoint, arch);
            }
            if(hasVscnt)
            {
                createSingleWaitInst(
                    GFX::s_wait_storecnt, waitData.vscnt, comment, irBuilder, insertPoint, arch);
            }
            if(hasCombinedDs)
            {
                createSingleWaitInst(
                    GFX::s_wait_dscnt, combinedDscnt, comment, irBuilder, insertPoint, arch);
            }
            if(hasKmcnt)
            {
                createSingleWaitInst(
                    GFX::s_wait_kmcnt, waitData.kmcnt, comment, irBuilder, insertPoint, arch);
            }

            // Remove the old s_waitcnt instruction
            irlist.erase(waitCntIter);
        }

        void legalizeBarrier(IRList::iterator     barrierIter,
                             StinkyInstIRBuilder& irBuilder,
                             IRList&              irlist,
                             GfxArchID            arch)
        {
            // On gfx1250, s_barrier is deprecated and replaced with:
            // - s_barrier_signal <barrier_id>: Signal a barrier
            // - s_barrier_wait <barrier_id>: Wait on a barrier
            //
            // The barrier_id is typically -1 for a global barrier where all threads
            // participate. Split barriers (where only some threads participate) use
            // specific barrier IDs (0-31).

            auto insertPoint = barrierIter;

            // Create s_barrier_signal -1 (signal global barrier)
            {
                const HwInstDesc*  desc = getMCIDByUOp(GFX::s_barrier_signal, arch);
                StinkyInstruction* inst = irBuilder.createStinkyInstBefore(insertPoint, desc);
                inst->addSrcReg(StinkyRegister(-1)); // -1 = global barrier
            }

            // Create s_barrier_wait -1 (wait on global barrier)
            {
                const HwInstDesc*  desc = getMCIDByUOp(GFX::s_barrier_wait, arch);
                StinkyInstruction* inst = irBuilder.createStinkyInstBefore(insertPoint, desc);
                inst->addSrcReg(StinkyRegister(-1)); // -1 = global barrier
            }

            // Remove the old s_barrier instruction
            irlist.erase(barrierIter);
        }

        void createSingleWaitInst(GFX                               unifiedOpcode,
                                  int8_t                            count,
                                  const std::optional<std::string>& comment,
                                  StinkyInstIRBuilder&              irBuilder,
                                  IRList::iterator                  insertPoint,
                                  GfxArchID                         arch)
        {
            const HwInstDesc* desc = getMCIDByUOp(unifiedOpcode, arch);

            StinkyInstruction* inst = irBuilder.createStinkyInstBefore(insertPoint, desc);

            // Add the count as a literal operand
            // The wait instruction format is: s_wait_xxx <count>
            inst->addSrcReg(StinkyRegister(static_cast<int>(count)));
            if(comment && !comment->empty())
            {
                inst->addModifier<CommentData>(CommentData{*comment});
            }
        }

        void createCombinedWaitInst(GFX                               unifiedOpcode,
                                    int8_t                            count1,
                                    int8_t                            count2,
                                    const std::optional<std::string>& comment,
                                    StinkyInstIRBuilder&              irBuilder,
                                    IRList::iterator                  insertPoint,
                                    GfxArchID                         arch)
        {
            const HwInstDesc* desc = getMCIDByUOp(unifiedOpcode, arch);

            StinkyInstruction* inst = irBuilder.createStinkyInstBefore(insertPoint, desc);

            // Combined wait instruction format: s_wait_xxx_yyy <count1_count2>
            // The encoding combines two 8-bit counts into a single 16-bit immediate
            // SIMM16[15:8] = first count (load/store)
            // SIMM16[7:0] = second count (DS)
            uint16_t combinedCount = ((count1 & 0xFF) << 8) | (count2 & 0xFF);
            inst->addSrcReg(StinkyRegister(static_cast<int>(combinedCount)));
            if(comment && !comment->empty())
            {
                inst->addModifier<CommentData>(CommentData{*comment});
            }
        }
    };

    char WaitCntLegalizationPassImpl::ID = 0;

} // anonymous namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createWaitCntLegalizationPass()
    {
        return std::make_unique<WaitCntLegalizationPassImpl>();
    }

} // namespace stinkytofu
