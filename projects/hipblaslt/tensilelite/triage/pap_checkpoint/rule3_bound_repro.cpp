// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPassTestSupport.hpp"

using namespace stinkytofu;

int main(int argc, char** argv) {
    const int maxHops = argc > 1 ? std::atoi(argv[1]) : 0;
    const bool leadingDrain = argc > 2 && std::atoi(argv[2]);
    const GfxArchID arch = GfxArchID::Gfx1250;
    Function func("bounded_rule3_cluster_wait");
    BasicBlock* bb = func.createBasicBlock("entry");
    AsmIRBuilder builder(*bb, arch);
    auto barrier = [&](GFX opcode, int id) {
        auto* inst = builder.create(getMCIDByUOp(opcode, arch));
        inst->addSrcReg(StinkyRegister(id));
        return inst;
    };
    auto mov = [&] {
        auto* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, arch));
        inst->addDestReg(StinkyRegister("s", 20, 1));
        inst->addSrcReg(StinkyRegister(1));
        return inst;
    };
    builder.createLabel("label_loop");
    auto* loopHead = dyn_cast<StinkyInstruction>(std::prev(bb->end()).getNodePtr());
    barrier(GFX::s_barrier_wait, -3);
    mov();
    StinkyInstruction* waitAnchor = nullptr;
    if (leadingDrain) waitAnchor = barrier(GFX::s_wait_tensorcnt, 0);
    auto* trigger = barrier(GFX::s_barrier_signal, -1);
    if (!waitAnchor) waitAnchor = trigger;
    barrier(GFX::s_barrier_wait, -1);
    auto* load = builder.create(getMCIDByUOp(GFX::tensor_load_to_lds, arch));
    load->addSrcReg(StinkyRegister("s", 0, 4));
    load->addSrcReg(StinkyRegister("s", 4, 8));
    mov();
    auto* branch = builder.create(getMCIDByUOp(GFX::s_branch, arch));
    branch->addSrcReg(StinkyRegister(std::string("label_loop")));
    branch->addModifier<LabelData>(LabelData{"label_loop"});
    std::unordered_map<const StinkyInstruction*, uint32_t> cycles;
    uint32_t clock = 0;
    for (IRBase& ir : *bb)
        if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) cycles[inst] = clock++;
    const auto result = cluster_barrier::test::findRule3SignalAnchorByCycleLeadForUnitTest(
        waitAnchor, std::next(BasicBlock::iterator(loopHead)), waitAnchor, cycles,
        100, 900, {}, maxHops, loopHead);
    if (result.anchor != waitAnchor || result.hops != 0 || result.outOfSegmentNomination) {
        std::cerr << "unexpected out-of-bound nomination or hop\n";
        return 2;
    }
    std::cout << "PASS maxHops=" << maxHops << " leadingDrain=" << leadingDrain << '\n';
}
