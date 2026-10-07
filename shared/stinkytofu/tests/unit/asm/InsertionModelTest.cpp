// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Each insertion model inserts exactly what its pass inserts: run the pass on one copy of
// every function of the pass's filecheck input, the model on another, and compare the
// instruction sequences block by block.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/PrefetchBridgeSubstitutionPass.hpp"
#include "transforms/asm/coissue/InsertionModels.hpp"

using namespace stinkytofu;
using namespace stinkytofu::coissue;

namespace {

constexpr std::array<int, 3> kArch{12, 5, 0};

std::string readFilecheck(const std::string& name) {
    const auto path =
        std::filesystem::path(__FILE__).parent_path() / ".." / ".." / "filecheck" / name;
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// What an instruction is, as far as the inserting passes are concerned.
std::string signature(const StinkyInstruction& inst) {
    std::ostringstream os;
    os << inst.getHwInstDesc()->mnemonic;
    auto regs = [&](const std::vector<StinkyRegister>& list) {
        for (const StinkyRegister& r : list) {
            if (r.isRegister())
                os << " r" << static_cast<int>(r.reg.type) << ":" << r.reg.idx << "x" << r.reg.num;
            else if (r.dataType == StinkyRegister::Type::LiteralInt)
                os << " #" << r.getLiteralInt();
            else if (r.dataType == StinkyRegister::Type::LiteralString)
                os << " '" << r.literalValue << "'";
        }
    };
    regs(inst.getDestRegs());
    os << " <-";
    regs(inst.getSrcRegs());
    if (const auto* w = inst.getModifier<SWaitAluData>()) {
        for (auto f : {SWaitAluData::VA_VDST, SWaitAluData::VM_VSRC, SWaitAluData::HOLD_CNT})
            os << " f" << static_cast<int>(f) << "=" << (w->hasField(f) ? int(w->getField(f)) : -1);
    }
    return os.str();
}

using ModelFactory = std::function<std::unique_ptr<InsertionModel>(
    Function&, const std::vector<const BasicBlock*>&, const PassContext&, InstructionPool&)>;

struct Case {
    const char* file;
    bool buildCfg;
    std::function<std::unique_ptr<Pass>()> pass;
    ModelFactory model;
    VgprMsbMode msbMode = VgprMsbMode::None;
    /// The input itself instead of a filecheck file.
    const char* text = nullptr;
    /// Only the block with this label is the model's scope; every block is otherwise.
    const char* scopeLabel = nullptr;
};

// Instructions the pass adds that the model leaves out on purpose: the mode2 enable at the
// kernel entry is not inside any loop.
bool ignored(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_setreg_IMM32_b32;
}

void checkCase(const Case& c) {
    const std::string text = c.text != nullptr ? c.text : readFilecheck(c.file);
    ASSERT_FALSE(text.empty()) << c.file;
    auto inScope = [&](const BasicBlock& bb) {
        return c.scopeLabel == nullptr || bb.getLabel() == c.scopeLabel;
    };
    MultiParseResult forPass = parseAllSourceStringsWithDiagnostics(text);
    MultiParseResult forModel = parseAllSourceStringsWithDiagnostics(text);
    ASSERT_FALSE(forPass.hasErrors()) << c.file;
    ASSERT_EQ(forPass.functions.size(), forModel.functions.size());
    const GfxArchID arch = getGfxArchID(kArch[0], kArch[1], kArch[2]);

    int insertionsSeen = 0;
    for (size_t f = 0; f < forPass.functions.size(); ++f) {
        const std::string& name = forPass.functions[f]->funcName;
        Function passFunc(name), modelFunc(name);
        ASSERT_EQ(
            StinkyIRConverter::populateFunctionFromParsed(*forPass.functions[f], passFunc, arch),
            StinkyErrorCode::SUCCESS);
        ASSERT_EQ(
            StinkyIRConverter::populateFunctionFromParsed(*forModel.functions[f], modelFunc, arch),
            StinkyErrorCode::SUCCESS);
        GemmTileConfig config;
        config.arch = kArch;
        passFunc.setGemmTileConfig(config);
        modelFunc.setGemmTileConfig(config);
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        AsmCapsConfig caps;
        caps.vgprMsbMode = c.msbMode;
        ctx.setAsmCapsConfig(caps);
        AnalysisManager amPass, amModel;
        registerAllAnalyses(amPass);
        registerAllAnalyses(amModel);
        if (c.buildCfg) {
            createCFGBuilderPass()->run(passFunc, ctx, amPass);
            createCFGBuilderPass()->run(modelFunc, ctx, amModel);
        }

        std::vector<const BasicBlock*> scope;
        std::vector<std::vector<const StinkyInstruction*>> orders;
        for (BasicBlock& bb : modelFunc) {
            if (!inScope(bb)) continue;
            scope.push_back(&bb);
            auto& order = orders.emplace_back();
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) order.push_back(inst);
        }
        InstructionPool pool(arch);
        PredictedBlock predicted;
        predicted.blocks = scope;
        predicted.seqs = orders;
        c.model(modelFunc, scope, ctx, pool)->apply(predicted);

        c.pass()->run(passFunc, ctx, amPass);

        size_t b = 0;
        for (BasicBlock& bb : passFunc) {
            if (!inScope(bb)) continue;
            ASSERT_LT(b, predicted.seqs.size()) << c.file << " @" << name;
            std::vector<std::string> real, model;
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node))
                    if (!ignored(*inst)) real.push_back(signature(*inst));
            for (const StinkyInstruction* inst : predicted.seqs[b]) {
                model.push_back(signature(*inst));
                insertionsSeen += pool.owns(inst) ? 1 : 0;
            }
            EXPECT_EQ(model, real) << c.file << " @" << name << " block " << b;
            ++b;
        }
        EXPECT_EQ(b, predicted.seqs.size()) << c.file << " @" << name;
    }
    // The inputs exercise the passes: some function must get an insertion.
    EXPECT_GT(insertionsSeen, 0) << c.file;
}

ModelFactory vgprMsb(VgprMsbMode mode) {
    return [mode](Function&, const std::vector<const BasicBlock*>&, const PassContext&,
                  InstructionPool& pool) { return makeVgprMsbModel(mode, pool); };
}

ModelFactory waitAlu(InsertWaitAluOptions opts) {
    return
        [opts](Function& func, const std::vector<const BasicBlock*>& scope, const PassContext& ctx,
               InstructionPool& pool) { return makeWaitAluModel(func, scope, ctx, opts, pool); };
}

std::function<std::unique_ptr<Pass>()> waitAluPass(InsertWaitAluOptions opts) {
    return [opts] { return createInsertWaitAluPass(opts); };
}

}  // namespace

TEST(InsertionModelTest, VgprMsbMatchesPass) {
    for (const char* file : {"insert_vgpr_msb_call_reset.stir", "insert_vgpr_msb_multi_func.stir",
                             "insert_vgpr_msb_prefer_after.stir"})
        checkCase({file, false, [] { return createInsertVgprMsbPass(); },
                   vgprMsb(VgprMsbMode::Msb8), VgprMsbMode::Msb8});
}

TEST(InsertionModelTest, BridgeMatchesPass) {
    checkCase({"PrefetchBridgeSubstitutionPass_test.stir", true,
               [] { return createPrefetchBridgeSubstitutionPass(); },
               [](Function& func, const std::vector<const BasicBlock*>& scope,
                  const PassContext& ctx, InstructionPool& pool) {
                   return makeBridgeModel(func, scope, ctx.getHWModel().waitHide.vmVsrcBridge,
                                          pool);
               }});
}

TEST(InsertionModelTest, WaitAluMatchesPass) {
    const InsertWaitAluOptions none{};
    const InsertWaitAluOptions vsrc{true, false, false};
    const InsertWaitAluOptions shared{true, true, false};
    const InsertWaitAluOptions nextWmma{true, false, true};
    checkCase({"InsertWaitAluPass_test.stir", false, waitAluPass(vsrc), waitAlu(vsrc)});
    checkCase({"InsertWaitAluPass_anchor_test.stir", true, waitAluPass(none), waitAlu(none)});
    checkCase(
        {"InsertWaitAluPass_paired_flat_join_test.stir", true, waitAluPass(none), waitAlu(none)});
    checkCase(
        {"InsertWaitAluPass_shared_order_test.stir", true, waitAluPass(shared), waitAlu(shared)});
    checkCase({"InsertWaitAluPass_vmvsrc_hide_test.stir", false, waitAluPass(none), waitAlu(none)});
    checkCase({"InsertWaitAluPass_xdl_hide_test.stir", false, waitAluPass(vsrc), waitAlu(vsrc)});
    checkCase({"InsertWaitAluPass_xdl_next_wmma_test.stir", false, waitAluPass(nextWmma),
               waitAlu(nextWmma)});
}

// The scope is the loop alone, yet what runs before it still decides its waits: the load
// in front of the loop reads v3, which the loop's first VALU overwrites.
TEST(InsertionModelTest, WaitAluSeesWhatRunsBeforeTheLoop) {
    const InsertWaitAluOptions vsrc{true, false, false};
    Case c{"inline", false, waitAluPass(vsrc), waitAlu(vsrc)};
    c.text = R"(
st.func @before_loop() {
^entry:
  v10 = "st.buffer_load_b32"(v3)
  Successors: ^label_loop
^label_loop:
  v3 = "st.v_mov_b32"(v4)
  s10, SCC0 = "st.s_add_i32"(s10, -1)
  SCC0 = "st.s_cmp_eq_u32"(s10, 0)
  "st.s_cbranch_scc0"(label_loop, SCC0)
  Successors: ^label_loop, ^label_end
^label_end:
  "st.s_endpgm"()
}
)";
    c.scopeLabel = "label_loop";
    checkCase(c);
}

TEST(InsertionModelTest, CoexecNopMatchesPass) {
    checkCase(
        {"InsertCoexecHazardPass_test.stir", false, [] { return createInsertCoexecHazardPass(); },
         [](Function& func, const std::vector<const BasicBlock*>&, const PassContext& ctx,
            InstructionPool& pool) { return makeCoexecNopModel(func, ctx.getHWModel(), pool); }});
}
