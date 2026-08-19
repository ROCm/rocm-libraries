// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// InsertAsanCheckPass -- debug-only, gfx1250-only.
//
// NOTE ON VALIDATION: this pass was authored and compile-verified against the
// real stinkytofu build, and smoke-tested via stinkytofu-opt on synthetic
// input (confirmed it fires only on tracked-SRD MUBUF accesses and resolves
// register operands to real physical indices). It has NOT been run on real
// gfx1250 hardware (no GPU available in the authoring environment). Treat the
// exact operand/carry-chain sequence below as a first draft pending
// on-hardware verification, per the plan's manual smoke-test step.
//
#include "stinkytofu/transforms/asm/InsertAsanCheckPass.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/AsmSetSymbolMap.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu {
namespace {

// Shadow memory formula reused from real ASan's AMDGPU/x86-64 "small" shadow
// mapping: shadow_addr = (addr >> kShadowScale) | kShadowOffset. This is only
// meaningful when the launching client process is itself built with
// -fsanitize=address (the shadow region is otherwise unmapped/garbage) -- see
// the AsanInstrument doc comment in bindings/python/Module.hpp.
constexpr uint32_t kShadowScale = 3;
constexpr uint32_t kShadowOffset = 0x7fff8000u;

// Symbolic register names Tensile reserves (KernelWriter.py / defineSgpr,
// KernelWriterAssembly.py / RegSet) specifically for this pass. This pass only
// ever references these fixed names -- it never allocates its own scratch,
// per this codebase's convention (see InsertClusterBarrierPass's
// kLoopCounterLSymbol/kWaveIdxSymbol). Resolved to physical indices via the
// `.set` symbol table (see resolveOwnRegisters below) rather than emitted as
// StinkyRegister symbolic names, since symbolic-name printing is opt-in
// (--preserve-symbolic-regs) and off by default in normal codegen -- emitting
// symbolic here would silently print bogus operands otherwise.
constexpr const char* kReportBufSymbol = "sgprAsanReportBuf";
constexpr const char* kSgprTmpSymbol = "sgprAsanTmp";
constexpr const char* kVgprTmpSymbol = "vgprAsanTmp";
constexpr const char* kSkipLabelPrefix = "label_asanCheckOk_";

// Named tensor-buffer SRDs this pass knows how to bounds-check. Buffers not in
// this list (bias/E/scale/StreamK workspace/etc.) are not instrumented in v1
// -- documented scope reduction, not silent gap: these are the direct
// SRD-relative accesses covering the primary A/B/C/D GEMM footprint;
// GroupedGemm pointer-array batch offsets, StreamK partial-tile addressing,
// and TDM-based accesses use different addressing and are not covered here.
constexpr std::array<const char*, 4> kTrackedSrdSymbols = {"sgprSrdA", "sgprSrdB", "sgprSrdC",
                                                           "sgprSrdD"};

// Fixed offsets within the reserved AsanTmp scratch blocks (see doc comments
// on numVgprAsanTmp/numSgprAsanTmp in Tensile's KernelWriter.py).
constexpr int kVaLoOff = 0;        // vgprAsanTmp+0 : VA_lo / shadow_lo
constexpr int kVaHiOff = 1;        // vgprAsanTmp+1 : VA_hi / shadow_hi
constexpr int kShadowByteOff = 2;  // vgprAsanTmp+2 : loaded shadow byte
// NOTE: every 64-bit VGPR pair below must start at an EVEN offset -- AMDGPU
// rejects misaligned 64-bit VGPR operands (v[53:54] fails to assemble). This
// also requires Tensile to reserve the vgprAsanTmp block at an even base.
constexpr int kReportAddrLoOff = 4;  // vgprAsanTmp+4 : AsanReportBuf ptr (vgpr copy, lo)
constexpr int kReportAddrHiOff = 5;  // vgprAsanTmp+5 : AsanReportBuf ptr (vgpr copy, hi)
constexpr int kPcVgprLoOff = 6;      // vgprAsanTmp+6 : PC (vgpr copy, lo)
constexpr int kPcVgprHiOff = 7;      // vgprAsanTmp+7 : PC (vgpr copy, hi)
constexpr int kPcSgprLoOff = 0;      // sgprAsanTmp+0 : PC (s_getpc_b64 result, lo)
constexpr int kPcSgprHiOff = 1;      // sgprAsanTmp+1 : PC (s_getpc_b64 result, hi)

StinkyRegister vgpr(int idx, uint16_t num = 1) {
    return StinkyRegister(RegType::V, static_cast<uint32_t>(idx), num);
}

StinkyRegister sgpr(int idx, uint16_t num = 1) {
    return StinkyRegister(RegType::S, static_cast<uint32_t>(idx), num);
}

// gfx1250 runs wave32, so the carry-out / compare-result lane mask is vcc_lo
// (plain "vcc" is rejected by the assembler in wave32 mode).
StinkyRegister vccReg() {
    return StinkyRegister(RegType::VCC_LO, 0u, 1);
}

// The "off" keyword occupying a global_* saddr slot: the address is fully
// supplied by the 64-bit VGPR-pair vaddr. Emitted as a LiteralString (same
// mechanism MUBUF uses for an "off" vaddr, cf. tests/filecheck/mubuf_off_vaddr.stir),
// which also makes the verifier skip register-width checks on that operand.
StinkyRegister offReg() {
    return StinkyRegister(std::string("off"));
}

// Which operand slots hold vaddr/saddr/soffset differs between MUBUF loads
// (getSrcParams() == {vaddr, saddr, soffset}, rocisa mem.hpp MUBUFReadInstruction)
// and MUBUF stores (getSrcParams() == {srcData, vaddr, saddr, soffset},
// MUBUFStoreInstruction) -- the store's vdata occupies slot 0.
struct MubufAddrOperands {
    StinkyRegister vaddr;
    StinkyRegister saddr;
    StinkyRegister soffset;
};

MubufAddrOperands getMubufAddrOperands(const StinkyInstruction& inst, bool isStore) {
    size_t base = isStore ? 1 : 0;
    return MubufAddrOperands{inst.getSrcReg(base + 0), inst.getSrcReg(base + 1),
                             inst.getSrcReg(base + 2)};
}

// Resolves whether `saddr` (the 4-SGPR SRD range of a MUBUF instruction) is one
// of the tracked tensor buffers, via the `.set sgprSrdX, <idx>` symbols Tensile
// already emits for every named register (rocisa ValueSet). Instruction
// operands carry only a physical index post-conversion (no symbolic name is
// preserved for SGPR ranges), so identity is recovered by matching the
// physical start index against the resolved symbol table -- the same pattern
// SwInstructionPrefetchAbsStaticPass/DynamicPass already use for ShadowLimitA/B.
bool isTrackedSrd(const StinkyRegister& saddr,
                  const std::unordered_map<std::string, int64_t>& symbols) {
    if (!saddr.isRegister() || saddr.reg.type != RegType::S) return false;
    for (const char* name : kTrackedSrdSymbols) {
        auto it = symbols.find(name);
        if (it != symbols.end() && static_cast<int64_t>(saddr.reg.idx) == it->second) return true;
    }
    return false;
}

// Physical base indices for this pass's own reserved registers, resolved once
// per function from the `.set` symbol table.
struct OwnRegisters {
    bool valid = false;
    int reportBufBase = -1;  // sgprAsanReportBuf: 2 SGPRs
    int sgprTmpBase = -1;    // sgprAsanTmp: 2 SGPRs
    int vgprTmpBase = -1;    // vgprAsanTmp: 8 VGPRs
};

OwnRegisters resolveOwnRegisters(const std::unordered_map<std::string, int64_t>& symbols) {
    OwnRegisters own;
    auto reportIt = symbols.find(kReportBufSymbol);
    auto sgprIt = symbols.find(kSgprTmpSymbol);
    auto vgprIt = symbols.find(kVgprTmpSymbol);
    if (reportIt == symbols.end() || sgprIt == symbols.end() || vgprIt == symbols.end()) {
        return own;  // AsanInstrument requested but Tensile didn't reserve these -- no-op.
    }
    own.reportBufBase = static_cast<int>(reportIt->second);
    own.sgprTmpBase = static_cast<int>(sgprIt->second);
    own.vgprTmpBase = static_cast<int>(vgprIt->second);
    own.valid = true;
    return own;
}

// Emits the shadow-memory check + violation-report block before `anchor`
// (the real MUBUF load/store instruction). Uses only the fixed scratch
// registers Tensile reserved (own.sgprTmpBase/own.vgprTmpBase) -- this pass
// never allocates its own registers, matching this codebase's convention that
// StinkyTofu passes are schedulers, not code generators.
void emitAsanCheck(IRBase* anchor, const MubufAddrOperands& ops, const OwnRegisters& own,
                   AsmIRBuilder& irBuilder, GfxArchID archId, int& labelCounter) {
    const StinkyRegister vaLo = vgpr(own.vgprTmpBase + kVaLoOff);
    const StinkyRegister vaHi = vgpr(own.vgprTmpBase + kVaHiOff);
    const StinkyRegister vaPair = vgpr(own.vgprTmpBase + kVaLoOff, 2);
    const StinkyRegister shadowByte = vgpr(own.vgprTmpBase + kShadowByteOff);
    const StinkyRegister reportAddrPair = vgpr(own.vgprTmpBase + kReportAddrLoOff, 2);
    const StinkyRegister pcVgprPair = vgpr(own.vgprTmpBase + kPcVgprLoOff, 2);
    const StinkyRegister pcSgprPair = sgpr(own.sgprTmpBase + kPcSgprLoOff, 2);
    const int srdBaseIdx = static_cast<int>(ops.saddr.reg.idx);

    // v1 only handles a soffset that is literally 0 (the common case -- GEMM
    // address math is folded into vaddr against a zero soffset). A nonzero
    // register/literal soffset is not folded into the check; the comment
    // documents this rather than silently under-checking.
    bool soffsetIsZero = ops.soffset.dataType == StinkyRegister::Type::LiteralInt &&
                         ops.soffset.getLiteralInt() == 0;

    // VA = srdBase(64-bit) + zext(vaddr, 64) [+ soffset, only when literal 0 i.e. no-op]
    StinkyInstruction* addLo = irBuilder.create(getMCIDByUOp(GFX::v_add_co_u32, archId), anchor);
    addLo->addDestReg(vaLo);
    addLo->addDestReg(vccReg());
    addLo->addSrcReg(sgpr(srdBaseIdx));
    addLo->addSrcReg(ops.vaddr);
    addLo->addModifier<CommentData>(CommentData{
        soffsetIsZero
            ? "AsanCheck: VA_lo = SrdBase_lo + vaddr"
            : "AsanCheck: VA_lo = SrdBase_lo + vaddr (soffset != 0 not folded, v1 limitation)"});

    StinkyInstruction* addHi = irBuilder.create(getMCIDByUOp(GFX::v_add_co_ci_u32, archId), anchor);
    addHi->addDestReg(vaHi);
    addHi->addDestReg(vccReg());
    addHi->addSrcReg(sgpr(srdBaseIdx + 1));
    addHi->addSrcReg(StinkyRegister(0));
    addHi->addSrcReg(vccReg());
    addHi->addModifier<CommentData>(CommentData{"AsanCheck: VA_hi = SrdBase_hi + carry"});

    // shadow = (VA >> 3) | 0x7fff8000, computed as a 64-bit quantity via a
    // manual funnel shift (v_lshl_or_b32), then in-place offset add.
    StinkyInstruction* shiftLo = irBuilder.create(getMCIDByUOp(GFX::v_lshrrev_b32, archId), anchor);
    shiftLo->addDestReg(vaLo);
    shiftLo->addSrcReg(StinkyRegister(static_cast<int>(kShadowScale)));
    shiftLo->addSrcReg(vaLo);
    shiftLo->addModifier<CommentData>(CommentData{"AsanCheck: tmp = VA_lo >> 3"});

    StinkyInstruction* funnel = irBuilder.create(getMCIDByUOp(GFX::v_lshl_or_b32, archId), anchor);
    funnel->addDestReg(vaLo);
    funnel->addSrcReg(vaHi);
    funnel->addSrcReg(StinkyRegister(32 - static_cast<int>(kShadowScale)));
    funnel->addSrcReg(vaLo);
    funnel->addModifier<CommentData>(
        CommentData{"AsanCheck: shadow_lo = (VA_hi << 29) | tmp  (== (VA >> 3) low word)"});

    StinkyInstruction* shiftHi = irBuilder.create(getMCIDByUOp(GFX::v_lshrrev_b32, archId), anchor);
    shiftHi->addDestReg(vaHi);
    shiftHi->addSrcReg(StinkyRegister(static_cast<int>(kShadowScale)));
    shiftHi->addSrcReg(vaHi);
    shiftHi->addModifier<CommentData>(CommentData{"AsanCheck: shadow_hi = VA_hi >> 3"});

    // Real ASan ORs the offset in rather than adding (Mapping.OrShadowOffset),
    // so only the low word changes and no carry propagation is needed.
    StinkyInstruction* orOffsetLo = irBuilder.create(getMCIDByUOp(GFX::v_or_b32, archId), anchor);
    orOffsetLo->addDestReg(vaLo);
    orOffsetLo->addSrcReg(StinkyRegister(static_cast<int>(kShadowOffset)));
    orOffsetLo->addSrcReg(vaLo);
    orOffsetLo->addModifier<CommentData>(CommentData{"AsanCheck: shadow_lo |= 0x7fff8000"});

    // Load the shadow byte. Wait-count for this load is inserted automatically
    // by StinkyWaitCntInsertionPass, which this pass must run before (see
    // Gfx1250Backend.cpp pipeline placement).
    StinkyInstruction* loadShadow =
        irBuilder.create(getMCIDByUOp(GFX::global_load_u8, archId), anchor);
    loadShadow->addDestReg(shadowByte);
    loadShadow->addSrcReg(vaPair);
    loadShadow->addSrcReg(offReg());
    loadShadow->addModifier<CommentData>(CommentData{"AsanCheck: load ASan shadow byte"});

    // Explicit wait rather than relying on StinkyWaitCntInsertionPass: that pass
    // runs at region scope (inside the KernelToRegions adaptor in
    // Gfx1250Backend), while this pass runs at kernel scope afterwards, so it
    // would never see this load. Debug-only path, so an unconditional drain is fine.
    StinkyInstruction* wait = irBuilder.create(getMCIDByUOp(GFX::s_wait_loadcnt, archId), anchor);
    wait->addSrcReg(StinkyRegister(0));
    wait->addModifier<CommentData>(CommentData{"AsanCheck: wait for the shadow byte"});

    StinkyInstruction* cmp = irBuilder.create(getMCIDByUOp(GFX::v_cmp_ne_u32, archId), anchor);
    cmp->addDestReg(vccReg());
    cmp->addSrcReg(StinkyRegister(0));
    cmp->addSrcReg(shadowByte);
    cmp->addModifier<CommentData>(
        CommentData{"AsanCheck: vcc = (shadow byte != 0, i.e. poisoned)"});

    const std::string skipLabel = std::string(kSkipLabelPrefix) + std::to_string(labelCounter++);
    StinkyInstruction* br = irBuilder.create(getMCIDByUOp(GFX::s_cbranch_vccz, archId), anchor);
    br->addSrcReg(StinkyRegister(skipLabel));
    br->addModifier<LabelData>(LabelData{skipLabel});
    br->addModifier<CommentData>(
        CommentData{"AsanCheck: skip report+trap if no active lane poisoned"});

    // Violation block: capture PC, write it to AsanReportBuf, halt the wave.
    // Deliberately does NOT call real __asan_report_*/hostcall -- see plan doc
    // (mossy-cooking-wozniak.md) for why: that needs the full AMDGPU
    // non-kernel function-call ABI (private-segment-buffer, flat-scratch init,
    // SP/RA convention) these kernels don't have.
    StinkyInstruction* getpc = irBuilder.create(getMCIDByUOp(GFX::s_getpc_b64, archId), anchor);
    getpc->addDestReg(pcSgprPair);
    getpc->addModifier<CommentData>(CommentData{"AsanCheck: capture PC of the failing check"});

    StinkyInstruction* movAddrLo = irBuilder.create(getMCIDByUOp(GFX::v_mov_b32, archId), anchor);
    movAddrLo->addDestReg(vgpr(own.vgprTmpBase + kReportAddrLoOff));
    movAddrLo->addSrcReg(sgpr(own.reportBufBase + 0));

    StinkyInstruction* movAddrHi = irBuilder.create(getMCIDByUOp(GFX::v_mov_b32, archId), anchor);
    movAddrHi->addDestReg(vgpr(own.vgprTmpBase + kReportAddrHiOff));
    movAddrHi->addSrcReg(sgpr(own.reportBufBase + 1));

    StinkyInstruction* movPcLo = irBuilder.create(getMCIDByUOp(GFX::v_mov_b32, archId), anchor);
    movPcLo->addDestReg(vgpr(own.vgprTmpBase + kPcVgprLoOff));
    movPcLo->addSrcReg(sgpr(own.sgprTmpBase + kPcSgprLoOff));

    StinkyInstruction* movPcHi = irBuilder.create(getMCIDByUOp(GFX::v_mov_b32, archId), anchor);
    movPcHi->addDestReg(vgpr(own.vgprTmpBase + kPcVgprHiOff));
    movPcHi->addSrcReg(sgpr(own.sgprTmpBase + kPcSgprHiOff));

    StinkyInstruction* storePc =
        irBuilder.create(getMCIDByUOp(GFX::global_store_b64, archId), anchor);
    storePc->addSrcReg(reportAddrPair);
    storePc->addSrcReg(pcVgprPair);
    storePc->addSrcReg(offReg());
    storePc->addModifier<CommentData>(CommentData{"AsanCheck: AsanReportBuf[0:1] = failing PC"});

    // Without this, s_trap could fire before the store retires -- the wave
    // halting (or the whole queue faulting) does not guarantee the write
    // landed in memory, which would make the host-side readback see a stale
    // zero PC instead of the actual violation record.
    StinkyInstruction* storeWait =
        irBuilder.create(getMCIDByUOp(GFX::s_wait_storecnt, archId), anchor);
    storeWait->addSrcReg(StinkyRegister(0));
    storeWait->addModifier<CommentData>(
        CommentData{"AsanCheck: wait for the report write to land before trapping"});

    // Same trap used by device-side assert()/abort() on any normal HIP
    // process (no host-ASan-build dependency for the halt itself, unlike the
    // shadow check above).
    StinkyInstruction* trap = irBuilder.create(getMCIDByUOp(GFX::s_trap, archId), anchor);
    trap->addSrcReg(StinkyRegister(2));
    trap->addModifier<CommentData>(
        CommentData{"AsanCheck: halt wave after shadow-poison violation"});

    // AsmIRBuilder::createLabel() has no insertBefore parameter -- it always
    // appends at the end of the block, which would place every check's skip
    // label after the LAST target in the block instead of right before this
    // check's own anchor. With more than one tracked access in a block, a
    // passing shadow check would then branch clean past every later real
    // access (and even past s_endpgm). Build the LABEL instruction directly,
    // anchored like every other instruction above, so it lands immediately
    // before `anchor` -- the branch skips only the report+trap block.
    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};
    StinkyInstruction* label = irBuilder.create(&labelMCID, anchor);
    label->addModifier<LabelData>(LabelData{skipLabel, /*alignment=*/1});
}

class InsertAsanCheckPassImpl : public Pass {
   public:
    static char ID;

    InsertAsanCheckPassImpl() = default;

    const char* getName() const override {
        return "Insert ASan Check";
    }

    Pass::ID getPassID() const override {
        return &InsertAsanCheckPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        std::unordered_map<std::string, int64_t> symbols;
        collectAsmSetSymbolValues(func, symbols);
        OwnRegisters own = resolveOwnRegisters(symbols);
        if (!own.valid) {
            // AsanInstrument was requested but Tensile did not reserve/emit
            // AsanReportBuf/AsanTmp (e.g. mismatched build config) -- no-op
            // rather than emit checks referencing undeclared registers.
            return PreservedAnalyses::none();
        }

        int labelCounter = 0;
        for (BasicBlock& bb : func) {
            // Snapshot real (non-PHI) instructions first: emitAsanCheck inserts
            // new instructions before `anchor`, which would otherwise be
            // visited again by a live iterator walking the same block.
            std::vector<StinkyInstruction*> targets;
            for (auto it = bb.begin(); it != bb.end(); ++it) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (inst == nullptr) continue;
                bool isLoad = isMUBUFLoad(*inst);
                bool isStore = isMUBUFStore(*inst);
                if (!isLoad && !isStore) continue;
                MubufAddrOperands ops = getMubufAddrOperands(*inst, isStore);
                if (!isTrackedSrd(ops.saddr, symbols)) continue;
                targets.push_back(inst);
            }
            if (targets.empty()) continue;

            AsmIRBuilder irBuilder(bb, archId);
            for (StinkyInstruction* inst : targets) {
                bool isStore = isMUBUFStore(*inst);
                MubufAddrOperands ops = getMubufAddrOperands(*inst, isStore);
                emitAsanCheck(inst, ops, own, irBuilder, archId, labelCounter);
            }
        }

        return PreservedAnalyses::none();
    }
};

char InsertAsanCheckPassImpl::ID = 0;

}  // namespace

std::unique_ptr<Pass> createInsertAsanCheckPass() {
    return std::make_unique<InsertAsanCheckPassImpl>();
}

}  // namespace stinkytofu
