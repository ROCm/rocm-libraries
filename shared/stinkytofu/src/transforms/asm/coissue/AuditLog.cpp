// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "AuditLog.hpp"

#include <map>
#include <mutex>
#include <sstream>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"

namespace stinkytofu::coissue {
namespace {

std::mutex& logMutex() {
    static std::mutex m;
    return m;
}

std::map<std::pair<std::string, std::string>, PredictedLoop>& log() {
    static std::map<std::pair<std::string, std::string>, PredictedLoop> entries;
    return entries;
}

}  // namespace

std::string auditSignature(const StinkyInstruction& inst) {
    std::ostringstream os;
    os << inst.getHwInstDesc()->mnemonic;
    const int op = inst.getUnifiedOpcode();
    if (op == GFX::s_set_vgpr_msb || op == GFX::s_nop) {
        for (const StinkyRegister& r : inst.getSrcRegs())
            if (r.dataType == StinkyRegister::Type::LiteralInt) {
                os << " " << r.getLiteralInt();
                break;
            }
    }
    if (const auto* w = inst.getModifier<SWaitAluData>()) {
        for (auto f : {SWaitAluData::VA_VDST, SWaitAluData::VM_VSRC, SWaitAluData::HOLD_CNT})
            os << " " << (w->hasField(f) ? static_cast<int>(w->getField(f)) : -1);
    }
    return os.str();
}

void recordPrediction(const std::string& function, const std::string& header,
                      PredictedLoop prediction) {
    std::lock_guard<std::mutex> lock(logMutex());
    log()[{function, header}] = std::move(prediction);
}

std::optional<PredictedLoop> takePrediction(const std::string& function,
                                            const std::string& header) {
    std::lock_guard<std::mutex> lock(logMutex());
    auto it = log().find({function, header});
    if (it == log().end()) return std::nullopt;
    PredictedLoop p = std::move(it->second);
    log().erase(it);
    return p;
}

}  // namespace stinkytofu::coissue
