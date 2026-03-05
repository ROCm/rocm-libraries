#include "race-emulator/Instruction.h"
#include "race-emulator/Util.h"
#include "race-emulator/Wave.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raceemulator {

namespace {

// SOPP: Wait for memory counts (s_waitcnt)
class SOPP_WaitCnt : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    // Examples: s_waitcnt lgkmcnt(0), s_waitcnt vmcnt(0) expcnt(0)
    auto partitioned = getPartitioned(line);

    int vmcnt = -1;
    int lgkmcnt = -1;

    // Iterate through tokens starting after the mnemonic
    for (size_t i = 1; i < partitioned.size(); ++i) {
      std::string_view token = partitioned[i];

      constexpr const char *vmcntPrefix = "vmcnt(";
      auto lenVmcntPrefix = std::strlen(vmcntPrefix);

      constexpr const char *lgkmcntPrefix = "lgkmcnt(";
      auto lenLgkmcntPrefix = std::strlen(lgkmcntPrefix);

      if (token.substr(0, lenVmcntPrefix) == vmcntPrefix) {
        size_t end = token.find(')');
        assert(end != std::string_view::npos && "Malformed vmcnt token");
        auto valStr = token.substr(lenVmcntPrefix, end - lenVmcntPrefix);
        vmcnt = getIntFromView<int>(valStr);
      } else if (token.substr(0, lenLgkmcntPrefix) == lgkmcntPrefix) {
        size_t end = token.find(')');
        assert(end != std::string_view::npos && "Malformed lgkmcnt token");
        auto valStr = token.substr(lenLgkmcntPrefix, end - lenLgkmcntPrefix);
        lgkmcnt = getIntFromView<int>(valStr);
      } else {
        throw std::runtime_error("Unsupported s_waitcnt counter: " +
                                 std::string(token));
      }
    }

    return [&wave, vmcnt, lgkmcnt]() {
      if (vmcnt >= 0) {
        wave.sWaitCntVmcnt(vmcnt);
      }
      if (lgkmcnt >= 0) {
        wave.sWaitCntLgkmcnt(lgkmcnt);
      }
      return wave.getPc() + 1;
    };
  }
};

// SOPP: Synchronization barrier (s_barrier)
class SOPP_Barrier : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 1);

    return [&wave]() {
      wave.flushWaveCompleteMemoryEvents();
      return wave.getPc() + 1;
    };
  }
};

// SOPP: Save execution mask (s_and_saveexec_b64, etc.)
template <typename Op> class SOPP_SaveExec : public Instruction {
  Op op;

public:
  SOPP_SaveExec(Op op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 3);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::SGPR &&
           "SaveExec dest must be SGPR");

    auto src0 = wave.parseOperand<uint64_t>(partitioned[2]);
    if (!src0.isLiteral) {
      assert(src0.reg.type == CommonRegister::Type::SGPR &&
             "SaveExec src must be SGPR");
    }

    auto operation = this->op;

    return [&wave, dst, src0, operation]() {
      uint64_t originalExec = wave.getExecU64();
      uint64_t srcVal = wave.getSgprOrLiteralValue(src0);
      uint64_t newExec = operation(srcVal, originalExec);

      wave.setExecU64(newExec);
      wave.setSgpr64(dst.index, originalExec);
      wave.setScc(newExec != 0);

      return wave.getPc() + 1;
    };
  }
};

// SOPP: Conditional branch (s_branch, s_cbranch_*)
template <typename Op> class SOPP_Branch : public Instruction {
  Op op;

public:
  SOPP_Branch(Op op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 2);

    auto label = partitioned[1];
    const auto &labels = wave.getLabels();
    auto labelIndex = labels.at(std::string(label));
    auto operation = this->op;

    return [&wave, operation, labelIndex]() {
      if (operation(wave)) {
        return labelIndex;
      }
      return wave.getPc() + 1;
    };
  }
};

// SOPP: Program counter manipulation (s_getpc, s_setpc)
class SOPP_GetPc : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = partitioned[1];

    return [&wave, dst]() {
      uint64_t currentPc = static_cast<uint64_t>(wave.getPc());
      auto dstReg = wave.getFirstRegister(dst);
      assert(dstReg.type == CommonRegister::Type::SGPR);

      wave.setSgpr64(dstReg.index, 4 * (currentPc + 1));
      return wave.getPc() + 1;
    };
  }
};

class SOPP_SetPc : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto src = partitioned[1];

    return [&wave, src]() {
      auto srcReg = wave.getFirstRegister(src);
      assert(srcReg.type == CommonRegister::Type::SGPR);
      uint64_t targetPc = wave.getSgpr64(srcReg.index);
      return static_cast<int>(targetPc / 4);
    };
  }
};

// SOPP: No operation (s_nop, s_endpgm)
class SOPP_NoOp : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view /*line*/) const final {
    return [&]() { return wave.getPc() + 1; };
  }
};

// Debugging utility (Pseudo-instruction)
class DebugPrint : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    std::string type(partitioned[1]);
    std::string regString = std::string(partitioned[2]);
    auto reg = wave.getFirstRegister(regString);
    int lane = getIntFromView<int>(partitioned[3]);

    std::string comment;
    constexpr auto minLength = 5;
    if (partitioned.size() >= minLength) {
      comment = std::string(partitioned[4]);
    }

    return [&wave, type, reg, lane, comment, regString]() {
      std::ostringstream oss;
      oss << "DebugPrint: " << type << " " << regString << ": ";

      if (reg.type == CommonRegister::Type::VGPR) {
        oss << wave.getVgpr(reg.index, lane);
      } else if (reg.type == CommonRegister::Type::SGPR) {
        oss << wave.getSgpr(reg.index);
      } else {
        oss << "(unknown register type)";
      }

      if (!comment.empty()) {
        oss << "   " << comment;
      }

      std::cout << oss.str() << "\n";
      return wave.getPc() + 1;
    };
  }
};

// Helpers for templates
template <typename Op>
static std::unique_ptr<Instruction> makeSOPP_SaveExec(Op op) {
  return std::make_unique<SOPP_SaveExec<Op>>(std::move(op));
}

template <typename Op>
static std::unique_ptr<Instruction> makeSOPP_Branch(Op op) {
  return std::make_unique<SOPP_Branch<Op>>(std::move(op));
}

// Registration

// Standard adapter
template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    InstructionRegistry::instance().add(
        name, std::make_unique<InstT>(std::forward<Args>(args)...));
  }
};

// Factory adapter
struct RegisterFactory {
  RegisterFactory(const std::string &name, std::unique_ptr<Instruction> inst) {
    InstructionRegistry::instance().add(name, std::move(inst));
  }
};

// --- Program Control ---
const Register<SOPP_GetPc> s_getpc("s_getpc_b64");
const Register<SOPP_SetPc> s_setpc("s_setpc_b64");
const Register<SOPP_NoOp> s_nop("s_nop");
const Register<SOPP_NoOp> s_setprio("s_setprio");
const Register<SOPP_NoOp> s_endpgm("s_endpgm");
const Register<SOPP_WaitCnt> s_waitcnt("s_waitcnt");
const Register<SOPP_Barrier> s_barrier("s_barrier");
const Register<DebugPrint> debug_print("print");

// --- Execution Mask (SaveExec) ---
const RegisterFactory
    s_and_exec("s_and_saveexec_b64",
               makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & b; }));

const RegisterFactory s_andn2_exec(
    "s_andn2_saveexec_b64",
    makeSOPP_SaveExec([](uint64_t a, uint64_t b) { return a & (~b); }));

// --- Branching ---
const RegisterFactory s_cb_scc0("s_cbranch_scc0", makeSOPP_Branch([](Wave &w) {
                                  return w.getScc() == 0;
                                }));

const RegisterFactory s_cb_scc1("s_cbranch_scc1", makeSOPP_Branch([](Wave &w) {
                                  return w.getScc() != 0;
                                }));

const RegisterFactory s_branch("s_branch",
                               makeSOPP_Branch([](Wave &) { return true; }));

const RegisterFactory s_cb_ez("s_cbranch_execz", makeSOPP_Branch([](Wave &w) {
                                return w.getExecU64() == 0;
                              }));

const RegisterFactory s_cb_enz("s_cbranch_execnz", makeSOPP_Branch([](Wave &w) {
                                 return w.getExecU64() != 0;
                               }));

const RegisterFactory s_cb_vnz("s_cbranch_vccnz", makeSOPP_Branch([](Wave &w) {
                                 return w.getVccU64() != 0;
                               }));

const RegisterFactory s_cb_vz("s_cbranch_vccz", makeSOPP_Branch([](Wave &w) {
                                return w.getVccU64() == 0;
                              }));

} // namespace
} // namespace raceemulator
