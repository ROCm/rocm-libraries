#include "race-emulator/Instruction.h"
#include "race-emulator/Util.h"
#include "race-emulator/Wave.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace raceemulator {

namespace {

// Instruction implementations

// SOP1 instructions
template <typename T, typename OpFunc, bool ConditionalOnScc = false>
class SOP1_Op : public Instruction {
  OpFunc op;

public:
  SOP1_Op(OpFunc op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 3);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::SGPR && "SOP1 dest must be SGPR");

    auto src0 = wave.parseOperand<T>(partitioned[2]);
    auto operation = this->op;
    auto resIdx = dst.index;

    return [&wave, resIdx, src0, operation]() {
      T val0 = wave.getSgprOrLiteralValue(src0);

      if (ConditionalOnScc && !wave.getScc()) {
        return wave.getPc() + 1;
      }

      T result = operation(val0, wave);

      if constexpr (sizeof(T) == 8) {
        wave.setSgpr64(resIdx, std::bit_cast<uint64_t>(result));
      } else {
        wave.setSgpr(resIdx, std::bit_cast<uint32_t>(result));
      }
      return wave.getPc() + 1;
    };
  }
};

// Scalar moves and conversions
template <typename T_DST, typename T_SRC> class SOP1_Move : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::SGPR && "S_MOV dest must be SGPR");
    auto src = wave.parseOperand<T_SRC>(partitioned[2]);

    return [&wave, dst, src]() {
      T_SRC inVal = wave.getSgprOrLiteralValue(src);
      T_DST outVal = static_cast<T_DST>(inVal);

      if constexpr (sizeof(T_DST) == 8) {
        wave.setSgpr64(dst.index, std::bit_cast<uint64_t>(outVal));
      } else {
        wave.setSgpr(dst.index, std::bit_cast<uint32_t>(outVal));
      }

      return wave.getPc() + 1;
    };
  }
};

// SOP2 instructions
template <typename T, typename OpFunc> class SOP2_Op : public Instruction {
  OpFunc op;

public:
  SOP2_Op(OpFunc op) : op(std::move(op)) {}

  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() >= 4);

    auto dst = wave.getFirstRegister(partitioned[1]);
    assert(dst.type == CommonRegister::Type::SGPR && "SOP2 dest must be SGPR");

    auto src0 = wave.parseOperand<T>(partitioned[2]);
    auto src1 = wave.parseOperand<T>(partitioned[3]);
    auto operation = this->op;
    auto resIdx = dst.index;

    return [&wave, resIdx, src0, src1, operation]() {
      T val0 = wave.getSgprOrLiteralValue(src0);
      T val1 = wave.getSgprOrLiteralValue(src1);

      T result = operation(val0, val1, wave);

      if constexpr (sizeof(T) == 8) {
        wave.setSgpr64(resIdx, std::bit_cast<uint64_t>(result));
      } else {
        wave.setSgpr(resIdx, std::bit_cast<uint32_t>(result));
      }
      return wave.getPc() + 1;
    };
  }
};

// SOPC scalar comparisons
template <typename T, typename Op> class SOPC_Compare : public Instruction {
  Op op;

public:
  SOPC_Compare(Op op) : op(std::move(op)) {}
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 3);

    auto src0 = wave.parseOperand<T>(partitioned[1]);
    auto src1 = wave.parseOperand<T>(partitioned[2]);
    auto operation = this->op;

    return [&wave, src0, src1, operation]() {
      T val0 = wave.getSgprOrLiteralValue(src0);
      T val1 = wave.getSgprOrLiteralValue(src1);

      bool result = operation(val0, val1);
      wave.setScc(result);
      return wave.getPc() + 1;
    };
  }
};

// SOPK scalar constant instructions
template <typename T, typename Op> class SOPK_Compare : public Instruction {
  Op op;

public:
  SOPK_Compare(Op op) : op(std::move(op)) {}
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 3);

    auto src0 = wave.parseOperand<T>(partitioned[1]);
    assert(!src0.isLiteral && "SOPK operand 0 cannot be a literal");

    auto sgprIndex = src0.reg.index;
    T literalValue = getIntFromView<T>(partitioned[2]);
    auto operation = this->op;

    return [&wave, sgprIndex, literalValue, operation]() {
      auto val0 = wave.getSgpr(sgprIndex);
      bool result = operation(val0, literalValue);
      wave.setScc(result);
      return wave.getPc() + 1;
    };
  }
};

// Registration

template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    InstructionRegistry::instance().add(
        name, std::make_unique<InstT>(std::forward<Args>(args)...));
  }
};

template <typename T, bool Cond = false> struct RegisterSOP1 {
  template <typename Op> RegisterSOP1(const std::string &name, Op op) {
    // Forward declaration of SOP1_Op class defined below
    using InstType = SOP1_Op<T, Op, Cond>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterSOP2 {
  template <typename Op> RegisterSOP2(const std::string &name, Op op) {
    using InstType = SOP2_Op<T, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterSOPC {
  template <typename Op> RegisterSOPC(const std::string &name, Op op) {
    using InstType = SOPC_Compare<T, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

template <typename T> struct RegisterSOPK {
  template <typename Op> RegisterSOPK(const std::string &name, Op op) {
    using InstType = SOPK_Compare<T, Op>;
    InstructionRegistry::instance().add(name, std::make_unique<InstType>(op));
  }
};

// SOP1 moves and conversions
static Register<SOP1_Move<uint32_t, uint32_t>> s_mov_32("s_mov_b32");
static Register<SOP1_Move<uint64_t, uint64_t>> s_mov_64("s_mov_b64");
static Register<SOP1_Move<int32_t, int16_t>> s_movk("s_movk_i32");

// SOP1 logic and arithmetic
static RegisterSOP1<uint32_t> s_ff1("s_ff1_i32_b32", [](uint32_t s0, Wave &) {
  return (s0 == 0) ? 0xFFFFFFFF : static_cast<uint32_t>(std::countr_zero(s0));
});
// s_ctz_i32_b32: RDNA3+ name for s_ff1_i32_b32 (count trailing zeros).
static RegisterSOP1<uint32_t> s_ctz("s_ctz_i32_b32", [](uint32_t s0, Wave &) {
  return (s0 == 0) ? 0xFFFFFFFF : static_cast<uint32_t>(std::countr_zero(s0));
});

static RegisterSOP1<uint32_t> s_sext("s_sext_i32_i16", [](uint32_t s0, Wave &) {
  return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(s0)));
});

// s_abs_i32 s3, s14
// D.i = abs (S0.i). SCC=result not zero.
static RegisterSOP1<int32_t> s_abs("s_abs_i32", [](int32_t s0, Wave &w) {
  auto out = (s0 < 0) ? -s0 : s0;
  auto isZero = (out == 0);
  w.setScc(!isZero);
  return out;
});

static RegisterSOP1<uint32_t, true>
    s_cmov("s_cmov_b32", [](uint32_t s0, Wave &) { return s0; });

// SOP2 logic
static RegisterSOP2<uint32_t> s_and("s_and_b32",
                                    [](uint32_t a, uint32_t b, Wave &w) {
                                      auto r = a & b;
                                      w.setScc(r != 0);
                                      return r;
                                    });
static RegisterSOP2<uint64_t> s_and64("s_and_b64",
                                      [](uint64_t a, uint64_t b, Wave &w) {
                                        auto r = a & b;
                                        w.setScc(r != 0);
                                        return r;
                                      });

static auto andn2_fn = [](uint32_t a, uint32_t b, Wave &w) {
  auto r = a & (~b);
  w.setScc(r != 0);
  return r;
};
static RegisterSOP2<uint32_t> s_andn2("s_andn2_b32", andn2_fn);
// RDNA name for s_andn2_b32
static RegisterSOP2<uint32_t> s_and_not1("s_and_not1_b32", andn2_fn);

static RegisterSOP2<uint32_t> s_lshl("s_lshl_b32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       uint32_t r = a << (b & 0x1F);
                                       w.setScc(r != 0);
                                       return r;
                                     });

static RegisterSOP2<uint64_t> s_lshl64("s_lshl_b64",
                                       [](uint64_t a, uint64_t b, Wave &w) {
                                         uint64_t r = a << (b & 0x3F);
                                         w.setScc(r != 0);
                                         return r;
                                       });

static RegisterSOP2<uint32_t> s_ashr("s_ashr_i32", [](uint32_t a, uint32_t b,
                                                      Wave &w) {
  int32_t r = std::bit_cast<int32_t>(a) >> (b & 0x1F);
  w.setScc(r != 0);
  return std::bit_cast<uint32_t>(r);
});

static RegisterSOP2<uint32_t> s_lshr("s_lshr_b32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       uint32_t r = a >> (b & 0x1F);
                                       w.setScc(r != 0);
                                       return r;
                                     });

static RegisterSOP2<uint64_t> s_andn2_64("s_andn2_b64",
                                         [](uint64_t a, uint64_t b, Wave &w) {
                                           uint64_t r = a & (~b);
                                           w.setScc(r != 0);
                                           return r;
                                         });

static RegisterSOP2<uint32_t> s_or("s_or_b32",
                                   [](uint32_t a, uint32_t b, Wave &w) {
                                     auto r = a | b;
                                     w.setScc(r != 0);
                                     return r;
                                   });

static RegisterSOP2<uint64_t> s_orn2("s_orn2_b64",
                                     [](uint64_t a, uint64_t b, Wave &w) {
                                       uint64_t r = a | (~b);
                                       w.setScc(r != 0);
                                       return r;
                                     });

static RegisterSOP2<uint32_t> s_xor32("s_xor_b32",
                                      [](uint32_t a, uint32_t b, Wave &w) {
                                        uint32_t r = a ^ b;
                                        w.setScc(r != 0);
                                        return r;
                                      });
static RegisterSOP2<uint64_t> s_xor64("s_xor_b64",
                                      [](uint64_t a, uint64_t b, Wave &w) {
                                        uint64_t r = a ^ b;
                                        w.setScc(r != 0);
                                        return r;
                                      });

static RegisterSOP2<uint64_t> s_or64("s_or_b64",
                                     [](uint64_t a, uint64_t b, Wave &w) {
                                       uint64_t r = a | b;
                                       w.setScc(r != 0);
                                       return r;
                                     });

static RegisterSOP2<uint32_t> s_csel("s_cselect_b32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       return w.getScc() ? a : b;
                                     });

static RegisterSOP2<uint32_t> s_bfe("s_bfe_u32", [](uint32_t val, uint32_t c,
                                                    Wave &w) {
  uint32_t off = c & 0x1F;
  uint32_t width = (c >> 16) & 0x7F;
  uint32_t r = (width == 0) ? 0
                            : (width >= 32 ? val >> off
                                           : (val >> off) & ((1 << width) - 1));
  w.setScc(r != 0);
  return r;
});

// s_bfm_b32: bitfield mask. D = ((1 << S0[4:0]) - 1) << S1[4:0].
static RegisterSOP2<uint32_t> s_bfm("s_bfm_b32", [](uint32_t a, uint32_t b,
                                                    Wave &) {
  uint32_t width = a & 0x1F;
  uint32_t offset = b & 0x1F;
  uint32_t mask =
      (width == 0) ? 0 : ((width >= 32) ? 0xFFFFFFFF : ((1u << width) - 1));
  return mask << offset;
});

// s_lshl2_add_u32: D = (S0 << 2) + S1. SCC = carry out.
static RegisterSOP2<uint32_t>
    s_lshl2_add("s_lshl2_add_u32", [](uint32_t a, uint32_t b, Wave &w) {
      uint64_t r = (static_cast<uint64_t>(a) << 2) + static_cast<uint64_t>(b);
      w.setScc(r > 0xFFFFFFFF);
      return static_cast<uint32_t>(r);
    });

// SOP2 arithmetic
static RegisterSOP2<int> s_mul_i("s_mul_i32",
                                 [](int a, int b, Wave &) { return a * b; });

static RegisterSOP2<uint32_t>
    s_mul_hi_u("s_mul_hi_u32", [](uint32_t a, uint32_t b, Wave &) {
      return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
    });

static RegisterSOP2<int32_t> s_mul_hi_i("s_mul_hi_i32", [](int32_t a, int32_t b,
                                                           Wave &) {
  return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 32);
});

static RegisterSOP2<uint32_t> s_add("s_add_i32",
                                    [](uint32_t a, uint32_t b, Wave &w) {
                                      uint32_t r = a + b;
                                      bool s0 = a >> 31;
                                      bool s1 = b >> 31;
                                      bool sr = r >> 31;
                                      w.setScc(s0 == s1 && s0 != sr);
                                      return r;
                                    });

static RegisterSOP2<int32_t> s_sub("s_sub_i32",
                                   [](int32_t a, int32_t b, Wave &w) {
                                     int32_t r = a - b;
                                     bool s0 = a >> 31;
                                     bool s1 = b >> 31;
                                     bool sr = r >> 31;
                                     w.setScc(s0 != s1 && s0 != sr);
                                     return r;
                                   });

static RegisterSOP2<uint32_t> s_addu("s_add_u32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       uint64_t r = (uint64_t)a + b;
                                       w.setScc((r >> 32) != 0);
                                       return (uint32_t)r;
                                     });

static RegisterSOP2<uint32_t> s_subu("s_sub_u32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       w.setScc(b > a);
                                       return a - b;
                                     });

static RegisterSOP2<uint32_t> s_addc("s_addc_u32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       uint64_t c = w.getScc() ? 1 : 0;
                                       uint64_t r = (uint64_t)a + b + c;
                                       w.setScc((r >> 32) != 0);
                                       return (uint32_t)r;
                                     });

static RegisterSOP2<uint32_t> s_subb("s_subb_u32",
                                     [](uint32_t a, uint32_t b, Wave &w) {
                                       uint32_t c = w.getScc() ? 1 : 0;
                                       uint32_t r = a - b - c;
                                       w.setScc((uint64_t)b + c > a);
                                       return r;
                                     });

// SOPC scalar comparisons
static RegisterSOPC<uint32_t> s_cmp_eq_u("s_cmp_eq_u32",
                                         std::equal_to<uint32_t>());
static RegisterSOPC<int32_t> s_cmp_eq_i("s_cmp_eq_i32",
                                        std::equal_to<int32_t>());
static RegisterSOPC<int32_t> s_cmp_lt_i("s_cmp_lt_i32", std::less<int32_t>());
static RegisterSOPC<uint32_t> s_cmp_lt_u("s_cmp_lt_u32", std::less<uint32_t>());
static RegisterSOPC<int32_t> s_cmp_le_i("s_cmp_le_i32",
                                        std::less_equal<int32_t>());
static RegisterSOPC<uint32_t> s_cmp_le_u("s_cmp_le_u32",
                                         std::less_equal<uint32_t>());
static RegisterSOPC<uint32_t> s_cmp_lg_u("s_cmp_lg_u32",
                                         [](uint32_t a, uint32_t b) {
                                           return a < b || a > b;
                                         });
static RegisterSOPC<int32_t> s_cmp_ge_i("s_cmp_ge_i32",
                                        std::greater_equal<int32_t>());
static RegisterSOPC<uint64_t> s_cmp_eq_64("s_cmp_eq_u64",
                                          std::equal_to<uint64_t>());
static RegisterSOPC<int32_t> s_cmp_gt_i("s_cmp_gt_i32",
                                        std::greater<int32_t>());
static RegisterSOPC<uint32_t> s_cmp_gt_u("s_cmp_gt_u32",
                                         std::greater<uint32_t>());
static RegisterSOPC<uint32_t> s_cmp_ge_u("s_cmp_ge_u32",
                                         std::greater_equal<uint32_t>());

// s_addk_i32: D = D + signext(imm16), SCC = overflow.
// SOPK format: the destination register is also the source.
class SAddkI32 : public Instruction {
public:
  std::function<int()> getExecutor(Wave &wave,
                                   std::string_view line) const final {
    auto partitioned = getPartitioned(line);
    assert(partitioned.size() == 3);
    auto dst = wave.getFirstRegister(partitioned[1]);
    int16_t imm = getIntFromView<int16_t>(partitioned[2]);

    return [&wave, dst, imm]() {
      int32_t old = static_cast<int32_t>(wave.getSgpr(dst.index));
      int64_t result64 = static_cast<int64_t>(old) + static_cast<int64_t>(imm);
      int32_t result = static_cast<int32_t>(result64);
      wave.setSgpr(dst.index, static_cast<uint32_t>(result));
      wave.setScc(result64 != static_cast<int64_t>(result));
      return wave.getPc() + 1;
    };
  }
};
static Register<SAddkI32> s_addk("s_addk_i32");

// s_add_f32: scalar float add (RDNA3.5+). D.f32 = S0.f32 + S1.f32.
static RegisterSOP2<uint32_t> s_add_f32("s_add_f32", [](uint32_t a, uint32_t b,
                                                        Wave &) {
  float fa = std::bit_cast<float>(a);
  float fb = std::bit_cast<float>(b);
  return std::bit_cast<uint32_t>(fa + fb);
});

// SOPK constant comparisons
static RegisterSOPK<int32_t> s_cmpk_eq_i("s_cmpk_eq_i32",
                                         [](int32_t s, int16_t i) {
                                           return s == (int32_t)i;
                                         });
static RegisterSOPK<uint32_t> s_cmpk_eq_u("s_cmpk_eq_u32",
                                          [](uint32_t s, uint16_t i) {
                                            return s == (uint32_t)i;
                                          });
static RegisterSOPK<uint32_t> s_cmpk_gt_u("s_cmpk_gt_u32",
                                          [](uint32_t s, uint16_t i) {
                                            return s > (uint32_t)i;
                                          });
static RegisterSOPK<uint32_t> s_cmpk_lg_u("s_cmpk_lg_u32",
                                          [](uint32_t s, uint16_t i) {
                                            return s != (uint32_t)i;
                                          });

} // namespace
} // namespace raceemulator
