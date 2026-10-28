// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
#define ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP

#include <tensilelite/Codegen/Registers.hpp>
#include <tensilelite/Codegen/Target.hpp>

#include "code.hpp"
#include "instruction/common.hpp"
#include "instruction/instruction.hpp"

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace rocm::tensilelite::codegen {

/// An instruction source: a register or a 32-bit immediate, held as the rocisa
/// InstructionInput that instruction constructors take, so an Operand is passed
/// to them as is. A register is its RegisterContainer, of any width, so a pair
/// is one Operand; an immediate is the input's int. Converts implicitly from a
/// RegisterRange, a RegisterRef or an int32_t, so builder calls read like
/// builder.addi32(dst, a, 4); the 64-bit ops sign-extend an immediate. 64-bit
/// and unsigned integers do not convert, so a wide constant is a compile error
/// rather than a silent truncation: write a 32-bit pattern as its int32_t
/// value, such as -1 for 0xFFFFFFFF, and build a wider constant in a register
/// pair with two movb32.
class Operand : public InstructionInput {
public:
  Operand(const RegisterRange &range) : Operand(range.ref()) {}
  Operand(RegisterRef ref)
      : InstructionInput(std::shared_ptr<rocisa::Container>(ref.container())) {}
  Operand(int32_t immediate) : InstructionInput(immediate) {}
  Operand(int64_t) = delete;
  Operand(uint32_t) = delete;
  Operand(uint64_t) = delete;

  bool isRegister() const {
    return std::holds_alternative<std::shared_ptr<rocisa::Container>>(
        static_cast<const InstructionInput &>(*this));
  }
  /// True if this is a register of the given kind.
  bool isRegister(RegisterKind kind) const {
    return isRegister() && reg().kind == kind;
  }
  bool isImmediate() const {
    return std::holds_alternative<int32_t>(
        static_cast<const InstructionInput &>(*this));
  }

  /// The register, read back from its container; asserts isRegister().
  RegisterRef reg() const;
  /// The immediate; asserts isImmediate().
  int32_t immediate() const {
    assert(isImmediate() && "operand is not an immediate");
    return std::get<int32_t>(static_cast<const InstructionInput &>(*this));
  }
};

static_assert(sizeof(Operand) == sizeof(InstructionInput),
              "operand adds no state to the instruction input");

/// Outstanding-operation counts that a wait instruction waits down to. An
/// empty counter is not waited on. Each field maps to one rocisa SWaitCnt
/// counter, and rocisa combines them into the target's wait instructions.
struct WaitCounts {
  /// Vector memory loads (rocisa vlcnt).
  std::optional<int32_t> vectorLoads;
  /// Vector memory stores (rocisa vscnt).
  std::optional<int32_t> vectorStores;
  /// LDS operations (rocisa dscnt).
  std::optional<int32_t> lds;
  /// Scalar memory loads and messages (rocisa kmcnt).
  std::optional<int32_t> scalarMemory;
};

/// Integer comparison predicates. The s and u prefixes compare as signed or
/// unsigned values; eq and ne are sign-agnostic.
enum class Predicate { eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge };

/// The size of a memory access. b8 and b16 use one register and loads
/// zero-extend into it; b32 to b128 use one register per dword.
enum class MemoryWidth { b8, b16, b32, b64, b96, b128 };

/// The encoding controls of a buffer load or store, all off by default.
struct BufferAccess {
  /// Unsigned byte offset added to the address, 0 to 4095.
  int32_t offset = 0;
  /// The gfx950 cache-policy bits (rocisa's glc, slc and nt). They select the
  /// cache coherence scope and the non-temporal hint as in the CDNA3 ISA.
  bool sc0 = false;
  bool sc1 = false;
  bool nt = false;
};

/// Appends rocisa instructions to an insertion point, in the style of LLVM's
/// IRBuilder.
///
/// The builder never owns IR. It holds non-owning references to the target,
/// the register allocator and the insertion-point module, all of which must
/// outlive it. Like LLVM's BasicBlock pointer, the insertion point is a plain
/// pointer: rocisa itself links items to their parent through a raw Item
/// pointer, so a pointer lets a kernel own its body Module by value and lets
/// nested Modules, kept alive by the shared_ptr in their parent's item list,
/// serve as insertion points without forcing heap allocation or shared
/// ownership. It is a pointer rather than a reference so setInsertionPoint can
/// rebind it.
///
/// rocisa keeps its selected ISA per thread, so the constructor selects the
/// target for the calling thread and the builder must only be used on the
/// thread that created it. The builder is not thread-safe. Label names are
/// unique per builder, so it is not copyable.
class IRBuilder {
public:
  /// RAII guard that restores the builder's insertion point on destruction.
  class InsertionGuard {
  public:
    explicit InsertionGuard(IRBuilder &builder);
    ~InsertionGuard();
    InsertionGuard(const InsertionGuard &) = delete;
    InsertionGuard &operator=(const InsertionGuard &) = delete;

  private:
    IRBuilder &builder;
    rocisa::Module *savedInsertionPoint;
  };

  IRBuilder(const Target &target, RegisterAllocator &allocator,
            rocisa::Module &insertionPoint);
  IRBuilder(const IRBuilder &) = delete;
  IRBuilder &operator=(const IRBuilder &) = delete;

  const Target &target() const { return *targetInfo; }
  RegisterAllocator &allocator() const { return *registers; }
  rocisa::Module &insertionPoint() const { return *module; }
  void setInsertionPoint(rocisa::Module &newInsertionPoint) {
    module = &newInsertionPoint;
  }

  /// Constructs an Inst from args, appends it at the insertion point and
  /// returns it.
  template <typename Inst, typename... Args>
    requires std::derived_from<Inst, rocisa::Item>
  std::shared_ptr<Inst> create(Args &&...args) {
    return insert(std::make_shared<Inst>(std::forward<Args>(args)...));
  }

  /// Appends an item created earlier, such as a label from createLabel, at the
  /// insertion point and returns it. Throws std::invalid_argument if item is
  /// null and std::logic_error if it is already in a module.
  template <typename T>
    requires std::derived_from<T, rocisa::Item>
  std::shared_ptr<T> insert(std::shared_ptr<T> item) {
    insertItem(item);
    return item;
  }

  /// Creates a label named name_N, unique within this builder, without
  /// inserting it, so that forward branches can target it before insert places
  /// it. Branch instructions take label->getLabelName().
  std::shared_ptr<rocisa::Label> createLabel(std::string_view name);
  /// Appends a single-line comment.
  std::shared_ptr<rocisa::TextBlock> createComment(std::string_view text);
  /// Appends the wait instructions for counts.
  std::shared_ptr<rocisa::SWaitCnt>
  createWaitCnt(WaitCounts counts, std::string_view comment = {});
  /// Appends s_endpgm.
  std::shared_ptr<rocisa::SEndpgm> createEndProgram();

  /// Allocates count VGPRs through the allocator; see
  /// RegisterAllocator::allocate.
  RegisterRange allocateVgpr(int32_t count, int32_t alignment = 1,
                             std::string_view tag = {});
  /// Allocates count SGPRs through the allocator; see
  /// RegisterAllocator::allocate.
  RegisterRange allocateSgpr(int32_t count, int32_t alignment = 1,
                             std::string_view tag = {});
  /// Allocates count AGPRs through the allocator; see
  /// RegisterAllocator::allocate.
  RegisterRange allocateAgpr(int32_t count, int32_t alignment = 1,
                             std::string_view tag = {});

  // The moves below copy one value into a register.
  //
  // movb32 and movb64 pick the instruction from the register kinds: s_mov_b32
  // and s_mov_b64 for an SGPR dst, v_mov_b32 and v_mov_b64 for a VGPR dst,
  // v_accvgpr_write_b32 for an AGPR dst and v_accvgpr_read_b32 for an AGPR src.
  // There is no implicit readfirstlane, so an SGPR dst rejects a VGPR or AGPR
  // src. An AGPR dst takes a VGPR, an SGPR or an inline constant, but no
  // literal (copy it to a VGPR first) and no AGPR (rocisa has no
  // v_accvgpr_mov_b32). Throws std::invalid_argument, before emitting
  // anything, if the combination is not supported or if dst or a register src
  // has the wrong number of registers.

  /// Emits dst = src for one register; an immediate is its 32-bit pattern.
  void movb32(RegisterRef dst, Operand src);
  /// Emits dst = src for a register pair; an immediate is an int32_t
  /// sign-extended to 64 bits. One 64-bit mov is used when both pairs are
  /// even-aligned, the kinds allow it and an immediate is an inline constant or
  /// non-negative (a 64-bit mov zero-extends a literal); otherwise two movb32
  /// on the halves, which covers misaligned pairs, AGPRs and other immediates.
  /// Overlapping pairs are copied correctly.
  void movb64(RegisterRef dst, Operand src);

  // The 32-bit integer ops below share these rules. Results wrap modulo 2^32,
  // so the ops are sign-agnostic, except that mulhsi32 and the min and max ops
  // are explicitly signed or unsigned.
  //
  // The op is a SALU instruction (s_add_u32, s_sub_u32, s_mul_i32,
  // s_mul_hi_i32, s_min_i32, s_max_i32, s_min_u32, s_max_u32) iff dst and every
  // register operand are SGPRs; SALU add, sub, min and max write SCC, and
  // s_mul_i32 and s_mul_hi_i32 do not. A VGPR dst gives a VALU instruction
  // (v_add_u32, v_sub_u32, v_mul_lo_u32, v_mul_hi_i32, v_min_i32, v_max_i32,
  // v_min_u32, v_max_u32). SGPR and immediate operands are legalized for gfx9
  // (one SGPR or literal per instruction, no literals in VOP3, VOP2 src1 a
  // VGPR) by swapping commutative operands or copying an operand into a
  // temporary register, which is released after the op. The copies are moves
  // and write neither SCC nor VCC.
  //
  // Immediates are 32-bit patterns. dst may alias an operand.
  //
  // Throws std::invalid_argument, before emitting anything, if dst or a
  // register operand is not exactly one register or is an AGPR, or if dst is
  // an SGPR and an operand is a VGPR.

  /// Emits dst = a + b.
  void addi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a - b.
  void subi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the low 32 bits of a * b.
  /// An immediate 2^k in either operand (INT32_MIN counts as 2^31) gives a move
  /// for k == 0 and otherwise a shift, s_lshl_b32 or v_lshlrev_b32. Unlike
  /// s_mul_i32, the SALU shift writes SCC. A multiplication by 1 is a move of
  /// the other operand, or no instruction at all when that operand is dst, so
  /// it writes no SCC.
  void muli32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the high 32 bits of the signed 64-bit product a * b, as an
  /// s_mul_hi_i32 or a v_mul_hi_i32 (VOP3: literals are copied to VGPRs). It
  /// writes no SCC, and no VCC. The operands are signed, so for unsigned
  /// values the result differs from the unsigned high product whenever an
  /// operand is at or above 2^31.
  void mulhsi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the smaller of a and b as signed values. The SALU form,
  /// s_min_i32, writes SCC; the VALU form is v_min_i32.
  void minsi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the larger of a and b as signed values. The SALU form,
  /// s_max_i32, writes SCC; the VALU form is v_max_i32.
  void maxsi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the smaller of a and b as unsigned values. The SALU form,
  /// s_min_u32, writes SCC; the VALU form is v_min_u32.
  void minui32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the larger of a and b as unsigned values. The SALU form,
  /// s_max_u32, writes SCC; the VALU form is v_max_u32.
  void maxui32(RegisterRef dst, Operand a, Operand b);

  // The 64-bit integer ops below share these rules. Results wrap modulo 2^64,
  // so the ops are sign-agnostic.
  //
  // dst and the register operands are register pairs, and an immediate is an
  // int32_t sign-extended to 64 bits, as in C; build a wider constant in a
  // register pair with two movb32. The op is a SALU sequence iff dst and
  // every register operand are SGPRs: add is s_add_u32 + s_addc_u32, sub is
  // s_sub_u32 + s_subb_u32 and mul is s_mul_i32, s_mul_hi_u32 and s_add_u32;
  // they write SCC. A VGPR dst gives a VALU sequence: add is v_add_co_u32 +
  // v_addc_co_u32 and sub is v_sub_co_u32 + v_subb_co_u32, carrying through a
  // temporary SGPR pair so that VCC is untouched, with s_nop 1 before the
  // carry-in read; mul is v_mul_lo_u32, v_mul_hi_u32 and v_add3_u32.
  //
  // Every instruction is 32-bit, so pairs may start at any index and an
  // immediate is split into 32-bit halves; no 64-bit literal is ever emitted.
  // Operands that gfx9 cannot encode are copied into temporaries, which are
  // released after the op. dst may alias or overlap a or b.
  //
  // Throws std::invalid_argument, before emitting anything, if dst or a
  // register operand is not exactly two registers or is an AGPR, or if dst is
  // an SGPR pair and an operand is a VGPR.

  /// Emits dst = a + b.
  void addi64(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a - b.
  void subi64(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the low 64 bits of a * b.
  /// An immediate 2^k (k 0..30) in either operand gives movb64 (k == 0) or
  /// shli64: one s_lshl_b64 or v_lshlrev_b64 for even-aligned pairs, staged as
  /// shli64 documents otherwise. The SALU form writes SCC like the other
  /// 64-bit SALU sequences.
  void muli64(RegisterRef dst, Operand a, Operand b);

  // The bitwise and shift ops below share these rules.
  //
  // The 32-bit forms take single registers, and an immediate is a 32-bit
  // pattern. The 64-bit forms take register pairs for dst and a, and for b of
  // andi64 and ori64; a pair immediate is an int32_t sign-extended to 64 bits.
  //
  // For the shifts, a is the value and b the shift amount, which is always one
  // register or an int32_t immediate, also in the 64-bit forms. Only the low 5
  // bits (32-bit) or 6 bits (64-bit) of b are used, as the hardware does, so
  // shli32(dst, a, 33) shifts by 1.
  //
  // The op is a SALU instruction iff dst and every register operand are SGPRs:
  // s_and_b32/b64, s_or_b32/b64, s_lshl_b32/b64, s_lshr_b32/b64 and
  // s_ashr_i32/i64; they write SCC, except that a 32-bit shift by an immediate
  // amount whose low 5 bits are 0 is a move of a, or no instruction when a is
  // dst, and so writes no SCC. A VGPR dst gives v_and_b32, v_or_b32,
  // v_lshlrev_b32/b64, v_lshrrev_b32/b64 or v_ashrrev_i32/i64; the builder
  // hides the reversed operand order. gfx9 has no 64-bit VALU and/or, so
  // andi64 and ori64 into VGPRs are one 32-bit op per half, and so is the SALU
  // form when a pair is at an odd index or an immediate is a negative literal.
  // The 64-bit shifts need even-aligned pairs: a misaligned value is copied
  // into a temporary pair and a misaligned dst is written through one. Other
  // operands gfx9 cannot encode are legalized as for the arithmetic ops, by
  // swapping and/or operands or copying into temporaries, which are released
  // after the op.
  //
  // dst may alias or overlap an operand.
  //
  // Throws std::invalid_argument, before emitting anything, if dst or a
  // register operand is an AGPR or has the wrong number of registers, or if
  // dst is an SGPR and an operand is a VGPR.

  /// Emits dst = a & b.
  void andi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a | b.
  void ori32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a << (b & 31).
  void shli32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a >> (b & 31), shifting in zeros.
  void shrui32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a >> (b & 31), shifting in copies of the sign bit.
  void shrsi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a & b for register pairs.
  void andi64(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a | b for register pairs.
  void ori64(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a << (b & 63).
  void shli64(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a >> (b & 63), shifting in zeros.
  void shrui64(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a >> (b & 63), shifting in copies of the sign bit.
  void shrsi64(RegisterRef dst, Operand a, Operand b);

  // Division.
  //
  // The signed divisions below share these rules.
  //
  // useIEEE selects one of LLVM's AMDGPU expansions. true is the exact one,
  // correct for all inputs: an f32 reciprocal of |b| scaled by 2^32 - 512, one
  // Newton step and two corrections on the magnitudes, then the sign. false is
  // LLVM's 24-bit expansion, which divides in f32 and applies one +-1
  // correction; it is exact only when |a| < 2^23 and |b| < 2^23, and other
  // inputs give unspecified results.
  //
  // An immediate b = 2^k for k 0..30 (1 to 2^30) uses LLVM's shift expansions
  // instead, exact for every a and for either useIEEE. The quotient is
  // (a + ((a >> 31) >>> (32 - k))) >> k: four instructions, three for b = 2
  // and a copy for b = 1. The remainder is a - ((a + that bias) & -2^k): five
  // instructions, four for b = 2 and a move of 0 for b = 1. These are plain
  // SALU or VALU ops, so the SALU form then has no readfirstlane and no
  // active-lane precondition.
  //
  // Any other immediate b >= 3 (a constant that is not a power of two) uses
  // LLVM's magic-number expansion, also exact for every a, including INT32_MIN
  // and INT32_MAX, and for either useIEEE. The quotient is the high half of
  // a * M (mulhsi32, with M the magic multiplier of b), plus a when M is
  // negative, shifted right arithmetically by the magic shift when that is not
  // 0, plus the sign bit of that result: three to five instructions, one
  // temporary register for the sign bit, and one more when dst aliases a (the
  // intermediate values then live in a temporary, so dst is written only by the
  // last instruction). The remainder is a - q * b for that quotient q in a
  // temporary: two instructions more. These are plain SALU or VALU ops with a
  // literal multiplier legalized as for mulhsi32, so the SALU form has no
  // readfirstlane and no active-lane precondition, and the VALU form writes no
  // VCC. Negative divisors, including -1 and the negative powers of two, and
  // INT32_MIN use the general expansions.
  //
  // The op is a SALU sequence iff dst and every register operand are SGPRs
  // (the uniform form), and a VALU sequence for a VGPR dst, where SGPR and
  // immediate operands are legalized. The uniform form of the general
  // expansions computes the f32 reciprocal in temporary VGPRs and moves it to
  // an SGPR with v_readfirstlane_b32, as LLVM lowers uniform division, so it
  // needs at least one active lane: readfirstlane with exec == 0 is undefined,
  // and LLVM has the same precondition. The SALU forms write SCC, and the fast
  // path reads exec.
  //
  // Neither form writes exec or VCC. In the general expansions every
  // correction is a compare and a select, with masks in temporary SGPR pairs.
  // Temporary VGPRs, SGPRs and an SGPR pair are allocated and released inside
  // the op.
  //
  // dst is written only by the last instruction, so dst may alias a or b.
  //
  // Throws std::invalid_argument, before emitting anything, if dst or a
  // register operand is not exactly one register or is an AGPR, or if dst is
  // an SGPR and an operand is a VGPR.

  /// Emits dst = a / b for signed 32-bit values, with the quotient truncated
  /// toward zero like C and LLVM's sdiv. INT32_MIN / -1 wraps to INT32_MIN.
  /// Division by zero gives an unspecified value, but never faults or hangs.
  void divsi32(RegisterRef dst, Operand a, Operand b, bool useIEEE);
  /// Emits dst = a % b for signed 32-bit values with C semantics: the
  /// remainder has the sign of a and a == (a / b) * b + a % b. INT32_MIN % -1
  /// is 0. Division by zero gives an unspecified value, but never faults or
  /// hangs. The exact form is LLVM's srem expansion; the 24-bit form computes
  /// the quotient and returns a - q * b.
  void modsi32(RegisterRef dst, Operand a, Operand b, bool useIEEE);

  // The comparisons below share these rules.
  //
  // A destination of one SGPR holds a uniform bool, 0 or 1, and needs scalar
  // operands (SGPRs and immediates). An SGPR pair at an even index holds a lane
  // mask, one bit per lane; in the scalar form all ones or all zeros. The
  // scalar form is a SALU sequence; any VGPR operand gives the VALU form, which
  // needs a lane mask destination and ends with s_nop 1 so that any following
  // instruction can read the mask (VALU reads of an SGPR a VALU instruction
  // wrote need two wait states). Operands that gfx9 cannot encode, such as
  // literals in VOP3 or SGPR pairs at an odd index, are copied into temporary
  // registers, which are released afterwards.
  //
  // dst may alias an operand. Throws std::invalid_argument, before emitting
  // anything, if dst is not one of the destinations above, if an operand is an
  // AGPR or has the wrong number of registers, or if a uniform bool
  // destination has a VGPR operand.

  /// Emits dst = a pred b for one-register operands; immediates are 32-bit
  /// patterns.
  void cmpi32(Predicate pred, RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a pred b for register-pair operands; an immediate is an
  /// int32_t sign-extended to 64 bits. The scalar ordered predicates use a SALU
  /// borrow chain, so they do not depend on exec.
  void cmpi64(Predicate pred, RegisterRef dst, Operand a, Operand b);

  // The selects below compute dst = cond ? a : b. cond is an SGPR holding a
  // uniform bool, where nonzero is true, or an SGPR pair at an even index
  // holding a lane mask. An SGPR dst needs a uniform bool cond and scalar
  // operands. A VGPR dst takes either cond, widening a uniform bool to a
  // temporary mask, and any SGPR, VGPR or immediate operands. A mask written
  // by a VALU instruction outside the builder must already satisfy the two
  // wait states; masks from cmpi32 and cmpi64 do.
  //
  // Throws std::invalid_argument, before emitting anything, if dst is not the
  // right number of registers (1 for selectb32, 2 for selectb64) or is an
  // AGPR, if cond or an operand is an AGPR, if cond has the wrong kind or
  // shape, if an operand has the wrong number of registers, or if an SGPR dst
  // has a mask cond or a VGPR operand.

  /// Emits dst = cond ? a : b for one register; immediates are 32-bit
  /// patterns.
  void selectb32(RegisterRef dst, RegisterRef cond, Operand a, Operand b);
  /// Emits dst = cond ? a : b for a register pair; an immediate is
  /// sign-extended to 64 bits.
  void selectb64(RegisterRef dst, RegisterRef cond, Operand a, Operand b);

  // The selects below have no cond: the condition is implicit and the caller
  // must have set it. An SGPR dst gives s_cselect_b32 (one per word, or one
  // s_cselect_b64 for an even-aligned pair), which consumes the current SCC
  // with no s_cmp re-test: SCC set selects a, clear selects b. A VGPR dst gives
  // one v_cndmask_b32 per word that reads the current VCC as the lane mask, one
  // bit per lane with a set bit selecting a, with no test and no widening. SCC
  // and VCC must be live and unclobbered up to the select: the builder never
  // writes VCC, and its legalization copies are moves, which write neither SCC
  // nor VCC, but the SALU forms of addi32, subi32, muli32 by a power of two,
  // andi32, ori32, the shifts (but not a shift by zero), minsi32, maxsi32,
  // minui32, maxui32, the 64-bit ops and the comparisons write SCC, so the
  // caller must place the select before them. A mask that a VALU instruction
  // wrote to VCC must already satisfy the two wait states; the builder emits
  // no s_nop here.
  //
  // The operand rules are those of the selects with a cond: an SGPR dst needs
  // scalar operands, which gfx9 cannot always encode and are then copied into
  // temporary SGPRs; a VGPR dst takes SGPR, VGPR or immediate operands, copying
  // all but VGPRs and inline constants into temporary VGPRs, since the implicit
  // VCC read already uses the constant bus. dst may alias or overlap an operand
  // as for the selects with a cond.
  //
  // Throws std::invalid_argument, before emitting anything, if dst is not the
  // right number of registers (1 for selectb32, 2 for selectb64) or is an
  // AGPR, if an operand is an AGPR or has the wrong number of registers, or if
  // an SGPR dst has a VGPR operand.

  /// Emits dst = SCC (SGPR dst) or VCC (VGPR dst) ? a : b for one register;
  /// immediates are 32-bit patterns.
  void selectb32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = SCC (SGPR dst) or VCC (VGPR dst) ? a : b for a register pair;
  /// an immediate is sign-extended to 64 bits. A VGPR pair is one v_cndmask_b32
  /// per word, both reading the same VCC.
  void selectb64(RegisterRef dst, Operand a, Operand b);

  // Buffer resource descriptors.

  /// Emits a raw buffer resource descriptor (V#) for buffer_load and
  /// buffer_store into dst, which must be 4 SGPRs starting at an index that is
  /// a multiple of 4. The layout is: dst[0:1] = pointer, the 48-bit base
  /// address with stride and swizzle 0, so offsets are plain bytes; dst+2 =
  /// sizeBytes, the num_records the hardware checks offsets against; dst+3 =
  /// the target's fixed upper word from rocisa::SrdUpperValue (0x00020000, a
  /// 32-bit data format, on gfx9). An access whose byte offset is >= sizeBytes
  /// is out of bounds: loads return 0 and stores are dropped, so sizeBytes 0
  /// makes every access out of bounds.
  ///
  /// pointer must be an SGPR pair. An immediate pointer is rejected because a
  /// descriptor base is a runtime address; use sizeBytes 0 for a descriptor
  /// that never touches memory. sizeBytes is one SGPR or an immediate, the
  /// 32-bit pattern of the unsigned size (-1 is 0xFFFFFFFF). A descriptor must
  /// be uniform, so VGPRs and AGPRs are rejected. dst may overlap pointer or
  /// sizeBytes; each word is read before it is overwritten, and a word whose
  /// source already is its destination is not copied. Throws
  /// std::invalid_argument, before emitting anything, if an operand has the
  /// wrong kind, width or alignment.
  void makeRsrc(RegisterRef dst, Operand pointer, Operand sizeBytes);

  // Memory.
  //
  // The loads and stores below only issue the instruction. Waiting for the
  // result of a load or the completion of a store is the caller's job, with
  // createWaitCnt: WaitCounts::lds for DS, WaitCounts::vectorLoads and
  // vectorStores for buffer.
  //
  // dst and data are VGPRs or AGPRs with as many registers as the width needs:
  // 1 for b8, b16 and b32, 2 for b64, 3 for b96 and 4 for b128. Two or more
  // registers must start at an even index, the gfx950 tuple rule. b8 and b16
  // loads zero-extend into the register; b8 and b16 stores write only the low
  // byte or short.
  //
  // Throws std::invalid_argument, before emitting anything, if dst, data,
  // address, rsrc, voffset or soffset has the wrong kind, count or alignment,
  // if soffset is an immediate outside 0 to 64, or if an offset is out of
  // range.

  /// Emits a DS load (ds_read_*) of width from the LDS byte address in address,
  /// which is one VGPR, plus offset, an unsigned 16-bit immediate (0 to 65535).
  /// The access must lie inside the kernel's LDS allocation
  /// (Kernel::setLdsSize). gfx950 has no GDS, so offset is the only DS control.
  void loadDs(RegisterRef dst, RegisterRef address, MemoryWidth width,
              int32_t offset = 0);
  /// Emits a DS store (ds_write_*) of data at address plus offset, with the
  /// rules of loadDs.
  void storeDs(RegisterRef address, RegisterRef data, MemoryWidth width,
               int32_t offset = 0);
  /// Emits a buffer load (buffer_load_*) of width through the descriptor rsrc,
  /// which is 4 SGPRs at an index that is a multiple of 4, as makeRsrc writes.
  /// The byte offset is voffset + soffset + access.offset: voffset is one VGPR
  /// (there is no form without one, so pass a VGPR holding 0 if needed),
  /// soffset is one SGPR or an immediate 0 to 64 (it is an unsigned byte
  /// offset, so a negative value is never valid) and
  /// access.offset is 0 to 4095, which the builder checks because the
  /// assembler silently mis-encodes larger values. An access is out of bounds
  /// when that whole offset, soffset included, is >= the descriptor's
  /// sizeBytes: loads return 0 and stores are dropped. Accesses that straddle
  /// the end are not specified, so keep sizes a multiple of the access size.
  void loadBuffer(RegisterRef dst, RegisterRef rsrc, RegisterRef voffset,
                  Operand soffset, MemoryWidth width, BufferAccess access = {});
  /// Emits a buffer store (buffer_store_*) of data, with the rules of
  /// loadBuffer.
  void storeBuffer(RegisterRef data, RegisterRef rsrc, RegisterRef voffset,
                   Operand soffset, MemoryWidth width,
                   BufferAccess access = {});

private:
  void insertItem(const std::shared_ptr<rocisa::Item> &item);

  const Target *targetInfo;
  RegisterAllocator *registers;
  rocisa::Module *module;
  int32_t nextLabelId = 0;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
