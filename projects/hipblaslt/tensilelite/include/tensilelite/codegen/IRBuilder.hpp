// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
#define ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP

#include <tensilelite/codegen/Operand.hpp>
#include <tensilelite/codegen/Registers.hpp>
#include <tensilelite/codegen/Target.hpp>

#include "code.hpp"
#include "instruction/common.hpp"

#include <concepts>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace rocm::tensilelite::codegen {

/// Outstanding-operation counts that a wait instruction waits down to. An
/// empty counter is not waited on. Each field maps to one rocisa SWaitCnt
/// counter, and rocisa combines them into the target's wait instructions.
struct WaitCounts {
  /// Vector memory loads (rocisa vlcnt).
  std::optional<int> vectorLoads;
  /// Vector memory stores (rocisa vscnt).
  std::optional<int> vectorStores;
  /// LDS operations (rocisa dscnt).
  std::optional<int> lds;
  /// Scalar memory loads and messages (rocisa kmcnt).
  std::optional<int> scalarMemory;
};

/// Integer comparison predicates. The s and u prefixes compare as signed or
/// unsigned values; eq and ne are sign-agnostic.
enum class Predicate { eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge };

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
  RegisterRange allocateVgpr(int count, int alignment = 1,
                             std::string_view tag = {});
  /// Allocates count SGPRs through the allocator; see
  /// RegisterAllocator::allocate.
  RegisterRange allocateSgpr(int count, int alignment = 1,
                             std::string_view tag = {});
  /// Allocates count AGPRs through the allocator; see
  /// RegisterAllocator::allocate.
  RegisterRange allocateAgpr(int count, int alignment = 1,
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
  // anything, if the combination is not supported, if dst or a register src
  // has the wrong number of registers, or if an immediate does not fit.

  /// Emits dst = src for one register. An immediate must lie in
  /// [INT32_MIN, UINT32_MAX] and is used as a 32-bit pattern.
  void movb32(RegisterRef dst, Operand src);
  /// Emits dst = src for a register pair; an immediate is the full 64-bit
  /// value. One 64-bit mov is used when both pairs are even-aligned and the
  /// kinds allow it, and an immediate is an inline constant or in
  /// [0, UINT32_MAX]; otherwise two movb32 on the halves, which covers
  /// misaligned pairs, AGPRs and other immediates. Overlapping pairs are
  /// copied correctly.
  void movb64(RegisterRef dst, Operand src);

  // The 32-bit integer ops below share these rules. Results wrap modulo 2^32,
  // so the ops are sign-agnostic.
  //
  // The op is a SALU instruction (s_add_u32, s_sub_u32, s_mul_i32) iff dst and
  // every register operand are SGPRs; SALU add and sub write SCC. A VGPR dst
  // gives a VALU instruction (v_add_u32, v_sub_u32, v_mul_lo_u32). SGPR and
  // immediate operands are legalized for gfx9 (one SGPR or literal per
  // instruction, no literals in VOP3, VOP2 src1 a VGPR) by swapping
  // commutative operands or copying an operand into a temporary register,
  // which is released after the op.
  //
  // Immediates must lie in [INT32_MIN, UINT32_MAX] and are used as 32-bit
  // patterns. dst may alias an operand.
  //
  // Throws std::invalid_argument, before emitting anything, if dst or a
  // register operand is not exactly one register or is an AGPR, if an
  // immediate does not fit 32 bits, or if dst is an SGPR and an operand is a
  // VGPR.

  /// Emits dst = a + b.
  void addi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a - b.
  void subi32(RegisterRef dst, Operand a, Operand b);
  /// Emits dst = the low 32 bits of a * b.
  void muli32(RegisterRef dst, Operand a, Operand b);

  // The 64-bit integer ops below share these rules. Results wrap modulo 2^64,
  // so the ops are sign-agnostic.
  //
  // dst and the register operands are register pairs, and an immediate is any
  // int64 value, its full 64-bit pattern. The op is a SALU sequence iff dst and
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
  void muli64(RegisterRef dst, Operand a, Operand b);

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
  // AGPR or has the wrong number of registers, if an immediate does not fit,
  // or if a uniform bool destination has a VGPR operand.

  /// Emits dst = a pred b for one-register operands. Immediates must lie in
  /// [INT32_MIN, UINT32_MAX] and are used as 32-bit patterns.
  void cmpi32(Predicate pred, RegisterRef dst, Operand a, Operand b);
  /// Emits dst = a pred b for register-pair operands; immediates are full
  /// 64-bit values. The scalar ordered predicates use a SALU borrow chain, so
  /// they do not depend on exec.
  void cmpi64(Predicate pred, RegisterRef dst, Operand a, Operand b);

  /// Emits dst = cond ? a : b. The width comes from dst, 1 or 2 registers, and
  /// register operands must match it; immediates are 32-bit patterns (as for
  /// cmpi32) for one register and full 64-bit values for a pair. cond is an
  /// SGPR holding a uniform bool, where nonzero is true, or an SGPR pair at an
  /// even index holding a lane mask. An SGPR dst needs a uniform bool cond and
  /// scalar operands. A VGPR dst takes either cond, widening a uniform bool to
  /// a temporary mask, and any SGPR, VGPR or immediate operands. A mask
  /// written by a VALU instruction outside the builder must already satisfy
  /// the two wait states; masks from cmpi32 and cmpi64 do. Throws
  /// std::invalid_argument, before emitting anything, if dst is not 1 or 2
  /// registers, if dst, cond or an operand is an AGPR, if cond has the wrong
  /// kind or shape, if an operand has the wrong number of registers, if an
  /// immediate does not fit, or if an SGPR dst has a mask cond or a VGPR
  /// operand.
  void select(RegisterRef dst, RegisterRef cond, Operand a, Operand b);

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
  /// that never touches memory. sizeBytes is one SGPR or an immediate in
  /// [0, UINT32_MAX]. A descriptor must be uniform, so VGPRs and AGPRs are
  /// rejected. dst may overlap pointer or sizeBytes; each word is read before
  /// it is overwritten, and a word whose source already is its destination is
  /// not copied. Throws std::invalid_argument, before emitting anything, if an
  /// operand has the wrong kind, width or alignment, or sizeBytes is out of
  /// range.
  void makeRsrc(RegisterRef dst, Operand pointer, Operand sizeBytes);

private:
  void insertItem(const std::shared_ptr<rocisa::Item> &item);

  const Target *targetInfo;
  RegisterAllocator *registers;
  rocisa::Module *module;
  int nextLabelId = 0;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
