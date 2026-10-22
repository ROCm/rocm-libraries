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

private:
  void insertItem(const std::shared_ptr<rocisa::Item> &item);

  const Target *targetInfo;
  RegisterAllocator *registers;
  rocisa::Module *module;
  int nextLabelId = 0;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
