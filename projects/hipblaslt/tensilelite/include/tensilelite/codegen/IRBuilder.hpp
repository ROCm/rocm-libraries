// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
#define ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP

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

private:
  void insertItem(const std::shared_ptr<rocisa::Item> &item);

  const Target *targetInfo;
  RegisterAllocator *registers;
  rocisa::Module *module;
  int nextLabelId = 0;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_IRBUILDER_HPP
