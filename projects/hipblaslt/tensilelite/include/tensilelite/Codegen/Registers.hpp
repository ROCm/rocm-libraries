// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_REGISTERS_HPP
#define ROCM_TENSILELITE_CODEGEN_REGISTERS_HPP

#include <tensilelite/Codegen/Target.hpp>

#include "container.hpp"
#include "register.hpp"

#include <memory>
#include <string_view>

namespace rocm::tensilelite::codegen {

/// Kind of general-purpose register. An agpr is an accumulation VGPR; AGPRs
/// follow the arch VGPRs in the unified register file.
enum class RegisterKind { sgpr, vgpr, agpr };

/// Non-owning reference to count registers of kind starting at index start,
/// such as one half of a register pair. It does not keep the registers
/// allocated, so it must not be used after the range it came from is released.
struct RegisterRef {
  RegisterKind kind;
  int start;
  int count;

  /// Returns a container naming the registers, such as v[4:7], s2 or acc5.
  std::shared_ptr<rocisa::RegisterContainer> container() const;

  /// Returns a reference to count registers starting offset registers into
  /// this one, such as slice(1) for the high half of a pair. Throws
  /// std::out_of_range if they are not inside this reference.
  RegisterRef slice(int offset, int count = 1) const;

  bool operator==(const RegisterRef &) const = default;
};

class RegisterAllocator;

/// Move-only RAII handle on a contiguous range of registers of one kind; a
/// single register is a range of one. The destructor returns the registers to
/// the allocator. A moved-from or released handle owns nothing, and only
/// destruction, assignment and release() are valid on it.
class RegisterRange {
public:
  RegisterRange(RegisterRange &&other) noexcept;
  /// Releases the registers this handle owns, then takes over other's.
  RegisterRange &operator=(RegisterRange &&other) noexcept;
  RegisterRange(const RegisterRange &) = delete;
  RegisterRange &operator=(const RegisterRange &) = delete;
  ~RegisterRange();

  RegisterKind kind() const;
  /// Index of the first register.
  int start() const;
  /// Number of registers.
  int count() const;

  /// Returns a container naming the whole range, such as v[4:7], s2 or
  /// acc[0:3].
  std::shared_ptr<rocisa::RegisterContainer> container() const;
  /// Same as ref(offset, count).container(), so it throws the same exception.
  std::shared_ptr<rocisa::RegisterContainer> sub(int offset,
                                                 int count = 1) const;

  /// Returns a reference to the whole range.
  RegisterRef ref() const;
  /// Returns a reference to count registers starting offset registers into
  /// the range. Throws std::out_of_range if they are not inside the range.
  RegisterRef ref(int offset, int count = 1) const;

  /// Returns the registers to the allocator before the handle goes out of
  /// scope. Does nothing on a handle that owns nothing.
  void release();

private:
  friend class RegisterAllocator;
  RegisterRange(RegisterAllocator &allocator, RegisterKind kind, int start,
                int count);

  RegisterAllocator *allocator; // Null when the handle owns nothing.
  RegisterKind rangeKind;
  int rangeStart;
  int rangeCount;
};

/// Allocates the SGPRs, VGPRs and AGPRs of one kernel.
///
/// Wraps one rocisa::RegisterPool per kind, the same allocator the Python
/// kernel writer uses. Pools start empty and grow on demand, so their size is
/// the high-water mark; the target's limits are checked by checkLimits when
/// the kernel is finalized, not on every allocation.
///
/// The allocator must outlive every RegisterRange it hands out; destroying it
/// while ranges are alive is a programming error, caught by an assertion in
/// debug builds. It is not thread-safe.
class RegisterAllocator {
public:
  explicit RegisterAllocator(const Target &target);
  ~RegisterAllocator();
  RegisterAllocator(const RegisterAllocator &) = delete;
  RegisterAllocator &operator=(const RegisterAllocator &) = delete;

  /// Allocates count registers of kind at the lowest free index that is a
  /// multiple of alignment. Throws std::invalid_argument if count or alignment
  /// is not positive.
  RegisterRange allocate(RegisterKind kind, int count, int alignment = 1,
                         std::string_view tag = {});

  /// Reserves count registers of kind at exactly index start, for registers
  /// whose index the hardware ABI fixes (the kernarg pointer in s[0:1], for
  /// example). Reserve ABI registers before allocating anything else so their
  /// indices are still free. Throws std::invalid_argument if start is
  /// negative, count is not positive, or any requested register is in use; the
  /// returned range always starts at start.
  RegisterRange reserve(RegisterKind kind, int start, int count,
                        std::string_view tag = {});

  /// Returns one past the highest register index of kind ever handed out; the
  /// kernel descriptor's next_free_sgpr/next_free_vgpr, and the AGPR count in
  /// the descriptor and metadata, come from this.
  int highWaterMark(RegisterKind kind) const;

  /// Index of the first AGPR in the unified register file, the descriptor's
  /// accum_offset: the arch VGPR high-water mark, at least 1, rounded up to a
  /// multiple of 8 as rocisa's KernelBody::setGprs does.
  int accumOffset() const;

  /// Throws std::runtime_error if the high-water mark of any kind exceeds the
  /// target's per-kind limit, or if the accum offset plus the AGPRs exceed the
  /// unified register file.
  void checkLimits() const;

private:
  friend class RegisterRange;
  void release(RegisterKind kind, int start);
  rocisa::RegisterPool &pool(RegisterKind kind);
  const rocisa::RegisterPool &pool(RegisterKind kind) const;
  void requireFree(RegisterKind kind, int start, int count) const;
  void checkLimit(RegisterKind kind, int limit) const;
  void checkUnifiedLimit() const;

  Target target;
  rocisa::RegisterPool sgprPool;
  rocisa::RegisterPool vgprPool;
  rocisa::RegisterPool agprPool;
  int liveRanges = 0; // Ranges handed out and not yet released.
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_REGISTERS_HPP
