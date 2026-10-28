// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_KERNEL_HPP
#define ROCM_TENSILELITE_CODEGEN_KERNEL_HPP

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/KernelAbi.hpp>
#include <tensilelite/Codegen/Registers.hpp>
#include <tensilelite/Codegen/Target.hpp>

#include "code.hpp"
#include "instruction/mem.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocm::tensilelite::codegen {

/// How a kernel argument is passed.
enum class KernelArgumentKind {
  /// A pointer to global memory (8 bytes, value_kind global_buffer).
  globalPointer,
  /// A scalar copied into the kernarg segment (value_kind by_value).
  byValue,
  /// Hidden arguments the runtime fills from the launch. These value kinds
  /// exist from code object v5 on: a Kernel at v4 rejects a signature that
  /// contains them, and the runtime does not fill them in conformant v4
  /// objects. The block count is the number of workgroups in that dimension.
  hiddenBlockCountX,
  hiddenBlockCountY,
  hiddenBlockCountZ,
  /// The group size is the workgroup size in that dimension.
  hiddenGroupSizeX,
  hiddenGroupSizeY,
  hiddenGroupSizeZ,
};

/// One argument in the kernarg segment.
struct KernelArgument {
  /// Empty for hidden arguments.
  std::string name;
  KernelArgumentKind kind;
  /// Size in bytes.
  int32_t size;
  /// Byte offset from the start of the kernarg segment.
  int32_t offset;
};

/// Ordered kernel arguments laid out in the kernarg segment with natural
/// alignment: each argument starts at a multiple of its size.
class KernelSignature {
public:
  /// Alignment of the kernarg segment, and the granule its size is rounded to.
  static constexpr int32_t segmentAlignment = 8;

  /// Appends a global pointer argument and returns its offset, which the
  /// kernel body loads it from with s_load. Throws std::invalid_argument if
  /// name is not an identifier or is already used.
  int32_t addPointer(std::string_view name);
  /// Appends a by-value scalar of size 4 or 8 bytes and returns its offset.
  /// Throws std::invalid_argument for other sizes and for bad names, as
  /// addPointer does.
  int32_t addScalar(std::string_view name, int32_t size);
  /// Appends a hidden argument, which the runtime fills, and returns its
  /// offset. Block counts take 4 bytes; group sizes take 2 bytes as LLVM emits
  /// them, or 4; with a 4-byte slot only the low 16 bits are defined. Throws
  /// std::invalid_argument if kind is not hidden, size is not allowed for it,
  /// or kind was already added.
  int32_t addHidden(KernelArgumentKind kind, int32_t size);

  const std::vector<KernelArgument> &arguments() const { return args; }
  /// End of the last argument rounded up to segmentAlignment; 0 without
  /// arguments.
  int32_t segmentSize() const;

private:
  int32_t addExplicit(std::string_view name, KernelArgumentKind kind,
                      int32_t size);
  int32_t append(std::string name, KernelArgumentKind kind, int32_t size);

  std::vector<KernelArgument> args;
};

/// A kernel under construction, emitted as a complete assembly file.
///
/// The ABI (code object version, enabled workgroup ids and kernarg preloading)
/// and the complete kernarg signature are fixed at construction, so the
/// registers the hardware initializes are reserved before the body allocates
/// anything.
///
/// Kernarg preloading: the hardware can copy the first kernarg dwords into the
/// SGPRs right after the kernarg segment pointer at kernel entry. The window is
/// the longest prefix of whole explicit arguments that fits the request and the
/// user SGPRs, so s[2, 2 + preloadDwords()) holds kernarg bytes
/// [0, 4 * preloadDwords()) at entry, and the workgroup ids follow it. Every
/// argument inside the window has one RegisterRange, pre-initialized by the
/// hardware at s[2 + offset / 4], that is held until it is taken, released or
/// the Kernel is destroyed. The allocator cannot hand out a held range, so
/// nothing aliases a live preloaded value. loadArgument copies a held argument
/// with s_mov and falls back to s_load once it is not held, which is always
/// correct because the kernarg segment stays intact. The request
/// (KernelAbi::maxPreloadDwords) is an upper bound: clamping it never affects
/// correctness, only how many loads are saved.
///
/// A held range is released like every other register, in emission order, not
/// in execution order. Do not release a preloaded range, or take it and drop
/// it, inside a loop body when an earlier part of that loop, or a later
/// iteration, still reads it.
///
/// It owns the body module, the register allocator and an IRBuilder whose
/// insertion point is the body. The builder and the allocator are not movable,
/// so the kernel is built in place and is neither copyable nor movable. Label
/// names are unique only per builder, so the body must be built through this
/// kernel's builder(). rocisa keeps its ISA state per thread, so the builder
/// must be used on the thread that constructed the kernel.
class Kernel {
public:
  /// Creates an empty kernel named name with the complete signature.
  /// workgroupSize is the largest workgroup it may be launched with
  /// (max_flat_workgroup_size). Reserves the ABI registers before anything
  /// else is allocated: the kernarg segment pointer in s[0:1], the preloaded
  /// kernarg arguments, then the enabled workgroup ids x, y, z in consecutive
  /// SGPRs. Preloading is off when abi.maxPreloadDwords is 0; otherwise
  /// preloadDwords() is the largest whole-argument prefix of at most
  /// abi.maxPreloadDwords dwords (and at most the user SGPRs left after the
  /// kernarg pointer), which may be 0. Throws std::invalid_argument if name is
  /// not an identifier, workgroupSize is not in [1, 1024], the code object
  /// version is not v4, v5 or v6, the signature has hidden arguments at v4,
  /// abi.maxPreloadDwords is negative, or it is positive and the target does
  /// not support kernarg preloading (even if the window would clamp to 0).
  Kernel(const Target &target, std::string_view name, int32_t workgroupSize,
         KernelSignature signature, const KernelOptions &options = {});
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;

  const Target &target() const { return kernelTarget; }
  std::string_view name() const { return kernelName; }
  int32_t workgroupSize() const { return maxWorkgroupSize; }
  CodeObjectVersion codeObjectVersion() const { return version; }
  const KernelSignature &signature() const { return kernelSignature; }
  IRBuilder &builder() { return irBuilder; }

  /// Number of kernarg dwords preloaded into s[2, 2 + preloadDwords()); 0 when
  /// preloading is off or no whole argument fits.
  int32_t preloadDwords() const { return preloadDwordCount; }
  /// Kernarg segment pointer in s[0:1], the first user SGPRs.
  const RegisterRange &kernargSegmentPtr() const { return kernargPtr; }
  /// Workgroup id x, the first system SGPR after the user SGPRs, at
  /// s[2 + preloadDwords()].
  const RegisterRange &workgroupIdX() const { return workgroupIdXReg; }
  bool hasWorkgroupIdY() const { return workgroupIdYReg.has_value(); }
  bool hasWorkgroupIdZ() const { return workgroupIdZReg.has_value(); }
  /// Workgroup id y, at s[3 + preloadDwords()]; throws std::logic_error unless
  /// enabled.
  const RegisterRange &workgroupIdY() const;
  /// Workgroup id z, at s[4 + preloadDwords()] with y enabled and at
  /// s[3 + preloadDwords()] otherwise; throws std::logic_error unless enabled.
  const RegisterRange &workgroupIdZ() const;
  /// Workitem id in v0. On gfx90a and later the ids are packed: x is bits
  /// [9:0] (y [19:10] and z [29:20] are not enabled), so mask with 0x3ff
  /// before using it as the x id.
  const RegisterRange &workitemId() const { return workitemIdReg; }

  /// Sets the LDS bytes the kernel allocates (group_segment_fixed_size); 0 by
  /// default. Throws std::invalid_argument unless 0 <= bytes <=
  /// target().maxLdsSize().
  void setLdsSize(int32_t bytes);
  int32_t ldsSize() const { return ldsBytes; }

  /// Makes dst.count dwords (1, 2 or 4) of the kernarg segment at byte offset,
  /// such as one returned by the signature, available in dst. Dwords of a
  /// preloaded argument that is still held are copied with s_mov (64-bit when
  /// both registers are even), and skipped when source and destination are the
  /// same register; every other dword, including those straddling the window
  /// edge or in padding, is loaded with s_load (the widest aligned loads). A
  /// request spanning several arguments is split per argument. It never
  /// releases a range, so it is safe inside a loop. The caller emits the wait,
  /// which only the s_load part needs and which is harmless otherwise. Throws
  /// std::invalid_argument if dst is not SGPRs aligned to its count, or the
  /// request is misaligned or outside the segment.
  void loadArgument(RegisterRef dst, int32_t offset);

  /// Moves the pre-initialized range of the argument starting at byte offset
  /// to the caller, who owns it like any allocated range. Returns std::nullopt
  /// if the argument is not preloaded or was already taken or released. Hidden
  /// arguments count as arguments here; they are never preloaded. Throws
  /// std::invalid_argument if offset is not the start of an argument.
  std::optional<RegisterRange> takePreloadedArgument(int32_t offset);

  /// Frees the pre-initialized range of the argument starting at byte offset
  /// so the allocator can reuse it; does nothing if it is not held. Throws
  /// std::invalid_argument if offset is not the start of an argument.
  void releasePreloadedArgument(int32_t offset);

  /// Returns the complete assembly file: target and code object version
  /// directives, the kernel symbol and body, the kernel descriptor and the
  /// .amdgpu_metadata. Throws std::runtime_error if the registers used exceed
  /// the target's limits.
  std::string toAssembly() const;

private:
  /// A preloaded explicit argument and, while held, its register range.
  struct PreloadedArgument {
    int32_t offset;
    int32_t size;
    std::optional<RegisterRange> range;
  };

  /// Returns the preloaded argument at byte offset if offset is the start of
  /// an argument, else throws std::invalid_argument; null if the argument is
  /// outside the window.
  PreloadedArgument *findPreloaded(int32_t offset);
  /// Returns the preloaded argument whose held range covers the dword at byte
  /// offset, or null.
  const PreloadedArgument *heldArgumentAt(int32_t offset) const;

  // Ranges, including the preloaded ones, are declared after the allocator so
  // they are released first, and the body before the builder that points into
  // it. The preload window size precedes the allocator because the workgroup
  // id reservations below depend on it.
  Target kernelTarget;
  std::string kernelName;
  int32_t maxWorkgroupSize;
  int32_t ldsBytes = 0;
  CodeObjectVersion version;
  KernelSignature kernelSignature;
  int32_t preloadDwordCount;
  RegisterAllocator registers;
  rocisa::Module bodyModule;
  IRBuilder irBuilder;
  RegisterRange kernargPtr;
  RegisterRange workgroupIdXReg;
  RegisterRange workitemIdReg;
  std::optional<RegisterRange> workgroupIdYReg;
  std::optional<RegisterRange> workgroupIdZReg;
  std::vector<PreloadedArgument> preloaded;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_KERNEL_HPP
