// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_KERNEL_HPP
#define ROCM_TENSILELITE_CODEGEN_KERNEL_HPP

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Registers.hpp>
#include <tensilelite/Codegen/Target.hpp>

#include "code.hpp"
#include "instruction/mem.hpp"

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
  /// Hidden arguments the runtime fills from the launch (code object v5+ value
  /// kinds). The block count is the number of workgroups in that dimension.
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
  int size;
  /// Byte offset from the start of the kernarg segment.
  int offset;
};

/// Ordered kernel arguments laid out in the kernarg segment with natural
/// alignment: each argument starts at a multiple of its size.
class KernelSignature {
public:
  /// Alignment of the kernarg segment, and the granule its size is rounded to.
  static constexpr int segmentAlignment = 8;

  /// Appends a global pointer argument and returns its offset, which the
  /// kernel body loads it from with s_load. Throws std::invalid_argument if
  /// name is not an identifier or is already used.
  int addPointer(std::string_view name);
  /// Appends a by-value scalar of size 4 or 8 bytes and returns its offset.
  /// Throws std::invalid_argument for other sizes and for bad names, as
  /// addPointer does.
  int addScalar(std::string_view name, int size);
  /// Appends a hidden argument, which the runtime fills, and returns its
  /// offset. Block counts take 4 bytes; group sizes take 2 bytes as LLVM emits
  /// them, or 4; with a 4-byte slot only the low 16 bits are defined. Throws
  /// std::invalid_argument if kind is not hidden, size is not allowed for it,
  /// or kind was already added.
  int addHidden(KernelArgumentKind kind, int size);

  const std::vector<KernelArgument> &arguments() const { return args; }
  /// End of the last argument rounded up to segmentAlignment; 0 without
  /// arguments.
  int segmentSize() const;

private:
  int addExplicit(std::string_view name, KernelArgumentKind kind, int size);
  int append(std::string name, KernelArgumentKind kind, int size);

  std::vector<KernelArgument> args;
};

/// A kernel under construction, emitted as a complete assembly file.
///
/// It owns the body module, the register allocator and an IRBuilder whose
/// insertion point is the body. The builder and the allocator are not movable,
/// so the kernel is built in place and is neither copyable nor movable. Label
/// names are unique only per builder, so the body must be built through this
/// kernel's builder(). rocisa keeps its ISA state per thread, so the builder
/// must be used on the thread that constructed the kernel.
class Kernel {
public:
  /// Creates an empty kernel named name. workgroupSize is the largest
  /// workgroup it may be launched with (max_flat_workgroup_size). Reserves the
  /// ABI registers before anything else is allocated. Throws
  /// std::invalid_argument if name is not an identifier, workgroupSize is not
  /// in [1, 1024], or codeObjectVersion is not v4, v5 or v6.
  Kernel(const Target &target, std::string_view name, int workgroupSize,
         CodeObjectVersion codeObjectVersion = CodeObjectVersion::v6);
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;

  const Target &target() const { return kernelTarget; }
  std::string_view name() const { return kernelName; }
  int workgroupSize() const { return maxWorkgroupSize; }
  CodeObjectVersion codeObjectVersion() const { return version; }
  KernelSignature &signature() { return kernelSignature; }
  const KernelSignature &signature() const { return kernelSignature; }
  IRBuilder &builder() { return irBuilder; }

  /// Kernarg segment pointer in s[0:1], the only user SGPRs enabled.
  const RegisterRange &kernargSegmentPtr() const { return kernargPtr; }
  /// Workgroup id x in s2, the first system SGPR after the user SGPRs.
  const RegisterRange &workgroupIdX() const { return workgroupIdXReg; }
  /// Enables the workgroup ids y and z, which the hardware places in the
  /// system SGPRs right after workgroup id x in x, y, z order (s3 and s4 when
  /// both are enabled). Call it before allocating SGPRs so those indices are
  /// free. Throws std::invalid_argument if neither is requested or a register
  /// is in use, and std::logic_error if called twice.
  void enableWorkgroupIds(bool y, bool z);
  bool hasWorkgroupIdY() const { return workgroupIdYReg.has_value(); }
  bool hasWorkgroupIdZ() const { return workgroupIdZReg.has_value(); }
  /// Workgroup id y; throws std::logic_error unless enabled.
  const RegisterRange &workgroupIdY() const;
  /// Workgroup id z; throws std::logic_error unless enabled.
  const RegisterRange &workgroupIdZ() const;
  /// Workitem id in v0. On gfx90a and later the ids are packed: x is bits
  /// [9:0] (y [19:10] and z [29:20] are not enabled), so mask with 0x3ff
  /// before using it as the x id.
  const RegisterRange &workitemId() const { return workitemIdReg; }

  /// Sets the LDS bytes the kernel allocates (group_segment_fixed_size); 0 by
  /// default. Throws std::invalid_argument unless 0 <= bytes <=
  /// target().maxLdsSize().
  void setLdsSize(int bytes);
  int ldsSize() const { return ldsBytes; }

  /// Emits an s_load of dst.count dwords (1, 2 or 4) from the kernarg segment
  /// at byte offset, such as one returned by the signature. The caller emits
  /// the wait. Throws std::invalid_argument if dst is not SGPRs aligned to its
  /// count, or the load is misaligned or outside the segment.
  std::shared_ptr<rocisa::SMemLoadInstruction> loadArgument(RegisterRef dst,
                                                            int offset);

  /// Returns the complete assembly file: target and code object version
  /// directives, the kernel symbol and body, the kernel descriptor and the
  /// .amdgpu_metadata. Throws std::runtime_error if the registers used exceed
  /// the target's limits.
  std::string toAssembly() const;

private:
  // Ranges are declared after the allocator so they are released first, and
  // the body before the builder that points into it.
  Target kernelTarget;
  std::string kernelName;
  int maxWorkgroupSize;
  int ldsBytes = 0;
  bool workgroupIdsEnabled = false;
  CodeObjectVersion version;
  KernelSignature kernelSignature;
  RegisterAllocator registers;
  rocisa::Module bodyModule;
  IRBuilder irBuilder;
  RegisterRange kernargPtr;
  RegisterRange workgroupIdXReg;
  RegisterRange workitemIdReg;
  std::optional<RegisterRange> workgroupIdYReg;
  std::optional<RegisterRange> workgroupIdZReg;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_KERNEL_HPP
