# RUN: %stinkytofu-opt --arch gfx1250 %s 2>&1; %stinkytofu-opt --arch gfx1250 %s --whole-function --RaiseVgprMsbPass --InsertVgprMsbPass 2>&1; echo "exit=$?"
#
# A hand-written kernel's header and metadata round-trip as written. The
# header used to be regenerated from the fields stinkytofu models: it added
# .amdhsa_float_denorm_mode_32/_16_64 3 that the input left at the assembler
# default, dropped the header comments, and emitted a second, empty
# .amdgpu_metadata block. The metadata after the body lost its YAML
# indentation, and its "---" line counted as an instruction kept as text, so
# --whole-function passes refused every such kernel.

# CHECK:      .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
# CHECK-NOT:  Begin Kernel
# CHECK:      .amdhsa_kernel hand_kernel
# CHECK-NEXT:   .amdhsa_user_sgpr_kernarg_segment_ptr 1
# CHECK-NEXT:   .amdhsa_next_free_vgpr 8
# CHECK-NEXT:   .amdhsa_next_free_sgpr 8
# CHECK-NEXT:   .amdhsa_wavefront_size32 1
# CHECK-NEXT: .end_amdhsa_kernel
# CHECK-NEXT: // tile notes stay with the header
# CHECK-NEXT: .text
# CHECK-NEXT: hand_kernel:
# CHECK-NOT:  .amdgpu_metadata
# CHECK:      s_endpgm
# CHECK-NEXT: .amdgpu_metadata
# CHECK-NEXT: ---
# CHECK-NEXT: amdhsa.version: [1, 1]
# CHECK-NEXT: amdhsa.kernels:
# CHECK-NEXT:   - .name: hand_kernel
# CHECK-NEXT:     .symbol: hand_kernel.kd
# CHECK-NEXT:     .wavefront_size: 32
# CHECK-NEXT: ...
# CHECK-NEXT: .end_amdgpu_metadata
# CHECK-NOT:  .amdgpu_metadata

# The passes run over the whole function; the metadata does not stop them.
# CHECK-NOT:  kept as text
# CHECK:      hand_kernel:
# CHECK:      s_endpgm
# CHECK-NEXT: .amdgpu_metadata
# CHECK:      exit=0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.protected hand_kernel
.globl hand_kernel
.p2align 8
.type hand_kernel,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel hand_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 8
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
// tile notes stay with the header
.text
hand_kernel:
  v_mov_b32 v1, v2
  s_endpgm
.amdgpu_metadata
---
amdhsa.version: [1, 1]
amdhsa.kernels:
  - .name: hand_kernel
    .symbol: hand_kernel.kd
    .wavefront_size: 32
...
.end_amdgpu_metadata
