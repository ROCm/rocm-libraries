// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal, hand-written gfx1250 kernel BODY with ONE deliberate
// out-of-bounds buffer access, used by tests/AsanReproTest.cpp to
// demonstrate InsertAsanCheckPass end to end.
//
// This is the RAW kernel body -- what a Tensile-generated kernel's code
// looks like BEFORE the ASan pass runs. It reserves the registers
// InsertAsanCheckPass expects (sgprAsanReportBuf/sgprAsanTmp/vgprAsanTmp)
// via `.set`, but never uses them itself -- only the pass emits
// instructions that do.
//
// Deliberately split from kernel_descriptor.s (the .amdhsa_kernel /
// .amdgpu_metadata block): stinkytofu-opt's raw-.s parser is lexical, not
// YAML-aware, and tries to parse anything in the file as instructions --
// including outside the --from-label/--to-label region -- so a YAML
// metadata block's `-` list bullets get misread as mnemonics. It also
// always synthesizes its own (empty/placeholder) kernel descriptor for
// whatever function it recognized in the input, which would collide with a
// real one placed in the same file. See tests/CMakeLists.txt for how the
// two files get combined: run this one through
//
//   stinkytofu-opt --arch gfx1250 kernel_body.s \
//       --from-label region_start --to-label region_end \
//       --InsertAsanCheckPass --emit-asm
//
// strip stinkytofu-opt's own auto-generated descriptor block (uniquely
// identified by its `.section .rodata,#alloc` opener, vs. kernel_descriptor.s's
// plain `.rodata`), then concatenate kernel_descriptor.s onto the result
// before assembling with amdclang++.
//
// Kernel ABI (matches AsanReproTest.cpp's KernArgs struct and
// kernel_descriptor.s's .args: list):
//   offset 0: A_ptr             (8 bytes, global_buffer) -- heap-allocated,
//                                ASan-poisoned host buffer, host-registered
//                                so the GPU can address it.
//   offset 8: AsanReportBuf_ptr (8 bytes, global_buffer) -- 8-byte device
//                                buffer InsertAsanCheckPass writes the
//                                failing PC into on a shadow-poison hit.
//
// The kernel builds a raw/typeless gfx1250 buffer descriptor (SRD) for A_ptr
// and issues a single buffer_load_b32 at byte offset 64 -- one byte past the
// small host buffer the test allocates, landing squarely in real ASan's
// redzone. sgprSrdA is the tracked SRD name the pass looks for (see
// kTrackedSrdSymbols in InsertAsanCheckPass.cpp); any of sgprSrdA/B/C/D works
// identically, this repro only needs one.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl asan_repro_kernel
.p2align 8
.type asan_repro_kernel,@function
asan_repro_kernel:
region_start:
  // Fixed register plan for this repro (a real Tensile kernel reserves the
  // same three Asan* symbols via KernelWriter.py/KernelWriterAssembly.py --
  // see numSgprAsanReportBuf/numSgprAsanTmp/numVgprAsanTmp and the doc
  // comments on them).
  .set sgprKernArgAddress, 0   // s[0:1], HW-provided kernarg segment pointer
  .set sgprA, 4                // s[4:5]: A_ptr, loaded from kernarg offset 0
  .set sgprAsanReportBuf, 6    // s[6:7]: AsanReportBuf_ptr, kernarg offset 8
  .set sgprSrdA, 8             // s[8:11]: raw buffer descriptor built from A
  .set sgprAsanTmp, 12         // s[12:13]: pass scratch (PC capture)
  .set vgprAsanTmp, 8          // v[8:15]: pass scratch, must be even-aligned

  s_load_b64 s[sgprA:sgprA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0
  s_load_b64 s[sgprAsanReportBuf:sgprAsanReportBuf+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 8
  s_wait_kmcnt 0

  // Raw/typeless gfx1250 SRD: base ptr, unbounded byte limit (hardware SRD
  // bounds never trap on this arch anyway -- see the plan doc), Srd127_96=0
  // (gfx1250's SrdUpperValue125X zero-initializes every subfield; confirmed
  // against a real compiled kernel dump, see KernelWriterAssembly.py:1744-1746).
  s_mov_b32 s[sgprSrdA], s[sgprA]
  s_mov_b32 s[sgprSrdA+1], s[sgprA+1]
  s_mov_b32 s[sgprSrdA+2], 0xffffffff
  s_mov_b32 s[sgprSrdA+3], 0x0

  // THE illegal access: byte offset 64 is one past the small host buffer
  // AsanReproTest.cpp allocates -- inside real ASan's redzone. The offset
  // MUST be a real vaddr VGPR (offen), not folded into the `offset:`
  // immediate -- InsertAsanCheckPass computes VA = SrdBase + vaddr and does
  // not account for the immediate-offset modifier (see the "soffset != 0
  // not folded" v1 limitation noted in the pass), so an immediate-only
  // access would make the check add garbage instead of the real offset.
  v_mov_b32 v1, 64
  buffer_load_b32 v0, v1, s[sgprSrdA:sgprSrdA+3], 0 offen
  s_wait_loadcnt 0

  s_endpgm
region_end:
