/* HIP fat binary layout fragment -- see RPP_HIP_FATBIN_LAYOUT in the top-level CMakeLists.
 *
 * By default the linker places .hip_fatbin between .gcc_except_table and .data. The fatbin
 * holds one copy of device code per --offload-arch, so it dominates librpp.so (83% of a
 * Release build) and grows with the GPU_TARGETS list. C++ exception tables encode their
 * typeinfo references as DW_EH_PE_pcrel|sdata4 -- 4-byte PC-relative displacements, capped
 * at +/-2GB by the x86-64 small code model. Once the fatbin pushes .data more than 2GB past
 * .gcc_except_table those relocations overflow and the link fails with
 *
 *     ld.lld: error: ...: relocation R_X86_64_PC32 out of range: ... references section '.data'
 *
 * Moving .hip_fatbin past .bss puts every relocatable section on the near side of it, so the
 * distances that matter stay small no matter how large the fatbin gets. Device code is
 * addressed by the HIP runtime via symbol lookup, not by PC-relative code references, so it
 * has no such range constraint itself.
 *
 * This is an INSERT fragment: it augments the linker's built-in script rather than replacing
 * it, so all default section placement is preserved. Supported by both ld.lld and GNU ld.
 */
SECTIONS
{
  .hip_fatbin : { *(.hip_fatbin) }
} INSERT AFTER .bss;
