# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --emit-asm 2>/dev/null
#
# Symbolic matrix scale formats are kept as written on a no-pass raw-assembly
# round trip. Numeric forms remain numeric for rocisa parity.

# CHECK:      v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[16:23], v[0:7], v[24:25], v[26:27] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
# CHECK-NEXT: v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[16:23], v[0:7], v[24:25], v[26:27] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:2 matrix_b_scale_fmt:2

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[16:23], v[0:7], v[24:25], v[26:27] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[16:23], v[0:7], v[24:25], v[26:27] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:2 matrix_b_scale_fmt:2
s_endpgm
