# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end
#
# The assembler reads a leading-zero literal as octal (010 == 8) and accepts
# 0b binary. Both must survive a no-pass round trip with their value intact:
# 010 used to be re-printed as decimal 10, and 0b101 was split into "0" plus an
# identifier and lost.

# CHECK-LABEL: region_begin:
# CHECK-NEXT:  s_mov_b32 s0, 010
# CHECK-NEXT:  s_mov_b32 s1, 0b101
# CHECK-NEXT:  s_mov_b32 s2, 0x10
# CHECK-NEXT:  s_mov_b32 s3, -7
# CHECK-LABEL: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_mov_b32 s0, 010
s_mov_b32 s1, 0b101
s_mov_b32 s2, 0x10
s_mov_b32 s3, -7
region_end:
s_endpgm
