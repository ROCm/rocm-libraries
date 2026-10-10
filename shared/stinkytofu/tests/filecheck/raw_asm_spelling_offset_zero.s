# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --emit-asm 2>/dev/null
#
# GLOBAL/FLAT/SMEM drop a zero offset on emit; an `offset:0` the input wrote is
# kept as written (byte identity on a round trip with no passes). Before: the
# global, flat and SMEM forms lost it. Without one, nothing is added.

# CHECK:      global_store_b64 v1, v[2:3], s[0:1] offset:0{{$}}
# CHECK-NEXT: global_load_b32 v1, v2, s[0:1] offset:0x0{{$}}
# CHECK-NEXT: global_load_b32 v1, v2, s[0:1]{{$}}
# CHECK-NEXT: global_load_b32 v1, v2, s[0:1] offset:16{{$}}
# CHECK-NEXT: flat_load_b32 v1, v[2:3] offset:0{{$}}
# CHECK-NEXT: s_load_b32 s4, s[0:1], s5 offset:0{{$}}
# CHECK-NEXT: ds_load_b32 v1, v2 offset:0{{$}}

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
global_store_b64 v1, v[2:3], s[0:1] offset:0
global_load_b32 v1, v2, s[0:1] offset:0x0
global_load_b32 v1, v2, s[0:1]
global_load_b32 v1, v2, s[0:1] offset:16
flat_load_b32 v1, v[2:3] offset:0
s_load_b32 s4, s[0:1], s5 offset:0
ds_load_b32 v1, v2 offset:0
s_endpgm
