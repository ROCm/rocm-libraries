# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --Gfx1250HazardPass --from-label region_begin --to-label region_end
#
# Hazard checks read immediates through the shared integer decoder, so hex
# spellings count: `s_sleep 0x8000` is a forever sleep (XNACK replay drain
# before it), and `s_wait_loadcnt 0x0` is a full translation drain (no extra
# s_wait_xcnt before the source-clobbering flat load after it).

# CHECK-LABEL: region_begin:
# CHECK:       global_load_b32 v1, v2, s[0:1]
# CHECK-NEXT:  s_wait_xcnt 0
# CHECK-NEXT:  s_sleep 0x8000
# CHECK:       flat_load_b32 v9, v[0:1]
# CHECK-NEXT:  s_wait_loadcnt 0x0
# CHECK-NEXT:  flat_load_b32 v0, v[2:3]
# CHECK-LABEL: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
global_load_b32 v1, v2, s[0:1]
s_sleep 0x8000
flat_load_b32 v9, v[0:1]
s_wait_loadcnt 0x0
flat_load_b32 v0, v[2:3]
region_end:
s_endpgm
