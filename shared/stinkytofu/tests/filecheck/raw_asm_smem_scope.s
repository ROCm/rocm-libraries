# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm
#
# The raw-asm parser keeps the cache scope of an s_load. A glc next to a scope is not
# printed back.
#
# CHECK: s_load_b64 s[0:1], s[2:3], 0x0 scope:SCOPE_SYS
# CHECK-NOT: glc
# CHECK: s_load_b32 s4, s[2:3], 0x8 scope:SCOPE_DEV
# CHECK-NOT: glc
# CHECK: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

s_load_b64 s[0:1], s[2:3], 0x0 scope:SCOPE_SYS
s_load_b32 s4, s[2:3], 0x8 glc scope:SCOPE_DEV
s_endpgm
