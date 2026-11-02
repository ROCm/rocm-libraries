# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm
#
# tensor_load_to_lds carries the gfx1250 cache-policy fields th / scope / nv.
# The raw-asm parser maps them onto TENSORModifiers, so the
# line stays a TENSOR instruction that the waitcnt and scheduling passes can see,
# and it is re-emitted in the LLVM printer order (th, scope, nv) whatever order
# the input used: the second line below proves the instruction was modelled
# rather than passed through as TEXTBLOCK. A token TENSORModifiers does not model
# (offset:0) keeps its line verbatim as TEXTBLOCK.
#
# CHECK: tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_NT_HT scope:SCOPE_DEV nv
# CHECK: tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_NT scope:SCOPE_SE
# CHECK: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_LOAD_HT
# CHECK: tensor_load_to_lds s[0:3], s[4:11] nv
# CHECK: tensor_load_to_lds s[0:3], s[4:11] offset:0
# CHECK: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_NT_HT scope:SCOPE_DEV nv
tensor_load_to_lds s[0:3], s[4:11] scope:SCOPE_SE th:TH_LOAD_NT
tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_LOAD_HT
tensor_load_to_lds s[0:3], s[4:11] nv
tensor_load_to_lds s[0:3], s[4:11] offset:0
s_endpgm
