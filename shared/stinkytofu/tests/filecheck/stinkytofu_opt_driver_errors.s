# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --NoSuchPass 2>&1; echo "exit=$?"; %stinkytofu-opt --arch gfx1250 %s 2>/dev/null | head -n 1
#
# An unknown pass name is an error (it used to be skipped with exit 0, so a
# typo silently ran nothing). The banner goes to stderr: stdout starts with
# the assembly.

# CHECK:      Error: Unknown pass 'NoSuchPass'
# CHECK:      exit=1
# CHECK-NEXT: .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
v_add_f32 v1, v2, v3
s_endpgm
