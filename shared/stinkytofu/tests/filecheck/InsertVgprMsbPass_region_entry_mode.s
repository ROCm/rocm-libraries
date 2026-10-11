# RUN: %stinkytofu-opt --arch gfx1250 %s --vgpr-msb-mode=msb16 --CFGBuilderPass --RaiseVgprMsbPass --InsertVgprMsbPass --from-label label_a --to-label label_end 2>/dev/null; echo "--- whole-function"; %stinkytofu-opt --arch gfx1250 %s --vgpr-msb-mode=msb16 --CFGBuilderPass --RaiseVgprMsbPass --InsertVgprMsbPass --whole-function 2>/dev/null
#
# Code before the region enters label_a (fall-through) and label_b (branch) in
# mode 0x41, which the input's setters after both labels declare in their
# previous-mode byte (0x4145). The region used to assume mode 0 at every label:
# setters after them said previous mode 0 (69) and it set 0 before falling into
# label_b (17664), so label_b's edges arrived in 0x41 and 0. A region label now
# takes the input's entry mode, and every edge the region makes into it carries
# that mode. The whole-function run owns every edge and keeps mode 0.

# CHECK:      s_cbranch_scc1 label_b
# CHECK-NEXT: label_a:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 16709{{$}}
# CHECK-NEXT: v_add_f32 v[261-256], v[262-256], v[263-256]
# CHECK-NEXT: s_set_vgpr_msb 17729{{$}}
# CHECK-NEXT: label_b:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 16709{{$}}
# CHECK-NEXT: v_add_f32 v[264-256], v[265-256], v[266-256]
# CHECK-NEXT: s_set_vgpr_msb 17729{{$}}
# CHECK-NEXT: label_end:
# CHECK-NEXT: s_set_vgpr_msb 0x4100
# CHECK-NEXT: s_endpgm

# CHECK:      --- whole-function
# CHECK:      s_set_vgpr_msb 16640{{$}}
# CHECK-NEXT: s_cbranch_scc1 label_b
# CHECK-NEXT: label_a:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 69{{$}}
# CHECK:      s_set_vgpr_msb 17664{{$}}
# CHECK-NEXT: label_b:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 69{{$}}
# CHECK:      s_set_vgpr_msb 17664{{$}}
# CHECK-NEXT: label_end:
# CHECK-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
s_set_vgpr_msb 0x41
v_add_f32 v1, v2, v3
s_cbranch_scc1 label_b
label_a:
s_nop 0
s_set_vgpr_msb 0x4145
v_add_f32 v5, v6, v7
s_set_vgpr_msb 0x4541
label_b:
s_nop 0
s_set_vgpr_msb 0x4145
v_add_f32 v8, v9, v10
s_set_vgpr_msb 0x4541
label_end:
s_set_vgpr_msb 0x4100
s_endpgm
