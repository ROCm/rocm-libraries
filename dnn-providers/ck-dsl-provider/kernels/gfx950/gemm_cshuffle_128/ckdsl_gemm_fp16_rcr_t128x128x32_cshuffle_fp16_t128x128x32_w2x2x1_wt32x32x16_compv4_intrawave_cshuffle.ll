target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle = internal unnamed_addr addrspace(3) global [128 x [32 x half]] poison, align 4
@B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle = internal unnamed_addr addrspace(3) global [128 x [32 x half]] poison, align 4
@C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle = internal unnamed_addr addrspace(3) global [128 x [128 x half]] poison, align 4

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare void @llvm.amdgcn.s.barrier()
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)

define amdgpu_kernel void @ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle(ptr addrspace(1) noalias readonly nocapture align 16 %A, ptr addrspace(1) noalias readonly nocapture align 16 %B, ptr addrspace(1) noalias writeonly nocapture align 16 %C, i32 %M, i32 %N, i32 %K) #0 {
entry:
  %tid7 = call i32 @llvm.amdgcn.workitem.id.x()
  %div8 = sdiv i32 %tid7, 64
  %div9 = sdiv i32 %div8, 2
  %mod10 = srem i32 %div8, 2
  %mod11 = srem i32 %tid7, 64
  %bid15 = call i32 @llvm.amdgcn.workgroup.id.y()
  %mul16 = mul nsw i32 %bid15, 128
  %ufm17 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %mul16)
  %sgpr18 = call i32 asm "", "=s,0"(i32 %ufm17)
  %bid19 = call i32 @llvm.amdgcn.workgroup.id.x()
  %mul20 = mul nsw i32 %bid19, 128
  %ufm21 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %mul20)
  %sgpr22 = call i32 asm "", "=s,0"(i32 %ufm21)
  %cz1625 = select i1 true, <16 x float> zeroinitializer, <16 x float> zeroinitializer
  br label %for.header.1
for.header.1:
  %k0 = phi i32 [ 0, %entry ], [ %iv.next.for.header.1, %for.latch.3 ]
  %acc_m0_n0 = phi <16 x float> [ %cz1625, %entry ], [ %acc_m0_n0.next.for.header.1, %for.latch.3 ]
  %acc_m0_n1 = phi <16 x float> [ %cz1625, %entry ], [ %acc_m0_n1.next.for.header.1, %for.latch.3 ]
  %acc_m1_n0 = phi <16 x float> [ %cz1625, %entry ], [ %acc_m1_n0.next.for.header.1, %for.latch.3 ]
  %acc_m1_n1 = phi <16 x float> [ %cz1625, %entry ], [ %acc_m1_n1.next.for.header.1, %for.latch.3 ]
  %cmp.1 = icmp slt i32 %k0, %K
  br i1 %cmp.1, label %for.body.2, label %for.exit.4
for.body.2:
  %mul38 = mul nsw i32 0, 256
  %add39 = add nsw i32 %mul38, %tid7
  %div40 = sdiv i32 %add39, 4
  %mod41 = srem i32 %add39, 4
  %mul42 = mul nsw i32 %mod41, 8
  %add44 = add nsw i32 0, 0
  %add45 = add nsw i32 %sgpr18, %div40
  %add46 = add nsw i32 %k0, %mul42
  %mul47 = mul nsw i32 %add45, %K
  %add48 = add nsw i32 %add44, %mul47
  %add49 = add nsw i32 %add48, %add46
  %gep.2 = getelementptr inbounds half, ptr addrspace(1) %A, i32 %add49
  %gv850 = load <8 x half>, ptr addrspace(1) %gep.2, align 16
  %add51 = add nsw i32 0, %div40
  %add52 = add nsw i32 0, %mul42
  %gep.3 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add51, i32 %add52
  store <8 x half> %gv850, ptr addrspace(3) %gep.3, align 16
  %mul54 = mul nsw i32 1, 256
  %add55 = add nsw i32 %mul54, %tid7
  %div56 = sdiv i32 %add55, 4
  %mod57 = srem i32 %add55, 4
  %mul58 = mul nsw i32 %mod57, 8
  %add60 = add nsw i32 0, 0
  %add61 = add nsw i32 %sgpr18, %div56
  %add62 = add nsw i32 %k0, %mul58
  %mul63 = mul nsw i32 %add61, %K
  %add64 = add nsw i32 %add60, %mul63
  %add65 = add nsw i32 %add64, %add62
  %gep.4 = getelementptr inbounds half, ptr addrspace(1) %A, i32 %add65
  %gv866 = load <8 x half>, ptr addrspace(1) %gep.4, align 16
  %add67 = add nsw i32 0, %div56
  %add68 = add nsw i32 0, %mul58
  %gep.5 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add67, i32 %add68
  store <8 x half> %gv866, ptr addrspace(3) %gep.5, align 16
  %mul70 = mul nsw i32 0, 256
  %add71 = add nsw i32 %mul70, %tid7
  %div72 = sdiv i32 %add71, 4
  %mod73 = srem i32 %add71, 4
  %mul74 = mul nsw i32 %mod73, 8
  %add76 = add nsw i32 0, 0
  %add77 = add nsw i32 %sgpr22, %div72
  %add78 = add nsw i32 %k0, %mul74
  %mul79 = mul nsw i32 %add77, %K
  %add80 = add nsw i32 %add76, %mul79
  %add81 = add nsw i32 %add80, %add78
  %gep.6 = getelementptr inbounds half, ptr addrspace(1) %B, i32 %add81
  %gv882 = load <8 x half>, ptr addrspace(1) %gep.6, align 16
  %add83 = add nsw i32 0, %div72
  %add84 = add nsw i32 0, %mul74
  %gep.7 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add83, i32 %add84
  store <8 x half> %gv882, ptr addrspace(3) %gep.7, align 16
  %mul86 = mul nsw i32 1, 256
  %add87 = add nsw i32 %mul86, %tid7
  %div88 = sdiv i32 %add87, 4
  %mod89 = srem i32 %add87, 4
  %mul90 = mul nsw i32 %mod89, 8
  %add92 = add nsw i32 0, 0
  %add93 = add nsw i32 %sgpr22, %div88
  %add94 = add nsw i32 %k0, %mul90
  %mul95 = mul nsw i32 %add93, %K
  %add96 = add nsw i32 %add92, %mul95
  %add97 = add nsw i32 %add96, %add94
  %gep.8 = getelementptr inbounds half, ptr addrspace(1) %B, i32 %add97
  %gv898 = load <8 x half>, ptr addrspace(1) %gep.8, align 16
  %add99 = add nsw i32 0, %div88
  %add100 = add nsw i32 0, %mul90
  %gep.9 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add99, i32 %add100
  store <8 x half> %gv898, ptr addrspace(3) %gep.9, align 16
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  %mod102 = srem i32 %mod11, 32
  %div104 = sdiv i32 %mod11, 32
  %mod106 = srem i32 %mod11, 32
  %mul108 = mul nsw i32 %div9, 64
  %mul110 = mul nsw i32 %mod10, 64
  %mul112 = mul nsw i32 %div104, 8
  %add114 = add nsw i32 %mul112, 0
  %add116 = add nsw i32 0, %mod102
  %add117 = add nsw i32 %mul108, %add116
  %smem.base.10 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add117, i32 %add114
  %av8118 = load <8 x half>, ptr addrspace(3) %smem.base.10, align 16
  %add120 = add nsw i32 %mul112, 0
  %add122 = add nsw i32 32, %mod102
  %add123 = add nsw i32 %mul108, %add122
  %smem.base.11 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add123, i32 %add120
  %av8124 = load <8 x half>, ptr addrspace(3) %smem.base.11, align 16
  %add126 = add nsw i32 %mul112, 0
  %add128 = add nsw i32 0, %mod106
  %add129 = add nsw i32 %mul110, %add128
  %smem.base.12 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add129, i32 %add126
  %av8130 = load <8 x half>, ptr addrspace(3) %smem.base.12, align 16
  %add132 = add nsw i32 %mul112, 0
  %add134 = add nsw i32 32, %mod106
  %add135 = add nsw i32 %mul110, %add134
  %smem.base.13 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add135, i32 %add132
  %av8136 = load <8 x half>, ptr addrspace(3) %smem.base.13, align 16
  %acc137 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8118, <8 x half> %av8130, <16 x float> %acc_m0_n0, i32 0, i32 0, i32 0)
  %acc138 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8118, <8 x half> %av8136, <16 x float> %acc_m0_n1, i32 0, i32 0, i32 0)
  %acc139 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8124, <8 x half> %av8130, <16 x float> %acc_m1_n0, i32 0, i32 0, i32 0)
  %acc140 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8124, <8 x half> %av8136, <16 x float> %acc_m1_n1, i32 0, i32 0, i32 0)
  %add142 = add nsw i32 %mul112, 16
  %add144 = add nsw i32 0, %mod102
  %add145 = add nsw i32 %mul108, %add144
  %smem.base.14 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add145, i32 %add142
  %av8146 = load <8 x half>, ptr addrspace(3) %smem.base.14, align 16
  %add148 = add nsw i32 %mul112, 16
  %add150 = add nsw i32 32, %mod102
  %add151 = add nsw i32 %mul108, %add150
  %smem.base.15 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add151, i32 %add148
  %av8152 = load <8 x half>, ptr addrspace(3) %smem.base.15, align 16
  %add154 = add nsw i32 %mul112, 16
  %add156 = add nsw i32 0, %mod106
  %add157 = add nsw i32 %mul110, %add156
  %smem.base.16 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add157, i32 %add154
  %av8158 = load <8 x half>, ptr addrspace(3) %smem.base.16, align 16
  %add160 = add nsw i32 %mul112, 16
  %add162 = add nsw i32 32, %mod106
  %add163 = add nsw i32 %mul110, %add162
  %smem.base.17 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add163, i32 %add160
  %av8164 = load <8 x half>, ptr addrspace(3) %smem.base.17, align 16
  %acc165 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8146, <8 x half> %av8158, <16 x float> %acc137, i32 0, i32 0, i32 0)
  %acc166 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8146, <8 x half> %av8164, <16 x float> %acc138, i32 0, i32 0, i32 0)
  %acc167 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8152, <8 x half> %av8158, <16 x float> %acc139, i32 0, i32 0, i32 0)
  %acc168 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8152, <8 x half> %av8164, <16 x float> %acc140, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 4, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  br label %for.latch.3
for.latch.3:
  %iv.next.for.header.1 = add nsw i32 %k0, 32
  %acc_m0_n0.next.for.header.1 = bitcast <16 x float> %acc165 to <16 x float>
  %acc_m0_n1.next.for.header.1 = bitcast <16 x float> %acc166 to <16 x float>
  %acc_m1_n0.next.for.header.1 = bitcast <16 x float> %acc167 to <16 x float>
  %acc_m1_n1.next.for.header.1 = bitcast <16 x float> %acc168 to <16 x float>
  br label %for.header.1
for.exit.4:
  %for29 = bitcast <16 x float> %acc_m0_n0 to <16 x float>
  %for30 = bitcast <16 x float> %acc_m0_n1 to <16 x float>
  %for31 = bitcast <16 x float> %acc_m1_n0 to <16 x float>
  %for32 = bitcast <16 x float> %acc_m1_n1 to <16 x float>
  %mul171 = mul nsw i32 %div9, 64
  %mul173 = mul nsw i32 %mod10, 64
  %mod175 = srem i32 %mod11, 32
  %div176 = sdiv i32 %mod11, 32
  %add180 = add nsw i32 0, 0
  %mul182 = mul nsw i32 %div176, 4
  %add183 = add nsw i32 %add180, %mul182
  %mul185 = mul nsw i32 0, 8
  %add186 = add nsw i32 %add183, %mul185
  %add188 = add nsw i32 0, %mod175
  %add192 = add nsw i32 0, 1
  %mul194 = mul nsw i32 %div176, 4
  %add195 = add nsw i32 %add192, %mul194
  %mul197 = mul nsw i32 0, 8
  %add198 = add nsw i32 %add195, %mul197
  %add200 = add nsw i32 0, %mod175
  %add204 = add nsw i32 0, 2
  %mul206 = mul nsw i32 %div176, 4
  %add207 = add nsw i32 %add204, %mul206
  %mul209 = mul nsw i32 0, 8
  %add210 = add nsw i32 %add207, %mul209
  %add212 = add nsw i32 0, %mod175
  %add216 = add nsw i32 0, 3
  %mul218 = mul nsw i32 %div176, 4
  %add219 = add nsw i32 %add216, %mul218
  %mul221 = mul nsw i32 0, 8
  %add222 = add nsw i32 %add219, %mul221
  %add224 = add nsw i32 0, %mod175
  %add228 = add nsw i32 0, 0
  %mul230 = mul nsw i32 %div176, 4
  %add231 = add nsw i32 %add228, %mul230
  %mul233 = mul nsw i32 1, 8
  %add234 = add nsw i32 %add231, %mul233
  %add236 = add nsw i32 0, %mod175
  %add240 = add nsw i32 0, 1
  %mul242 = mul nsw i32 %div176, 4
  %add243 = add nsw i32 %add240, %mul242
  %mul245 = mul nsw i32 1, 8
  %add246 = add nsw i32 %add243, %mul245
  %add248 = add nsw i32 0, %mod175
  %add252 = add nsw i32 0, 2
  %mul254 = mul nsw i32 %div176, 4
  %add255 = add nsw i32 %add252, %mul254
  %mul257 = mul nsw i32 1, 8
  %add258 = add nsw i32 %add255, %mul257
  %add260 = add nsw i32 0, %mod175
  %add264 = add nsw i32 0, 3
  %mul266 = mul nsw i32 %div176, 4
  %add267 = add nsw i32 %add264, %mul266
  %mul269 = mul nsw i32 1, 8
  %add270 = add nsw i32 %add267, %mul269
  %add272 = add nsw i32 0, %mod175
  %add276 = add nsw i32 0, 0
  %mul278 = mul nsw i32 %div176, 4
  %add279 = add nsw i32 %add276, %mul278
  %mul281 = mul nsw i32 2, 8
  %add282 = add nsw i32 %add279, %mul281
  %add284 = add nsw i32 0, %mod175
  %add288 = add nsw i32 0, 1
  %mul290 = mul nsw i32 %div176, 4
  %add291 = add nsw i32 %add288, %mul290
  %mul293 = mul nsw i32 2, 8
  %add294 = add nsw i32 %add291, %mul293
  %add296 = add nsw i32 0, %mod175
  %add300 = add nsw i32 0, 2
  %mul302 = mul nsw i32 %div176, 4
  %add303 = add nsw i32 %add300, %mul302
  %mul305 = mul nsw i32 2, 8
  %add306 = add nsw i32 %add303, %mul305
  %add308 = add nsw i32 0, %mod175
  %add312 = add nsw i32 0, 3
  %mul314 = mul nsw i32 %div176, 4
  %add315 = add nsw i32 %add312, %mul314
  %mul317 = mul nsw i32 2, 8
  %add318 = add nsw i32 %add315, %mul317
  %add320 = add nsw i32 0, %mod175
  %add324 = add nsw i32 0, 0
  %mul326 = mul nsw i32 %div176, 4
  %add327 = add nsw i32 %add324, %mul326
  %mul329 = mul nsw i32 3, 8
  %add330 = add nsw i32 %add327, %mul329
  %add332 = add nsw i32 0, %mod175
  %add336 = add nsw i32 0, 1
  %mul338 = mul nsw i32 %div176, 4
  %add339 = add nsw i32 %add336, %mul338
  %mul341 = mul nsw i32 3, 8
  %add342 = add nsw i32 %add339, %mul341
  %add344 = add nsw i32 0, %mod175
  %add348 = add nsw i32 0, 2
  %mul350 = mul nsw i32 %div176, 4
  %add351 = add nsw i32 %add348, %mul350
  %mul353 = mul nsw i32 3, 8
  %add354 = add nsw i32 %add351, %mul353
  %add356 = add nsw i32 0, %mod175
  %add360 = add nsw i32 0, 3
  %mul362 = mul nsw i32 %div176, 4
  %add363 = add nsw i32 %add360, %mul362
  %mul365 = mul nsw i32 3, 8
  %add366 = add nsw i32 %add363, %mul365
  %add368 = add nsw i32 0, %mod175
  %add370 = add nsw i32 %mul171, 0
  %vh16371 = fptrunc <16 x float> %for29 to <16 x half>
  %add373 = add nsw i32 %mul173, 0
  %add374 = add nsw i32 %add373, %add368
  %add375 = add nsw i32 %add370, %add186
  %e376 = extractelement <16 x half> %vh16371, i32 0
  %gep.18 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add375, i32 %add374
  store half %e376, ptr addrspace(3) %gep.18, align 2
  %add377 = add nsw i32 %add370, %add198
  %e378 = extractelement <16 x half> %vh16371, i32 1
  %gep.19 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add377, i32 %add374
  store half %e378, ptr addrspace(3) %gep.19, align 2
  %add379 = add nsw i32 %add370, %add210
  %e380 = extractelement <16 x half> %vh16371, i32 2
  %gep.20 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add379, i32 %add374
  store half %e380, ptr addrspace(3) %gep.20, align 2
  %add381 = add nsw i32 %add370, %add222
  %e382 = extractelement <16 x half> %vh16371, i32 3
  %gep.21 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add381, i32 %add374
  store half %e382, ptr addrspace(3) %gep.21, align 2
  %add383 = add nsw i32 %add370, %add234
  %e384 = extractelement <16 x half> %vh16371, i32 4
  %gep.22 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add383, i32 %add374
  store half %e384, ptr addrspace(3) %gep.22, align 2
  %add385 = add nsw i32 %add370, %add246
  %e386 = extractelement <16 x half> %vh16371, i32 5
  %gep.23 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add385, i32 %add374
  store half %e386, ptr addrspace(3) %gep.23, align 2
  %add387 = add nsw i32 %add370, %add258
  %e388 = extractelement <16 x half> %vh16371, i32 6
  %gep.24 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add387, i32 %add374
  store half %e388, ptr addrspace(3) %gep.24, align 2
  %add389 = add nsw i32 %add370, %add270
  %e390 = extractelement <16 x half> %vh16371, i32 7
  %gep.25 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add389, i32 %add374
  store half %e390, ptr addrspace(3) %gep.25, align 2
  %add391 = add nsw i32 %add370, %add282
  %e392 = extractelement <16 x half> %vh16371, i32 8
  %gep.26 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add391, i32 %add374
  store half %e392, ptr addrspace(3) %gep.26, align 2
  %add393 = add nsw i32 %add370, %add294
  %e394 = extractelement <16 x half> %vh16371, i32 9
  %gep.27 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add393, i32 %add374
  store half %e394, ptr addrspace(3) %gep.27, align 2
  %add395 = add nsw i32 %add370, %add306
  %e396 = extractelement <16 x half> %vh16371, i32 10
  %gep.28 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add395, i32 %add374
  store half %e396, ptr addrspace(3) %gep.28, align 2
  %add397 = add nsw i32 %add370, %add318
  %e398 = extractelement <16 x half> %vh16371, i32 11
  %gep.29 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add397, i32 %add374
  store half %e398, ptr addrspace(3) %gep.29, align 2
  %add399 = add nsw i32 %add370, %add330
  %e400 = extractelement <16 x half> %vh16371, i32 12
  %gep.30 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add399, i32 %add374
  store half %e400, ptr addrspace(3) %gep.30, align 2
  %add401 = add nsw i32 %add370, %add342
  %e402 = extractelement <16 x half> %vh16371, i32 13
  %gep.31 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add401, i32 %add374
  store half %e402, ptr addrspace(3) %gep.31, align 2
  %add403 = add nsw i32 %add370, %add354
  %e404 = extractelement <16 x half> %vh16371, i32 14
  %gep.32 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add403, i32 %add374
  store half %e404, ptr addrspace(3) %gep.32, align 2
  %add405 = add nsw i32 %add370, %add366
  %e406 = extractelement <16 x half> %vh16371, i32 15
  %gep.33 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add405, i32 %add374
  store half %e406, ptr addrspace(3) %gep.33, align 2
  %vh16407 = fptrunc <16 x float> %for30 to <16 x half>
  %add409 = add nsw i32 %mul173, 32
  %add410 = add nsw i32 %add409, %add368
  %add411 = add nsw i32 %add370, %add186
  %e412 = extractelement <16 x half> %vh16407, i32 0
  %gep.34 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add411, i32 %add410
  store half %e412, ptr addrspace(3) %gep.34, align 2
  %add413 = add nsw i32 %add370, %add198
  %e414 = extractelement <16 x half> %vh16407, i32 1
  %gep.35 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add413, i32 %add410
  store half %e414, ptr addrspace(3) %gep.35, align 2
  %add415 = add nsw i32 %add370, %add210
  %e416 = extractelement <16 x half> %vh16407, i32 2
  %gep.36 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add415, i32 %add410
  store half %e416, ptr addrspace(3) %gep.36, align 2
  %add417 = add nsw i32 %add370, %add222
  %e418 = extractelement <16 x half> %vh16407, i32 3
  %gep.37 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add417, i32 %add410
  store half %e418, ptr addrspace(3) %gep.37, align 2
  %add419 = add nsw i32 %add370, %add234
  %e420 = extractelement <16 x half> %vh16407, i32 4
  %gep.38 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add419, i32 %add410
  store half %e420, ptr addrspace(3) %gep.38, align 2
  %add421 = add nsw i32 %add370, %add246
  %e422 = extractelement <16 x half> %vh16407, i32 5
  %gep.39 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add421, i32 %add410
  store half %e422, ptr addrspace(3) %gep.39, align 2
  %add423 = add nsw i32 %add370, %add258
  %e424 = extractelement <16 x half> %vh16407, i32 6
  %gep.40 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add423, i32 %add410
  store half %e424, ptr addrspace(3) %gep.40, align 2
  %add425 = add nsw i32 %add370, %add270
  %e426 = extractelement <16 x half> %vh16407, i32 7
  %gep.41 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add425, i32 %add410
  store half %e426, ptr addrspace(3) %gep.41, align 2
  %add427 = add nsw i32 %add370, %add282
  %e428 = extractelement <16 x half> %vh16407, i32 8
  %gep.42 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add427, i32 %add410
  store half %e428, ptr addrspace(3) %gep.42, align 2
  %add429 = add nsw i32 %add370, %add294
  %e430 = extractelement <16 x half> %vh16407, i32 9
  %gep.43 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add429, i32 %add410
  store half %e430, ptr addrspace(3) %gep.43, align 2
  %add431 = add nsw i32 %add370, %add306
  %e432 = extractelement <16 x half> %vh16407, i32 10
  %gep.44 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add431, i32 %add410
  store half %e432, ptr addrspace(3) %gep.44, align 2
  %add433 = add nsw i32 %add370, %add318
  %e434 = extractelement <16 x half> %vh16407, i32 11
  %gep.45 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add433, i32 %add410
  store half %e434, ptr addrspace(3) %gep.45, align 2
  %add435 = add nsw i32 %add370, %add330
  %e436 = extractelement <16 x half> %vh16407, i32 12
  %gep.46 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add435, i32 %add410
  store half %e436, ptr addrspace(3) %gep.46, align 2
  %add437 = add nsw i32 %add370, %add342
  %e438 = extractelement <16 x half> %vh16407, i32 13
  %gep.47 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add437, i32 %add410
  store half %e438, ptr addrspace(3) %gep.47, align 2
  %add439 = add nsw i32 %add370, %add354
  %e440 = extractelement <16 x half> %vh16407, i32 14
  %gep.48 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add439, i32 %add410
  store half %e440, ptr addrspace(3) %gep.48, align 2
  %add441 = add nsw i32 %add370, %add366
  %e442 = extractelement <16 x half> %vh16407, i32 15
  %gep.49 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add441, i32 %add410
  store half %e442, ptr addrspace(3) %gep.49, align 2
  %add444 = add nsw i32 %mul171, 32
  %vh16445 = fptrunc <16 x float> %for31 to <16 x half>
  %add447 = add nsw i32 %mul173, 0
  %add448 = add nsw i32 %add447, %add368
  %add449 = add nsw i32 %add444, %add186
  %e450 = extractelement <16 x half> %vh16445, i32 0
  %gep.50 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add449, i32 %add448
  store half %e450, ptr addrspace(3) %gep.50, align 2
  %add451 = add nsw i32 %add444, %add198
  %e452 = extractelement <16 x half> %vh16445, i32 1
  %gep.51 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add451, i32 %add448
  store half %e452, ptr addrspace(3) %gep.51, align 2
  %add453 = add nsw i32 %add444, %add210
  %e454 = extractelement <16 x half> %vh16445, i32 2
  %gep.52 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add453, i32 %add448
  store half %e454, ptr addrspace(3) %gep.52, align 2
  %add455 = add nsw i32 %add444, %add222
  %e456 = extractelement <16 x half> %vh16445, i32 3
  %gep.53 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add455, i32 %add448
  store half %e456, ptr addrspace(3) %gep.53, align 2
  %add457 = add nsw i32 %add444, %add234
  %e458 = extractelement <16 x half> %vh16445, i32 4
  %gep.54 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add457, i32 %add448
  store half %e458, ptr addrspace(3) %gep.54, align 2
  %add459 = add nsw i32 %add444, %add246
  %e460 = extractelement <16 x half> %vh16445, i32 5
  %gep.55 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add459, i32 %add448
  store half %e460, ptr addrspace(3) %gep.55, align 2
  %add461 = add nsw i32 %add444, %add258
  %e462 = extractelement <16 x half> %vh16445, i32 6
  %gep.56 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add461, i32 %add448
  store half %e462, ptr addrspace(3) %gep.56, align 2
  %add463 = add nsw i32 %add444, %add270
  %e464 = extractelement <16 x half> %vh16445, i32 7
  %gep.57 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add463, i32 %add448
  store half %e464, ptr addrspace(3) %gep.57, align 2
  %add465 = add nsw i32 %add444, %add282
  %e466 = extractelement <16 x half> %vh16445, i32 8
  %gep.58 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add465, i32 %add448
  store half %e466, ptr addrspace(3) %gep.58, align 2
  %add467 = add nsw i32 %add444, %add294
  %e468 = extractelement <16 x half> %vh16445, i32 9
  %gep.59 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add467, i32 %add448
  store half %e468, ptr addrspace(3) %gep.59, align 2
  %add469 = add nsw i32 %add444, %add306
  %e470 = extractelement <16 x half> %vh16445, i32 10
  %gep.60 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add469, i32 %add448
  store half %e470, ptr addrspace(3) %gep.60, align 2
  %add471 = add nsw i32 %add444, %add318
  %e472 = extractelement <16 x half> %vh16445, i32 11
  %gep.61 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add471, i32 %add448
  store half %e472, ptr addrspace(3) %gep.61, align 2
  %add473 = add nsw i32 %add444, %add330
  %e474 = extractelement <16 x half> %vh16445, i32 12
  %gep.62 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add473, i32 %add448
  store half %e474, ptr addrspace(3) %gep.62, align 2
  %add475 = add nsw i32 %add444, %add342
  %e476 = extractelement <16 x half> %vh16445, i32 13
  %gep.63 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add475, i32 %add448
  store half %e476, ptr addrspace(3) %gep.63, align 2
  %add477 = add nsw i32 %add444, %add354
  %e478 = extractelement <16 x half> %vh16445, i32 14
  %gep.64 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add477, i32 %add448
  store half %e478, ptr addrspace(3) %gep.64, align 2
  %add479 = add nsw i32 %add444, %add366
  %e480 = extractelement <16 x half> %vh16445, i32 15
  %gep.65 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add479, i32 %add448
  store half %e480, ptr addrspace(3) %gep.65, align 2
  %vh16481 = fptrunc <16 x float> %for32 to <16 x half>
  %add483 = add nsw i32 %mul173, 32
  %add484 = add nsw i32 %add483, %add368
  %add485 = add nsw i32 %add444, %add186
  %e486 = extractelement <16 x half> %vh16481, i32 0
  %gep.66 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add485, i32 %add484
  store half %e486, ptr addrspace(3) %gep.66, align 2
  %add487 = add nsw i32 %add444, %add198
  %e488 = extractelement <16 x half> %vh16481, i32 1
  %gep.67 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add487, i32 %add484
  store half %e488, ptr addrspace(3) %gep.67, align 2
  %add489 = add nsw i32 %add444, %add210
  %e490 = extractelement <16 x half> %vh16481, i32 2
  %gep.68 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add489, i32 %add484
  store half %e490, ptr addrspace(3) %gep.68, align 2
  %add491 = add nsw i32 %add444, %add222
  %e492 = extractelement <16 x half> %vh16481, i32 3
  %gep.69 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add491, i32 %add484
  store half %e492, ptr addrspace(3) %gep.69, align 2
  %add493 = add nsw i32 %add444, %add234
  %e494 = extractelement <16 x half> %vh16481, i32 4
  %gep.70 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add493, i32 %add484
  store half %e494, ptr addrspace(3) %gep.70, align 2
  %add495 = add nsw i32 %add444, %add246
  %e496 = extractelement <16 x half> %vh16481, i32 5
  %gep.71 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add495, i32 %add484
  store half %e496, ptr addrspace(3) %gep.71, align 2
  %add497 = add nsw i32 %add444, %add258
  %e498 = extractelement <16 x half> %vh16481, i32 6
  %gep.72 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add497, i32 %add484
  store half %e498, ptr addrspace(3) %gep.72, align 2
  %add499 = add nsw i32 %add444, %add270
  %e500 = extractelement <16 x half> %vh16481, i32 7
  %gep.73 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add499, i32 %add484
  store half %e500, ptr addrspace(3) %gep.73, align 2
  %add501 = add nsw i32 %add444, %add282
  %e502 = extractelement <16 x half> %vh16481, i32 8
  %gep.74 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add501, i32 %add484
  store half %e502, ptr addrspace(3) %gep.74, align 2
  %add503 = add nsw i32 %add444, %add294
  %e504 = extractelement <16 x half> %vh16481, i32 9
  %gep.75 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add503, i32 %add484
  store half %e504, ptr addrspace(3) %gep.75, align 2
  %add505 = add nsw i32 %add444, %add306
  %e506 = extractelement <16 x half> %vh16481, i32 10
  %gep.76 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add505, i32 %add484
  store half %e506, ptr addrspace(3) %gep.76, align 2
  %add507 = add nsw i32 %add444, %add318
  %e508 = extractelement <16 x half> %vh16481, i32 11
  %gep.77 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add507, i32 %add484
  store half %e508, ptr addrspace(3) %gep.77, align 2
  %add509 = add nsw i32 %add444, %add330
  %e510 = extractelement <16 x half> %vh16481, i32 12
  %gep.78 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add509, i32 %add484
  store half %e510, ptr addrspace(3) %gep.78, align 2
  %add511 = add nsw i32 %add444, %add342
  %e512 = extractelement <16 x half> %vh16481, i32 13
  %gep.79 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add511, i32 %add484
  store half %e512, ptr addrspace(3) %gep.79, align 2
  %add513 = add nsw i32 %add444, %add354
  %e514 = extractelement <16 x half> %vh16481, i32 14
  %gep.80 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add513, i32 %add484
  store half %e514, ptr addrspace(3) %gep.80, align 2
  %add515 = add nsw i32 %add444, %add366
  %e516 = extractelement <16 x half> %vh16481, i32 15
  %gep.81 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %add515, i32 %add484
  store half %e516, ptr addrspace(3) %gep.81, align 2
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  %tid517 = call i32 @llvm.amdgcn.workitem.id.x()
  %mul521 = mul nsw i32 0, 256
  %add522 = add nsw i32 %mul521, %tid517
  %div523 = sdiv i32 %add522, 16
  %mod524 = srem i32 %add522, 16
  %mul526 = mul nsw i32 %mod524, 8
  %add527 = add nsw i32 %sgpr18, %div523
  %add528 = add nsw i32 %sgpr22, %mul526
  %mul529 = mul nsw i32 %add527, %N
  %add530 = add nsw i32 %mul529, %add528
  %add531 = add nsw i32 0, %add530
  %smem.base.82 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div523, i32 %mul526
  %av8532 = load <8 x half>, ptr addrspace(3) %smem.base.82, align 16
  %gep.83 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add531
  store <8 x half> %av8532, ptr addrspace(1) %gep.83, align 16
  %mul534 = mul nsw i32 1, 256
  %add535 = add nsw i32 %mul534, %tid517
  %div536 = sdiv i32 %add535, 16
  %mod537 = srem i32 %add535, 16
  %mul539 = mul nsw i32 %mod537, 8
  %add540 = add nsw i32 %sgpr18, %div536
  %add541 = add nsw i32 %sgpr22, %mul539
  %mul542 = mul nsw i32 %add540, %N
  %add543 = add nsw i32 %mul542, %add541
  %add544 = add nsw i32 0, %add543
  %smem.base.84 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div536, i32 %mul539
  %av8545 = load <8 x half>, ptr addrspace(3) %smem.base.84, align 16
  %gep.85 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add544
  store <8 x half> %av8545, ptr addrspace(1) %gep.85, align 16
  %mul547 = mul nsw i32 2, 256
  %add548 = add nsw i32 %mul547, %tid517
  %div549 = sdiv i32 %add548, 16
  %mod550 = srem i32 %add548, 16
  %mul552 = mul nsw i32 %mod550, 8
  %add553 = add nsw i32 %sgpr18, %div549
  %add554 = add nsw i32 %sgpr22, %mul552
  %mul555 = mul nsw i32 %add553, %N
  %add556 = add nsw i32 %mul555, %add554
  %add557 = add nsw i32 0, %add556
  %smem.base.86 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div549, i32 %mul552
  %av8558 = load <8 x half>, ptr addrspace(3) %smem.base.86, align 16
  %gep.87 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add557
  store <8 x half> %av8558, ptr addrspace(1) %gep.87, align 16
  %mul560 = mul nsw i32 3, 256
  %add561 = add nsw i32 %mul560, %tid517
  %div562 = sdiv i32 %add561, 16
  %mod563 = srem i32 %add561, 16
  %mul565 = mul nsw i32 %mod563, 8
  %add566 = add nsw i32 %sgpr18, %div562
  %add567 = add nsw i32 %sgpr22, %mul565
  %mul568 = mul nsw i32 %add566, %N
  %add569 = add nsw i32 %mul568, %add567
  %add570 = add nsw i32 0, %add569
  %smem.base.88 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div562, i32 %mul565
  %av8571 = load <8 x half>, ptr addrspace(3) %smem.base.88, align 16
  %gep.89 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add570
  store <8 x half> %av8571, ptr addrspace(1) %gep.89, align 16
  %mul573 = mul nsw i32 4, 256
  %add574 = add nsw i32 %mul573, %tid517
  %div575 = sdiv i32 %add574, 16
  %mod576 = srem i32 %add574, 16
  %mul578 = mul nsw i32 %mod576, 8
  %add579 = add nsw i32 %sgpr18, %div575
  %add580 = add nsw i32 %sgpr22, %mul578
  %mul581 = mul nsw i32 %add579, %N
  %add582 = add nsw i32 %mul581, %add580
  %add583 = add nsw i32 0, %add582
  %smem.base.90 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div575, i32 %mul578
  %av8584 = load <8 x half>, ptr addrspace(3) %smem.base.90, align 16
  %gep.91 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add583
  store <8 x half> %av8584, ptr addrspace(1) %gep.91, align 16
  %mul586 = mul nsw i32 5, 256
  %add587 = add nsw i32 %mul586, %tid517
  %div588 = sdiv i32 %add587, 16
  %mod589 = srem i32 %add587, 16
  %mul591 = mul nsw i32 %mod589, 8
  %add592 = add nsw i32 %sgpr18, %div588
  %add593 = add nsw i32 %sgpr22, %mul591
  %mul594 = mul nsw i32 %add592, %N
  %add595 = add nsw i32 %mul594, %add593
  %add596 = add nsw i32 0, %add595
  %smem.base.92 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div588, i32 %mul591
  %av8597 = load <8 x half>, ptr addrspace(3) %smem.base.92, align 16
  %gep.93 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add596
  store <8 x half> %av8597, ptr addrspace(1) %gep.93, align 16
  %mul599 = mul nsw i32 6, 256
  %add600 = add nsw i32 %mul599, %tid517
  %div601 = sdiv i32 %add600, 16
  %mod602 = srem i32 %add600, 16
  %mul604 = mul nsw i32 %mod602, 8
  %add605 = add nsw i32 %sgpr18, %div601
  %add606 = add nsw i32 %sgpr22, %mul604
  %mul607 = mul nsw i32 %add605, %N
  %add608 = add nsw i32 %mul607, %add606
  %add609 = add nsw i32 0, %add608
  %smem.base.94 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div601, i32 %mul604
  %av8610 = load <8 x half>, ptr addrspace(3) %smem.base.94, align 16
  %gep.95 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add609
  store <8 x half> %av8610, ptr addrspace(1) %gep.95, align 16
  %mul612 = mul nsw i32 7, 256
  %add613 = add nsw i32 %mul612, %tid517
  %div614 = sdiv i32 %add613, 16
  %mod615 = srem i32 %add613, 16
  %mul617 = mul nsw i32 %mod615, 8
  %add618 = add nsw i32 %sgpr18, %div614
  %add619 = add nsw i32 %sgpr22, %mul617
  %mul620 = mul nsw i32 %add618, %N
  %add621 = add nsw i32 %mul620, %add619
  %add622 = add nsw i32 0, %add621
  %smem.base.96 = getelementptr inbounds [128 x [128 x half]], ptr addrspace(3) @C_smem169.ckdsl_gemm_fp16_rcr_t128x128x32_cshuffle_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle, i32 0, i32 %div614, i32 %mul617
  %av8623 = load <8 x half>, ptr addrspace(3) %smem.base.96, align 16
  %gep.97 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add622
  store <8 x half> %av8623, ptr addrspace(1) %gep.97, align 16
 ret void
}

attributes #0 = { "uniform-work-group-size"="true" "amdgpu-flat-work-group-size"="64,256" norecurse nounwind }
