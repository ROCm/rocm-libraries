target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@A_smem23.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default = internal unnamed_addr addrspace(3) global [64 x [32 x half]] poison, align 4
@B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default = internal unnamed_addr addrspace(3) global [128 x [32 x half]] poison, align 4

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare void @llvm.amdgcn.s.barrier()
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)

define amdgpu_kernel void @ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default(ptr addrspace(1) noalias readonly nocapture align 16 %A, ptr addrspace(1) noalias readonly nocapture align 16 %B, ptr addrspace(1) noalias writeonly nocapture align 16 %C, i32 %M, i32 %N, i32 %K) #0 {
entry:
  %tid7 = call i32 @llvm.amdgcn.workitem.id.x()
  %div8 = sdiv i32 %tid7, 64
  %div9 = sdiv i32 %div8, 2
  %mod10 = srem i32 %div8, 2
  %mod11 = srem i32 %tid7, 64
  %bid15 = call i32 @llvm.amdgcn.workgroup.id.y()
  %mul16 = mul nsw i32 %bid15, 64
  %ufm17 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %mul16)
  %sgpr18 = call i32 asm "", "=s,0"(i32 %ufm17)
  %bid19 = call i32 @llvm.amdgcn.workgroup.id.x()
  %mul20 = mul nsw i32 %bid19, 128
  %ufm21 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %mul20)
  %sgpr22 = call i32 asm "", "=s,0"(i32 %ufm21)
  %cz425 = select i1 true, <4 x float> zeroinitializer, <4 x float> zeroinitializer
  br label %for.header.1
for.header.1:
  %k0 = phi i32 [ 0, %entry ], [ %iv.next.for.header.1, %for.latch.3 ]
  %acc_m0_n0 = phi <4 x float> [ %cz425, %entry ], [ %acc_m0_n0.next.for.header.1, %for.latch.3 ]
  %acc_m0_n1 = phi <4 x float> [ %cz425, %entry ], [ %acc_m0_n1.next.for.header.1, %for.latch.3 ]
  %acc_m0_n2 = phi <4 x float> [ %cz425, %entry ], [ %acc_m0_n2.next.for.header.1, %for.latch.3 ]
  %acc_m0_n3 = phi <4 x float> [ %cz425, %entry ], [ %acc_m0_n3.next.for.header.1, %for.latch.3 ]
  %acc_m1_n0 = phi <4 x float> [ %cz425, %entry ], [ %acc_m1_n0.next.for.header.1, %for.latch.3 ]
  %acc_m1_n1 = phi <4 x float> [ %cz425, %entry ], [ %acc_m1_n1.next.for.header.1, %for.latch.3 ]
  %acc_m1_n2 = phi <4 x float> [ %cz425, %entry ], [ %acc_m1_n2.next.for.header.1, %for.latch.3 ]
  %acc_m1_n3 = phi <4 x float> [ %cz425, %entry ], [ %acc_m1_n3.next.for.header.1, %for.latch.3 ]
  %cmp.1 = icmp slt i32 %k0, %K
  br i1 %cmp.1, label %for.body.2, label %for.exit.4
for.body.2:
  %mul42 = mul nsw i32 0, 256
  %add43 = add nsw i32 %mul42, %tid7
  %div44 = sdiv i32 %add43, 4
  %mod45 = srem i32 %add43, 4
  %mul46 = mul nsw i32 %mod45, 8
  %add48 = add nsw i32 0, 0
  %add49 = add nsw i32 %sgpr18, %div44
  %add50 = add nsw i32 %k0, %mul46
  %mul51 = mul nsw i32 %add49, %K
  %add52 = add nsw i32 %add48, %mul51
  %add53 = add nsw i32 %add52, %add50
  %gep.2 = getelementptr inbounds half, ptr addrspace(1) %A, i32 %add53
  %gv854 = load <8 x half>, ptr addrspace(1) %gep.2, align 16
  %add55 = add nsw i32 0, %div44
  %add56 = add nsw i32 0, %mul46
  %gep.3 = getelementptr inbounds [64 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add55, i32 %add56
  store <8 x half> %gv854, ptr addrspace(3) %gep.3, align 16
  %mul58 = mul nsw i32 0, 256
  %add59 = add nsw i32 %mul58, %tid7
  %div60 = sdiv i32 %add59, 4
  %mod61 = srem i32 %add59, 4
  %mul62 = mul nsw i32 %mod61, 8
  %add64 = add nsw i32 0, 0
  %add65 = add nsw i32 %sgpr22, %div60
  %add66 = add nsw i32 %k0, %mul62
  %mul67 = mul nsw i32 %add65, %K
  %add68 = add nsw i32 %add64, %mul67
  %add69 = add nsw i32 %add68, %add66
  %gep.4 = getelementptr inbounds half, ptr addrspace(1) %B, i32 %add69
  %gv870 = load <8 x half>, ptr addrspace(1) %gep.4, align 16
  %add71 = add nsw i32 0, %div60
  %add72 = add nsw i32 0, %mul62
  %gep.5 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add71, i32 %add72
  store <8 x half> %gv870, ptr addrspace(3) %gep.5, align 16
  %mul74 = mul nsw i32 1, 256
  %add75 = add nsw i32 %mul74, %tid7
  %div76 = sdiv i32 %add75, 4
  %mod77 = srem i32 %add75, 4
  %mul78 = mul nsw i32 %mod77, 8
  %add80 = add nsw i32 0, 0
  %add81 = add nsw i32 %sgpr22, %div76
  %add82 = add nsw i32 %k0, %mul78
  %mul83 = mul nsw i32 %add81, %K
  %add84 = add nsw i32 %add80, %mul83
  %add85 = add nsw i32 %add84, %add82
  %gep.6 = getelementptr inbounds half, ptr addrspace(1) %B, i32 %add85
  %gv886 = load <8 x half>, ptr addrspace(1) %gep.6, align 16
  %add87 = add nsw i32 0, %div76
  %add88 = add nsw i32 0, %mul78
  %gep.7 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add87, i32 %add88
  store <8 x half> %gv886, ptr addrspace(3) %gep.7, align 16
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  %mod90 = srem i32 %mod11, 16
  %div92 = sdiv i32 %mod11, 16
  %mod94 = srem i32 %mod11, 16
  %mul96 = mul nsw i32 %div9, 32
  %mul98 = mul nsw i32 %mod10, 64
  %mul100 = mul nsw i32 %div92, 4
  %add102 = add nsw i32 %mul100, 0
  %add104 = add nsw i32 0, %mod90
  %add105 = add nsw i32 %mul96, %add104
  %smem.base.8 = getelementptr inbounds [64 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add105, i32 %add102
  %smem.ep.9 = getelementptr inbounds half, ptr addrspace(3) %smem.base.8, i32 0
  %smem.ld.10 = load half, ptr addrspace(3) %smem.ep.9, align 2
  %smem.ep.11 = getelementptr inbounds half, ptr addrspace(3) %smem.base.8, i32 1
  %smem.ld.12 = load half, ptr addrspace(3) %smem.ep.11, align 2
  %smem.ep.13 = getelementptr inbounds half, ptr addrspace(3) %smem.base.8, i32 2
  %smem.ld.14 = load half, ptr addrspace(3) %smem.ep.13, align 2
  %smem.ep.15 = getelementptr inbounds half, ptr addrspace(3) %smem.base.8, i32 3
  %smem.ld.16 = load half, ptr addrspace(3) %smem.ep.15, align 2
  %vec.17 = insertelement <4 x half> undef, half %smem.ld.10, i32 0
  %vec.18 = insertelement <4 x half> %vec.17, half %smem.ld.12, i32 1
  %vec.19 = insertelement <4 x half> %vec.18, half %smem.ld.14, i32 2
  %a106 = insertelement <4 x half> %vec.19, half %smem.ld.16, i32 3
  %add108 = add nsw i32 %mul100, 0
  %add110 = add nsw i32 16, %mod90
  %add111 = add nsw i32 %mul96, %add110
  %smem.base.20 = getelementptr inbounds [64 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add111, i32 %add108
  %smem.ep.21 = getelementptr inbounds half, ptr addrspace(3) %smem.base.20, i32 0
  %smem.ld.22 = load half, ptr addrspace(3) %smem.ep.21, align 2
  %smem.ep.23 = getelementptr inbounds half, ptr addrspace(3) %smem.base.20, i32 1
  %smem.ld.24 = load half, ptr addrspace(3) %smem.ep.23, align 2
  %smem.ep.25 = getelementptr inbounds half, ptr addrspace(3) %smem.base.20, i32 2
  %smem.ld.26 = load half, ptr addrspace(3) %smem.ep.25, align 2
  %smem.ep.27 = getelementptr inbounds half, ptr addrspace(3) %smem.base.20, i32 3
  %smem.ld.28 = load half, ptr addrspace(3) %smem.ep.27, align 2
  %vec.29 = insertelement <4 x half> undef, half %smem.ld.22, i32 0
  %vec.30 = insertelement <4 x half> %vec.29, half %smem.ld.24, i32 1
  %vec.31 = insertelement <4 x half> %vec.30, half %smem.ld.26, i32 2
  %a112 = insertelement <4 x half> %vec.31, half %smem.ld.28, i32 3
  %add114 = add nsw i32 %mul100, 0
  %add116 = add nsw i32 0, %mod94
  %add117 = add nsw i32 %mul98, %add116
  %smem.base.32 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add117, i32 %add114
  %smem.ep.33 = getelementptr inbounds half, ptr addrspace(3) %smem.base.32, i32 0
  %smem.ld.34 = load half, ptr addrspace(3) %smem.ep.33, align 2
  %smem.ep.35 = getelementptr inbounds half, ptr addrspace(3) %smem.base.32, i32 1
  %smem.ld.36 = load half, ptr addrspace(3) %smem.ep.35, align 2
  %smem.ep.37 = getelementptr inbounds half, ptr addrspace(3) %smem.base.32, i32 2
  %smem.ld.38 = load half, ptr addrspace(3) %smem.ep.37, align 2
  %smem.ep.39 = getelementptr inbounds half, ptr addrspace(3) %smem.base.32, i32 3
  %smem.ld.40 = load half, ptr addrspace(3) %smem.ep.39, align 2
  %vec.41 = insertelement <4 x half> undef, half %smem.ld.34, i32 0
  %vec.42 = insertelement <4 x half> %vec.41, half %smem.ld.36, i32 1
  %vec.43 = insertelement <4 x half> %vec.42, half %smem.ld.38, i32 2
  %a118 = insertelement <4 x half> %vec.43, half %smem.ld.40, i32 3
  %add120 = add nsw i32 %mul100, 0
  %add122 = add nsw i32 16, %mod94
  %add123 = add nsw i32 %mul98, %add122
  %smem.base.44 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add123, i32 %add120
  %smem.ep.45 = getelementptr inbounds half, ptr addrspace(3) %smem.base.44, i32 0
  %smem.ld.46 = load half, ptr addrspace(3) %smem.ep.45, align 2
  %smem.ep.47 = getelementptr inbounds half, ptr addrspace(3) %smem.base.44, i32 1
  %smem.ld.48 = load half, ptr addrspace(3) %smem.ep.47, align 2
  %smem.ep.49 = getelementptr inbounds half, ptr addrspace(3) %smem.base.44, i32 2
  %smem.ld.50 = load half, ptr addrspace(3) %smem.ep.49, align 2
  %smem.ep.51 = getelementptr inbounds half, ptr addrspace(3) %smem.base.44, i32 3
  %smem.ld.52 = load half, ptr addrspace(3) %smem.ep.51, align 2
  %vec.53 = insertelement <4 x half> undef, half %smem.ld.46, i32 0
  %vec.54 = insertelement <4 x half> %vec.53, half %smem.ld.48, i32 1
  %vec.55 = insertelement <4 x half> %vec.54, half %smem.ld.50, i32 2
  %a124 = insertelement <4 x half> %vec.55, half %smem.ld.52, i32 3
  %add126 = add nsw i32 %mul100, 0
  %add128 = add nsw i32 32, %mod94
  %add129 = add nsw i32 %mul98, %add128
  %smem.base.56 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add129, i32 %add126
  %smem.ep.57 = getelementptr inbounds half, ptr addrspace(3) %smem.base.56, i32 0
  %smem.ld.58 = load half, ptr addrspace(3) %smem.ep.57, align 2
  %smem.ep.59 = getelementptr inbounds half, ptr addrspace(3) %smem.base.56, i32 1
  %smem.ld.60 = load half, ptr addrspace(3) %smem.ep.59, align 2
  %smem.ep.61 = getelementptr inbounds half, ptr addrspace(3) %smem.base.56, i32 2
  %smem.ld.62 = load half, ptr addrspace(3) %smem.ep.61, align 2
  %smem.ep.63 = getelementptr inbounds half, ptr addrspace(3) %smem.base.56, i32 3
  %smem.ld.64 = load half, ptr addrspace(3) %smem.ep.63, align 2
  %vec.65 = insertelement <4 x half> undef, half %smem.ld.58, i32 0
  %vec.66 = insertelement <4 x half> %vec.65, half %smem.ld.60, i32 1
  %vec.67 = insertelement <4 x half> %vec.66, half %smem.ld.62, i32 2
  %a130 = insertelement <4 x half> %vec.67, half %smem.ld.64, i32 3
  %add132 = add nsw i32 %mul100, 0
  %add134 = add nsw i32 48, %mod94
  %add135 = add nsw i32 %mul98, %add134
  %smem.base.68 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add135, i32 %add132
  %smem.ep.69 = getelementptr inbounds half, ptr addrspace(3) %smem.base.68, i32 0
  %smem.ld.70 = load half, ptr addrspace(3) %smem.ep.69, align 2
  %smem.ep.71 = getelementptr inbounds half, ptr addrspace(3) %smem.base.68, i32 1
  %smem.ld.72 = load half, ptr addrspace(3) %smem.ep.71, align 2
  %smem.ep.73 = getelementptr inbounds half, ptr addrspace(3) %smem.base.68, i32 2
  %smem.ld.74 = load half, ptr addrspace(3) %smem.ep.73, align 2
  %smem.ep.75 = getelementptr inbounds half, ptr addrspace(3) %smem.base.68, i32 3
  %smem.ld.76 = load half, ptr addrspace(3) %smem.ep.75, align 2
  %vec.77 = insertelement <4 x half> undef, half %smem.ld.70, i32 0
  %vec.78 = insertelement <4 x half> %vec.77, half %smem.ld.72, i32 1
  %vec.79 = insertelement <4 x half> %vec.78, half %smem.ld.74, i32 2
  %a136 = insertelement <4 x half> %vec.79, half %smem.ld.76, i32 3
  %acc137 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a106, <4 x half> %a118, <4 x float> %acc_m0_n0, i32 0, i32 0, i32 0)
  %acc138 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a106, <4 x half> %a124, <4 x float> %acc_m0_n1, i32 0, i32 0, i32 0)
  %acc139 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a106, <4 x half> %a130, <4 x float> %acc_m0_n2, i32 0, i32 0, i32 0)
  %acc140 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a106, <4 x half> %a136, <4 x float> %acc_m0_n3, i32 0, i32 0, i32 0)
  %acc141 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a112, <4 x half> %a118, <4 x float> %acc_m1_n0, i32 0, i32 0, i32 0)
  %acc142 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a112, <4 x half> %a124, <4 x float> %acc_m1_n1, i32 0, i32 0, i32 0)
  %acc143 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a112, <4 x half> %a130, <4 x float> %acc_m1_n2, i32 0, i32 0, i32 0)
  %acc144 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a112, <4 x half> %a136, <4 x float> %acc_m1_n3, i32 0, i32 0, i32 0)
  %add146 = add nsw i32 %mul100, 16
  %add148 = add nsw i32 0, %mod90
  %add149 = add nsw i32 %mul96, %add148
  %smem.base.80 = getelementptr inbounds [64 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add149, i32 %add146
  %smem.ep.81 = getelementptr inbounds half, ptr addrspace(3) %smem.base.80, i32 0
  %smem.ld.82 = load half, ptr addrspace(3) %smem.ep.81, align 2
  %smem.ep.83 = getelementptr inbounds half, ptr addrspace(3) %smem.base.80, i32 1
  %smem.ld.84 = load half, ptr addrspace(3) %smem.ep.83, align 2
  %smem.ep.85 = getelementptr inbounds half, ptr addrspace(3) %smem.base.80, i32 2
  %smem.ld.86 = load half, ptr addrspace(3) %smem.ep.85, align 2
  %smem.ep.87 = getelementptr inbounds half, ptr addrspace(3) %smem.base.80, i32 3
  %smem.ld.88 = load half, ptr addrspace(3) %smem.ep.87, align 2
  %vec.89 = insertelement <4 x half> undef, half %smem.ld.82, i32 0
  %vec.90 = insertelement <4 x half> %vec.89, half %smem.ld.84, i32 1
  %vec.91 = insertelement <4 x half> %vec.90, half %smem.ld.86, i32 2
  %a150 = insertelement <4 x half> %vec.91, half %smem.ld.88, i32 3
  %add152 = add nsw i32 %mul100, 16
  %add154 = add nsw i32 16, %mod90
  %add155 = add nsw i32 %mul96, %add154
  %smem.base.92 = getelementptr inbounds [64 x [32 x half]], ptr addrspace(3) @A_smem23.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add155, i32 %add152
  %smem.ep.93 = getelementptr inbounds half, ptr addrspace(3) %smem.base.92, i32 0
  %smem.ld.94 = load half, ptr addrspace(3) %smem.ep.93, align 2
  %smem.ep.95 = getelementptr inbounds half, ptr addrspace(3) %smem.base.92, i32 1
  %smem.ld.96 = load half, ptr addrspace(3) %smem.ep.95, align 2
  %smem.ep.97 = getelementptr inbounds half, ptr addrspace(3) %smem.base.92, i32 2
  %smem.ld.98 = load half, ptr addrspace(3) %smem.ep.97, align 2
  %smem.ep.99 = getelementptr inbounds half, ptr addrspace(3) %smem.base.92, i32 3
  %smem.ld.100 = load half, ptr addrspace(3) %smem.ep.99, align 2
  %vec.101 = insertelement <4 x half> undef, half %smem.ld.94, i32 0
  %vec.102 = insertelement <4 x half> %vec.101, half %smem.ld.96, i32 1
  %vec.103 = insertelement <4 x half> %vec.102, half %smem.ld.98, i32 2
  %a156 = insertelement <4 x half> %vec.103, half %smem.ld.100, i32 3
  %add158 = add nsw i32 %mul100, 16
  %add160 = add nsw i32 0, %mod94
  %add161 = add nsw i32 %mul98, %add160
  %smem.base.104 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add161, i32 %add158
  %smem.ep.105 = getelementptr inbounds half, ptr addrspace(3) %smem.base.104, i32 0
  %smem.ld.106 = load half, ptr addrspace(3) %smem.ep.105, align 2
  %smem.ep.107 = getelementptr inbounds half, ptr addrspace(3) %smem.base.104, i32 1
  %smem.ld.108 = load half, ptr addrspace(3) %smem.ep.107, align 2
  %smem.ep.109 = getelementptr inbounds half, ptr addrspace(3) %smem.base.104, i32 2
  %smem.ld.110 = load half, ptr addrspace(3) %smem.ep.109, align 2
  %smem.ep.111 = getelementptr inbounds half, ptr addrspace(3) %smem.base.104, i32 3
  %smem.ld.112 = load half, ptr addrspace(3) %smem.ep.111, align 2
  %vec.113 = insertelement <4 x half> undef, half %smem.ld.106, i32 0
  %vec.114 = insertelement <4 x half> %vec.113, half %smem.ld.108, i32 1
  %vec.115 = insertelement <4 x half> %vec.114, half %smem.ld.110, i32 2
  %a162 = insertelement <4 x half> %vec.115, half %smem.ld.112, i32 3
  %add164 = add nsw i32 %mul100, 16
  %add166 = add nsw i32 16, %mod94
  %add167 = add nsw i32 %mul98, %add166
  %smem.base.116 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add167, i32 %add164
  %smem.ep.117 = getelementptr inbounds half, ptr addrspace(3) %smem.base.116, i32 0
  %smem.ld.118 = load half, ptr addrspace(3) %smem.ep.117, align 2
  %smem.ep.119 = getelementptr inbounds half, ptr addrspace(3) %smem.base.116, i32 1
  %smem.ld.120 = load half, ptr addrspace(3) %smem.ep.119, align 2
  %smem.ep.121 = getelementptr inbounds half, ptr addrspace(3) %smem.base.116, i32 2
  %smem.ld.122 = load half, ptr addrspace(3) %smem.ep.121, align 2
  %smem.ep.123 = getelementptr inbounds half, ptr addrspace(3) %smem.base.116, i32 3
  %smem.ld.124 = load half, ptr addrspace(3) %smem.ep.123, align 2
  %vec.125 = insertelement <4 x half> undef, half %smem.ld.118, i32 0
  %vec.126 = insertelement <4 x half> %vec.125, half %smem.ld.120, i32 1
  %vec.127 = insertelement <4 x half> %vec.126, half %smem.ld.122, i32 2
  %a168 = insertelement <4 x half> %vec.127, half %smem.ld.124, i32 3
  %add170 = add nsw i32 %mul100, 16
  %add172 = add nsw i32 32, %mod94
  %add173 = add nsw i32 %mul98, %add172
  %smem.base.128 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add173, i32 %add170
  %smem.ep.129 = getelementptr inbounds half, ptr addrspace(3) %smem.base.128, i32 0
  %smem.ld.130 = load half, ptr addrspace(3) %smem.ep.129, align 2
  %smem.ep.131 = getelementptr inbounds half, ptr addrspace(3) %smem.base.128, i32 1
  %smem.ld.132 = load half, ptr addrspace(3) %smem.ep.131, align 2
  %smem.ep.133 = getelementptr inbounds half, ptr addrspace(3) %smem.base.128, i32 2
  %smem.ld.134 = load half, ptr addrspace(3) %smem.ep.133, align 2
  %smem.ep.135 = getelementptr inbounds half, ptr addrspace(3) %smem.base.128, i32 3
  %smem.ld.136 = load half, ptr addrspace(3) %smem.ep.135, align 2
  %vec.137 = insertelement <4 x half> undef, half %smem.ld.130, i32 0
  %vec.138 = insertelement <4 x half> %vec.137, half %smem.ld.132, i32 1
  %vec.139 = insertelement <4 x half> %vec.138, half %smem.ld.134, i32 2
  %a174 = insertelement <4 x half> %vec.139, half %smem.ld.136, i32 3
  %add176 = add nsw i32 %mul100, 16
  %add178 = add nsw i32 48, %mod94
  %add179 = add nsw i32 %mul98, %add178
  %smem.base.140 = getelementptr inbounds [128 x [32 x half]], ptr addrspace(3) @B_smem24.ckdsl_gemm_fp16_rcr_t64x128x32_mem_fp16_t64x128x32_w2x2x1_wt16x16x16_mem_intrawave_default, i32 0, i32 %add179, i32 %add176
  %smem.ep.141 = getelementptr inbounds half, ptr addrspace(3) %smem.base.140, i32 0
  %smem.ld.142 = load half, ptr addrspace(3) %smem.ep.141, align 2
  %smem.ep.143 = getelementptr inbounds half, ptr addrspace(3) %smem.base.140, i32 1
  %smem.ld.144 = load half, ptr addrspace(3) %smem.ep.143, align 2
  %smem.ep.145 = getelementptr inbounds half, ptr addrspace(3) %smem.base.140, i32 2
  %smem.ld.146 = load half, ptr addrspace(3) %smem.ep.145, align 2
  %smem.ep.147 = getelementptr inbounds half, ptr addrspace(3) %smem.base.140, i32 3
  %smem.ld.148 = load half, ptr addrspace(3) %smem.ep.147, align 2
  %vec.149 = insertelement <4 x half> undef, half %smem.ld.142, i32 0
  %vec.150 = insertelement <4 x half> %vec.149, half %smem.ld.144, i32 1
  %vec.151 = insertelement <4 x half> %vec.150, half %smem.ld.146, i32 2
  %a180 = insertelement <4 x half> %vec.151, half %smem.ld.148, i32 3
  %acc181 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a150, <4 x half> %a162, <4 x float> %acc137, i32 0, i32 0, i32 0)
  %acc182 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a150, <4 x half> %a168, <4 x float> %acc138, i32 0, i32 0, i32 0)
  %acc183 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a150, <4 x half> %a174, <4 x float> %acc139, i32 0, i32 0, i32 0)
  %acc184 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a150, <4 x half> %a180, <4 x float> %acc140, i32 0, i32 0, i32 0)
  %acc185 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a156, <4 x half> %a162, <4 x float> %acc141, i32 0, i32 0, i32 0)
  %acc186 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a156, <4 x half> %a168, <4 x float> %acc142, i32 0, i32 0, i32 0)
  %acc187 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a156, <4 x half> %a174, <4 x float> %acc143, i32 0, i32 0, i32 0)
  %acc188 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a156, <4 x half> %a180, <4 x float> %acc144, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  br label %for.latch.3
for.latch.3:
  %iv.next.for.header.1 = add nsw i32 %k0, 32
  %acc_m0_n0.next.for.header.1 = bitcast <4 x float> %acc181 to <4 x float>
  %acc_m0_n1.next.for.header.1 = bitcast <4 x float> %acc182 to <4 x float>
  %acc_m0_n2.next.for.header.1 = bitcast <4 x float> %acc183 to <4 x float>
  %acc_m0_n3.next.for.header.1 = bitcast <4 x float> %acc184 to <4 x float>
  %acc_m1_n0.next.for.header.1 = bitcast <4 x float> %acc185 to <4 x float>
  %acc_m1_n1.next.for.header.1 = bitcast <4 x float> %acc186 to <4 x float>
  %acc_m1_n2.next.for.header.1 = bitcast <4 x float> %acc187 to <4 x float>
  %acc_m1_n3.next.for.header.1 = bitcast <4 x float> %acc188 to <4 x float>
  br label %for.header.1
for.exit.4:
  %for29 = bitcast <4 x float> %acc_m0_n0 to <4 x float>
  %for30 = bitcast <4 x float> %acc_m0_n1 to <4 x float>
  %for31 = bitcast <4 x float> %acc_m0_n2 to <4 x float>
  %for32 = bitcast <4 x float> %acc_m0_n3 to <4 x float>
  %for33 = bitcast <4 x float> %acc_m1_n0 to <4 x float>
  %for34 = bitcast <4 x float> %acc_m1_n1 to <4 x float>
  %for35 = bitcast <4 x float> %acc_m1_n2 to <4 x float>
  %for36 = bitcast <4 x float> %acc_m1_n3 to <4 x float>
  %mul190 = mul nsw i32 %div9, 32
  %mul192 = mul nsw i32 %mod10, 64
  %add193 = add nsw i32 %sgpr18, %mul190
  %add194 = add nsw i32 %sgpr22, %mul192
  %mod196 = srem i32 %mod11, 16
  %div197 = sdiv i32 %mod11, 16
  %add201 = add nsw i32 0, 0
  %mul203 = mul nsw i32 %div197, 4
  %add204 = add nsw i32 %add201, %mul203
  %mul206 = mul nsw i32 0, 16
  %add207 = add nsw i32 %add204, %mul206
  %add209 = add nsw i32 0, %mod196
  %add213 = add nsw i32 0, 1
  %mul215 = mul nsw i32 %div197, 4
  %add216 = add nsw i32 %add213, %mul215
  %mul218 = mul nsw i32 0, 16
  %add219 = add nsw i32 %add216, %mul218
  %add221 = add nsw i32 0, %mod196
  %add225 = add nsw i32 0, 2
  %mul227 = mul nsw i32 %div197, 4
  %add228 = add nsw i32 %add225, %mul227
  %mul230 = mul nsw i32 0, 16
  %add231 = add nsw i32 %add228, %mul230
  %add233 = add nsw i32 0, %mod196
  %add237 = add nsw i32 0, 3
  %mul239 = mul nsw i32 %div197, 4
  %add240 = add nsw i32 %add237, %mul239
  %mul242 = mul nsw i32 0, 16
  %add243 = add nsw i32 %add240, %mul242
  %add245 = add nsw i32 0, %mod196
  %add247 = add nsw i32 %add193, 0
  %vh4248 = fptrunc <4 x float> %for29 to <4 x half>
  %add250 = add nsw i32 0, %add245
  %add251 = add nsw i32 %add194, %add250
  %add252 = add nsw i32 %add247, %add207
  %mul253 = mul nsw i32 %add252, %N
  %add254 = add nsw i32 %mul253, %add251
  %add255 = add nsw i32 0, %add254
  %e256 = extractelement <4 x half> %vh4248, i32 0
  %gep.152 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add255
  store half %e256, ptr addrspace(1) %gep.152, align 2
  %add257 = add nsw i32 %add247, %add219
  %mul258 = mul nsw i32 %add257, %N
  %add259 = add nsw i32 %mul258, %add251
  %add260 = add nsw i32 0, %add259
  %e261 = extractelement <4 x half> %vh4248, i32 1
  %gep.153 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add260
  store half %e261, ptr addrspace(1) %gep.153, align 2
  %add262 = add nsw i32 %add247, %add231
  %mul263 = mul nsw i32 %add262, %N
  %add264 = add nsw i32 %mul263, %add251
  %add265 = add nsw i32 0, %add264
  %e266 = extractelement <4 x half> %vh4248, i32 2
  %gep.154 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add265
  store half %e266, ptr addrspace(1) %gep.154, align 2
  %add267 = add nsw i32 %add247, %add243
  %mul268 = mul nsw i32 %add267, %N
  %add269 = add nsw i32 %mul268, %add251
  %add270 = add nsw i32 0, %add269
  %e271 = extractelement <4 x half> %vh4248, i32 3
  %gep.155 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add270
  store half %e271, ptr addrspace(1) %gep.155, align 2
  %vh4272 = fptrunc <4 x float> %for30 to <4 x half>
  %add274 = add nsw i32 16, %add245
  %add275 = add nsw i32 %add194, %add274
  %add276 = add nsw i32 %add247, %add207
  %mul277 = mul nsw i32 %add276, %N
  %add278 = add nsw i32 %mul277, %add275
  %add279 = add nsw i32 0, %add278
  %e280 = extractelement <4 x half> %vh4272, i32 0
  %gep.156 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add279
  store half %e280, ptr addrspace(1) %gep.156, align 2
  %add281 = add nsw i32 %add247, %add219
  %mul282 = mul nsw i32 %add281, %N
  %add283 = add nsw i32 %mul282, %add275
  %add284 = add nsw i32 0, %add283
  %e285 = extractelement <4 x half> %vh4272, i32 1
  %gep.157 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add284
  store half %e285, ptr addrspace(1) %gep.157, align 2
  %add286 = add nsw i32 %add247, %add231
  %mul287 = mul nsw i32 %add286, %N
  %add288 = add nsw i32 %mul287, %add275
  %add289 = add nsw i32 0, %add288
  %e290 = extractelement <4 x half> %vh4272, i32 2
  %gep.158 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add289
  store half %e290, ptr addrspace(1) %gep.158, align 2
  %add291 = add nsw i32 %add247, %add243
  %mul292 = mul nsw i32 %add291, %N
  %add293 = add nsw i32 %mul292, %add275
  %add294 = add nsw i32 0, %add293
  %e295 = extractelement <4 x half> %vh4272, i32 3
  %gep.159 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add294
  store half %e295, ptr addrspace(1) %gep.159, align 2
  %vh4296 = fptrunc <4 x float> %for31 to <4 x half>
  %add298 = add nsw i32 32, %add245
  %add299 = add nsw i32 %add194, %add298
  %add300 = add nsw i32 %add247, %add207
  %mul301 = mul nsw i32 %add300, %N
  %add302 = add nsw i32 %mul301, %add299
  %add303 = add nsw i32 0, %add302
  %e304 = extractelement <4 x half> %vh4296, i32 0
  %gep.160 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add303
  store half %e304, ptr addrspace(1) %gep.160, align 2
  %add305 = add nsw i32 %add247, %add219
  %mul306 = mul nsw i32 %add305, %N
  %add307 = add nsw i32 %mul306, %add299
  %add308 = add nsw i32 0, %add307
  %e309 = extractelement <4 x half> %vh4296, i32 1
  %gep.161 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add308
  store half %e309, ptr addrspace(1) %gep.161, align 2
  %add310 = add nsw i32 %add247, %add231
  %mul311 = mul nsw i32 %add310, %N
  %add312 = add nsw i32 %mul311, %add299
  %add313 = add nsw i32 0, %add312
  %e314 = extractelement <4 x half> %vh4296, i32 2
  %gep.162 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add313
  store half %e314, ptr addrspace(1) %gep.162, align 2
  %add315 = add nsw i32 %add247, %add243
  %mul316 = mul nsw i32 %add315, %N
  %add317 = add nsw i32 %mul316, %add299
  %add318 = add nsw i32 0, %add317
  %e319 = extractelement <4 x half> %vh4296, i32 3
  %gep.163 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add318
  store half %e319, ptr addrspace(1) %gep.163, align 2
  %vh4320 = fptrunc <4 x float> %for32 to <4 x half>
  %add322 = add nsw i32 48, %add245
  %add323 = add nsw i32 %add194, %add322
  %add324 = add nsw i32 %add247, %add207
  %mul325 = mul nsw i32 %add324, %N
  %add326 = add nsw i32 %mul325, %add323
  %add327 = add nsw i32 0, %add326
  %e328 = extractelement <4 x half> %vh4320, i32 0
  %gep.164 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add327
  store half %e328, ptr addrspace(1) %gep.164, align 2
  %add329 = add nsw i32 %add247, %add219
  %mul330 = mul nsw i32 %add329, %N
  %add331 = add nsw i32 %mul330, %add323
  %add332 = add nsw i32 0, %add331
  %e333 = extractelement <4 x half> %vh4320, i32 1
  %gep.165 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add332
  store half %e333, ptr addrspace(1) %gep.165, align 2
  %add334 = add nsw i32 %add247, %add231
  %mul335 = mul nsw i32 %add334, %N
  %add336 = add nsw i32 %mul335, %add323
  %add337 = add nsw i32 0, %add336
  %e338 = extractelement <4 x half> %vh4320, i32 2
  %gep.166 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add337
  store half %e338, ptr addrspace(1) %gep.166, align 2
  %add339 = add nsw i32 %add247, %add243
  %mul340 = mul nsw i32 %add339, %N
  %add341 = add nsw i32 %mul340, %add323
  %add342 = add nsw i32 0, %add341
  %e343 = extractelement <4 x half> %vh4320, i32 3
  %gep.167 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add342
  store half %e343, ptr addrspace(1) %gep.167, align 2
  %add345 = add nsw i32 %add193, 16
  %vh4346 = fptrunc <4 x float> %for33 to <4 x half>
  %add348 = add nsw i32 0, %add245
  %add349 = add nsw i32 %add194, %add348
  %add350 = add nsw i32 %add345, %add207
  %mul351 = mul nsw i32 %add350, %N
  %add352 = add nsw i32 %mul351, %add349
  %add353 = add nsw i32 0, %add352
  %e354 = extractelement <4 x half> %vh4346, i32 0
  %gep.168 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add353
  store half %e354, ptr addrspace(1) %gep.168, align 2
  %add355 = add nsw i32 %add345, %add219
  %mul356 = mul nsw i32 %add355, %N
  %add357 = add nsw i32 %mul356, %add349
  %add358 = add nsw i32 0, %add357
  %e359 = extractelement <4 x half> %vh4346, i32 1
  %gep.169 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add358
  store half %e359, ptr addrspace(1) %gep.169, align 2
  %add360 = add nsw i32 %add345, %add231
  %mul361 = mul nsw i32 %add360, %N
  %add362 = add nsw i32 %mul361, %add349
  %add363 = add nsw i32 0, %add362
  %e364 = extractelement <4 x half> %vh4346, i32 2
  %gep.170 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add363
  store half %e364, ptr addrspace(1) %gep.170, align 2
  %add365 = add nsw i32 %add345, %add243
  %mul366 = mul nsw i32 %add365, %N
  %add367 = add nsw i32 %mul366, %add349
  %add368 = add nsw i32 0, %add367
  %e369 = extractelement <4 x half> %vh4346, i32 3
  %gep.171 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add368
  store half %e369, ptr addrspace(1) %gep.171, align 2
  %vh4370 = fptrunc <4 x float> %for34 to <4 x half>
  %add372 = add nsw i32 16, %add245
  %add373 = add nsw i32 %add194, %add372
  %add374 = add nsw i32 %add345, %add207
  %mul375 = mul nsw i32 %add374, %N
  %add376 = add nsw i32 %mul375, %add373
  %add377 = add nsw i32 0, %add376
  %e378 = extractelement <4 x half> %vh4370, i32 0
  %gep.172 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add377
  store half %e378, ptr addrspace(1) %gep.172, align 2
  %add379 = add nsw i32 %add345, %add219
  %mul380 = mul nsw i32 %add379, %N
  %add381 = add nsw i32 %mul380, %add373
  %add382 = add nsw i32 0, %add381
  %e383 = extractelement <4 x half> %vh4370, i32 1
  %gep.173 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add382
  store half %e383, ptr addrspace(1) %gep.173, align 2
  %add384 = add nsw i32 %add345, %add231
  %mul385 = mul nsw i32 %add384, %N
  %add386 = add nsw i32 %mul385, %add373
  %add387 = add nsw i32 0, %add386
  %e388 = extractelement <4 x half> %vh4370, i32 2
  %gep.174 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add387
  store half %e388, ptr addrspace(1) %gep.174, align 2
  %add389 = add nsw i32 %add345, %add243
  %mul390 = mul nsw i32 %add389, %N
  %add391 = add nsw i32 %mul390, %add373
  %add392 = add nsw i32 0, %add391
  %e393 = extractelement <4 x half> %vh4370, i32 3
  %gep.175 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add392
  store half %e393, ptr addrspace(1) %gep.175, align 2
  %vh4394 = fptrunc <4 x float> %for35 to <4 x half>
  %add396 = add nsw i32 32, %add245
  %add397 = add nsw i32 %add194, %add396
  %add398 = add nsw i32 %add345, %add207
  %mul399 = mul nsw i32 %add398, %N
  %add400 = add nsw i32 %mul399, %add397
  %add401 = add nsw i32 0, %add400
  %e402 = extractelement <4 x half> %vh4394, i32 0
  %gep.176 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add401
  store half %e402, ptr addrspace(1) %gep.176, align 2
  %add403 = add nsw i32 %add345, %add219
  %mul404 = mul nsw i32 %add403, %N
  %add405 = add nsw i32 %mul404, %add397
  %add406 = add nsw i32 0, %add405
  %e407 = extractelement <4 x half> %vh4394, i32 1
  %gep.177 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add406
  store half %e407, ptr addrspace(1) %gep.177, align 2
  %add408 = add nsw i32 %add345, %add231
  %mul409 = mul nsw i32 %add408, %N
  %add410 = add nsw i32 %mul409, %add397
  %add411 = add nsw i32 0, %add410
  %e412 = extractelement <4 x half> %vh4394, i32 2
  %gep.178 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add411
  store half %e412, ptr addrspace(1) %gep.178, align 2
  %add413 = add nsw i32 %add345, %add243
  %mul414 = mul nsw i32 %add413, %N
  %add415 = add nsw i32 %mul414, %add397
  %add416 = add nsw i32 0, %add415
  %e417 = extractelement <4 x half> %vh4394, i32 3
  %gep.179 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add416
  store half %e417, ptr addrspace(1) %gep.179, align 2
  %vh4418 = fptrunc <4 x float> %for36 to <4 x half>
  %add420 = add nsw i32 48, %add245
  %add421 = add nsw i32 %add194, %add420
  %add422 = add nsw i32 %add345, %add207
  %mul423 = mul nsw i32 %add422, %N
  %add424 = add nsw i32 %mul423, %add421
  %add425 = add nsw i32 0, %add424
  %e426 = extractelement <4 x half> %vh4418, i32 0
  %gep.180 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add425
  store half %e426, ptr addrspace(1) %gep.180, align 2
  %add427 = add nsw i32 %add345, %add219
  %mul428 = mul nsw i32 %add427, %N
  %add429 = add nsw i32 %mul428, %add421
  %add430 = add nsw i32 0, %add429
  %e431 = extractelement <4 x half> %vh4418, i32 1
  %gep.181 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add430
  store half %e431, ptr addrspace(1) %gep.181, align 2
  %add432 = add nsw i32 %add345, %add231
  %mul433 = mul nsw i32 %add432, %N
  %add434 = add nsw i32 %mul433, %add421
  %add435 = add nsw i32 0, %add434
  %e436 = extractelement <4 x half> %vh4418, i32 2
  %gep.182 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add435
  store half %e436, ptr addrspace(1) %gep.182, align 2
  %add437 = add nsw i32 %add345, %add243
  %mul438 = mul nsw i32 %add437, %N
  %add439 = add nsw i32 %mul438, %add421
  %add440 = add nsw i32 0, %add439
  %e441 = extractelement <4 x half> %vh4418, i32 3
  %gep.183 = getelementptr inbounds half, ptr addrspace(1) %C, i32 %add440
  store half %e441, ptr addrspace(1) %gep.183, align 2
 ret void
}

attributes #0 = { "uniform-work-group-size"="true" "amdgpu-flat-work-group-size"="64,256" norecurse nounwind }
