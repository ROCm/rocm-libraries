target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle = internal unnamed_addr addrspace(3) global [64 x [72 x half]] poison, align 4
@B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle = internal unnamed_addr addrspace(3) global [64 x [72 x half]] poison, align 4
@C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle = internal unnamed_addr addrspace(3) global [64 x [64 x half]] poison, align 4

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare void @llvm.amdgcn.s.barrier()
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) nocapture readnone, i16, i64, i32)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)

define amdgpu_kernel void @ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle(ptr addrspace(1) noalias readonly nocapture align 16 %A, ptr addrspace(1) noalias readonly nocapture align 16 %B, ptr addrspace(1) noalias writeonly nocapture align 16 %D, i32 %A_bytes, i32 %B_bytes, i32 %D_bytes) #0 {
entry:
  %tid7 = call i32 @llvm.amdgcn.workitem.id.x()
  %mod8 = srem i32 %tid7, 64
  %div9 = sdiv i32 %tid7, 64
  %div10 = sdiv i32 %div9, 2
  %mod11 = srem i32 %div9, 2
  %bid13 = call i32 @llvm.amdgcn.workgroup.id.y()
  %mul14 = mul nsw i32 %bid13, 64
  %bid15 = call i32 @llvm.amdgcn.workgroup.id.x()
  %mul16 = mul nsw i32 %bid15, 64
  %cz1623 = select i1 true, <16 x float> zeroinitializer, <16 x float> zeroinitializer
  %nb64.1 = zext i32 %A_bytes to i64
  %rsrc24 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %A, i16 0, i64 %nb64.1, i32 159744)
  %nb64.2 = zext i32 %B_bytes to i64
  %rsrc26 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %B, i16 0, i64 %nb64.2, i32 159744)
  %nb64.3 = zext i32 %D_bytes to i64
  %rsrc28 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %D, i16 0, i64 %nb64.3, i32 159744)
  br label %for.header.1
for.header.1:
  %k0 = phi i32 [ 0, %entry ], [ %iv.next.for.header.1, %for.latch.3 ]
  %acc_m0_n0 = phi <16 x float> [ %cz1623, %entry ], [ %acc_m0_n0.next.for.header.1, %for.latch.3 ]
  %cmp.4 = icmp slt i32 %k0, 576
  br i1 %cmp.4, label %for.body.2, label %for.exit.4
for.body.2:
  %mul38 = mul nsw i32 0, 256
  %add39 = add nsw i32 %mul38, %tid7
  %div40 = sdiv i32 %add39, 8
  %mod41 = srem i32 %add39, 8
  %mul42 = mul nsw i32 %mod41, 8
  %add43 = add nsw i32 %mul14, %div40
  %add44 = add nsw i32 %k0, %mul42
  %div46 = sdiv i32 %add43, 3136
  %div48 = sdiv i32 %add43, 56
  %mod50 = srem i32 %div48, 56
  %mod52 = srem i32 %add43, 56
  %div54 = sdiv i32 %add44, 192
  %div56 = sdiv i32 %add44, 64
  %mod58 = srem i32 %div56, 3
  %mod60 = srem i32 %add44, 64
  %ge63 = icmp sge i32 %div54, 0
  %lt64 = icmp slt i32 %div54, 3
  %and65 = and i1 %ge63, %lt64
  %ge68 = icmp sge i32 %mod58, 0
  %lt69 = icmp slt i32 %mod58, 3
  %and70 = and i1 %ge68, %lt69
  %add71 = add nsw i32 %mod50, %div54
  %add73 = add nsw i32 %add71, -1
  %ge75 = icmp sge i32 %add73, 0
  %lt77 = icmp slt i32 %add73, 56
  %and78 = and i1 %ge75, %lt77
  %and79 = and i1 %and65, %and78
  %add80 = add nsw i32 %mod52, %mod58
  %add82 = add nsw i32 %add80, -1
  %ge84 = icmp sge i32 %add82, 0
  %lt86 = icmp slt i32 %add82, 56
  %and87 = and i1 %ge84, %lt86
  %and88 = and i1 %and70, %and87
  %mul90 = mul nsw i32 %div46, 200704
  %mul92 = mul nsw i32 %add73, 3584
  %add93 = add nsw i32 %mul90, %mul92
  %and94 = and i1 %and79, %and88
  %mul96 = mul nsw i32 %add82, 64
  %add97 = add nsw i32 %add93, %mul96
  %add98 = add nsw i32 %add97, %mod60
  %mul99 = mul nsw i32 %add98, 2
  %sel100 = select i1 %and94, i32 %mul99, i32 2147483647
  %blv4.5 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %rsrc24, i32 %sel100, i32 0, i32 0)
  %bl8101 = bitcast <4 x i32> %blv4.5 to <8 x half>
  %gep.6 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %div40, i32 %mul42
  store <8 x half> %bl8101, ptr addrspace(3) %gep.6, align 16
  %mul103 = mul nsw i32 1, 256
  %add104 = add nsw i32 %mul103, %tid7
  %div105 = sdiv i32 %add104, 8
  %mod106 = srem i32 %add104, 8
  %mul107 = mul nsw i32 %mod106, 8
  %add108 = add nsw i32 %mul14, %div105
  %add109 = add nsw i32 %k0, %mul107
  %div111 = sdiv i32 %add108, 3136
  %div113 = sdiv i32 %add108, 56
  %mod115 = srem i32 %div113, 56
  %mod117 = srem i32 %add108, 56
  %div119 = sdiv i32 %add109, 192
  %div121 = sdiv i32 %add109, 64
  %mod123 = srem i32 %div121, 3
  %mod125 = srem i32 %add109, 64
  %ge128 = icmp sge i32 %div119, 0
  %lt129 = icmp slt i32 %div119, 3
  %and130 = and i1 %ge128, %lt129
  %ge133 = icmp sge i32 %mod123, 0
  %lt134 = icmp slt i32 %mod123, 3
  %and135 = and i1 %ge133, %lt134
  %add136 = add nsw i32 %mod115, %div119
  %add138 = add nsw i32 %add136, -1
  %ge140 = icmp sge i32 %add138, 0
  %lt142 = icmp slt i32 %add138, 56
  %and143 = and i1 %ge140, %lt142
  %and144 = and i1 %and130, %and143
  %add145 = add nsw i32 %mod117, %mod123
  %add147 = add nsw i32 %add145, -1
  %ge149 = icmp sge i32 %add147, 0
  %lt151 = icmp slt i32 %add147, 56
  %and152 = and i1 %ge149, %lt151
  %and153 = and i1 %and135, %and152
  %mul155 = mul nsw i32 %div111, 200704
  %mul157 = mul nsw i32 %add138, 3584
  %add158 = add nsw i32 %mul155, %mul157
  %and159 = and i1 %and144, %and153
  %mul161 = mul nsw i32 %add147, 64
  %add162 = add nsw i32 %add158, %mul161
  %add163 = add nsw i32 %add162, %mod125
  %mul164 = mul nsw i32 %add163, 2
  %sel165 = select i1 %and159, i32 %mul164, i32 2147483647
  %blv4.7 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %rsrc24, i32 %sel165, i32 0, i32 0)
  %bl8166 = bitcast <4 x i32> %blv4.7 to <8 x half>
  %gep.8 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %div105, i32 %mul107
  store <8 x half> %bl8166, ptr addrspace(3) %gep.8, align 16
  %mul174 = mul nsw i32 0, 256
  %add175 = add nsw i32 %mul174, %tid7
  %div176 = sdiv i32 %add175, 8
  %mod177 = srem i32 %add175, 8
  %mul178 = mul nsw i32 %mod177, 8
  %add179 = add nsw i32 %mul16, %div176
  %add180 = add nsw i32 %k0, %mul178
  %div182 = sdiv i32 %add180, 192
  %div184 = sdiv i32 %add180, 64
  %mod186 = srem i32 %div184, 3
  %mod188 = srem i32 %add180, 64
  %ge191 = icmp sge i32 %div182, 0
  %lt192 = icmp slt i32 %div182, 3
  %and193 = and i1 %ge191, %lt192
  %ge196 = icmp sge i32 %mod186, 0
  %lt197 = icmp slt i32 %mod186, 3
  %and198 = and i1 %ge196, %lt197
  %mul200 = mul nsw i32 %add179, 576
  %mul202 = mul nsw i32 %div182, 192
  %add203 = add nsw i32 %mul200, %mul202
  %and204 = and i1 %and193, %and198
  %mul206 = mul nsw i32 %mod186, 64
  %add207 = add nsw i32 %add203, %mul206
  %add208 = add nsw i32 %add207, %mod188
  %mul209 = mul nsw i32 %add208, 2
  %sel210 = select i1 %and204, i32 %mul209, i32 2147483647
  %blv4.9 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %rsrc26, i32 %sel210, i32 0, i32 0)
  %bl8211 = bitcast <4 x i32> %blv4.9 to <8 x half>
  %gep.10 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %div176, i32 %mul178
  store <8 x half> %bl8211, ptr addrspace(3) %gep.10, align 16
  %mul213 = mul nsw i32 1, 256
  %add214 = add nsw i32 %mul213, %tid7
  %div215 = sdiv i32 %add214, 8
  %mod216 = srem i32 %add214, 8
  %mul217 = mul nsw i32 %mod216, 8
  %add218 = add nsw i32 %mul16, %div215
  %add219 = add nsw i32 %k0, %mul217
  %div221 = sdiv i32 %add219, 192
  %div223 = sdiv i32 %add219, 64
  %mod225 = srem i32 %div223, 3
  %mod227 = srem i32 %add219, 64
  %ge230 = icmp sge i32 %div221, 0
  %lt231 = icmp slt i32 %div221, 3
  %and232 = and i1 %ge230, %lt231
  %ge235 = icmp sge i32 %mod225, 0
  %lt236 = icmp slt i32 %mod225, 3
  %and237 = and i1 %ge235, %lt236
  %mul239 = mul nsw i32 %add218, 576
  %mul241 = mul nsw i32 %div221, 192
  %add242 = add nsw i32 %mul239, %mul241
  %and243 = and i1 %and232, %and237
  %mul245 = mul nsw i32 %mod225, 64
  %add246 = add nsw i32 %add242, %mul245
  %add247 = add nsw i32 %add246, %mod227
  %mul248 = mul nsw i32 %add247, 2
  %sel249 = select i1 %and243, i32 %mul248, i32 2147483647
  %blv4.11 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %rsrc26, i32 %sel249, i32 0, i32 0)
  %bl8250 = bitcast <4 x i32> %blv4.11 to <8 x half>
  %gep.12 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %div215, i32 %mul217
  store <8 x half> %bl8250, ptr addrspace(3) %gep.12, align 16
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  %mod253 = srem i32 %mod8, 32
  %mod254 = srem i32 %mod8, 32
  %div255 = sdiv i32 %mod8, 32
  %mul257 = mul nsw i32 %div10, 32
  %mul259 = mul nsw i32 %mod11, 32
  %mul261 = mul nsw i32 %div255, 8
  %add263 = add nsw i32 %mul261, 0
  %add265 = add nsw i32 0, %mod253
  %add266 = add nsw i32 %mul257, %add265
  %smem.base.13 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add266, i32 %add263
  %av8267 = load <8 x half>, ptr addrspace(3) %smem.base.13, align 16
  %add269 = add nsw i32 0, %mod254
  %add270 = add nsw i32 %mul259, %add269
  %smem.base.14 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add270, i32 %add263
  %av8271 = load <8 x half>, ptr addrspace(3) %smem.base.14, align 16
  %acc272 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8267, <8 x half> %av8271, <16 x float> %acc_m0_n0, i32 0, i32 0, i32 0)
  %mul274 = mul nsw i32 %div255, 8
  %add276 = add nsw i32 %mul274, 16
  %add278 = add nsw i32 0, %mod253
  %add279 = add nsw i32 %mul257, %add278
  %smem.base.15 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add279, i32 %add276
  %av8280 = load <8 x half>, ptr addrspace(3) %smem.base.15, align 16
  %add282 = add nsw i32 0, %mod254
  %add283 = add nsw i32 %mul259, %add282
  %smem.base.16 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add283, i32 %add276
  %av8284 = load <8 x half>, ptr addrspace(3) %smem.base.16, align 16
  %acc285 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8280, <8 x half> %av8284, <16 x float> %acc272, i32 0, i32 0, i32 0)
  %mul287 = mul nsw i32 %div255, 8
  %add289 = add nsw i32 %mul287, 32
  %add291 = add nsw i32 0, %mod253
  %add292 = add nsw i32 %mul257, %add291
  %smem.base.17 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add292, i32 %add289
  %av8293 = load <8 x half>, ptr addrspace(3) %smem.base.17, align 16
  %add295 = add nsw i32 0, %mod254
  %add296 = add nsw i32 %mul259, %add295
  %smem.base.18 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add296, i32 %add289
  %av8297 = load <8 x half>, ptr addrspace(3) %smem.base.18, align 16
  %acc298 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8293, <8 x half> %av8297, <16 x float> %acc285, i32 0, i32 0, i32 0)
  %mul300 = mul nsw i32 %div255, 8
  %add302 = add nsw i32 %mul300, 48
  %add304 = add nsw i32 0, %mod253
  %add305 = add nsw i32 %mul257, %add304
  %smem.base.19 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @A_smem21.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add305, i32 %add302
  %av8306 = load <8 x half>, ptr addrspace(3) %smem.base.19, align 16
  %add308 = add nsw i32 0, %mod254
  %add309 = add nsw i32 %mul259, %add308
  %smem.base.20 = getelementptr inbounds [64 x [72 x half]], ptr addrspace(3) @B_smem22.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add309, i32 %add302
  %av8310 = load <8 x half>, ptr addrspace(3) %smem.base.20, align 16
  %acc311 = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %av8306, <8 x half> %av8310, <16 x float> %acc298, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  br label %for.latch.3
for.latch.3:
  %iv.next.for.header.1 = add nsw i32 %k0, 64
  %acc_m0_n0.next.for.header.1 = bitcast <16 x float> %acc311 to <16 x float>
  br label %for.header.1
for.exit.4:
  %for30 = bitcast <16 x float> %acc_m0_n0 to <16 x float>
  %mul313 = mul nsw i32 %div10, 32
  %mul315 = mul nsw i32 %mod11, 32
  %mod318 = srem i32 %mod8, 32
  %div319 = sdiv i32 %mod8, 32
  %vh16320 = fptrunc <16 x float> %for30 to <16 x half>
  %add322 = add nsw i32 %mul315, 0
  %add323 = add nsw i32 %add322, %mod318
  %mul326 = mul nsw i32 %div319, 4
  %add327 = add nsw i32 0, %mul326
  %add329 = add nsw i32 %add327, 0
  %add331 = add nsw i32 %mul313, 0
  %add332 = add nsw i32 %add331, %add329
  %e333 = extractelement <16 x half> %vh16320, i32 0
  %gep.21 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add332, i32 %add323
  store half %e333, ptr addrspace(3) %gep.21, align 2
  %mul336 = mul nsw i32 %div319, 4
  %add337 = add nsw i32 0, %mul336
  %add339 = add nsw i32 %add337, 1
  %add341 = add nsw i32 %mul313, 0
  %add342 = add nsw i32 %add341, %add339
  %e343 = extractelement <16 x half> %vh16320, i32 1
  %gep.22 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add342, i32 %add323
  store half %e343, ptr addrspace(3) %gep.22, align 2
  %mul346 = mul nsw i32 %div319, 4
  %add347 = add nsw i32 0, %mul346
  %add349 = add nsw i32 %add347, 2
  %add351 = add nsw i32 %mul313, 0
  %add352 = add nsw i32 %add351, %add349
  %e353 = extractelement <16 x half> %vh16320, i32 2
  %gep.23 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add352, i32 %add323
  store half %e353, ptr addrspace(3) %gep.23, align 2
  %mul356 = mul nsw i32 %div319, 4
  %add357 = add nsw i32 0, %mul356
  %add359 = add nsw i32 %add357, 3
  %add361 = add nsw i32 %mul313, 0
  %add362 = add nsw i32 %add361, %add359
  %e363 = extractelement <16 x half> %vh16320, i32 3
  %gep.24 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add362, i32 %add323
  store half %e363, ptr addrspace(3) %gep.24, align 2
  %mul366 = mul nsw i32 %div319, 4
  %add367 = add nsw i32 8, %mul366
  %add369 = add nsw i32 %add367, 0
  %add371 = add nsw i32 %mul313, 0
  %add372 = add nsw i32 %add371, %add369
  %e373 = extractelement <16 x half> %vh16320, i32 4
  %gep.25 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add372, i32 %add323
  store half %e373, ptr addrspace(3) %gep.25, align 2
  %mul376 = mul nsw i32 %div319, 4
  %add377 = add nsw i32 8, %mul376
  %add379 = add nsw i32 %add377, 1
  %add381 = add nsw i32 %mul313, 0
  %add382 = add nsw i32 %add381, %add379
  %e383 = extractelement <16 x half> %vh16320, i32 5
  %gep.26 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add382, i32 %add323
  store half %e383, ptr addrspace(3) %gep.26, align 2
  %mul386 = mul nsw i32 %div319, 4
  %add387 = add nsw i32 8, %mul386
  %add389 = add nsw i32 %add387, 2
  %add391 = add nsw i32 %mul313, 0
  %add392 = add nsw i32 %add391, %add389
  %e393 = extractelement <16 x half> %vh16320, i32 6
  %gep.27 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add392, i32 %add323
  store half %e393, ptr addrspace(3) %gep.27, align 2
  %mul396 = mul nsw i32 %div319, 4
  %add397 = add nsw i32 8, %mul396
  %add399 = add nsw i32 %add397, 3
  %add401 = add nsw i32 %mul313, 0
  %add402 = add nsw i32 %add401, %add399
  %e403 = extractelement <16 x half> %vh16320, i32 7
  %gep.28 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add402, i32 %add323
  store half %e403, ptr addrspace(3) %gep.28, align 2
  %mul406 = mul nsw i32 %div319, 4
  %add407 = add nsw i32 16, %mul406
  %add409 = add nsw i32 %add407, 0
  %add411 = add nsw i32 %mul313, 0
  %add412 = add nsw i32 %add411, %add409
  %e413 = extractelement <16 x half> %vh16320, i32 8
  %gep.29 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add412, i32 %add323
  store half %e413, ptr addrspace(3) %gep.29, align 2
  %mul416 = mul nsw i32 %div319, 4
  %add417 = add nsw i32 16, %mul416
  %add419 = add nsw i32 %add417, 1
  %add421 = add nsw i32 %mul313, 0
  %add422 = add nsw i32 %add421, %add419
  %e423 = extractelement <16 x half> %vh16320, i32 9
  %gep.30 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add422, i32 %add323
  store half %e423, ptr addrspace(3) %gep.30, align 2
  %mul426 = mul nsw i32 %div319, 4
  %add427 = add nsw i32 16, %mul426
  %add429 = add nsw i32 %add427, 2
  %add431 = add nsw i32 %mul313, 0
  %add432 = add nsw i32 %add431, %add429
  %e433 = extractelement <16 x half> %vh16320, i32 10
  %gep.31 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add432, i32 %add323
  store half %e433, ptr addrspace(3) %gep.31, align 2
  %mul436 = mul nsw i32 %div319, 4
  %add437 = add nsw i32 16, %mul436
  %add439 = add nsw i32 %add437, 3
  %add441 = add nsw i32 %mul313, 0
  %add442 = add nsw i32 %add441, %add439
  %e443 = extractelement <16 x half> %vh16320, i32 11
  %gep.32 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add442, i32 %add323
  store half %e443, ptr addrspace(3) %gep.32, align 2
  %mul446 = mul nsw i32 %div319, 4
  %add447 = add nsw i32 24, %mul446
  %add449 = add nsw i32 %add447, 0
  %add451 = add nsw i32 %mul313, 0
  %add452 = add nsw i32 %add451, %add449
  %e453 = extractelement <16 x half> %vh16320, i32 12
  %gep.33 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add452, i32 %add323
  store half %e453, ptr addrspace(3) %gep.33, align 2
  %mul456 = mul nsw i32 %div319, 4
  %add457 = add nsw i32 24, %mul456
  %add459 = add nsw i32 %add457, 1
  %add461 = add nsw i32 %mul313, 0
  %add462 = add nsw i32 %add461, %add459
  %e463 = extractelement <16 x half> %vh16320, i32 13
  %gep.34 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add462, i32 %add323
  store half %e463, ptr addrspace(3) %gep.34, align 2
  %mul466 = mul nsw i32 %div319, 4
  %add467 = add nsw i32 24, %mul466
  %add469 = add nsw i32 %add467, 2
  %add471 = add nsw i32 %mul313, 0
  %add472 = add nsw i32 %add471, %add469
  %e473 = extractelement <16 x half> %vh16320, i32 14
  %gep.35 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add472, i32 %add323
  store half %e473, ptr addrspace(3) %gep.35, align 2
  %mul476 = mul nsw i32 %div319, 4
  %add477 = add nsw i32 24, %mul476
  %add479 = add nsw i32 %add477, 3
  %add481 = add nsw i32 %mul313, 0
  %add482 = add nsw i32 %add481, %add479
  %e483 = extractelement <16 x half> %vh16320, i32 15
  %gep.36 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %add482, i32 %add323
  store half %e483, ptr addrspace(3) %gep.36, align 2
  call void @llvm.amdgcn.s.waitcnt(i32 112)
 call void @llvm.amdgcn.s.barrier()
  %tid484 = call i32 @llvm.amdgcn.workitem.id.x()
  %mul491 = mul nsw i32 0, 256
  %add492 = add nsw i32 %mul491, %tid484
  %div493 = sdiv i32 %add492, 8
  %mod494 = srem i32 %add492, 8
  %mul496 = mul nsw i32 %mod494, 8
  %add497 = add nsw i32 %mul14, %div493
  %add498 = add nsw i32 %mul16, %mul496
  %lt499 = icmp slt i32 %add497, 25088
  %add501 = add nsw i32 %add498, 8
  %le502 = icmp sle i32 %add501, 64
  %and503 = and i1 %lt499, %le502
  %div505 = sdiv i32 %add497, 3136
  %div507 = sdiv i32 %add497, 56
  %mod509 = srem i32 %div507, 56
  %mod511 = srem i32 %add497, 56
  %mul513 = mul nsw i32 %div505, 200704
  %mul515 = mul nsw i32 %mod509, 3584
  %add516 = add nsw i32 %mul513, %mul515
  %mul518 = mul nsw i32 %mod511, 64
  %add519 = add nsw i32 %add516, %mul518
  %add520 = add nsw i32 %add519, %add498
  %mul521 = mul nsw i32 %add520, 2
  %sel523 = select i1 %and503, i32 %mul521, i32 2147483647
  %smem.base.37 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %div493, i32 %mul496
  %av8524 = load <8 x half>, ptr addrspace(3) %smem.base.37, align 16
  %bsbc.38 = bitcast <8 x half> %av8524 to <4 x i32>
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %bsbc.38, ptr addrspace(8) %rsrc28, i32 %sel523, i32 0, i32 0)
  %mul526 = mul nsw i32 1, 256
  %add527 = add nsw i32 %mul526, %tid484
  %div528 = sdiv i32 %add527, 8
  %mod529 = srem i32 %add527, 8
  %mul531 = mul nsw i32 %mod529, 8
  %add532 = add nsw i32 %mul14, %div528
  %add533 = add nsw i32 %mul16, %mul531
  %lt534 = icmp slt i32 %add532, 25088
  %add536 = add nsw i32 %add533, 8
  %le537 = icmp sle i32 %add536, 64
  %and538 = and i1 %lt534, %le537
  %div540 = sdiv i32 %add532, 3136
  %div542 = sdiv i32 %add532, 56
  %mod544 = srem i32 %div542, 56
  %mod546 = srem i32 %add532, 56
  %mul548 = mul nsw i32 %div540, 200704
  %mul550 = mul nsw i32 %mod544, 3584
  %add551 = add nsw i32 %mul548, %mul550
  %mul553 = mul nsw i32 %mod546, 64
  %add554 = add nsw i32 %add551, %mul553
  %add555 = add nsw i32 %add554, %add533
  %mul556 = mul nsw i32 %add555, 2
  %sel558 = select i1 %and538, i32 %mul556, i32 2147483647
  %smem.base.39 = getelementptr inbounds [64 x [64 x half]], ptr addrspace(3) @C_smem316.ck_dsl_ex08_bake_off_implicit_gemm_N8H56W56C64_K64R3S3_t64x64x64_w2x2_a32x32x16_mem_cshuffle, i32 0, i32 %div528, i32 %mul531
  %av8559 = load <8 x half>, ptr addrspace(3) %smem.base.39, align 16
  %bsbc.40 = bitcast <8 x half> %av8559 to <4 x i32>
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %bsbc.40, ptr addrspace(8) %rsrc28, i32 %sel558, i32 0, i32 0)
 ret void
}

attributes #0 = { "uniform-work-group-size"="true" "amdgpu-flat-work-group-size"="64,256" norecurse nounwind }
