// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.BF8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.BF6.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
 // T_LOAD_TILED 3 Source 2
 // Tensor.E8M0.d2 4, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
 // T_LOAD_TILED 5 Source 4
 // BlockScale(Separate, {1, 32}): Data: 1, Scale: 5
 // Tensor.E8M0.d2 7, (base=&120, sizes={&128 &136 }, strides={&144 &152 })
 // T_LOAD_TILED 8 Source 7
 // BlockScale(Separate, {32, 1}): Data: 3, Scale: 8
 // Tensor.Float.d2 10, (base=&160, sizes={&168 &176 }, strides={&184 &192 })
 // T_LOAD_TILED 11 Source 10
 // Scalar.Float.12
 // T_LOAD_SCALAR 13Source 12
 // Scalar.Float.14
 // T_LOAD_SCALAR 15Source 14
 // T_Mul 6 9 Value: Float
 // T_EXECUTE 13 16 15 11
 //   E_Mul 17, 15, 11
 //   E_Mul 18, 13, 16
 //   E_Add 19, 17, 18
 // Tensor.Float.d2 21, (base=&208, sizes={&216 &224 }, strides={&232 &240 })
 // T_STORE_TILED 22 Source 19 Dest 21

 // Command arguments:
 // Tensor_0_pointer: PointerGlobal: BF8(offset: 0, size: 8, read_write)
 // Tensor_0_size_0: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: BF6(offset: 40, size: 8, read_write)
 // Tensor_2_size_0: Value: Int64(offset: 48, size: 8, read_only)
 // Tensor_2_size_1: Value: Int64(offset: 56, size: 8, read_only)
 // Tensor_2_stride_0: Value: Int64(offset: 64, size: 8, read_only)
 // Tensor_2_stride_1: Value: Int64(offset: 72, size: 8, read_only)
 // Tensor_4_pointer: PointerGlobal: E8M0(offset: 80, size: 8, read_write)
 // Tensor_4_size_0: Value: Int64(offset: 88, size: 8, read_only)
 // Tensor_4_size_1: Value: Int64(offset: 96, size: 8, read_only)
 // Tensor_4_stride_0: Value: Int64(offset: 104, size: 8, read_only)
 // Tensor_4_stride_1: Value: Int64(offset: 112, size: 8, read_only)
 // Tensor_7_pointer: PointerGlobal: E8M0(offset: 120, size: 8, read_write)
 // Tensor_7_size_0: Value: Int64(offset: 128, size: 8, read_only)
 // Tensor_7_size_1: Value: Int64(offset: 136, size: 8, read_only)
 // Tensor_7_stride_0: Value: Int64(offset: 144, size: 8, read_only)
 // Tensor_7_stride_1: Value: Int64(offset: 152, size: 8, read_only)
 // Tensor_10_pointer: PointerGlobal: Float(offset: 160, size: 8, read_write)
 // Tensor_10_size_0: Value: Int64(offset: 168, size: 8, read_only)
 // Tensor_10_size_1: Value: Int64(offset: 176, size: 8, read_only)
 // Tensor_10_stride_0: Value: Int64(offset: 184, size: 8, read_only)
 // Tensor_10_stride_1: Value: Int64(offset: 192, size: 8, read_only)
 // user_Float_Value_12: Value: Float(offset: 200, size: 4, read_only)
 // user_Float_Value_14: Value: Float(offset: 204, size: 4, read_only)
 // Tensor_21_pointer: PointerGlobal: Float(offset: 208, size: 8, read_write)
 // Tensor_21_size_0: Value: Int64(offset: 216, size: 8, read_only)
 // Tensor_21_size_1: Value: Int64(offset: 224, size: 8, read_only)
 // Tensor_21_stride_0: Value: Int64(offset: 232, size: 8, read_only)
 // Tensor_21_stride_1: Value: Int64(offset: 240, size: 8, read_only)

 // Kernel Arguments:
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: BF6, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}
.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
 // Pad to make room for arg loading preamble.
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:
 // Kernel execution starts here. Use this label instead of the kernel name when setting a breakpoint.
 // Kernel Options:
 //   logLevel:                             Verbose
 //   alwaysWaitAfterLoad:                    false
 //   alwaysWaitAfterStore:                   false
 //   alwaysWaitBeforeBranch:                 false
 //   alwaysWaitZeroBeforeBarrier:            false
 //   systemPreloadedKernelArguments:            -1
 //   lazyLoadKernelArguments:                false
 //   maxACCVGPRs:                              256
 //   maxSGPRs:                                 102
 //   maxVGPRs:                                 256
 //   loadLocalWidth:                             4
 //   loadGlobalWidth:                            8
 //   storeLocalWidth:                            4
 //   storeGlobalWidth:                           4
 //   assertWaitCntState:                      true
 //   setNextFreeVGPRToMax:                   false
 //   deduplicateArguments:                    true
 //   minLaunchTimeExpressionComplexity:         20
 //   maxConcurrentSubExpressions:                1
 //   maxConcurrentControlOps:                 none
 //   dsObserver:                        DSMEMObserver
 //   enableFullDivision:                     false
 //   scaleSkipPermlane:                       None
 //   assertOpKind:                            NoOp
 //   removeSetCoordinate:                    false
 //   ldsSwizzleMode:                          None
 //   coexecutionEnabled:                      true
 //   favourF8F6F4OverF8MatrixInstruction:      true

 // Loading Kernel Arguments: 
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: BF6, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}

 // Kernel arguments: Loading 72 bytes
s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
 // Freeing Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15

 // Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]
 // Tensor_0_pointer: SGPR PointerGlobal: BF8 x 1: s[4:5]
 // Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]
 // Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]
 // Tensor_2_pointer: SGPR PointerGlobal: BF6 x 1: s[10:11]
 // Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]
 // user_Float_Value_14: SGPR Value: Float x 1: s14
 // user_Float_Value_12: SGPR Value: Float x 1: s15
 // Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]
 // Tensor_4_pointer: SGPR PointerGlobal: E8M0 x 1: s[22:23]
 // Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]
 // Tensor_7_pointer: SGPR PointerGlobal: E8M0 x 1: s[26:27]
 // Tensor_0_size_1_14: SGPR Value: Int64 x 1: s[28:29]
 // Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]
 // Tensor_21_pointer: SGPR PointerGlobal: Float x 1: s[32:33]
 // Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]
 // Tensor_10_pointer: SGPR PointerGlobal: Float x 1: s[36:37]
 // Extract 10 bit X coordinate
v_bfe_u32 v1, v0, 0, 10
 // Generate {Workitem Index X: v1:U32} into Workitem Index X: VGPR Value: UInt32 x 1: v1
 // reg expression
 // Extract 10 bit Y coordinate
v_bfe_u32 v2, v0, 10, 10
 // Generate {Workitem Index Y: v2:U32} into Workitem Index Y: VGPR Value: UInt32 x 1: v2
 // reg expression
 // Freeing Packed Workitem Index: 1 VGPR (Value: Raw32): v0

 // Freeing Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1

 // Marking Workgroup Index X: SGPR Value: UInt32 x 1: s16 as read-only
 // Marking Workgroup Index Y: SGPR Value: UInt32 x 1: s17 as read-only
 // Marking Workitem Index X: VGPR Value: UInt32 x 1: v1 as read-only
 // Marking Workitem Index Y: VGPR Value: UInt32 x 1: v2 as read-only
 // CodeGeneratorVisitor::generate() begin
 // UpdateParameters
 // IdentifyParallelDimensions
 // OrderMemory
 // AddLDS
 // LowerLinear
 // LowerTile
 // LowerTensorContraction
 // Simplify
 // AddLDSPadding
 // ConstantPropagation
 // FuseExpressions
 // ConnectWorkgroups
 // WorkgroupRemapXCC
 // UnrollLoops
 // FuseLoops
 // RemoveDuplicates
 // OrderEpilogueBlocks
 // Simplify
 // CleanLoops
 // SwizzleScale
 // AddPrefetch
 // AddF6LDSPadding
 // AddDirect2LDS
 // AddTDMToLDS
 // Simplify
 // AddPRNG
 // UpdateWavefrontParameters
 // AssignIndexExpressions
 // LoadPacked
 // AddConvert
 // NopExtraScopes
 // InlineInits
 // InlineIncrements
 // OrderMultiplyNodes
 // Simplify
 // Simplify
 // OrderExchangeNodes
 // AliasDataFlowTags
 // AddLDSBarriers
 // AddDeallocateDataFlow
 // CleanArguments
 // AddDeallocateArguments
 // MergeAdjacentDeallocates
 // Simplify
 // SortArguments
 // SetWorkitemCount
 // ModelAddresses
 // generate({1})
 // Kernel(1) BEGIN
 // (op 1) generate({})
 // (op 1) end: generate({})
 // (op 1) generate({783, 836, 889, 942, 1015, 1039, 1063, 1087})
 // (op 1) NOP(783) BEGIN
 // (op 783) generate({})
 // (op 783) end: generate({})
 // (op 1) NOP(783) END
 // (op 1) NOP(836) BEGIN
 // (op 836) generate({})
 // (op 836) end: generate({})
 // (op 1) NOP(836) END
 // (op 1) NOP(889) BEGIN
 // (op 889) generate({})
 // (op 889) end: generate({})
 // (op 1) NOP(889) END
 // (op 1) NOP(942) BEGIN
 // (op 942) generate({})
 // (op 942) end: generate({})
 // (op 1) NOP(942) END
 // (op 1) NOP(1015) BEGIN
 // (op 1015) generate({})
 // (op 1015) end: generate({})
 // (op 1) NOP(1015) END
 // (op 1) NOP(1039) BEGIN
 // (op 1039) generate({})
 // (op 1039) end: generate({})
 // (op 1) NOP(1039) END
 // (op 1) NOP(1063) BEGIN
 // (op 1063) generate({})
 // (op 1063) end: generate({})
 // (op 1) NOP(1063) END
 // (op 1) NOP(1087) BEGIN
 // (op 1087) generate({})
 // (op 1087) end: generate({})
 // (op 1) NOP(1087) END
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(787) BEGIN
 // (op 787) Assign dim(584) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 584: v**UNALLOCATED**
 // (op 787) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag584: VGPR Value: UInt32 x 1: (unallocated)
 // (op 787) reg expression
 // (op 787) BEGIN: Flatten[1]
 // (op 787) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(840) BEGIN
 // (op 840) Assign dim(600) = LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 600: v**UNALLOCATED**
 // (op 840) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag600: VGPR Value: UInt32 x 1: (unallocated)
 // (op 840) reg expression
 // (op 840) BEGIN: Flatten[1]
 // (op 840) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(893) BEGIN
 // (op 893) Assign dim(616) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 616: v**UNALLOCATED**
 // (op 893) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag616: VGPR Value: UInt32 x 1: (unallocated)
 // (op 893) reg expression
 // (op 893) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(946) BEGIN
 // (op 946) Assign dim(632) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 632: v**UNALLOCATED**
 // (op 946) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag632: VGPR Value: UInt32 x 1: (unallocated)
 // (op 946) reg expression
 // (op 946) BEGIN: Flatten[1]Tile
 // (op 946) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1019) BEGIN
 // (op 1019) Assign dim(654) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 654: v**UNALLOCATED**
 // (op 1019) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag654: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1019) reg expression
 // (op 1019) BEGIN: Flatten[1]
 // (op 1019) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1043) BEGIN
 // (op 1043) Assign dim(662) = LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 662: v**UNALLOCATED**
 // (op 1043) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag662: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1043) reg expression
 // (op 1043) BEGIN: Flatten[1]
 // (op 1043) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(1067) BEGIN
 // (op 1067) Assign dim(670) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 670: v**UNALLOCATED**
 // (op 1067) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag670: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1067) reg expression
 // (op 1067) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(1091) BEGIN
 // (op 1091) Assign dim(678) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 678: v**UNALLOCATED**
 // (op 1091) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag678: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1091) reg expression
 // (op 1091) BEGIN: Flatten[1]Tile
 // (op 1091) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 787): v0
v_and_b32 v0, 63, v1 // (op 787) 
 // (op 787) END: Flatten[1]
 // (op 787) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 787): v3
v_lshrrev_b32 v3, 4, v0 // (op 787) 
 // (op 787) BEGIN: Flatten[1]
 // (op 787) {Flatten[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 787): v4
v_and_b32 v4, 1, v3 // (op 787) 
 // (op 787) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 787): v3

 // (op 787) BEGIN: Flatten[1]
 // (op 787) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 787): v3
v_and_b32 v3, 15, v1 // (op 787) 
 // (op 787) END: Flatten[1]
 // (op 787) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 787): v5
v_lshl_add_u32 v5, v4, 4, v3 // (op 787) 
 // (op 787) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 787): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 787): v4

 // (op 787) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 787): v3
v_lshrrev_b32 v3, 6, v1 // (op 787) 
 // (op 787) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 787): v4
v_lshl_add_u32 v4, v3, 5, v5 // (op 787) 
 // (op 787) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 787): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 787): v3

 // (op 787) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 787): v3
v_lshrrev_b32 v3, 5, v0 // (op 787) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 787): v0

 // (op 787) BEGIN: TileTile
 // (op 787) {TileTile: ShiftL(v3:U32, 4:U32)U32}
 // (op 787) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 787)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 787)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 787): v0
v_lshlrev_b32 v0, 4, v3 // (op 787) 
 // (op 787) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 787): v3

 // (op 787) BEGIN: Tile
// Allocated DataFlowTag584: 1 VGPR (Value: UInt32) (op 787): v3
v_lshl_add_u32 v3, v4, 6, v0 // (op 787) 
 // (op 787) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 787): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 787): v4

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(787) END
// Allocated : 1 VGPR (Value: UInt32) (op 840): v0
v_and_b32 v0, 63, v1 // (op 840) 
 // (op 840) END: Flatten[1]
 // (op 840) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 840): v4
v_lshrrev_b32 v4, 4, v0 // (op 840) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 840): v0

 // (op 840) BEGIN: Flatten[1]
 // (op 840) {Flatten[1]: BitwiseAnd(v4:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 840): v0
v_and_b32 v0, 1, v4 // (op 840) 
 // (op 840) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 840): v4

 // (op 840) BEGIN: Flatten[1]
 // (op 840) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 840): v4
v_and_b32 v4, 15, v1 // (op 840) 
 // (op 840) END: Flatten[1]
 // (op 840) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 840): v5
v_lshl_add_u32 v5, v0, 4, v4 // (op 840) 
 // (op 840) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 840): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 840): v0

 // (op 840) BEGIN: TileTile
 // (op 840) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 840): v0
v_and_b32 v0, 32, v1 // (op 840) 
 // (op 840) END: TileTile
 // (op 840) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 840): v4
v_lshl_add_u32 v4, v5, 6, v0 // (op 840) 
 // (op 840) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 840): v0

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 840): v5

 // (op 840) Multiply({Tile: v4:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 840): v0
v_mul_lo_u32 v0, v4, 6 // (op 840) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 840): v4

 // (op 840) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag600: 1 VGPR (Value: UInt32) (op 840): v4
v_lshrrev_b32 v4, 3, v0 // (op 840) 
 // Freeing : 1 VGPR (Value: UInt32) (op 840): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(840) END
// Allocated : 1 VGPR (Value: UInt32) (op 893): v0
v_lshrrev_b32 v0, 6, v1 // (op 893) 
 // (op 893) BEGIN: Flatten[1]
 // (op 893) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 893): v5
v_and_b32 v5, 31, v1 // (op 893) 
 // (op 893) END: Flatten[1]
 // (op 893) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 893): v6
v_lshl_add_u32 v6, v0, 5, v5 // (op 893) 
 // (op 893) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 893): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 893): v0

 // (op 893) BEGIN: Flatten[1]
 // (op 893) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 893): v0
v_and_b32 v0, 63, v1 // (op 893) 
 // (op 893) END: Flatten[1]
 // (op 893) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 893): v5
v_lshrrev_b32 v5, 5, v0 // (op 893) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 893): v0

 // (op 893) BEGIN: Tile
// Allocated DataFlowTag616: 1 VGPR (Value: UInt32) (op 893): v0
v_lshl_add_u32 v0, v6, 1, v5 // (op 893) 
 // (op 893) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 893): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 893): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(893) END
// Allocated : 1 VGPR (Value: UInt32) (op 946): v5
v_and_b32 v5, 31, v1 // (op 946) 
 // (op 946) END: Flatten[1]Tile
 // (op 946) BEGIN: Flatten[1]
 // (op 946) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 946): v6
v_and_b32 v6, 63, v1 // (op 946) 
 // (op 946) END: Flatten[1]
 // (op 946) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 946): v7
v_lshrrev_b32 v7, 5, v6 // (op 946) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 946): v6

 // (op 946) BEGIN: Tile
// Allocated DataFlowTag632: 1 VGPR (Value: UInt32) (op 946): v6
v_lshl_add_u32 v6, v5, 1, v7 // (op 946) 
 // (op 946) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 946): v7

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 946): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(946) END
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v5
v_and_b32 v5, 63, v1 // (op 1019) 
 // (op 1019) END: Flatten[1]
 // (op 1019) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v7
v_lshrrev_b32 v7, 4, v5 // (op 1019) 
 // (op 1019) BEGIN: Flatten[1]
 // (op 1019) {Flatten[1]: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v8
v_and_b32 v8, 1, v7 // (op 1019) 
 // (op 1019) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1019): v7

 // (op 1019) BEGIN: Flatten[1]
 // (op 1019) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v7
v_and_b32 v7, 15, v1 // (op 1019) 
 // (op 1019) END: Flatten[1]
 // (op 1019) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v9
v_lshl_add_u32 v9, v8, 4, v7 // (op 1019) 
 // (op 1019) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1019): v7

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1019): v8

 // (op 1019) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v7
v_lshrrev_b32 v7, 6, v1 // (op 1019) 
 // (op 1019) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v8
v_lshl_add_u32 v8, v7, 5, v9 // (op 1019) 
 // (op 1019) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1019): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1019): v7

 // (op 1019) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v7
v_lshrrev_b32 v7, 5, v5 // (op 1019) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1019): v5

 // (op 1019) BEGIN: TileTile
 // (op 1019) {TileTile: ShiftL(v7:U32, 4:U32)U32}
 // (op 1019) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1019)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 1019)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1019): v5
v_lshlrev_b32 v5, 4, v7 // (op 1019) 
 // (op 1019) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 1019): v7

 // (op 1019) BEGIN: Tile
// Allocated DataFlowTag654: 1 VGPR (Value: UInt32) (op 1019): v7
v_lshl_add_u32 v7, v8, 6, v5 // (op 1019) 
 // (op 1019) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1019): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1019): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1019) END
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v5
v_and_b32 v5, 63, v1 // (op 1043) 
 // (op 1043) END: Flatten[1]
 // (op 1043) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
 // (op 1) Deallocate{}(1317) BEGIN
 // (op 1317) Deallocate 102
 // Deleting tag 102
 // (op 1) Deallocate{}(1317) END
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v8
v_lshrrev_b32 v8, 4, v5 // (op 1043) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1043): v5

 // (op 1043) BEGIN: Flatten[1]
 // (op 1043) {Flatten[1]: BitwiseAnd(v8:U32, 1:U32)U32}
 // (op 1) Assign VGPR 64:U32(788) BEGIN
 // (op 788) Assign dim(585) = 64:U32
 // (op 788) Assign stride expression
 // (op 1) Assign VGPR 64:U32(788) END
 // (op 1) Assign VGPR 64:U32(1020) BEGIN
 // (op 1020) Assign dim(655) = 64:U32
 // (op 1020) Assign stride expression
 // (op 1) Assign VGPR 64:U32(1020) END
 // (op 1) NOP(784) BEGIN
 // (op 784) generate({})
 // (op 784) end: generate({})
 // (op 1) NOP(784) END
 // (op 1) NOP(1016) BEGIN
 // (op 1016) generate({})
 // (op 1016) end: generate({})
 // (op 1) NOP(1016) END
 // (op 1) Assign VGPR 1:U32(792) BEGIN
 // (op 792) Assign dim(587) = 1:U32
 // (op 792) Assign stride expression
 // (op 1) Assign VGPR 1:U32(792) END
 // (op 1) Assign VGPR 1:U32(1024) BEGIN
 // (op 1024) Assign dim(657) = 1:U32
 // (op 1024) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1024) END
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v5
v_and_b32 v5, 1, v8 // (op 1043) 
 // (op 1043) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1043): v8

 // (op 1043) BEGIN: Flatten[1]
 // (op 1043) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v8
v_and_b32 v8, 15, v1 // (op 1043) 
 // (op 1043) END: Flatten[1]
 // (op 1043) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v9
v_lshl_add_u32 v9, v5, 4, v8 // (op 1043) 
 // (op 1043) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1043): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1043): v5

 // (op 1043) BEGIN: TileTile
 // (op 1043) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v5
v_and_b32 v5, 32, v1 // (op 1043) 
 // (op 1043) END: TileTile
 // (op 1043) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v8
v_lshl_add_u32 v8, v9, 6, v5 // (op 1043) 
 // (op 1043) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1043): v5

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1043): v9

 // (op 1043) Multiply({Tile: v8:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1043): v5
v_mul_lo_u32 v5, v8, 6 // (op 1043) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1043): v8

 // (op 1043) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag662: 1 VGPR (Value: UInt32) (op 1043): v8
v_lshrrev_b32 v8, 3, v5 // (op 1043) 
 // Freeing : 1 VGPR (Value: UInt32) (op 1043): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1043) END
// Allocated : 1 VGPR (Value: UInt32) (op 1067): v5
v_lshrrev_b32 v5, 6, v1 // (op 1067) 
 // (op 1067) BEGIN: Flatten[1]
 // (op 1067) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Deallocate{}(1320) BEGIN
 // (op 1320) Deallocate 168
 // Deleting tag 168
 // (op 1) Deallocate{}(1320) END
// Allocated : 1 VGPR (Value: UInt32) (op 1067): v9
v_and_b32 v9, 31, v1 // (op 1067) 
 // (op 1067) END: Flatten[1]
 // (op 1067) BEGIN: Tile
 // (op 1) Assign VGPR 48:U32(841) BEGIN
 // (op 841) Assign dim(601) = 48:U32
 // (op 841) Assign stride expression
 // (op 1) Assign VGPR 48:U32(841) END
 // (op 1) Assign VGPR 48:U32(1044) BEGIN
 // (op 1044) Assign dim(663) = 48:U32
 // (op 1044) Assign stride expression
 // (op 1) Assign VGPR 48:U32(1044) END
 // (op 1) NOP(837) BEGIN
 // (op 837) generate({})
 // (op 837) end: generate({})
 // (op 1) NOP(837) END
 // (op 1) NOP(1040) BEGIN
 // (op 1040) generate({})
 // (op 1040) end: generate({})
 // (op 1) NOP(1040) END
 // (op 1) Assign VGPR 0:U32(845) BEGIN
 // (op 845) Assign dim(603) = 0:U32
 // (op 845) Assign stride expression
 // (op 1) Assign VGPR 0:U32(845) END
 // (op 1) Assign VGPR 0:U32(1048) BEGIN
 // (op 1048) Assign dim(665) = 0:U32
 // (op 1048) Assign stride expression
 // (op 1) Assign VGPR 0:U32(1048) END
// Allocated : 1 VGPR (Value: UInt32) (op 1067): v10
v_lshl_add_u32 v10, v5, 5, v9 // (op 1067) 
 // (op 1067) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1067): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1067): v5

 // (op 1067) BEGIN: Flatten[1]
 // (op 1067) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1067): v5
v_and_b32 v5, 63, v1 // (op 1067) 
 // (op 1067) END: Flatten[1]
 // (op 1067) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1067): v9
v_lshrrev_b32 v9, 5, v5 // (op 1067) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1067): v5

 // (op 1067) BEGIN: Tile
// Allocated DataFlowTag670: 1 VGPR (Value: UInt32) (op 1067): v5
v_lshl_add_u32 v5, v10, 1, v9 // (op 1067) 
 // (op 1067) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1067): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1067): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(1067) END
// Allocated : 1 VGPR (Value: UInt32) (op 1091): v9
v_and_b32 v9, 31, v1 // (op 1091) 
 // (op 1091) END: Flatten[1]Tile
 // (op 1091) BEGIN: Flatten[1]
 // (op 1091) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1323) BEGIN
 // (op 1323) Deallocate 234
 // Deleting tag 234
 // (op 1) Deallocate{}(1323) END
// Allocated : 1 VGPR (Value: UInt32) (op 1091): v10
v_and_b32 v10, 63, v1 // (op 1091) 
 // (op 1091) END: Flatten[1]
 // (op 1091) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
 // (op 1) Assign VGPR 2:U32(894) BEGIN
 // (op 894) Assign dim(617) = 2:U32
 // (op 894) Assign stride expression
 // (op 1) Assign VGPR 2:U32(894) END
 // (op 1) Assign VGPR 2:U32(1068) BEGIN
 // (op 1068) Assign dim(671) = 2:U32
 // (op 1068) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1068) END
 // (op 1) NOP(890) BEGIN
 // (op 890) generate({})
 // (op 890) end: generate({})
 // (op 1) NOP(890) END
 // (op 1) NOP(1064) BEGIN
 // (op 1064) generate({})
 // (op 1064) end: generate({})
 // (op 1) NOP(1064) END
 // (op 1) Assign VGPR 1:U32(898) BEGIN
 // (op 898) Assign dim(619) = 1:U32
 // (op 898) Assign stride expression
 // (op 1) Assign VGPR 1:U32(898) END
 // (op 1) Assign VGPR 1:U32(1072) BEGIN
 // (op 1072) Assign dim(673) = 1:U32
 // (op 1072) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1072) END
// Allocated : 1 VGPR (Value: UInt32) (op 1091): v11
v_lshrrev_b32 v11, 5, v10 // (op 1091) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1091): v10

 // (op 1091) BEGIN: Tile
// Allocated DataFlowTag678: 1 VGPR (Value: UInt32) (op 1091): v10
v_lshl_add_u32 v10, v9, 1, v11 // (op 1091) 
 // (op 1091) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1091): v11

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 1091): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(1091) END
 // (op 1) Deallocate{}(1326) BEGIN
 // (op 1326) Deallocate 294
 // Deleting tag 294
 // (op 1) Deallocate{}(1326) END
 // (op 1) Assign VGPR 2:U32(947) BEGIN
 // (op 947) Assign dim(633) = 2:U32
 // (op 947) Assign stride expression
 // (op 1) Assign VGPR 2:U32(947) END
 // (op 1) Assign VGPR 2:U32(1092) BEGIN
 // (op 1092) Assign dim(679) = 2:U32
 // (op 1092) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1092) END
 // (op 1) NOP(943) BEGIN
 // (op 943) generate({})
 // (op 943) end: generate({})
 // (op 1) NOP(943) END
 // (op 1) NOP(1088) BEGIN
 // (op 1088) generate({})
 // (op 1088) end: generate({})
 // (op 1) NOP(1088) END
 // (op 1) Assign VGPR 1:U32(951) BEGIN
 // (op 951) Assign dim(635) = 1:U32
 // (op 951) Assign stride expression
 // (op 1) Assign VGPR 1:U32(951) END
 // (op 1) Assign VGPR 1:U32(1096) BEGIN
 // (op 1096) Assign dim(681) = 1:U32
 // (op 1096) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1096) END
 // (op 1) LoadVGPR Value: Float(2) BEGIN
 // tag 44: v**UNALLOCATED**
 // (op 2) GEN: LoadVGPR; scalar value
 // (op 1) LoadVGPR Value: Float(43) BEGIN
 // tag 53: v**UNALLOCATED**
 // (op 43) GEN: LoadVGPR; scalar value
// Allocated : 1 VGPR (Value: Float) (op 2): v9
v_mov_b32 v9, s14 // (op 2) Move value
 // (op 1) LoadVGPR Value: Float(2) END
// Allocated : 1 VGPR (Value: Float) (op 43): v11
v_mov_b32 v11, s15 // (op 43) Move value
 // (op 1) LoadVGPR Value: Float(43) END
 // (op 1) Deallocate{user_Float_Value_14}(1373) BEGIN
 // (op 1373) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1373) END
 // (op 1) Deallocate{user_Float_Value_12}(1375) BEGIN
 // (op 1375) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1375) END
 // (op 1) Scope(340) BEGIN
 // (op 340) Lock Scope 340
 // Lock Branch
 // (op 340) generate({345})
 // (op 340) NOP(345) BEGIN
 // (op 345) generate({})
 // (op 345) end: generate({})
 // (op 340) NOP(345) END
 // (op 340) Assign ACCVGPR 0.00000:S(63) BEGIN
 // (op 63) Assign dim(55) = 0.00000:S
 // tag 55: a**UNALLOCATED**
 // (op 63) Generate 0.00000:S into DataFlowTag55: ACCVGPR Value: Float x 16: (unallocated)
// Allocated DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 63): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
 // (op 340) Assign ACCVGPR 0.00000:S(63) END
 // (op 340) Scope(490) BEGIN
 // (op 490) Lock Scope 490
 // Lock Branch
 // (op 490) generate({485})
 // (op 490) NOP(485) BEGIN
 // (op 485) generate({})
 // (op 485) end: generate({})
 // (op 490) NOP(485) END
 // (op 490) Assign VGPR 64:U64(495) BEGIN
 // (op 495) Assign dim(498) = 64:U64
 // (op 495) Assign stride expression
 // (op 490) Assign VGPR 64:U64(495) END
 // (op 490) NOP(486) BEGIN
 // (op 486) generate({})
 // (op 486) end: generate({})
 // (op 490) NOP(486) END
 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(498) BEGIN
 // (op 498) Assign dim(499) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 499: v**UNALLOCATED**
 // (op 498) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag499: VGPR Value: UInt64 x 1: (unallocated)
 // (op 498) reg expression
 // (op 498) reg expression
 // (op 498) reg expression
 // (op 498) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 498): v12
v_lshrrev_b32 v12, 2, v1 // (op 498) 
 // (op 498) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 498): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 498) 
 // (op 498) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 498): v12

 // (op 498) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 498): v12
v_and_b32 v12, -4, v1 // (op 498) 
 // (op 498) BEGIN: Flatten[1]
 // (op 498) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 498): v14
v_sub_u32 v14, v1, v12 // (op 498) 
 // (op 498) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 498): v12

 // (op 498) BEGIN: TileTile
 // (op 498) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 498) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 498)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 498)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 498): v12
v_lshlrev_b32 v12, 4, v14 // (op 498) 
 // (op 498) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 498): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 498): v14, v15
v_mov_b32 v15, 0 // (op 498) convert
v_mov_b32 v14, v12 // (op 498) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 498): v12

 // (op 498) BEGIN: Split
 // (op 498) Multiply: dest (DataFlowTag499: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 498)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 498)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag499: 2 VGPRs (Value: UInt64) (op 498): v16, v17
v_mul_lo_u32 v17, v13, s3 // (op 498) most significant: low of low * high
 // (op 498) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 498): v12
v_mul_hi_u32 v12, v13, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 498) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 498): v12

 // (op 498) Add: dest (DataFlowTag499: VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 498)      lhs (DataFlowTag499: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 498)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 498): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 498) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 498) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 498): s0, s1

 // (op 498) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 498): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 498): v13

 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(498) END
 // (op 490) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(499) BEGIN
 // (op 499) Assign dim(500) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 499) Assign stride expression
 // (op 490) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(499) END
 // (op 490) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(500) BEGIN
 // (op 500) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 493: s**UNALLOCATED**
 // (op 500) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: (unallocated)
 // (op 500) reg expression
 // (op 500) reg expression
 // (op 500) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 500) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 500)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 500)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 500): s0, s1
s_mul_i32 s1, s2, s7 // (op 500) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 500): s14
s_mul_i32 s14, s3, s6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 500) least significant
// Allocated : 1 SGPR (Value: Int32) (op 500): s15
s_mul_hi_u32 s15, s2, s6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 500) most significant: sum
s_add_u32 s1, s1, s15 // (op 500) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 500): s15

 // Freeing : 1 SGPR (Value: Int32) (op 500): s14

 // (op 500) reg expression
 // (op 500) reg expression
// Allocated DataFlowTag493: 4 SGPRs (Buffer: None) (op 500): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 500) reg expression
s_mov_b32 s41, s5 // (op 500) reg expression
s_mov_b32 s42, s0 // (op 500) reg expression
s_mov_b32 s43, 131072 // (op 500) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 500): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(500) END
 // (op 490) NOP(487) BEGIN
 // (op 487) generate({})
 // (op 487) end: generate({})
 // (op 490) NOP(487) END
 // (op 490) Assign VGPR 1:U64(505) BEGIN
 // (op 505) Assign dim(502) = 1:U64
 // (op 505) Assign stride expression
 // (op 490) Assign VGPR 1:U64(505) END
 // (op 490) NOP(508) BEGIN
 // (op 508) generate({})
 // (op 508) end: generate({})
 // (op 490) NOP(508) END
 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(517) BEGIN
 // (op 517) Assign dim(503) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 503: v**UNALLOCATED**
 // (op 517) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)
 // (op 517) reg expression
 // (op 517) reg expression
 // (op 517) reg expression
 // (op 517) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 517): v12
v_lshrrev_b32 v12, 2, v1 // (op 517) 
 // (op 517) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 517): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 517) 
 // (op 517) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 517): v12

 // (op 517) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 517): v12
v_and_b32 v12, -4, v1 // (op 517) 
 // (op 517) BEGIN: Flatten[1]
 // (op 517) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 517): v14
v_sub_u32 v14, v1, v12 // (op 517) 
 // (op 517) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 517): v12

 // (op 517) BEGIN: TileTile
 // (op 517) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 517) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 517)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 517)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 517): v12
v_lshlrev_b32 v12, 4, v14 // (op 517) 
 // (op 517) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 517): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 517): v14, v15
v_mov_b32 v15, 0 // (op 517) convert
v_mov_b32 v14, v12 // (op 517) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 517): v12

 // (op 517) BEGIN: Split
 // (op 517) Multiply: dest (DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 517)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 517)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag503: 2 VGPRs (Value: UInt64) (op 517): v18, v19
v_mul_lo_u32 v19, v13, s3 // (op 517) most significant: low of low * high
 // (op 517) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 517): v12
v_mul_hi_u32 v12, v13, s2 // (op 517) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 517) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 517) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 517): v12

 // (op 517) Add: dest (DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 517)      lhs (DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 517)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 517): s0, s1
v_add_co_u32 v18, s[0:1], v18, v14 // (op 517) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 517) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 517): s0, s1

 // (op 517) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 517): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 517): v13

 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(517) END
 // (op 490) Assign VGPR 64:U64(518) BEGIN
 // (op 518) Assign dim(504) = 64:U64
 // (op 518) Assign stride expression
 // (op 490) Assign VGPR 64:U64(518) END
 // (op 490) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(519) BEGIN
 // (op 519) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 519) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // (op 519) reg expression
 // (op 519) reg expression
 // (op 519) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 519) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 519)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 519)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 519): s0, s1
s_mul_i32 s1, s2, s7 // (op 519) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 519): s14
s_mul_i32 s14, s3, s6 // (op 519) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 519) least significant
// Allocated : 1 SGPR (Value: Int32) (op 519): s15
s_mul_hi_u32 s15, s2, s6 // (op 519) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 519) most significant: sum
s_add_u32 s1, s1, s15 // (op 519) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 519): s15

 // Freeing : 1 SGPR (Value: Int32) (op 519): s14

 // (op 519) reg expression
 // (op 519) reg expression
s_mov_b32 s40, s4 // (op 519) reg expression
s_mov_b32 s41, s5 // (op 519) reg expression
s_mov_b32 s42, s0 // (op 519) reg expression
s_mov_b32 s43, 131072 // (op 519) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 519): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(519) END
 // (op 490) NOP(510) BEGIN
 // (op 510) generate({})
 // (op 510) end: generate({})
 // (op 490) NOP(510) END
 // (op 490) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(524) BEGIN
 // (op 524) Assign dim(506) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 524) Assign stride expression
 // (op 490) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(524) END
 // (op 490) NOP(511) BEGIN
 // (op 511) generate({})
 // (op 511) end: generate({})
 // (op 490) NOP(511) END
 // (op 490) Assign VGPR 1:U64(527) BEGIN
 // (op 527) Assign dim(508) = 1:U64
 // (op 527) Assign stride expression
 // (op 490) Assign VGPR 1:U64(527) END
 // (op 490) NOP(552) BEGIN
 // (op 552) generate({})
 // (op 552) end: generate({})
 // (op 490) NOP(552) END
 // (op 490) Assign VGPR 48:U64(559) BEGIN
 // (op 559) Assign dim(517) = 48:U64
 // (op 559) Assign stride expression
 // (op 490) Assign VGPR 48:U64(559) END
 // (op 490) NOP(553) BEGIN
 // (op 553) generate({})
 // (op 553) end: generate({})
 // (op 490) NOP(553) END
 // (op 490) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(562) BEGIN
 // (op 562) Assign dim(518) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 518: v**UNALLOCATED**
 // (op 562) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag518: VGPR Value: UInt32 x 1: (unallocated)
 // (op 562) reg expression
 // (op 562) reg expression
 // (op 562) reg expression
 // (op 562) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 562): v12
v_and_b32 v12, -2, v1 // (op 562) 
 // (op 562) BEGIN: Flatten[1]
 // (op 562) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 562): v13
v_sub_u32 v13, v1, v12 // (op 562) 
 // (op 562) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 562): v12

 // (op 562) BEGIN: TileTile
 // (op 562) {TileTile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 562) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 562)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 562)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 562): v12
v_lshlrev_b32 v12, 5, v13 // (op 562) 
 // (op 562) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 562): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 562): v14, v15
v_mov_b32 v15, 0 // (op 562) convert
v_mov_b32 v14, v12 // (op 562) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 562): v12

 // (op 562) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 562): v12
v_lshrrev_b32 v12, 1, v1 // (op 562) 
 // (op 562) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 562): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 562) 
 // (op 562) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 562): v12

 // (op 562) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 562) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 562)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 562)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 562): v20, v21
v_mul_lo_u32 v21, v13, s9 // (op 562) most significant: low of low * high
 // (op 562) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 562): v12
v_mul_hi_u32 v12, v13, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 562) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 562) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 562): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 562): v13

 // (op 562) BEGIN: Split
 // (op 562) {Split: Add({TileTile: v[14:15]:U64}, v[20:21]:I64)U64}
 // (op 562) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 562)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 562)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated : 2 VGPRs (Value: UInt64) (op 562): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 562): s0, s1
v_add_co_u32 v12, s[0:1], v14, v20 // (op 562) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 562) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 562): s0, s1

 // (op 562) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 562): v20, v21

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 562): v14, v15

 // (op 562) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 562) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 562)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 562)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 562) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 562): v14, v15
v_mul_lo_u32 v15, v13, 6 // (op 562) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 562): v20
v_mul_hi_u32 v20, v12, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 562) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 562) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 562): v20

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 562): v12, v13

 // (op 562) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 562): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 562) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 562): v14, v15

// Allocated DataFlowTag518: 1 VGPR (Value: UInt32) (op 562): v14
v_mov_b32 v14, v12 // (op 562) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 562): v12, v13

 // (op 490) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(562) END
 // (op 490) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(563) BEGIN
 // (op 563) Assign dim(519) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 563) Assign stride expression
 // (op 490) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(563) END
 // (op 490) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(564) BEGIN
 // (op 564) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 512: s**UNALLOCATED**
 // (op 564) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: (unallocated)
 // (op 564) reg expression
 // (op 564) reg expression
 // (op 564) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 564) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 564)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 564)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 564) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 564): s0, s1
s_mul_i32 s1, s13, 6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 564) least significant
// Allocated : 1 SGPR (Value: Int32) (op 564): s14
s_mul_hi_u32 s14, s12, 6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 564) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 564): s14

 // (op 564) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 564) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 564)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 564)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 564): s14, s15
s_mul_i32 s15, s8, s1 // (op 564) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 564): s18
s_mul_i32 s18, s9, s0 // (op 564) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 564) least significant
// Allocated : 1 SGPR (Value: Int32) (op 564): s19
s_mul_hi_u32 s19, s8, s0 // (op 564) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 564) most significant: sum
s_add_u32 s15, s15, s19 // (op 564) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 564): s19

 // Freeing : 1 SGPR (Value: Int32) (op 564): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 564): s0, s1

 // (op 564) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 564): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 564) 
 // Freeing : 2 SGPRs (Value: Int64) (op 564): s14, s15

 // (op 564) reg expression
 // (op 564) reg expression
// Allocated DataFlowTag512: 4 SGPRs (Buffer: None) (op 564): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 564) reg expression
s_mov_b32 s45, s11 // (op 564) reg expression
s_mov_b32 s46, s0 // (op 564) reg expression
s_mov_b32 s47, 131072 // (op 564) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 564): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(564) END
 // (op 490) NOP(554) BEGIN
 // (op 554) generate({})
 // (op 554) end: generate({})
 // (op 490) NOP(554) END
 // (op 490) Assign VGPR 0:U64(569) BEGIN
 // (op 569) Assign dim(521) = 0:U64
 // (op 569) Assign stride expression
 // (op 490) Assign VGPR 0:U64(569) END
 // (op 490) NOP(572) BEGIN
 // (op 572) generate({})
 // (op 572) end: generate({})
 // (op 490) NOP(572) END
 // (op 490) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(581) BEGIN
 // (op 581) Assign dim(522) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 522: v**UNALLOCATED**
 // (op 581) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag522: VGPR Value: UInt32 x 1: (unallocated)
 // (op 581) reg expression
 // (op 581) reg expression
 // (op 581) reg expression
 // (op 581) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 581): v15
v_and_b32 v15, -2, v1 // (op 581) 
 // (op 581) BEGIN: Flatten[1]
 // (op 581) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 581): v12
v_sub_u32 v12, v1, v15 // (op 581) 
 // (op 581) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 581): v15

 // (op 581) BEGIN: TileTile
 // (op 581) {TileTile: ShiftL({Flatten[1]: v12:U32}, 5:U32)U32}
 // (op 581) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 581)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 581)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 581): v13
v_lshlrev_b32 v13, 5, v12 // (op 581) 
 // (op 581) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 581): v12

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 581): v20, v21
v_mov_b32 v21, 0 // (op 581) convert
v_mov_b32 v20, v13 // (op 581) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 581): v13

 // (op 581) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 581): v15
v_lshrrev_b32 v15, 1, v1 // (op 581) 
 // (op 581) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 581): v12
v_lshl_add_u32 v12, s17, 5, v15 // (op 581) 
 // (op 581) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 581): v15

 // (op 581) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 581) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 581)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 581)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 581): v22, v23
v_mul_lo_u32 v23, v12, s9 // (op 581) most significant: low of low * high
 // (op 581) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 581): v13
v_mul_hi_u32 v13, v12, s8 // (op 581) most significant: high of low * low
v_mul_lo_u32 v22, v12, s8 // (op 581) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 581) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 581): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 581): v12

 // (op 581) BEGIN: Split
 // (op 581) {Split: Add({TileTile: v[20:21]:U64}, v[22:23]:I64)U64}
 // (op 581) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 581)      lhs (TileTile: VGPR Value: UInt64 x 1: v[20:21]) 
 // (op 581)      rhs (VGPR Value: Int64 x 1: v[22:23])
// Allocated : 2 VGPRs (Value: UInt64) (op 581): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 581): s0, s1
v_add_co_u32 v12, s[0:1], v20, v22 // (op 581) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 581) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 581): s0, s1

 // (op 581) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 581): v22, v23

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 581): v20, v21

 // (op 581) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 581) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 581)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 581)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 581) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 581): v20, v21
v_mul_lo_u32 v21, v13, 6 // (op 581) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 581): v15
v_mul_hi_u32 v15, v12, 6 // (op 581) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 581) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 581) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 581): v15

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 581): v12, v13

 // (op 581) LogicalShiftR(v[20:21]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 581): v12, v13
v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 581) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 581): v20, v21

// Allocated DataFlowTag522: 1 VGPR (Value: UInt32) (op 581): v15
v_mov_b32 v15, v12 // (op 581) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 581): v12, v13

 // (op 490) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(581) END
 // (op 490) Assign VGPR 48:U64(582) BEGIN
 // (op 582) Assign dim(523) = 48:U64
 // (op 582) Assign stride expression
 // (op 490) Assign VGPR 48:U64(582) END
 // (op 490) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(583) BEGIN
 // (op 583) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 583) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // (op 583) reg expression
 // (op 583) reg expression
 // (op 583) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 583) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 583)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 583)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 583) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 583): s0, s1
s_mul_i32 s1, s13, 6 // (op 583) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 583) least significant
// Allocated : 1 SGPR (Value: Int32) (op 583): s14
s_mul_hi_u32 s14, s12, 6 // (op 583) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 583) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 583): s14

 // (op 583) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 583) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 583)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 583)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 583): s14, s15
s_mul_i32 s15, s8, s1 // (op 583) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 583): s18
s_mul_i32 s18, s9, s0 // (op 583) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 583) least significant
// Allocated : 1 SGPR (Value: Int32) (op 583): s19
s_mul_hi_u32 s19, s8, s0 // (op 583) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 583) most significant: sum
s_add_u32 s15, s15, s19 // (op 583) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 583): s19

 // Freeing : 1 SGPR (Value: Int32) (op 583): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 583): s0, s1

 // (op 583) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 583): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 583) 
 // Freeing : 2 SGPRs (Value: Int64) (op 583): s14, s15

 // (op 583) reg expression
 // (op 583) reg expression
s_mov_b32 s44, s10 // (op 583) reg expression
s_mov_b32 s45, s11 // (op 583) reg expression
s_mov_b32 s46, s0 // (op 583) reg expression
s_mov_b32 s47, 131072 // (op 583) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 583): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(583) END
 // (op 490) NOP(574) BEGIN
 // (op 574) generate({})
 // (op 574) end: generate({})
 // (op 490) NOP(574) END
 // (op 490) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(588) BEGIN
 // (op 588) Assign dim(525) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 588) Assign stride expression
 // (op 490) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(588) END
 // (op 490) NOP(575) BEGIN
 // (op 575) generate({})
 // (op 575) end: generate({})
 // (op 490) NOP(575) END
 // (op 490) Assign VGPR 0:U64(591) BEGIN
 // (op 591) Assign dim(527) = 0:U64
 // (op 591) Assign stride expression
 // (op 490) Assign VGPR 0:U64(591) END
 // (op 490) NOP(616) BEGIN
 // (op 616) generate({})
 // (op 616) end: generate({})
 // (op 490) NOP(616) END
 // (op 490) Assign VGPR 2:U64(623) BEGIN
 // (op 623) Assign dim(536) = 2:U64
 // (op 623) Assign stride expression
 // (op 490) Assign VGPR 2:U64(623) END
 // (op 490) NOP(617) BEGIN
 // (op 617) generate({})
 // (op 617) end: generate({})
 // (op 490) NOP(617) END
 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(626) BEGIN
 // (op 626) Assign dim(537) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}
 // tag 537: v**UNALLOCATED**
 // (op 626) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32} into DataFlowTag537: VGPR Value: UInt32 x 1: (unallocated)
 // (op 626) reg expression
 // (op 626) reg expression
 // (op 626) reg expression
 // (op 626) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 626) convert
// Allocated : 1 VGPR (Value: UInt32) (op 626): v12
v_lshrrev_b32 v12, 1, v1 // (op 626) 
 // (op 626) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 626): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 626) 
 // (op 626) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 626): v12

 // (op 626) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 626): v12
v_and_b32 v12, -2, v1 // (op 626) 
 // (op 626) BEGIN: Flatten[1]TileTile
 // (op 626) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 626): v20
v_sub_u32 v20, v1, v12 // (op 626) 
 // (op 626) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 626): v12

 // (op 626) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag537: 1 VGPR (Value: UInt32) (op 626): v12
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v12, v13, s20 // (op 626) 
v_add_u32 v12, v12, v20 // (op 626) 
 // (op 626) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 626): v20

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 626): v13

 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(626) END
 // (op 490) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(627) BEGIN
 // (op 627) Assign dim(538) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 627) Assign stride expression
 // (op 490) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(627) END
 // (op 490) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(628) BEGIN
 // (op 628) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 531: s**UNALLOCATED**
 // (op 628) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: (unallocated)
 // (op 628) reg expression
 // (op 628) reg expression
 // (op 628) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 628) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 628)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 628)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 628): s0, s1
s_mul_i32 s1, s20, s7 // (op 628) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 628): s14
s_mul_i32 s14, s21, s6 // (op 628) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 628) least significant
// Allocated : 1 SGPR (Value: Int32) (op 628): s15
s_mul_hi_u32 s15, s20, s6 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 628): s15

 // Freeing : 1 SGPR (Value: Int32) (op 628): s14

 // (op 628) reg expression
 // (op 628) reg expression
// Allocated DataFlowTag531: 4 SGPRs (Buffer: None) (op 628): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 628) reg expression
s_mov_b32 s49, s23 // (op 628) reg expression
s_mov_b32 s50, s0 // (op 628) reg expression
s_mov_b32 s51, 131072 // (op 628) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 628): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(628) END
 // (op 490) NOP(618) BEGIN
 // (op 618) generate({})
 // (op 618) end: generate({})
 // (op 490) NOP(618) END
 // (op 490) Assign VGPR 1:U64(633) BEGIN
 // (op 633) Assign dim(540) = 1:U64
 // (op 633) Assign stride expression
 // (op 490) Assign VGPR 1:U64(633) END
 // (op 490) NOP(636) BEGIN
 // (op 636) generate({})
 // (op 636) end: generate({})
 // (op 490) NOP(636) END
 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(645) BEGIN
 // (op 645) Assign dim(541) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}
 // tag 541: v**UNALLOCATED**
 // (op 645) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32} into DataFlowTag541: VGPR Value: UInt32 x 1: (unallocated)
 // (op 645) reg expression
 // (op 645) reg expression
 // (op 645) reg expression
 // (op 645) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 645) convert
// Allocated : 1 VGPR (Value: UInt32) (op 645): v13
v_lshrrev_b32 v13, 1, v1 // (op 645) 
 // (op 645) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 645): v20
v_lshl_add_u32 v20, s16, 5, v13 // (op 645) 
 // (op 645) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 645): v13

 // (op 645) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 645): v13
v_and_b32 v13, -2, v1 // (op 645) 
 // (op 645) BEGIN: Flatten[1]TileTile
 // (op 645) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 645): v21
v_sub_u32 v21, v1, v13 // (op 645) 
 // (op 645) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 645): v13

 // (op 645) BEGIN: Split
// Allocated DataFlowTag541: 1 VGPR (Value: UInt32) (op 645): v13
v_mul_lo_u32 v13, v20, s20 // (op 645) 
v_add_u32 v13, v13, v21 // (op 645) 
 // (op 645) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 645): v21

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 645): v20

 // (op 490) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(645) END
 // (op 490) Assign VGPR 2:U64(646) BEGIN
 // (op 646) Assign dim(542) = 2:U64
 // (op 646) Assign stride expression
 // (op 490) Assign VGPR 2:U64(646) END
 // (op 490) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(647) BEGIN
 // (op 647) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 647) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // (op 647) reg expression
 // (op 647) reg expression
 // (op 647) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 647) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 647)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 647)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 647): s0, s1
s_mul_i32 s1, s20, s7 // (op 647) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 647): s14
s_mul_i32 s14, s21, s6 // (op 647) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 647) least significant
// Allocated : 1 SGPR (Value: Int32) (op 647): s15
s_mul_hi_u32 s15, s20, s6 // (op 647) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 647) most significant: sum
s_add_u32 s1, s1, s15 // (op 647) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 647): s15

 // Freeing : 1 SGPR (Value: Int32) (op 647): s14

 // (op 647) reg expression
 // (op 647) reg expression
s_mov_b32 s48, s22 // (op 647) reg expression
s_mov_b32 s49, s23 // (op 647) reg expression
s_mov_b32 s50, s0 // (op 647) reg expression
s_mov_b32 s51, 131072 // (op 647) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 647): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(647) END
 // (op 490) NOP(638) BEGIN
 // (op 638) generate({})
 // (op 638) end: generate({})
 // (op 490) NOP(638) END
 // (op 490) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(652) BEGIN
 // (op 652) Assign dim(544) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 652) Assign stride expression
 // (op 490) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(652) END
 // (op 490) NOP(639) BEGIN
 // (op 639) generate({})
 // (op 639) end: generate({})
 // (op 490) NOP(639) END
 // (op 490) Assign VGPR 1:U64(655) BEGIN
 // (op 655) Assign dim(546) = 1:U64
 // (op 655) Assign stride expression
 // (op 490) Assign VGPR 1:U64(655) END
 // (op 490) NOP(680) BEGIN
 // (op 680) generate({})
 // (op 680) end: generate({})
 // (op 490) NOP(680) END
 // (op 490) Assign VGPR 2:U64(687) BEGIN
 // (op 687) Assign dim(555) = 2:U64
 // (op 687) Assign stride expression
 // (op 490) Assign VGPR 2:U64(687) END
 // (op 490) NOP(681) BEGIN
 // (op 681) generate({})
 // (op 681) end: generate({})
 // (op 490) NOP(681) END
 // (op 490) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(690) BEGIN
 // (op 690) Assign dim(556) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 556: v**UNALLOCATED**
 // (op 690) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag556: VGPR Value: UInt32 x 1: (unallocated)
 // (op 690) reg expression
 // (op 690) reg expression
 // (op 690) reg expression
 // (op 690) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 690) convert
// Allocated : 1 VGPR (Value: UInt32) (op 690): v20
v_and_b32 v20, -2, v1 // (op 690) 
 // (op 690) BEGIN: Flatten[1]TileTile
 // (op 690) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 690): v21
v_sub_u32 v21, v1, v20 // (op 690) 
 // (op 690) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 690): v20

 // (op 690) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 690): v20
v_lshrrev_b32 v20, 1, v1 // (op 690) 
 // (op 690) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 690): v22
v_lshl_add_u32 v22, s17, 5, v20 // (op 690) 
 // (op 690) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 690): v20

 // (op 690) Multiply({Tile: v22:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 690): v20
v_mul_lo_u32 v20, v22, s24 // (op 690) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 690): v22

 // (op 690) BEGIN: Split
 // (op 690) {Split: Add({Flatten[1]TileTile: v21:U32}, v20:U32)U32}
// Allocated DataFlowTag556: 1 VGPR (Value: UInt32) (op 690): v22
v_add_u32 v22, v21, v20 // (op 690) 
 // (op 690) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 690): v20

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 690): v21

 // (op 490) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(690) END
 // (op 490) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(691) BEGIN
 // (op 691) Assign dim(557) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 691) Assign stride expression
 // (op 490) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(691) END
 // (op 490) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(692) BEGIN
 // (op 692) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 550: s**UNALLOCATED**
 // (op 692) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: (unallocated)
 // (op 692) reg expression
 // (op 692) reg expression
 // (op 692) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 692) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 692)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 692)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 692): s0, s1
s_mul_i32 s1, s24, s13 // (op 692) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 692): s14
s_mul_i32 s14, s25, s12 // (op 692) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 692) least significant
// Allocated : 1 SGPR (Value: Int32) (op 692): s15
s_mul_hi_u32 s15, s24, s12 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 692) most significant: sum
s_add_u32 s1, s1, s15 // (op 692) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 692): s15

 // Freeing : 1 SGPR (Value: Int32) (op 692): s14

 // (op 692) reg expression
 // (op 692) reg expression
// Allocated DataFlowTag550: 4 SGPRs (Buffer: None) (op 692): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 692) reg expression
s_mov_b32 s53, s27 // (op 692) reg expression
s_mov_b32 s54, s0 // (op 692) reg expression
s_mov_b32 s55, 131072 // (op 692) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 692): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(692) END
 // (op 490) NOP(682) BEGIN
 // (op 682) generate({})
 // (op 682) end: generate({})
 // (op 490) NOP(682) END
 // (op 490) Assign VGPR 1:U64(697) BEGIN
 // (op 697) Assign dim(559) = 1:U64
 // (op 697) Assign stride expression
 // (op 490) Assign VGPR 1:U64(697) END
 // (op 490) NOP(700) BEGIN
 // (op 700) generate({})
 // (op 700) end: generate({})
 // (op 490) NOP(700) END
 // (op 490) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(709) BEGIN
 // (op 709) Assign dim(560) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 560: v**UNALLOCATED**
 // (op 709) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 709) reg expression
 // (op 709) reg expression
 // (op 709) reg expression
 // (op 709) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 709) convert
// Allocated : 1 VGPR (Value: UInt32) (op 709): v20
v_and_b32 v20, -2, v1 // (op 709) 
 // (op 709) BEGIN: Flatten[1]TileTile
 // (op 709) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 709): v21
v_sub_u32 v21, v1, v20 // (op 709) 
 // (op 709) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v20

 // (op 709) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 709): v20
v_lshrrev_b32 v20, 1, v1 // (op 709) 
 // (op 709) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 709): v23
v_lshl_add_u32 v23, s17, 5, v20 // (op 709) 
 // (op 709) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v20

 // (op 709) Multiply({Tile: v23:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 709): v20
v_mul_lo_u32 v20, v23, s24 // (op 709) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 709): v23

 // (op 709) BEGIN: Split
 // (op 709) {Split: Add({Flatten[1]TileTile: v21:U32}, v20:U32)U32}
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 709): v23
v_add_u32 v23, v21, v20 // (op 709) 
 // (op 709) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 709): v20

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 709): v21

 // (op 490) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(709) END
 // (op 490) Assign VGPR 2:U64(710) BEGIN
 // (op 710) Assign dim(561) = 2:U64
 // (op 710) Assign stride expression
 // (op 490) Assign VGPR 2:U64(710) END
 // (op 490) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(711) BEGIN
 // (op 711) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 711) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // (op 711) reg expression
 // (op 711) reg expression
 // (op 711) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 711) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 711)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 711)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 711): s0, s1
s_mul_i32 s1, s24, s13 // (op 711) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 711): s14
s_mul_i32 s14, s25, s12 // (op 711) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 711) least significant
// Allocated : 1 SGPR (Value: Int32) (op 711): s15
s_mul_hi_u32 s15, s24, s12 // (op 711) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 711) most significant: sum
s_add_u32 s1, s1, s15 // (op 711) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 711): s15

 // Freeing : 1 SGPR (Value: Int32) (op 711): s14

 // (op 711) reg expression
 // (op 711) reg expression
s_mov_b32 s52, s26 // (op 711) reg expression
s_mov_b32 s53, s27 // (op 711) reg expression
s_mov_b32 s54, s0 // (op 711) reg expression
s_mov_b32 s55, 131072 // (op 711) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 711): s0, s1

 // (op 490) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(711) END
 // (op 490) NOP(702) BEGIN
 // (op 702) generate({})
 // (op 702) end: generate({})
 // (op 490) NOP(702) END
 // (op 490) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(716) BEGIN
 // (op 716) Assign dim(563) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 716) Assign stride expression
 // (op 490) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(716) END
 // (op 490) NOP(703) BEGIN
 // (op 703) generate({})
 // (op 703) end: generate({})
 // (op 490) NOP(703) END
 // (op 490) Assign VGPR 1:U64(719) BEGIN
 // (op 719) Assign dim(565) = 1:U64
 // (op 719) Assign stride expression
 // (op 490) Assign VGPR 1:U64(719) END
 // (op 490) NOP(360) BEGIN
 // (op 360) generate({})
 // (op 360) end: generate({})
 // (op 490) NOP(360) END
 // (op 490) SetCoordinate(365) BEGIN
 // (op 365) generate({})
 // (op 365) end: generate({})
 // (op 365) generate({366})
 // (op 365) LoadTiled Value: BF6x16(366) BEGIN
 // (op 366) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 366) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 366) Generate 0:U64 into nullptr
 // tag 9: v**UNALLOCATED**
 // (op 366) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 366) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag518: VGPR Value: UInt32 x 1: v14
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 366) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: BF6x16) (op 366): v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[24:27], v14, s[44:47], 0 offen // (op 366) Load value
// VMEM: Expected complete at 811 (current 331)
buffer_load_dwordx2 v[28:29], v14, s[44:47], 0 offen offset:16 // (op 366) Load value
// VMEM: Expected complete at 812 (current 332)
 // (op 365) LoadTiled Value: BF6x16(366) END
 // (op 365) end: generate({366})
 // (op 490) SetCoordinate(365) END
 // (op 490) SetCoordinate(368) BEGIN
 // (op 368) generate({})
 // (op 368) end: generate({})
 // (op 368) generate({369})
 // (op 368) LoadTiled Value: E8M0(369) BEGIN
 // (op 369) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 369) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 369) Generate 1:U64 into nullptr
 // tag 11: v**UNALLOCATED**
 // (op 369) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 369) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v22
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 369) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 369): v20
buffer_load_ubyte v20, v22, s[52:55], 0 offen offset: 0 // (op 369) Load value
// VMEM: Expected complete at 813 (current 333)
 // (op 368) LoadTiled Value: E8M0(369) END
 // (op 368) end: generate({369})
 // (op 490) SetCoordinate(368) END
 // (op 490) SetCoordinate(371) BEGIN
 // (op 371) generate({})
 // (op 371) end: generate({})
 // (op 371) generate({771})
 // (op 371) Scope(771) BEGIN
 // (op 771) Lock Scope 771
 // Lock Branch
 // (op 771) generate({768})
 // (op 771) NOP(768) BEGIN
 // (op 768) generate({})
 // (op 768) end: generate({})
 // (op 771) NOP(768) END
 // (op 771) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(775) BEGIN
 // (op 775) Assign dim(580) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 580: s**UNALLOCATED**
 // (op 775) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag580: SGPR Value: UInt32 x 1: (unallocated)
 // (op 775) reg expression
 // (op 775) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 775): v21
v_and_b32 v21, -4, v1 // (op 775) 
 // (op 775) BEGIN: Tile[1]
 // (op 775) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v21:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 775): v30
v_sub_u32 v30, v1, v21 // (op 775) 
 // (op 775) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 775): v21

 // (op 775) BEGIN: Flatten
 // (op 775) {Flatten: ShiftL({Tile[1]: v30:U32}, 4:U32)U32}
 // (op 775) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 775)         value (Tile[1]: VGPR Value: UInt32 x 1: v30) 
 // (op 775)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 775): v21
v_lshlrev_b32 v21, 4, v30 // (op 775) 
 // (op 775) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 775): v30

 // (op 775) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 775): v30
v_lshrrev_b32 v30, 2, v1 // (op 775) 
 // (op 775) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 775): v31
v_lshl_add_u32 v31, v30, 6, v21 // (op 775) 
 // (op 775) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 775): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 775): v30

// Allocated DataFlowTag580: 1 SGPR (Value: UInt32) (op 775): s0
s_nop 0
v_readfirstlane_b32 s0, v31 // (op 775) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 775): v31

 // (op 771) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(775) END
 // (op 771) Assign VGPR 1024:U32(776) BEGIN
 // (op 776) Assign dim(581) = 1024:U32
 // (op 776) Assign stride expression
 // (op 771) Assign VGPR 1024:U32(776) END
 // (op 771) NOP(769) BEGIN
 // (op 769) generate({})
 // (op 769) end: generate({})
 // (op 771) NOP(769) END
 // (op 771) Assign VGPR 1:U32(780) BEGIN
 // (op 780) Assign dim(583) = 1:U32
 // (op 780) Assign stride expression
 // (op 771) Assign VGPR 1:U32(780) END
 // (op 771) LoadTileDirect2LDS Value: BF8x4(454) BEGIN
 // (op 454) GEN: LoadTileDirect2LDS
 // (op 454) GEN: loadMacroTileDirect2LDS OP 454 LDS 17 MacroTile 8
 // (op 454) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 454) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride500: VGPR Value: UInt64 x 1: (unallocated)
 // (op 454) reg expression
 // (op 454) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 454) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 454)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 454)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 454): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 454) 
// Allocated Stride500: 2 VGPRs (Value: UInt64) (op 454): v30, v31
v_mov_b64 v[30:31], s[14:15] // (op 454) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 454): s14, s15

 // (op 454) Generate 1:U64 into nullptr
 // (op 454) Generate 1024:U32 into nullptr
 // (op 454) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride500: VGPR Value: UInt64 x 1: v[30:31]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 454): v32, v33
v_mov_b64 v[32:33], v[16:17] // (op 454) 
 // (op 454) Generate Add(0:U32, {DataFlowTag580Flatten: s0:U32})U32 into nullptr
 // (op 454)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride500: v[30:31]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 454) Generate {DataFlowTag580Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 454) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds
// VMEM: Expected stall of 467. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1292 (current 812)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 454) Generate Add(v[32:33]:U64, v30:R)U64 into VGPR Value: UInt64 x 1: v[32:33]
 // (op 454) reg expression
 // (op 454) reg expression
 // (op 454) Add(v[32:33]:U64, v30:R)U64
 // (op 454) Add: dest (VGPR Value: UInt64 x 1: v[32:33]) = 
 // (op 454)      lhs (VGPR Value: UInt64 x 1: v[32:33]) 
 // (op 454)      rhs (VGPR Value: Raw32 x 1: v30)
// Allocated : 2 SGPRs (Value: Bool64) (op 454): s14, s15
v_add_co_u32 v32, s[14:15], v32, v30 // (op 454) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 454) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 454): s14, s15

 // (op 454) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 454) reg expression
 // (op 454) Add(m0:U32, 1024:U32)U32
 // (op 454) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 454) 
 // (op 454) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 454) Load value direct to lds
// VMEM: Expected complete at 1296 (current 816)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride500: 2 VGPRs (Value: UInt64) (op 454): v30, v31

 // Freeing : 2 VGPRs (Value: UInt64) (op 454): v32, v33

 // (op 771) LoadTileDirect2LDS Value: BF8x4(454) END
 // (op 771) Deallocate{}(1311) BEGIN
 // (op 1311) Deallocate 580
 // Freeing DataFlowTag580: 1 SGPR (Value: UInt32) (op 775): s0

 // Deleting tag 580
 // (op 1311) Deallocate 581
 // Deleting tag 581
 // (op 1311) Deallocate 583
 // Deleting tag 583
 // (op 1311) Deallocate 582
 // Deleting tag 582
 // (op 771) Deallocate{}(1311) END
 // (op 771) end: generate({768})
 // (op 771) Unlock Scope 771
 // Unlock None
 // (op 371) Scope(771) END
 // (op 371) end: generate({771})
 // (op 490) SetCoordinate(371) END
 // (op 490) SetCoordinate(374) BEGIN
 // (op 374) generate({})
 // (op 374) end: generate({})
 // (op 374) generate({375})
 // (op 374) LoadTiled Value: E8M0(375) BEGIN
 // (op 375) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 375) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 375) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 375) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 375) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag537: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 375) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 375): v21
buffer_load_ubyte v21, v12, s[48:51], 0 offen offset: 0 // (op 375) Load value
// VMEM: Expected complete at 1297 (current 817)
 // (op 374) LoadTiled Value: E8M0(375) END
 // (op 374) end: generate({375})
 // (op 490) SetCoordinate(374) END
 // (op 490) SetCoordinate(377) BEGIN
 // (op 377) generate({})
 // (op 377) end: generate({})
 // (op 377) generate({378})
 // (op 377) LoadTiled Value: BF6x16(378) BEGIN
 // (op 378) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 378) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 378) Generate 0:U64 into nullptr
 // tag 470: v**UNALLOCATED**
 // (op 378) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 378) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag518: VGPR Value: UInt32 x 1: v14
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 48
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 378) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: BF6x16) (op 378): v30, v31, v32, v33, v34, v35
buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:48 // (op 378) Load value
// VMEM: Expected stall of 475. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1773 (current 1293)
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen offset:64 // (op 378) Load value
// VMEM: Expected stall of 3. CBNW: 4, Inc: 3
// VMEM: Expected complete at 1777 (current 1297)
 // (op 377) LoadTiled Value: BF6x16(378) END
 // (op 377) end: generate({378})
 // (op 490) SetCoordinate(377) END
 // (op 490) Deallocate{}(1291) BEGIN
 // (op 1291) Deallocate 519
 // Deleting tag 519
 // (op 1291) Deallocate 521
 // Deleting tag 521
 // (op 1291) Deallocate 518
 // Freeing DataFlowTag518: 1 VGPR (Value: UInt32) (op 562): v14

 // Deleting tag 518
 // (op 1291) Deallocate 520
 // Deleting tag 520
 // (op 1291) Deallocate 517
 // Deleting tag 517
 // (op 490) Deallocate{}(1291) END
 // (op 490) SetCoordinate(380) BEGIN
 // (op 380) generate({})
 // (op 380) end: generate({})
 // (op 380) generate({381})
 // (op 380) LoadTiled Value: E8M0(381) BEGIN
 // (op 381) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 381) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 381) Generate 1:U64 into nullptr
 // tag 478: v**UNALLOCATED**
 // (op 381) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 381) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v22
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 381) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 381): v14
buffer_load_ubyte v14, v22, s[52:55], 0 offen offset:2 // (op 381) Load value
// VMEM: Expected complete at 1778 (current 1298)
 // (op 380) LoadTiled Value: E8M0(381) END
 // (op 380) end: generate({381})
 // (op 490) SetCoordinate(380) END
 // (op 490) Deallocate{}(1293) BEGIN
 // (op 1293) Deallocate 558
 // Deleting tag 558
 // (op 1293) Deallocate 556
 // Freeing DataFlowTag556: 1 VGPR (Value: UInt32) (op 690): v22

 // Deleting tag 556
 // (op 1293) Deallocate 555
 // Deleting tag 555
 // (op 1293) Deallocate 557
 // Deleting tag 557
 // (op 1293) Deallocate 559
 // Deleting tag 559
 // (op 490) Deallocate{}(1293) END
 // (op 490) SetCoordinate(383) BEGIN
 // (op 383) generate({})
 // (op 383) end: generate({})
 // (op 383) generate({1003})
 // (op 383) Scope(1003) BEGIN
 // (op 1003) Lock Scope 1003
 // Lock Branch
 // (op 1003) generate({1000})
 // (op 1003) NOP(1000) BEGIN
 // (op 1000) generate({})
 // (op 1000) end: generate({})
 // (op 1003) NOP(1000) END
 // (op 1003) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1007) BEGIN
 // (op 1007) Assign dim(650) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 650: s**UNALLOCATED**
 // (op 1007) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag650: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1007) reg expression
 // (op 1007) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1007): v22
v_and_b32 v22, -4, v1 // (op 1007) 
 // (op 1007) BEGIN: Tile[1]
 // (op 1007) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1007): v36
v_sub_u32 v36, v1, v22 // (op 1007) 
 // (op 1007) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1007): v22

 // (op 1007) BEGIN: Flatten
 // (op 1007) {Flatten: ShiftL({Tile[1]: v36:U32}, 4:U32)U32}
 // (op 1007) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1007)         value (Tile[1]: VGPR Value: UInt32 x 1: v36) 
 // (op 1007)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1007): v22
v_lshlrev_b32 v22, 4, v36 // (op 1007) 
 // (op 1007) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1007): v36

 // (op 1007) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1007): v36
v_lshrrev_b32 v36, 2, v1 // (op 1007) 
 // (op 1007) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1007): v37
v_lshl_add_u32 v37, v36, 6, v22 // (op 1007) 
 // (op 1007) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1007): v22

 // Freeing : 1 VGPR (Value: UInt32) (op 1007): v36

// Allocated DataFlowTag650: 1 SGPR (Value: UInt32) (op 1007): s0
s_nop 0
v_readfirstlane_b32 s0, v37 // (op 1007) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1007): v37

 // (op 1003) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1007) END
 // (op 1003) Assign VGPR 1024:U32(1008) BEGIN
 // (op 1008) Assign dim(651) = 1024:U32
 // (op 1008) Assign stride expression
 // (op 1003) Assign VGPR 1024:U32(1008) END
 // (op 1003) NOP(1001) BEGIN
 // (op 1001) generate({})
 // (op 1001) end: generate({})
 // (op 1003) NOP(1001) END
 // (op 1003) Assign VGPR 1:U32(1012) BEGIN
 // (op 1012) Assign dim(653) = 1:U32
 // (op 1012) Assign stride expression
 // (op 1003) Assign VGPR 1:U32(1012) END
 // (op 1003) LoadTileDirect2LDS Value: BF8x4(459) BEGIN
 // (op 459) GEN: LoadTileDirect2LDS
 // (op 459) GEN: loadMacroTileDirect2LDS OP 459 LDS 468 MacroTile 466
 // (op 459) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 459) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride500: VGPR Value: UInt64 x 1: (unallocated)
 // (op 459) reg expression
 // (op 459) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 459) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 459)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 459)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 459): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 459) 
// Allocated Stride500: 2 VGPRs (Value: UInt64) (op 459): v36, v37
v_mov_b64 v[36:37], s[14:15] // (op 459) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 459): s14, s15

 // (op 459) Generate 1:U64 into nullptr
 // (op 459) Generate 1024:U32 into nullptr
 // (op 459) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride500: VGPR Value: UInt64 x 1: v[36:37]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 459): v38, v39
v_mov_b64 v[38:39], v[16:17] // (op 459) 
 // (op 459) Generate Add(2048:U32, {DataFlowTag650Flatten: s0:U32})U32 into nullptr
 // (op 459) reg expression
 // (op 459) Add(2048:U32, {DataFlowTag650Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 459): s1
s_add_u32 s1, 2048, s0 // (op 459) 
 // (op 459)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride500: v[36:37]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 459) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 459) reg expression
buffer_load_dwordx4 v38, s[40:43], 64 offen offset: 0 lds // (op 459) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2254 (current 1774)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 459) Generate Add(v[38:39]:U64, v36:R)U64 into VGPR Value: UInt64 x 1: v[38:39]
 // (op 459) reg expression
 // (op 459) reg expression
 // (op 459) Add(v[38:39]:U64, v36:R)U64
 // (op 459) Add: dest (VGPR Value: UInt64 x 1: v[38:39]) = 
 // (op 459)      lhs (VGPR Value: UInt64 x 1: v[38:39]) 
 // (op 459)      rhs (VGPR Value: Raw32 x 1: v36)
// Allocated : 2 SGPRs (Value: Bool64) (op 459): s14, s15
v_add_co_u32 v38, s[14:15], v38, v36 // (op 459) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 459) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 459): s14, s15

 // (op 459) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 459) reg expression
 // (op 459) Add(m0:U32, 1024:U32)U32
 // (op 459) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 459) 
 // (op 459) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v38, s[40:43], 64 offen offset: 0 lds // (op 459) Load value direct to lds
// VMEM: Expected complete at 2258 (current 1778)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing Stride500: 2 VGPRs (Value: UInt64) (op 459): v36, v37

 // Freeing : 2 VGPRs (Value: UInt64) (op 459): v38, v39

 // Freeing : 1 SGPR (Value: UInt32) (op 459): s1

 // (op 1003) LoadTileDirect2LDS Value: BF8x4(459) END
 // (op 1003) Deallocate{}(1313) BEGIN
 // (op 1313) Deallocate 650
 // Freeing DataFlowTag650: 1 SGPR (Value: UInt32) (op 1007): s0

 // Deleting tag 650
 // (op 1313) Deallocate 652
 // Deleting tag 652
 // (op 1313) Deallocate 651
 // Deleting tag 651
 // (op 1313) Deallocate 653
 // Deleting tag 653
 // (op 1003) Deallocate{}(1313) END
 // (op 1003) end: generate({1000})
 // (op 1003) Unlock Scope 1003
 // Unlock None
 // (op 383) Scope(1003) END
 // (op 383) end: generate({1003})
 // (op 490) SetCoordinate(383) END
 // (op 490) Deallocate{}(1295) BEGIN
 // (op 1295) Deallocate 500
 // Deleting tag 500
 // (op 1295) Deallocate 498
 // Deleting tag 498
 // (op 1295) Deallocate 499
 // Freeing DataFlowTag499: 2 VGPRs (Value: UInt64) (op 498): v16, v17

 // Deleting tag 499
 // (op 1295) Deallocate 502
 // Deleting tag 502
 // (op 1295) Deallocate 501
 // Deleting tag 501
 // (op 490) Deallocate{}(1295) END
 // (op 490) SetCoordinate(386) BEGIN
 // (op 386) generate({})
 // (op 386) end: generate({})
 // (op 386) generate({387})
 // (op 386) LoadTiled Value: E8M0(387) BEGIN
 // (op 387) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 387) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 387) Generate 1:U64 into nullptr
 // tag 474: v**UNALLOCATED**
 // (op 387) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 387) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag537: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 387) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 387): v22
buffer_load_ubyte v22, v12, s[48:51], 0 offen offset:2 // (op 387) Load value
// VMEM: Expected complete at 2259 (current 1779)
 // (op 386) LoadTiled Value: E8M0(387) END
 // (op 386) end: generate({387})
 // (op 490) SetCoordinate(386) END
 // (op 490) Deallocate{}(1297) BEGIN
 // (op 1297) Deallocate 538
 // Deleting tag 538
 // (op 1297) Deallocate 536
 // Deleting tag 536
 // (op 1297) Deallocate 537
 // Freeing DataFlowTag537: 1 VGPR (Value: UInt32) (op 626): v12

 // Deleting tag 537
 // (op 1297) Deallocate 540
 // Deleting tag 540
 // (op 1297) Deallocate 539
 // Deleting tag 539
 // (op 490) Deallocate{}(1297) END
 // (op 490) Scope(823) BEGIN
 // (op 823) Lock Scope 823
 // Lock Branch
 // (op 823) generate({820})
 // (op 823) NOP(820) BEGIN
 // (op 820) generate({})
 // (op 820) end: generate({})
 // (op 823) NOP(820) END
 // (op 823) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(828) BEGIN
 // (op 828) Assign dim(596) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 596: v**UNALLOCATED**
 // (op 828) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag596: VGPR Value: UInt32 x 1: (unallocated)
 // (op 828) reg expression
 // (op 828) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 828): v12
v_and_b32 v12, -2, v1 // (op 828) 
 // (op 828) BEGIN: Tile[1]
 // (op 828) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 828): v16
v_sub_u32 v16, v1, v12 // (op 828) 
 // (op 828) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 828): v12

 // (op 828) BEGIN: Flatten
 // (op 828) {Flatten: ShiftL({Tile[1]: v16:U32}, 5:U32)U32}
 // (op 828) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 828)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 828)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 828): v12
v_lshlrev_b32 v12, 5, v16 // (op 828) 
 // (op 828) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 828): v16

 // (op 828) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 828): v16
v_lshrrev_b32 v16, 1, v1 // (op 828) 
 // (op 828) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 828): v17
v_lshl_add_u32 v17, v16, 6, v12 // (op 828) 
 // (op 828) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 828): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 828): v16

 // (op 828) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 828): v12
v_mul_lo_u32 v12, v17, 6 // (op 828) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 828): v17

 // (op 828) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag596: 1 VGPR (Value: UInt32) (op 828): v16
v_lshrrev_b32 v16, 3, v12 // (op 828) 
 // Freeing : 1 VGPR (Value: UInt32) (op 828): v12

 // (op 823) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(828) END
 // (op 823) Assign VGPR 48:U32(829) BEGIN
 // (op 829) Assign dim(597) = 48:U32
 // (op 829) Assign stride expression
 // (op 823) Assign VGPR 48:U32(829) END
 // (op 823) NOP(821) BEGIN
 // (op 821) generate({})
 // (op 821) end: generate({})
 // (op 823) NOP(821) END
 // (op 823) Assign VGPR 0:U32(833) BEGIN
 // (op 833) Assign dim(599) = 0:U32
 // (op 833) Assign stride expression
 // (op 823) Assign VGPR 0:U32(833) END
 // (op 823) StoreLDSTile Value: BF6x16(389) BEGIN
 // (op 389) GEN: StoreLDSTile
 // (op 389) GEN: storeMacroTileLDS OP 389 LDS 25 MacroTile 9
 // (op 389) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 389) Generate 0:U32 into nullptr
 // (op 389) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 389) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag596: VGPR Value: UInt32 x 1: v16
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 389) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 389) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 1883 (current 1788)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 389) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 1885 (current 1790)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 823) StoreLDSTile Value: BF6x16(389) END
 // (op 823) Deallocate{}(1299) BEGIN
 // (op 1299) Deallocate 599
 // Deleting tag 599
 // (op 1299) Deallocate 597
 // Deleting tag 597
 // (op 1299) Deallocate 596
 // Freeing DataFlowTag596: 1 VGPR (Value: UInt32) (op 828): v16

 // Deleting tag 596
 // (op 1299) Deallocate 598
 // Deleting tag 598
 // (op 823) Deallocate{}(1299) END
 // (op 823) end: generate({820})
 // (op 823) Unlock Scope 823
 // Unlock None
 // (op 490) Scope(823) END
 // (op 490) Scope(929) BEGIN
 // (op 929) Lock Scope 929
 // Lock Branch
 // (op 929) generate({926})
 // (op 929) NOP(926) BEGIN
 // (op 926) generate({})
 // (op 926) end: generate({})
 // (op 929) NOP(926) END
 // (op 929) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(934) BEGIN
 // (op 934) Assign dim(628) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 628: v**UNALLOCATED**
 // (op 934) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag628: VGPR Value: UInt32 x 1: (unallocated)
 // (op 934) reg expression
 // (op 934) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 934): v12
v_and_b32 v12, -2, v1 // (op 934) 
 // (op 934) BEGIN: Tile[1]Flatten
 // (op 934) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 934): v16
v_sub_u32 v16, v1, v12 // (op 934) 
 // (op 934) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 934): v12

 // (op 934) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 934): v12
v_and_b32 v12, 4294967294, v1 // (op 934) 
 // (op 934) BEGIN: Flatten
 // (op 934) {Flatten: Add(v12:U32, {Tile[1]Flatten: v16:U32})U32}
// Allocated DataFlowTag628: 1 VGPR (Value: UInt32) (op 934): v17
v_add_u32 v17, v12, v16 // (op 934) 
 // (op 934) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 934): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 934): v12

 // (op 929) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(934) END
 // (op 929) Assign VGPR 2:U32(935) BEGIN
 // (op 935) Assign dim(629) = 2:U32
 // (op 935) Assign stride expression
 // (op 929) Assign VGPR 2:U32(935) END
 // (op 929) NOP(927) BEGIN
 // (op 927) generate({})
 // (op 927) end: generate({})
 // (op 929) NOP(927) END
 // (op 929) Assign VGPR 1:U32(939) BEGIN
 // (op 939) Assign dim(631) = 1:U32
 // (op 939) Assign stride expression
 // (op 929) Assign VGPR 1:U32(939) END
 // (op 929) StoreLDSTile Value: E8M0(390) BEGIN
 // (op 390) GEN: StoreLDSTile
 // (op 390) GEN: storeMacroTileLDS OP 390 LDS 41 MacroTile 11
 // (op 390) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 390) Generate 1:U32 into nullptr
 // (op 390) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 390) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = DataFlowTag628: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 390) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v17, v20 offset:5632 // (op 390) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1891 (current 1796)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 929) StoreLDSTile Value: E8M0(390) END
 // (op 929) Deallocate{}(1301) BEGIN
 // (op 1301) Deallocate 630
 // Deleting tag 630
 // (op 1301) Deallocate 629
 // Deleting tag 629
 // (op 1301) Deallocate 631
 // Deleting tag 631
 // (op 1301) Deallocate 628
 // Freeing DataFlowTag628: 1 VGPR (Value: UInt32) (op 934): v17

 // Deleting tag 628
 // (op 929) Deallocate{}(1301) END
 // (op 929) end: generate({926})
 // (op 929) Unlock Scope 929
 // Unlock None
 // (op 490) Scope(929) END
 // (op 490) NOP(456) BEGIN
 // (op 456) generate({})
 // (op 456) end: generate({})
 // (op 490) NOP(456) END
 // (op 490) Scope(876) BEGIN
 // (op 876) Lock Scope 876
 // Lock Branch
 // (op 876) generate({873})
 // (op 876) NOP(873) BEGIN
 // (op 873) generate({})
 // (op 873) end: generate({})
 // (op 876) NOP(873) END
 // (op 876) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(881) BEGIN
 // (op 881) Assign dim(612) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 612: v**UNALLOCATED**
 // (op 881) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag612: VGPR Value: UInt32 x 1: (unallocated)
 // (op 881) reg expression
 // (op 881) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 881): v12
v_and_b32 v12, -2, v1 // (op 881) 
 // (op 881) BEGIN: Tile[1]Flatten
 // (op 881) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 881): v16
v_sub_u32 v16, v1, v12 // (op 881) 
 // (op 881) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 881): v12

 // (op 881) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 881): v12
v_and_b32 v12, 4294967294, v1 // (op 881) 
 // (op 881) BEGIN: Flatten
 // (op 881) {Flatten: Add(v12:U32, {Tile[1]Flatten: v16:U32})U32}
// Allocated DataFlowTag612: 1 VGPR (Value: UInt32) (op 881): v17
v_add_u32 v17, v12, v16 // (op 881) 
 // (op 881) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 881): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 881): v12

 // (op 876) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(881) END
 // (op 876) Assign VGPR 2:U32(882) BEGIN
 // (op 882) Assign dim(613) = 2:U32
 // (op 882) Assign stride expression
 // (op 876) Assign VGPR 2:U32(882) END
 // (op 876) NOP(874) BEGIN
 // (op 874) generate({})
 // (op 874) end: generate({})
 // (op 876) NOP(874) END
 // (op 876) Assign VGPR 1:U32(886) BEGIN
 // (op 886) Assign dim(615) = 1:U32
 // (op 886) Assign stride expression
 // (op 876) Assign VGPR 1:U32(886) END
 // (op 876) StoreLDSTile Value: E8M0(392) BEGIN
 // (op 392) GEN: StoreLDSTile
 // (op 392) GEN: storeMacroTileLDS OP 392 LDS 33 MacroTile 10
 // (op 392) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 392) Generate 1:U32 into nullptr
 // (op 392) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 392) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = DataFlowTag612: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 392) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v17, v21 offset:5696 // (op 392) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 83. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1980 (current 1885)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 876) StoreLDSTile Value: E8M0(392) END
 // (op 876) Deallocate{}(1303) BEGIN
 // (op 1303) Deallocate 615
 // Deleting tag 615
 // (op 1303) Deallocate 612
 // Freeing DataFlowTag612: 1 VGPR (Value: UInt32) (op 881): v17

 // Deleting tag 612
 // (op 1303) Deallocate 613
 // Deleting tag 613
 // (op 1303) Deallocate 614
 // Deleting tag 614
 // (op 876) Deallocate{}(1303) END
 // (op 876) end: generate({873})
 // (op 876) Unlock Scope 876
 // Unlock None
 // (op 490) Scope(876) END
 // (op 490) NOP(363) BEGIN
 // (op 363) generate({})
 // (op 363) end: generate({})
 // (op 490) NOP(363) END
 // (op 490) NOP(743) BEGIN
 // (op 743) generate({})
 // (op 743) end: generate({})
 // (op 490) NOP(743) END
 // (op 490) NOP(740) BEGIN
 // (op 740) generate({})
 // (op 740) end: generate({})
 // (op 490) NOP(740) END
 // (op 490) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(747) BEGIN
 // (op 747) Assign dim(572) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 572: s**UNALLOCATED**
 // (op 747) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag572: SGPR Value: UInt32 x 1: (unallocated)
 // (op 747) reg expression
 // (op 747) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 747): v12
v_and_b32 v12, -4, v1 // (op 747) 
 // (op 747) BEGIN: Tile[1]
 // (op 747) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 747): v16
v_sub_u32 v16, v1, v12 // (op 747) 
 // (op 747) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 747): v12

 // (op 747) BEGIN: Flatten
 // (op 747) {Flatten: ShiftL({Tile[1]: v16:U32}, 4:U32)U32}
 // (op 747) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 747)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 747)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 747): v12
v_lshlrev_b32 v12, 4, v16 // (op 747) 
 // (op 747) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 747): v16

 // (op 747) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 747): v16
v_lshrrev_b32 v16, 2, v1 // (op 747) 
 // (op 747) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 747): v17
v_lshl_add_u32 v17, v16, 6, v12 // (op 747) 
 // (op 747) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 747): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 747): v16

// Allocated DataFlowTag572: 1 SGPR (Value: UInt32) (op 747): s0
s_nop 0
v_readfirstlane_b32 s0, v17 // (op 747) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 747): v17

 // (op 490) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(747) END
 // (op 490) Assign VGPR 1024:U32(748) BEGIN
 // (op 748) Assign dim(573) = 1024:U32
 // (op 748) Assign stride expression
 // (op 490) Assign VGPR 1024:U32(748) END
 // (op 490) NOP(741) BEGIN
 // (op 741) generate({})
 // (op 741) end: generate({})
 // (op 490) NOP(741) END
 // (op 490) Assign VGPR 1:U32(752) BEGIN
 // (op 752) Assign dim(575) = 1:U32
 // (op 752) Assign stride expression
 // (op 490) Assign VGPR 1:U32(752) END
 // (op 490) NOP(794) BEGIN
 // (op 794) generate({})
 // (op 794) end: generate({})
 // (op 490) NOP(794) END
 // (op 490) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(799) BEGIN
 // (op 799) Assign dim(588) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 588: v**UNALLOCATED**
 // (op 799) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag588: VGPR Value: UInt32 x 1: (unallocated)
 // (op 799) reg expression
 // (op 799) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 799): v12
v_and_b32 v12, -2, v1 // (op 799) 
 // (op 799) BEGIN: Tile[1]
 // (op 799) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 799): v16
v_sub_u32 v16, v1, v12 // (op 799) 
 // (op 799) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 799): v12

 // (op 799) BEGIN: Flatten
 // (op 799) {Flatten: ShiftL({Tile[1]: v16:U32}, 5:U32)U32}
 // (op 799) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 799)         value (Tile[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 799)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 799): v12
v_lshlrev_b32 v12, 5, v16 // (op 799) 
 // (op 799) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 799): v16

 // (op 799) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 799): v16
v_lshrrev_b32 v16, 1, v1 // (op 799) 
 // (op 799) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 799): v17
v_lshl_add_u32 v17, v16, 6, v12 // (op 799) 
 // (op 799) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 799): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 799): v16

 // (op 799) Multiply({Flatten: v17:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 799): v12
v_mul_lo_u32 v12, v17, 6 // (op 799) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 799): v17

 // (op 799) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag588: 1 VGPR (Value: UInt32) (op 799): v16
v_lshrrev_b32 v16, 3, v12 // (op 799) 
 // Freeing : 1 VGPR (Value: UInt32) (op 799): v12

 // (op 490) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(799) END
 // (op 490) Assign VGPR 48:U32(800) BEGIN
 // (op 800) Assign dim(589) = 48:U32
 // (op 800) Assign stride expression
 // (op 490) Assign VGPR 48:U32(800) END
 // (op 490) NOP(795) BEGIN
 // (op 795) generate({})
 // (op 795) end: generate({})
 // (op 490) NOP(795) END
 // (op 490) Assign VGPR 0:U32(804) BEGIN
 // (op 804) Assign dim(591) = 0:U32
 // (op 804) Assign stride expression
 // (op 490) Assign VGPR 0:U32(804) END
 // (op 490) NOP(847) BEGIN
 // (op 847) generate({})
 // (op 847) end: generate({})
 // (op 490) NOP(847) END
 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(852) BEGIN
 // (op 852) Assign dim(604) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 604: v**UNALLOCATED**
 // (op 852) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag604: VGPR Value: UInt32 x 1: (unallocated)
 // (op 852) reg expression
 // (op 852) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 852): v12
v_and_b32 v12, -2, v1 // (op 852) 
 // (op 852) BEGIN: Tile[1]Flatten
 // (op 852) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 852): v17
v_sub_u32 v17, v1, v12 // (op 852) 
 // (op 852) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 852): v12

 // (op 852) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 852): v12
v_and_b32 v12, 4294967294, v1 // (op 852) 
 // (op 852) BEGIN: Flatten
 // (op 852) {Flatten: Add(v12:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag604: 1 VGPR (Value: UInt32) (op 852): v36
v_add_u32 v36, v12, v17 // (op 852) 
 // (op 852) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 852): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 852): v12

 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(852) END
 // (op 490) Assign VGPR 2:U32(853) BEGIN
 // (op 853) Assign dim(605) = 2:U32
 // (op 853) Assign stride expression
 // (op 490) Assign VGPR 2:U32(853) END
 // (op 490) NOP(848) BEGIN
 // (op 848) generate({})
 // (op 848) end: generate({})
 // (op 490) NOP(848) END
 // (op 490) Assign VGPR 1:U32(857) BEGIN
 // (op 857) Assign dim(607) = 1:U32
 // (op 857) Assign stride expression
 // (op 490) Assign VGPR 1:U32(857) END
 // (op 490) NOP(900) BEGIN
 // (op 900) generate({})
 // (op 900) end: generate({})
 // (op 490) NOP(900) END
 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(905) BEGIN
 // (op 905) Assign dim(620) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 620: v**UNALLOCATED**
 // (op 905) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag620: VGPR Value: UInt32 x 1: (unallocated)
 // (op 905) reg expression
 // (op 905) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 905): v12
v_and_b32 v12, -2, v1 // (op 905) 
 // (op 905) BEGIN: Tile[1]Flatten
 // (op 905) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 905): v17
v_sub_u32 v17, v1, v12 // (op 905) 
 // (op 905) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 905): v12

 // (op 905) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 905): v12
v_and_b32 v12, 4294967294, v1 // (op 905) 
 // (op 905) BEGIN: Flatten
 // (op 905) {Flatten: Add(v12:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag620: 1 VGPR (Value: UInt32) (op 905): v37
v_add_u32 v37, v12, v17 // (op 905) 
 // (op 905) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 905): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 905): v12

 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(905) END
 // (op 490) Assign VGPR 2:U32(906) BEGIN
 // (op 906) Assign dim(621) = 2:U32
 // (op 906) Assign stride expression
 // (op 490) Assign VGPR 2:U32(906) END
 // (op 490) NOP(901) BEGIN
 // (op 901) generate({})
 // (op 901) end: generate({})
 // (op 490) NOP(901) END
 // (op 490) Assign VGPR 1:U32(910) BEGIN
 // (op 910) Assign dim(623) = 1:U32
 // (op 910) Assign stride expression
 // (op 490) Assign VGPR 1:U32(910) END
 // (op 490) NOP(987) BEGIN
 // (op 987) generate({})
 // (op 987) end: generate({})
 // (op 490) NOP(987) END
 // (op 490) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(992) BEGIN
 // (op 992) Assign dim(646) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 646: s**UNALLOCATED**
 // (op 992) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag646: SGPR Value: UInt32 x 1: (unallocated)
 // (op 992) reg expression
 // (op 992) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 992): v12
v_and_b32 v12, -4, v1 // (op 992) 
 // (op 992) BEGIN: Tile[1]
 // (op 992) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 992): v17
v_sub_u32 v17, v1, v12 // (op 992) 
 // (op 992) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 992): v12

 // (op 992) BEGIN: Flatten
 // (op 992) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 992) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 992)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 992)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 992): v12
v_lshlrev_b32 v12, 4, v17 // (op 992) 
 // (op 992) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 992): v17

 // (op 992) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 992): v17
v_lshrrev_b32 v17, 2, v1 // (op 992) 
 // (op 992) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 992): v38
v_lshl_add_u32 v38, v17, 6, v12 // (op 992) 
 // (op 992) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 992): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 992): v17

// Allocated DataFlowTag646: 1 SGPR (Value: UInt32) (op 992): s1
s_nop 0
v_readfirstlane_b32 s1, v38 // (op 992) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 992): v38

 // (op 490) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(992) END
 // (op 490) Assign VGPR 1024:U32(993) BEGIN
 // (op 993) Assign dim(647) = 1024:U32
 // (op 993) Assign stride expression
 // (op 490) Assign VGPR 1024:U32(993) END
 // (op 490) NOP(988) BEGIN
 // (op 988) generate({})
 // (op 988) end: generate({})
 // (op 490) NOP(988) END
 // (op 490) Assign VGPR 1:U32(997) BEGIN
 // (op 997) Assign dim(649) = 1:U32
 // (op 997) Assign stride expression
 // (op 490) Assign VGPR 1:U32(997) END
 // (op 490) NOP(1026) BEGIN
 // (op 1026) generate({})
 // (op 1026) end: generate({})
 // (op 490) NOP(1026) END
 // (op 490) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1031) BEGIN
 // (op 1031) Assign dim(658) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 658: v**UNALLOCATED**
 // (op 1031) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag658: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1031) reg expression
 // (op 1031) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v12
v_and_b32 v12, -2, v1 // (op 1031) 
 // (op 1031) BEGIN: Tile[1]
 // (op 1031) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v17
v_sub_u32 v17, v1, v12 // (op 1031) 
 // (op 1031) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1031): v12

 // (op 1031) BEGIN: Flatten
 // (op 1031) {Flatten: ShiftL({Tile[1]: v17:U32}, 5:U32)U32}
 // (op 1031) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1031)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 1031)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v12
v_lshlrev_b32 v12, 5, v17 // (op 1031) 
 // (op 1031) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1031): v17

 // (op 1031) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v17
v_lshrrev_b32 v17, 1, v1 // (op 1031) 
 // (op 1031) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v38
v_lshl_add_u32 v38, v17, 6, v12 // (op 1031) 
 // (op 1031) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1031): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 1031): v17

 // (op 1031) Multiply({Flatten: v38:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1031): v12
v_mul_lo_u32 v12, v38, 6 // (op 1031) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1031): v38

 // (op 1031) LogicalShiftR(v12:U32, 3:U32)U32
// Allocated DataFlowTag658: 1 VGPR (Value: UInt32) (op 1031): v17
v_lshrrev_b32 v17, 3, v12 // (op 1031) 
 // Freeing : 1 VGPR (Value: UInt32) (op 1031): v12

 // (op 490) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1031) END
 // (op 490) Assign VGPR 48:U32(1032) BEGIN
 // (op 1032) Assign dim(659) = 48:U32
 // (op 1032) Assign stride expression
 // (op 490) Assign VGPR 48:U32(1032) END
 // (op 490) NOP(1027) BEGIN
 // (op 1027) generate({})
 // (op 1027) end: generate({})
 // (op 490) NOP(1027) END
 // (op 490) Assign VGPR 0:U32(1036) BEGIN
 // (op 1036) Assign dim(661) = 0:U32
 // (op 1036) Assign stride expression
 // (op 490) Assign VGPR 0:U32(1036) END
 // (op 490) NOP(1050) BEGIN
 // (op 1050) generate({})
 // (op 1050) end: generate({})
 // (op 490) NOP(1050) END
 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(1055) BEGIN
 // (op 1055) Assign dim(666) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 666: v**UNALLOCATED**
 // (op 1055) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag666: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1055) reg expression
 // (op 1055) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1055): v12
v_and_b32 v12, -2, v1 // (op 1055) 
 // (op 1055) BEGIN: Tile[1]Flatten
 // (op 1055) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1055): v38
v_sub_u32 v38, v1, v12 // (op 1055) 
 // (op 1055) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1055): v12

 // (op 1055) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1055): v12
v_and_b32 v12, 4294967294, v1 // (op 1055) 
 // (op 1055) BEGIN: Flatten
 // (op 1055) {Flatten: Add(v12:U32, {Tile[1]Flatten: v38:U32})U32}
// Allocated DataFlowTag666: 1 VGPR (Value: UInt32) (op 1055): v39
v_add_u32 v39, v12, v38 // (op 1055) 
 // (op 1055) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1055): v38

 // Freeing : 1 VGPR (Value: UInt32) (op 1055): v12

 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(1055) END
 // (op 490) Assign VGPR 2:U32(1056) BEGIN
 // (op 1056) Assign dim(667) = 2:U32
 // (op 1056) Assign stride expression
 // (op 490) Assign VGPR 2:U32(1056) END
 // (op 490) NOP(1051) BEGIN
 // (op 1051) generate({})
 // (op 1051) end: generate({})
 // (op 490) NOP(1051) END
 // (op 490) Assign VGPR 1:U32(1060) BEGIN
 // (op 1060) Assign dim(669) = 1:U32
 // (op 1060) Assign stride expression
 // (op 490) Assign VGPR 1:U32(1060) END
 // (op 490) NOP(1074) BEGIN
 // (op 1074) generate({})
 // (op 1074) end: generate({})
 // (op 490) NOP(1074) END
 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(1079) BEGIN
 // (op 1079) Assign dim(674) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 674: v**UNALLOCATED**
 // (op 1079) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag674: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1079) reg expression
 // (op 1079) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1079): v12
v_and_b32 v12, -2, v1 // (op 1079) 
 // (op 1079) BEGIN: Tile[1]Flatten
 // (op 1079) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1079): v38
v_sub_u32 v38, v1, v12 // (op 1079) 
 // (op 1079) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1079): v12

 // (op 1079) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1079): v12
v_and_b32 v12, 4294967294, v1 // (op 1079) 
 // (op 1079) BEGIN: Flatten
 // (op 1079) {Flatten: Add(v12:U32, {Tile[1]Flatten: v38:U32})U32}
// Allocated DataFlowTag674: 1 VGPR (Value: UInt32) (op 1079): v40
v_add_u32 v40, v12, v38 // (op 1079) 
 // (op 1079) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1079): v38

 // Freeing : 1 VGPR (Value: UInt32) (op 1079): v12

 // (op 490) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(1079) END
 // (op 490) Assign VGPR 2:U32(1080) BEGIN
 // (op 1080) Assign dim(675) = 2:U32
 // (op 1080) Assign stride expression
 // (op 490) Assign VGPR 2:U32(1080) END
 // (op 490) NOP(1075) BEGIN
 // (op 1075) generate({})
 // (op 1075) end: generate({})
 // (op 490) NOP(1075) END
 // (op 490) Assign VGPR 1:U32(1084) BEGIN
 // (op 1084) Assign dim(677) = 1:U32
 // (op 1084) Assign stride expression
 // (op 490) Assign VGPR 1:U32(1084) END
 // (op 490) ForLoopOp KLoop: LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(58) BEGIN
 // (op 58) Initialize For Loop
 // (op 58) generate({59})
 // (op 58) Assign SGPR 0:U32(59) BEGIN
 // (op 59) Assign dim(395) = 0:U32
 // tag 395: s**UNALLOCATED**
 // (op 59) Generate 0:U32 into DataFlowTag395: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag395: 1 SGPR (Value: UInt32) (op 59): s14
s_mov_b32 s14, 0 // (op 59) call()
 // (op 58) Assign SGPR 0:U32(59) END
 // (op 58) end: generate({59})
 // (op 58) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 58) 
 // (op 58) Generate LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 58) LessThan({DataFlowTag395: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 58) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 58) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 58): s18, s19
s_ashr_i64 s[18:19], s[28:29], 6 // (op 58) 
// Allocated : 1 SGPR (Value: UInt32) (op 58): s15
s_mov_b32 s15, s18 // (op 58) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 58): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 58): s18
s_and_b32 s18, s15, 4294967294 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s15

s_cmp_lt_u32 s14, s18 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 58) 
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:
 // (op 58) 
 // (op 58) generate({60, 288, 1128})
 // (op 58) Assign SGPR Add(DataFlowTag(395)U32, 2:U32)U32(60) BEGIN
 // (op 60) Assign dim(395) = Add(DataFlowTag(395)U32, 2:U32)U32
 // (op 60) Generate Add(DataFlowTag(395)U32, 2:U32)U32 into DataFlowTag395: SGPR Value: UInt32 x 1: s14
 // (op 60) reg expression
 // (op 60) Add({DataFlowTag395: s14:U32}, 2:U32)U32
 // (op 58) SetCoordinate(288) BEGIN
 // (op 288) generate({})
 // (op 288) end: generate({})
 // (op 288) generate({10})
 // (op 288) LoadTiled Value: BF6x16(10) BEGIN
 // (op 10) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 10) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 10) Generate 0:U64 into nullptr
 // (op 10) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 524: v**UNALLOCATED**
 // (op 58) Barrier(1128) BEGIN
s_barrier  // (op 1128) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 58) Barrier(1128) END
s_add_u32 s14, s14, 2 // (op 60) 
 // (op 58) Assign SGPR Add(DataFlowTag(395)U32, 2:U32)U32(60) END
 // (op 58) SetCoordinate(68) BEGIN
 // (op 68) generate({})
 // (op 68) end: generate({})
 // (op 68) generate({70})
 // (op 68) LoadLDSTile{Value: BF8x4}(70) BEGIN
 // (op 70) GEN: loadMacroTileWAVELDS OP 70 LDS 17 WaveTile 103
 // (op 70) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 70) Generate 1:U32 into nullptr
 // tag 15: v**UNALLOCATED**
 // (op 70) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 70) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 70) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated Offset524: 1 VGPR (Value: UInt32) (op 10): v12
v_mov_b32 v12, v15 // (op 10) 
 // (op 10) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = Offset524: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 96
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 10) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 8 VGPRs (Value: BF8x4) (op 70): v42, v43, v44, v45, v46, v47, v48, v49
ds_read_b128 v[42:45], v3 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2035 (current 1940)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[46:49], v3 offset:32 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2036 (current 1941)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 68) LoadLDSTile{Value: BF8x4}(70) END
 // (op 68) end: generate({70})
 // (op 58) SetCoordinate(68) END
buffer_load_dwordx4 v[24:27], v12, s[44:47], 0 offen offset:96 // (op 10) Load value
// VMEM: Expected stall of 313. CBNW: 3, Inc: 3
// VMEM: Expected complete at 2735 (current 2255)
 // (op 58) Barrier(1141) BEGIN
buffer_load_dwordx2 v[28:29], v12, s[44:47], 0 offen offset:112 // (op 10) Load value
// VMEM: Expected stall of 3. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2739 (current 2259)
 // (op 288) LoadTiled Value: BF6x16(10) END
 // (op 288) end: generate({10})
 // (op 58) SetCoordinate(288) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1141) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 58) Barrier(1141) END
 // (op 58) SetCoordinate(298) BEGIN
 // (op 298) generate({})
 // (op 298) end: generate({})
 // (op 298) generate({25})
 // (op 298) LoadTiled Value: E8M0(25) BEGIN
 // (op 25) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 25) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 25) Generate 1:U64 into nullptr
 // (op 25) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 562: v**UNALLOCATED**
 // (op 58) SetCoordinate(73) BEGIN
 // (op 73) generate({})
 // (op 73) end: generate({})
 // (op 73) generate({75})
 // (op 73) LoadLDSTile{Value: BF6x16}(75) BEGIN
 // (op 75) GEN: loadMacroTileWAVELDS OP 75 LDS 25 WaveTile 169
 // (op 75) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 75) Generate 0:U32 into nullptr
 // tag 23: v**UNALLOCATED**
 // (op 75) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 75) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 75) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated Offset562: 1 VGPR (Value: UInt32) (op 25): v38
v_mov_b32 v38, v23 // (op 25) 
 // (op 25) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = Offset562: VGPR Value: UInt32 x 1: v38
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 25) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v20, v38, s[52:55], 0 offen offset:4 // (op 25) Load value
// VMEM: Expected complete at 2743 (current 2263)
 // (op 298) LoadTiled Value: E8M0(25) END
 // (op 298) end: generate({25})
 // (op 58) SetCoordinate(298) END
// Allocated : 6 VGPRs (Value: BF6x16) (op 75): v50, v51, v52, v53, v54, v55
ds_read_b128 v[50:53], v4 offset:4096 // (op 75) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 2359 (current 2264)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 58) Barrier(1131) BEGIN
ds_read_b64 v[54:55], v4 offset:4112 // (op 75) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 2360 (current 2265)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 73) LoadLDSTile{Value: BF6x16}(75) END
 // (op 73) end: generate({75})
 // (op 58) SetCoordinate(73) END
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1131) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 58) Barrier(1131) END
 // (op 58) Barrier(1155) BEGIN
 // (op 58) SetCoordinate(283) BEGIN
 // (op 283) generate({})
 // (op 283) end: generate({})
 // (op 283) generate({438})
 // (op 283) LoadTileDirect2LDS Value: BF8x4(438) BEGIN
 // (op 438) GEN: LoadTileDirect2LDS
 // (op 438) GEN: loadMacroTileDirect2LDS OP 438 LDS 17 MacroTile 8
 // (op 438) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 438) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride506: VGPR Value: UInt64 x 1: (unallocated)
 // (op 438) reg expression
 // (op 438) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 438) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 438)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 438)         shiftAmount (Literal Value: UInt32 x 0: 4)
s_barrier  // (op 1155) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 58) Barrier(1155) END
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 438): s18, s19
s_lshl_b64 s[18:19], s[2:3], 4 // (op 438) 
 // (op 58) SetCoordinate(78) BEGIN
 // (op 78) generate({})
 // (op 78) end: generate({})
 // (op 78) generate({80})
 // (op 78) LoadLDSTile{Value: E8M0}(80) BEGIN
 // (op 80) GEN: loadMacroTileWAVELDS OP 80 LDS 33 WaveTile 235
 // (op 80) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 80) Generate 1:U32 into nullptr
 // tag 31: v**UNALLOCATED**
 // (op 80) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 80) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 80) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 80): v41
ds_read_u8 v41, v0 offset:5696 // (op 80) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2365 (current 2270)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 78) LoadLDSTile{Value: E8M0}(80) END
 // (op 78) end: generate({80})
 // (op 58) SetCoordinate(78) END
// Allocated Stride506: 2 VGPRs (Value: UInt64) (op 438): v56, v57
v_mov_b64 v[56:57], s[18:19] // (op 438) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 438): s18, s19

 // (op 438) Generate 1:U64 into nullptr
 // (op 438) Generate 1024:U32 into nullptr
 // (op 438) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride506: VGPR Value: UInt64 x 1: v[56:57]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 505: v**UNALLOCATED**
 // (op 58) Barrier(1169) BEGIN
s_barrier  // (op 1169) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 58) Barrier(1169) END
// Allocated Offset505: 2 VGPRs (Value: UInt64) (op 438): v58, v59
v_mov_b64 v[58:59], v[18:19] // (op 438) 
 // (op 58) SetCoordinate(83) BEGIN
 // (op 83) generate({})
 // (op 83) end: generate({})
 // (op 83) generate({85})
 // (op 83) LoadLDSTile{Value: E8M0}(85) BEGIN
 // (op 85) GEN: loadMacroTileWAVELDS OP 85 LDS 41 WaveTile 295
 // (op 85) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 85) Generate 1:U32 into nullptr
 // tag 39: v**UNALLOCATED**
 // (op 85) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 85) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v6
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 85) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 2 VGPRs (Value: UInt64) (op 438): v60, v61
v_mov_b64 v[60:61], v[58:59] // (op 438) 
 // (op 438) Generate Add(0:U32, {DataFlowTag572Flatten: s0:U32})U32 into nullptr
 // (op 438)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride506: v[56:57]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 438) Generate {DataFlowTag572Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 438) reg expression
 // (op 438) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 438): s15
s_mov_b32 s15, 128 // (op 438) call()
// 128 is not a supported value!
// Allocated : 1 VGPR (Value: E8M0) (op 85): v62
ds_read_u8 v62, v6 offset:5632 // (op 85) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 83. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2455 (current 2360)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 83) LoadLDSTile{Value: E8M0}(85) END
 // (op 83) end: generate({85})
 // (op 58) SetCoordinate(83) END
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 438) Load value direct to lds
// VMEM: Expected stall of 375. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3216 (current 2736)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 438) Generate Add(v[60:61]:U64, v56:R)U64 into VGPR Value: UInt64 x 1: v[60:61]
 // (op 438) reg expression
 // (op 438) reg expression
 // (op 438) Add(v[60:61]:U64, v56:R)U64
 // (op 438) Add: dest (VGPR Value: UInt64 x 1: v[60:61]) = 
 // (op 438)      lhs (VGPR Value: UInt64 x 1: v[60:61]) 
 // (op 438)      rhs (VGPR Value: Raw32 x 1: v56)
 // (op 58) Multiply(88) BEGIN
 // (op 88) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile16xBF6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xBF6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
// Allocated : 2 SGPRs (Value: Bool64) (op 438): s18, s19
v_add_co_u32 v60, s[18:19], v60, v56 // (op 438) least significant half
v_addc_co_u32 v61, s[18:19], 0, v61, s[18:19] // (op 438) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 438): s18, s19

 // (op 438) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 438) reg expression
 // (op 438) Add(m0:U32, 1024:U32)U32
 // (op 438) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 438) 
 // (op 438) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v60, s[40:43], s15 offen offset: 0 lds // (op 438) Load value direct to lds
// VMEM: Expected complete at 3220 (current 2740)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 438): s15

 // Freeing Stride506: 2 VGPRs (Value: UInt64) (op 438): v56, v57

 // Freeing : 2 VGPRs (Value: UInt64) (op 438): v60, v61

 // (op 283) LoadTileDirect2LDS Value: BF8x4(438) END
 // (op 283) Deallocate{}(1305) BEGIN
 // (op 1305) Deallocate 574
 // Deleting tag 574
 // (op 283) Deallocate{}(1305) END
 // (op 283) end: generate({438})
 // (op 58) SetCoordinate(283) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v41, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 88) 
 // (op 58) Multiply(88) END
 // (op 58) SetCoordinate(293) BEGIN
 // (op 293) generate({})
 // (op 293) end: generate({})
 // (op 293) generate({17})
 // (op 293) LoadTiled Value: E8M0(17) BEGIN
 // (op 17) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 17) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 17) Generate 1:U64 into nullptr
 // (op 17) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 543: v**UNALLOCATED**
 // (op 58) Deallocate{}(1219) BEGIN
 // (op 1219) Deallocate 15
 // Freeing : 8 VGPRs (Value: BF8x4) (op 70): v42, v43, v44, v45, v46, v47, v48, v49

 // Deleting tag 15
 // (op 1219) Deallocate 31
 // Freeing : 1 VGPR (Value: E8M0) (op 80): v41

 // Deleting tag 31
 // (op 1219) Deallocate 39
 // Freeing : 1 VGPR (Value: E8M0) (op 85): v62

 // Deleting tag 39
 // (op 1219) Deallocate 23
 // Freeing : 6 VGPRs (Value: BF6x16) (op 75): v50, v51, v52, v53, v54, v55

 // Deleting tag 23
 // (op 58) Deallocate{}(1219) END
// Allocated Offset543: 1 VGPR (Value: UInt32) (op 17): v41
v_mov_b32 v41, v13 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = Offset543: VGPR Value: UInt32 x 1: v41
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 17) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v21, v41, s[48:51], 0 offen offset:4 // (op 17) Load value
// VMEM: Expected complete at 3224 (current 2744)
 // (op 293) LoadTiled Value: E8M0(17) END
 // (op 293) end: generate({17})
 // (op 58) SetCoordinate(293) END
 // (op 58) Barrier(1192) BEGIN
s_barrier  // (op 1192) 
 // (op 58) Barrier(1192) END
 // (op 58) StoreLDSTile Value: BF6x16(253) BEGIN
 // (op 253) GEN: StoreLDSTile
 // (op 253) GEN: storeMacroTileLDS OP 253 LDS 472 MacroTile 470
 // (op 253) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 253) Generate 0:U32 into nullptr
 // (op 253) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 253) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = DataFlowTag658: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 253) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(11)
ds_write_b128 v17, v[30:33] offset:5760 // (op 253) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 2842 (current 2747)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 5760, Size: 1536)
s_waitcnt vmcnt(10)
ds_write_b64 v17, v[34:35] offset:5776 // (op 253) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 2844 (current 2749)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 58) StoreLDSTile Value: BF6x16(253) END
 // (op 58) Deallocate{}(1273) BEGIN
 // (op 1273) Deallocate 660
 // Deleting tag 660
 // (op 58) Deallocate{}(1273) END
 // (op 58) Barrier(1204) BEGIN
s_barrier  // (op 1204) 
 // (op 58) Barrier(1204) END
 // (op 58) StoreLDSTile Value: E8M0(255) BEGIN
 // (op 255) GEN: StoreLDSTile
 // (op 255) GEN: storeMacroTileLDS OP 255 LDS 480 MacroTile 478
 // (op 255) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 255) Generate 1:U32 into nullptr
 // (op 255) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 255) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag674: VGPR Value: UInt32 x 1: v40
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 255) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v40, v14 offset:7296 // (op 255) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2847 (current 2752)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 7296, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(255) END
 // (op 58) Deallocate{}(1277) BEGIN
 // (op 1277) Deallocate 676
 // Deleting tag 676
 // (op 58) Deallocate{}(1277) END
 // (op 58) NOP(451) BEGIN
 // (op 451) generate({})
 // (op 451) end: generate({})
 // (op 58) NOP(451) END
 // (op 58) Barrier(1198) BEGIN
s_barrier  // (op 1198) 
 // (op 58) Barrier(1198) END
 // (op 58) StoreLDSTile Value: E8M0(254) BEGIN
 // (op 254) GEN: StoreLDSTile
 // (op 254) GEN: storeMacroTileLDS OP 254 LDS 476 MacroTile 474
 // (op 254) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 254) Generate 1:U32 into nullptr
 // (op 254) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 254) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag666: VGPR Value: UInt32 x 1: v39
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 254) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v39, v22 offset:7360 // (op 254) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2939 (current 2844)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 7360, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(254) END
 // (op 58) Deallocate{}(1275) BEGIN
 // (op 1275) Deallocate 668
 // Deleting tag 668
 // (op 58) Deallocate{}(1275) END
 // (op 58) NOP(406) BEGIN
 // (op 406) generate({})
 // (op 406) end: generate({})
 // (op 58) NOP(406) END
 // (op 58) SetCoordinate(309) BEGIN
 // (op 309) generate({})
 // (op 309) end: generate({})
 // (op 309) generate({256})
 // (op 309) LoadTiled Value: BF6x16(256) BEGIN
 // (op 256) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 256) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 256) Generate 0:U64 into nullptr
 // (op 256) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 256) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[30:35]
 // 	info.rowOffsetReg = Offset524: VGPR Value: UInt32 x 1: v12
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 144
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 256) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) Barrier(1183) BEGIN
s_barrier  // (op 1183) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 58) Barrier(1183) END
buffer_load_dwordx4 v[30:33], v12, s[44:47], 0 offen offset:144 // (op 256) Load value
// VMEM: Expected stall of 371. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3697 (current 3217)
 // (op 58) SetCoordinate(244) BEGIN
 // (op 244) generate({})
 // (op 244) end: generate({})
 // (op 244) generate({245})
 // (op 244) LoadLDSTile{Value: BF8x4}(245) BEGIN
 // (op 245) GEN: loadMacroTileWAVELDS OP 245 LDS 468 WaveTile 103
 // (op 245) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 245) Generate 1:U32 into nullptr
 // tag 482: v**UNALLOCATED**
 // (op 245) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2048
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 245) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag654: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2048
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 245) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: BF8x4) (op 245): v42, v43, v44, v45, v46, v47, v48, v49
ds_read_b128 v[42:45], v7 offset:2048 // (op 245) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3313 (current 3218)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
ds_read_b128 v[46:49], v7 offset:2080 // (op 245) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3314 (current 3219)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 244) LoadLDSTile{Value: BF8x4}(245) END
 // (op 244) Deallocate{}(1251) BEGIN
 // (op 1251) Deallocate 656
 // Deleting tag 656
 // (op 244) Deallocate{}(1251) END
 // (op 244) end: generate({245})
 // (op 58) SetCoordinate(244) END
buffer_load_dwordx2 v[34:35], v12, s[44:47], 0 offen offset:160 // (op 256) Load value
// VMEM: Expected stall of 1. CBNW: 4, Inc: 3
// VMEM: Expected complete at 3701 (current 3221)
 // (op 309) LoadTiled Value: BF6x16(256) END
 // (op 309) end: generate({256})
 // (op 58) SetCoordinate(309) END
 // (op 58) Barrier(1189) BEGIN
 // (op 58) Deallocate{}(1281) BEGIN
 // (op 1281) Deallocate 524
 // Freeing Offset524: 1 VGPR (Value: UInt32) (op 10): v12

 // Deleting tag 524
 // (op 1281) Deallocate 526
 // Deleting tag 526
 // (op 58) Deallocate{}(1281) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1189) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 58) Barrier(1189) END
 // (op 58) SetCoordinate(319) BEGIN
 // (op 319) generate({})
 // (op 319) end: generate({})
 // (op 319) generate({258})
 // (op 319) LoadTiled Value: E8M0(258) BEGIN
 // (op 258) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 258) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 258) Generate 1:U64 into nullptr
 // (op 258) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 258) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = Offset562: VGPR Value: UInt32 x 1: v38
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 258) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) Assign VGPR Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32(573) BEGIN
 // (op 573) Assign dim(522) = Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32
 // (op 573) Generate Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32 into DataFlowTag522: VGPR Value: UInt32 x 1: v15
 // (op 573) reg expression
 // (op 573) Add({DataFlowTag522: v15:U32}, 96:U32)U32
 // (op 58) SetCoordinate(246) BEGIN
 // (op 246) generate({})
 // (op 246) end: generate({})
 // (op 246) generate({247})
 // (op 246) LoadLDSTile{Value: BF6x16}(247) BEGIN
 // (op 247) GEN: loadMacroTileWAVELDS OP 247 LDS 472 WaveTile 169
 // (op 247) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 247) Generate 0:U32 into nullptr
 // tag 484: v**UNALLOCATED**
 // (op 247) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 247) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag662: VGPR Value: UInt32 x 1: v8
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5760
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 247) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

v_add_u32 v15, 96, v15 // (op 573) 
 // (op 58) Assign VGPR Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32(573) END
buffer_load_ubyte v14, v38, s[52:55], 0 offen offset:6 // (op 258) Load value
// VMEM: Expected complete at 3705 (current 3225)
 // (op 319) LoadTiled Value: E8M0(258) END
 // (op 319) end: generate({258})
 // (op 58) SetCoordinate(319) END
// Allocated : 6 VGPRs (Value: BF6x16) (op 247): v50, v51, v52, v53, v54, v55
ds_read_b128 v[50:53], v8 offset:5760 // (op 247) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 3321 (current 3226)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 58) Deallocate{}(1285) BEGIN
 // (op 1285) Deallocate 562
 // Freeing Offset562: 1 VGPR (Value: UInt32) (op 25): v38

 // Deleting tag 562
 // (op 1285) Deallocate 564
 // Deleting tag 564
 // (op 58) Deallocate{}(1285) END
ds_read_b64 v[54:55], v8 offset:5776 // (op 247) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected stall of 87. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 3409 (current 3314)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 5760, Size: 1536)
 // (op 246) LoadLDSTile{Value: BF6x16}(247) END
 // (op 246) Deallocate{}(1257) BEGIN
 // (op 1257) Deallocate 664
 // Deleting tag 664
 // (op 246) Deallocate{}(1257) END
 // (op 246) end: generate({247})
 // (op 58) SetCoordinate(246) END
 // (op 58) Assign VGPR Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32(701) BEGIN
 // (op 701) Assign dim(560) = Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32
 // (op 701) Generate Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32 into DataFlowTag560: VGPR Value: UInt32 x 1: v23
 // (op 701) reg expression
 // (op 701) Add({DataFlowTag560: v23:U32}, 4:U32)U32
 // (op 58) Barrier(1186) BEGIN
 // (op 58) Deallocate{}(1253) BEGIN
 // (op 1253) Deallocate 472
 // Deleting tag 472
 // (op 58) Deallocate{}(1253) END
v_add_u32 v23, 4, v23 // (op 701) 
 // (op 58) Assign VGPR Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32(701) END
 // (op 58) Barrier(1195) BEGIN
 // (op 58) Barrier(1255) BEGIN
s_barrier  // (op 1255) 
 // (op 58) Barrier(1255) END
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1195) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7360, Size: 64)
 // (op 58) Barrier(1195) END
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1186) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 58) Barrier(1186) END
 // (op 58) SetCoordinate(248) BEGIN
 // (op 248) generate({})
 // (op 248) end: generate({})
 // (op 248) generate({249})
 // (op 248) LoadLDSTile{Value: E8M0}(249) BEGIN
 // (op 249) GEN: loadMacroTileWAVELDS OP 249 LDS 476 WaveTile 235
 // (op 249) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 249) Generate 1:U32 into nullptr
 // tag 486: v**UNALLOCATED**
 // (op 249) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 249) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag670: VGPR Value: UInt32 x 1: v5
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 249) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) SetCoordinate(304) BEGIN
 // (op 304) generate({})
 // (op 304) end: generate({})
 // (op 304) generate({449})
 // (op 304) LoadTileDirect2LDS Value: BF8x4(449) BEGIN
 // (op 449) GEN: LoadTileDirect2LDS
 // (op 449) GEN: loadMacroTileDirect2LDS OP 449 LDS 468 MacroTile 466
 // (op 449) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 449) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride506: VGPR Value: UInt64 x 1: (unallocated)
 // (op 449) reg expression
 // (op 449) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 449) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 449)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 449)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: E8M0) (op 249): v12
ds_read_u8 v12, v5 offset:7360 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3416 (current 3321)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7360, Size: 64)
 // (op 248) LoadLDSTile{Value: E8M0}(249) END
 // (op 248) Deallocate{}(1263) BEGIN
 // (op 1263) Deallocate 672
 // Deleting tag 672
 // (op 248) Deallocate{}(1263) END
 // (op 248) end: generate({249})
 // (op 58) SetCoordinate(248) END
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 449): s18, s19
s_lshl_b64 s[18:19], s[2:3], 4 // (op 449) 
 // (op 58) Deallocate{}(1259) BEGIN
 // (op 1259) Deallocate 476
 // Deleting tag 476
 // (op 58) Deallocate{}(1259) END
// Allocated Stride506: 2 VGPRs (Value: UInt64) (op 449): v56, v57
v_mov_b64 v[56:57], s[18:19] // (op 449) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 449): s18, s19

 // (op 449) Generate 1:U64 into nullptr
 // (op 449) Generate 1024:U32 into nullptr
 // (op 449) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride506: VGPR Value: UInt64 x 1: v[56:57]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // (op 58) Barrier(1201) BEGIN
 // (op 58) Barrier(1261) BEGIN
s_barrier  // (op 1201) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7296, Size: 64)
 // (op 58) Barrier(1201) END
s_barrier  // (op 1261) 
 // (op 58) Barrier(1261) END
 // (op 58) SetCoordinate(250) BEGIN
 // (op 250) generate({})
 // (op 250) end: generate({})
 // (op 250) generate({251})
 // (op 250) LoadLDSTile{Value: E8M0}(251) BEGIN
 // (op 251) GEN: loadMacroTileWAVELDS OP 251 LDS 480 WaveTile 295
 // (op 251) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 251) Generate 1:U32 into nullptr
 // tag 488: v**UNALLOCATED**
 // (op 251) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 251) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag678: VGPR Value: UInt32 x 1: v10
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7296
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 251) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 251): v38
ds_read_u8 v38, v10 offset:7296 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3421 (current 3326)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 7296, Size: 64)
 // (op 250) LoadLDSTile{Value: E8M0}(251) END
 // (op 250) Deallocate{}(1269) BEGIN
 // (op 1269) Deallocate 680
 // Deleting tag 680
 // (op 250) Deallocate{}(1269) END
 // (op 250) end: generate({251})
 // (op 58) SetCoordinate(250) END
// Allocated : 2 VGPRs (Value: UInt64) (op 449): v60, v61
v_mov_b64 v[60:61], v[58:59] // (op 449) 
 // (op 449) Generate Add(2048:U32, {DataFlowTag646Flatten: s1:U32})U32 into nullptr
 // (op 449) reg expression
 // (op 449) Add(2048:U32, {DataFlowTag646Flatten: s1:U32})U32
 // (op 58) Deallocate{}(1265) BEGIN
 // (op 1265) Deallocate 480
 // Deleting tag 480
 // (op 58) Deallocate{}(1265) END
// Allocated : 1 SGPR (Value: UInt32) (op 449): s15
s_add_u32 s15, 2048, s1 // (op 449) 
 // (op 449)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride506: v[56:57]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 449) Generate s15:U32 into M0 Value: UInt32 x 1: m0
 // (op 58) Multiply(252) BEGIN
 // (op 252) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile16xBF6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xBF6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 58) Barrier(1267) BEGIN
s_barrier  // (op 1267) 
 // (op 58) Barrier(1267) END
s_mov_b32 m0, s15 // (op 449) reg expression
 // (op 449) Generate 192:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 192 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 449): s18
s_mov_b32 s18, 192 // (op 449) call()
// 192 is not a supported value!
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[42:49], v[50:55], a[0:15], v12, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 252) 
 // (op 58) Multiply(252) END
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 449) Load value direct to lds
// VMEM: Expected stall of 364. CBNW: 6, Inc: 3
// VMEM: Expected complete at 4178 (current 3698)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 449) Generate Add(v[60:61]:U64, v56:R)U64 into VGPR Value: UInt64 x 1: v[60:61]
 // (op 449) reg expression
 // (op 449) reg expression
 // (op 449) Add(v[60:61]:U64, v56:R)U64
 // (op 449) Add: dest (VGPR Value: UInt64 x 1: v[60:61]) = 
 // (op 449)      lhs (VGPR Value: UInt64 x 1: v[60:61]) 
 // (op 449)      rhs (VGPR Value: Raw32 x 1: v56)
 // (op 58) Deallocate{}(1271) BEGIN
 // (op 1271) Deallocate 486
 // Freeing : 1 VGPR (Value: E8M0) (op 249): v12

 // Deleting tag 486
 // (op 1271) Deallocate 488
 // Freeing : 1 VGPR (Value: E8M0) (op 251): v38

 // Deleting tag 488
 // (op 1271) Deallocate 484
 // Freeing : 6 VGPRs (Value: BF6x16) (op 247): v50, v51, v52, v53, v54, v55

 // Deleting tag 484
 // (op 1271) Deallocate 482
 // Freeing : 8 VGPRs (Value: BF8x4) (op 245): v42, v43, v44, v45, v46, v47, v48, v49

 // Deleting tag 482
 // (op 58) Deallocate{}(1271) END
// Allocated : 2 SGPRs (Value: Bool64) (op 449): s38, s39
v_add_co_u32 v60, s[38:39], v60, v56 // (op 449) least significant half
v_addc_co_u32 v61, s[38:39], 0, v61, s[38:39] // (op 449) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 449): s38, s39

 // (op 449) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 449) reg expression
 // (op 449) Add(m0:U32, 1024:U32)U32
 // (op 449) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 449) 
 // (op 449) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v60, s[40:43], s18 offen offset: 0 lds // (op 449) Load value direct to lds
// VMEM: Expected complete at 4182 (current 3702)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 449): s18

 // Freeing Stride506: 2 VGPRs (Value: UInt64) (op 449): v56, v57

 // Freeing : 2 VGPRs (Value: UInt64) (op 449): v60, v61

 // Freeing : 1 SGPR (Value: UInt32) (op 449): s15

 // (op 304) LoadTileDirect2LDS Value: BF8x4(449) END
 // (op 304) Deallocate{}(1309) BEGIN
 // (op 1309) Deallocate 648
 // Deleting tag 648
 // (op 304) Deallocate{}(1309) END
 // (op 304) end: generate({449})
 // (op 58) SetCoordinate(304) END
 // (op 58) Deallocate{}(1279) BEGIN
 // (op 1279) Deallocate 505
 // Freeing Offset505: 2 VGPRs (Value: UInt64) (op 438): v58, v59

 // Deleting tag 505
 // (op 1279) Deallocate 507
 // Deleting tag 507
 // (op 58) Deallocate{}(1279) END
 // (op 58) SetCoordinate(314) BEGIN
 // (op 314) generate({})
 // (op 314) end: generate({})
 // (op 314) generate({257})
 // (op 314) LoadTiled Value: E8M0(257) BEGIN
 // (op 257) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 257) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 257) Generate 1:U64 into nullptr
 // (op 257) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 257) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = Offset543: VGPR Value: UInt32 x 1: v41
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 257) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) Assign VGPR Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64(509) BEGIN
 // (op 509) Assign dim(503) = Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64
 // (op 509) Generate Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64 into DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]
 // (op 509) reg expression
 // (op 509) Add({DataFlowTag503: v[18:19]:U64}, 128:U64)U64
 // (op 509) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 509)      lhs (DataFlowTag503: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 509)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v18, vcc, 128, v18 // (op 509) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 509) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 58) Assign VGPR Add(DataFlowTag(503)U64, ShiftL(DataFlowTag(504)U64, 1:U32)U64)U64(509) END
buffer_load_ubyte v22, v41, s[48:51], 0 offen offset:6 // (op 257) Load value
// VMEM: Expected stall of 1. CBNW: 8, Inc: 3
// VMEM: Expected complete at 4186 (current 3706)
 // (op 314) LoadTiled Value: E8M0(257) END
 // (op 314) end: generate({257})
 // (op 58) SetCoordinate(314) END
 // (op 58) Deallocate{}(1283) BEGIN
 // (op 1283) Deallocate 543
 // Freeing Offset543: 1 VGPR (Value: UInt32) (op 17): v41

 // Deleting tag 543
 // (op 1283) Deallocate 545
 // Deleting tag 545
 // (op 58) Deallocate{}(1283) END
 // (op 58) Assign VGPR Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32(637) BEGIN
 // (op 637) Assign dim(541) = Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32
 // (op 637) Generate Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32 into DataFlowTag541: VGPR Value: UInt32 x 1: v13
 // (op 637) reg expression
 // (op 637) Add({DataFlowTag541: v13:U32}, 4:U32)U32
 // (op 58) Barrier(1144) BEGIN
s_barrier  // (op 1144) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 58) Barrier(1144) END
v_add_u32 v13, 4, v13 // (op 637) 
 // (op 58) Assign VGPR Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32(637) END
 // (op 58) StoreLDSTile Value: BF6x16(13) BEGIN
 // (op 13) GEN: StoreLDSTile
 // (op 13) GEN: storeMacroTileLDS OP 13 LDS 25 MacroTile 9
 // (op 13) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 13) Generate 0:U32 into nullptr
 // (op 13) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 13) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v16
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 13) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(11)
ds_write_b128 v16, v[24:27] offset:4096 // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3895 (current 3800)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[28:29] offset:4112 // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3897 (current 3802)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 58) StoreLDSTile Value: BF6x16(13) END
 // (op 58) Deallocate{}(1207) BEGIN
 // (op 1207) Deallocate 590
 // Deleting tag 590
 // (op 58) Deallocate{}(1207) END
 // (op 58) Barrier(1172) BEGIN
s_barrier  // (op 1172) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 58) Barrier(1172) END
 // (op 58) StoreLDSTile Value: E8M0(30) BEGIN
 // (op 30) GEN: StoreLDSTile
 // (op 30) GEN: storeMacroTileLDS OP 30 LDS 41 MacroTile 11
 // (op 30) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 30) Generate 1:U32 into nullptr
 // (op 30) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 30) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v37
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 30) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v37, v20 offset:5632 // (op 30) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3900 (current 3805)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(30) END
 // (op 58) Deallocate{}(1211) BEGIN
 // (op 1211) Deallocate 622
 // Deleting tag 622
 // (op 58) Deallocate{}(1211) END
 // (op 58) NOP(440) BEGIN
 // (op 440) generate({})
 // (op 440) end: generate({})
 // (op 58) NOP(440) END
 // (op 58) Barrier(1158) BEGIN
s_barrier  // (op 1158) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 58) Barrier(1158) END
 // (op 58) StoreLDSTile Value: E8M0(21) BEGIN
 // (op 21) GEN: StoreLDSTile
 // (op 21) GEN: storeMacroTileLDS OP 21 LDS 33 MacroTile 10
 // (op 21) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 21) Generate 1:U32 into nullptr
 // (op 21) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 21) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v21
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v36
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 21) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v36, v21 offset:5696 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 3992 (current 3897)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(21) END
 // (op 58) Deallocate{}(1209) BEGIN
 // (op 1209) Deallocate 606
 // Deleting tag 606
 // (op 58) Deallocate{}(1209) END
 // (op 58) NOP(407) BEGIN
 // (op 407) generate({})
 // (op 407) end: generate({})
 // (op 58) NOP(407) END
 // (op 58) end: generate({60, 288, 1128})
 // (op 58) For Loop Increment
 // (op 58) generate({})
 // (op 58) end: generate({})
 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
 // (op 58) Generate LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 58) LessThan({DataFlowTag395: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 58) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 58) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 58): s18, s19
s_ashr_i64 s[18:19], s[28:29], 6 // (op 58) 
// Allocated : 1 SGPR (Value: UInt32) (op 58): s15
s_mov_b32 s15, s18 // (op 58) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 58): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 58): s18
s_and_b32 s18, s15, 4294967294 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s15

s_cmp_lt_u32 s14, s18 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 58) 
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:
 // (op 58) 
 // (op 58) Unlock For Loop
 // Unlock None
 // (op 490) ForLoopOp KLoop: LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(58) END
 // (op 490) Deallocate{}(1215) BEGIN
 // (op 1215) Deallocate 506
 // Deleting tag 506
 // (op 1215) Deallocate 508
 // Deleting tag 508
 // (op 1215) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 375): v21

 // Deleting tag 10
 // (op 1215) Deallocate 607
 // Deleting tag 607
 // (op 1215) Deallocate 620
 // Freeing DataFlowTag620: 1 VGPR (Value: UInt32) (op 905): v37

 // Deleting tag 620
 // (op 1215) Deallocate 646
 // Freeing DataFlowTag646: 1 SGPR (Value: UInt32) (op 992): s1

 // Deleting tag 646
 // (op 1215) Deallocate 647
 // Deleting tag 647
 // (op 1215) Deallocate 649
 // Deleting tag 649
 // (op 1215) Deallocate 658
 // Freeing DataFlowTag658: 1 VGPR (Value: UInt32) (op 1031): v17

 // Deleting tag 658
 // (op 1215) Deallocate 667
 // Deleting tag 667
 // (op 1215) Deallocate 669
 // Deleting tag 669
 // (op 1215) Deallocate 666
 // Freeing DataFlowTag666: 1 VGPR (Value: UInt32) (op 1055): v39

 // Deleting tag 666
 // (op 1215) Deallocate 621
 // Deleting tag 621
 // (op 1215) Deallocate 674
 // Freeing DataFlowTag674: 1 VGPR (Value: UInt32) (op 1079): v40

 // Deleting tag 674
 // (op 1215) Deallocate 677
 // Deleting tag 677
 // (op 1215) Deallocate 659
 // Deleting tag 659
 // (op 1215) Deallocate 661
 // Deleting tag 661
 // (op 1215) Deallocate 623
 // Deleting tag 623
 // (op 1215) Deallocate 675
 // Deleting tag 675
 // (op 1215) Deallocate 589
 // Deleting tag 589
 // (op 1215) Deallocate 503
 // Freeing DataFlowTag503: 2 VGPRs (Value: UInt64) (op 517): v18, v19

 // Deleting tag 503
 // (op 1215) Deallocate 522
 // Freeing DataFlowTag522: 1 VGPR (Value: UInt32) (op 581): v15

 // Deleting tag 522
 // (op 1215) Deallocate 8
 // Deleting tag 8
 // (op 1215) Deallocate 470
 // Deleting tag 470
 // (op 1215) Deallocate 478
 // Deleting tag 478
 // (op 1215) Deallocate 605
 // Deleting tag 605
 // (op 1215) Deallocate 466
 // Deleting tag 466
 // (op 1215) Deallocate 468
 // Deleting tag 468
 // (op 1215) Deallocate 604
 // Freeing DataFlowTag604: 1 VGPR (Value: UInt32) (op 852): v36

 // Deleting tag 604
 // (op 1215) Deallocate 572
 // Freeing DataFlowTag572: 1 SGPR (Value: UInt32) (op 747): s0

 // Deleting tag 572
 // (op 1215) Deallocate 573
 // Deleting tag 573
 // (op 1215) Deallocate 575
 // Deleting tag 575
 // (op 1215) Deallocate 588
 // Freeing DataFlowTag588: 1 VGPR (Value: UInt32) (op 799): v16

 // Deleting tag 588
 // (op 1215) Deallocate 591
 // Deleting tag 591
 // (op 1215) Deallocate 565
 // Deleting tag 565
 // (op 1215) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 709): v23

 // Deleting tag 560
 // (op 1215) Deallocate 542
 // Deleting tag 542
 // (op 1215) Deallocate 541
 // Freeing DataFlowTag541: 1 VGPR (Value: UInt32) (op 645): v13

 // Deleting tag 541
 // (op 1215) Deallocate 504
 // Deleting tag 504
 // (op 1215) Deallocate 546
 // Deleting tag 546
 // (op 1215) Deallocate 9
 // Freeing : 6 VGPRs (Value: BF6x16) (op 366): v24, v25, v26, v27, v28, v29

 // Deleting tag 9
 // (op 1215) Deallocate 523
 // Deleting tag 523
 // (op 1215) Deallocate 474
 // Deleting tag 474
 // (op 1215) Deallocate 544
 // Deleting tag 544
 // (op 1215) Deallocate 561
 // Deleting tag 561
 // (op 1215) Deallocate 395
 // Freeing DataFlowTag395: 1 SGPR (Value: UInt32) (op 59): s14

 // Deleting tag 395
 // (op 1215) Deallocate 525
 // Deleting tag 525
 // (op 1215) Deallocate 11
 // Freeing : 1 VGPR (Value: E8M0) (op 369): v20

 // Deleting tag 11
 // (op 1215) Deallocate 563
 // Deleting tag 563
 // (op 1215) Deallocate 527
 // Deleting tag 527
 // (op 490) Deallocate{}(1215) END
 // (op 490) Barrier(1217) BEGIN
s_barrier  // (op 1217) 
 // (op 490) Barrier(1217) END
 // (op 490) end: generate({485})
 // (op 490) Unlock Scope 490
 // Unlock None
 // (op 340) Scope(490) END
 // (op 340) SetCoordinate(217) BEGIN
 // (op 217) generate({})
 // (op 217) end: generate({})
 // (op 217) generate({218})
 // (op 217) SetCoordinate(218) BEGIN
 // (op 218) generate({})
 // (op 218) end: generate({})
 // (op 218) generate({467})
 // (op 218) Scope(467) BEGIN
 // (op 467) Lock Scope 467
 // Lock Branch
 // (op 467) generate({461})
 // (op 467) NOP(461) BEGIN
 // (op 461) generate({})
 // (op 461) end: generate({})
 // (op 467) NOP(461) END
 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(472) BEGIN
 // (op 472) Assign dim(490) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}
 // tag 490: v**UNALLOCATED**
 // (op 472) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64} into DataFlowTag490: VGPR Value: UInt64 x 1: (unallocated)
 // (op 472) reg expression
 // (op 472) reg expression
 // (op 472) reg expression
 // (op 472) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 472) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 472): v12
v_lshrrev_b32 v12, 2, v1 // (op 472) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 472): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 472) 
 // (op 472) BEGIN: Tile
 // (op 472) convert
// Allocated : 1 VGPR (Value: UInt32) (op 472): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 472) 
 // (op 472) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 472): v12

 // (op 472) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 472) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 472): s14
s_lshr_b32 s14, s0, 1 // (op 472) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 472): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 472): v12
v_and_b32 v12, -4, v1 // (op 472) 
 // (op 472) reg expression
 // (op 472) BEGIN: Flatten[1]
 // (op 472) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 472): v15
v_sub_u32 v15, v1, v12 // (op 472) 
 // (op 472) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 472): v12

 // (op 472) BEGIN: Tile
 // (op 472) {Tile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 472) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 472)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 472)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 472): v12
v_lshlrev_b32 v12, 4, v15 // (op 472) 
 // (op 472) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 472): v15

 // (op 472) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 472): v15
v_lshl_add_u32 v15, s14, 7, v12 // (op 472) 
 // (op 472) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 472): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 472): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 472): v16, v17
v_mov_b32 v17, 0 // (op 472) convert
v_mov_b32 v16, v15 // (op 472) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 472): v15

 // (op 472) BEGIN: Split
 // (op 472) Multiply: dest (DataFlowTag490: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 472)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 472)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag490: 2 VGPRs (Value: UInt64) (op 472): v18, v19
v_mul_lo_u32 v19, v13, s3 // (op 472) most significant: low of low * high
 // (op 472) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 472): v12
v_mul_hi_u32 v12, v13, s2 // (op 472) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 472) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 472) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 472): v12

 // (op 472) Add: dest (DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 472)      lhs (DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 472)      rhs (Tile: VGPR Value: UInt64 x 1: v[16:17])
// Allocated : 2 SGPRs (Value: Bool64) (op 472): s0, s1
v_add_co_u32 v18, s[0:1], v18, v16 // (op 472) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 472) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 472): s0, s1

 // (op 472) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 472): v16, v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 472): v13

 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, LogicalShiftR(DataFlowTag(76)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(472) END
 // (op 467) Assign VGPR 64:U64(473) BEGIN
 // (op 473) Assign dim(491) = 64:U64
 // (op 473) Assign stride expression
 // (op 467) Assign VGPR 64:U64(473) END
 // (op 467) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(474) BEGIN
 // (op 474) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 474) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // (op 474) reg expression
 // (op 474) reg expression
 // (op 474) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 474) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 474)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 474)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 474): s0, s1
s_mul_i32 s1, s2, s7 // (op 474) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 474): s14
s_mul_i32 s14, s3, s6 // (op 474) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 474) least significant
// Allocated : 1 SGPR (Value: Int32) (op 474): s15
s_mul_hi_u32 s15, s2, s6 // (op 474) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 474) most significant: sum
s_add_u32 s1, s1, s15 // (op 474) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 474): s15

 // Freeing : 1 SGPR (Value: Int32) (op 474): s14

 // (op 474) reg expression
 // (op 474) reg expression
s_mov_b32 s40, s4 // (op 474) reg expression
s_mov_b32 s41, s5 // (op 474) reg expression
s_mov_b32 s42, s0 // (op 474) reg expression
s_mov_b32 s43, 131072 // (op 474) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 474): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(474) END
 // (op 467) NOP(463) BEGIN
 // (op 463) generate({})
 // (op 463) end: generate({})
 // (op 467) NOP(463) END
 // (op 467) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(479) BEGIN
 // (op 479) Assign dim(495) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 479) Assign stride expression
 // (op 467) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(479) END
 // (op 467) NOP(464) BEGIN
 // (op 464) generate({})
 // (op 464) end: generate({})
 // (op 467) NOP(464) END
 // (op 467) Assign VGPR 1:U64(482) BEGIN
 // (op 482) Assign dim(497) = 1:U64
 // (op 482) Assign stride expression
 // (op 467) Assign VGPR 1:U64(482) END
 // (op 467) NOP(530) BEGIN
 // (op 530) generate({})
 // (op 530) end: generate({})
 // (op 467) NOP(530) END
 // (op 467) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(539) BEGIN
 // (op 539) Assign dim(509) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 509: v**UNALLOCATED**
 // (op 539) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag509: VGPR Value: UInt32 x 1: (unallocated)
 // (op 539) reg expression
 // (op 539) reg expression
 // (op 539) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 539) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 539): v12
v_and_b32 v12, -2, v1 // (op 539) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 539): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 539) 
 // (op 539) BEGIN: Flatten[1]
 // (op 539) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
 // (op 539) convert
// Allocated : 1 VGPR (Value: UInt32) (op 539): v13
v_sub_u32 v13, v1, v12 // (op 539) 
 // (op 539) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 539): v12

 // (op 539) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 539) BEGIN: Tile
 // (op 539) {Tile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 539) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 539)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 539)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 SGPR (Value: UInt32) (op 539): s14
s_lshr_b32 s14, s0, 1 // (op 539) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 539): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 539): v12
v_lshlrev_b32 v12, 5, v13 // (op 539) 
 // (op 539) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 539): v13

 // (op 539) reg expression
 // (op 539) BEGIN: Tile
 // (op 539) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 539): v13
v_lshl_add_u32 v13, s14, 7, v12 // (op 539) 
 // (op 539) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 539): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 539): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 539): v16, v17
v_mov_b32 v17, 0 // (op 539) convert
v_mov_b32 v16, v13 // (op 539) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 539): v13

 // (op 539) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 539): v15
v_lshrrev_b32 v15, 1, v1 // (op 539) 
 // (op 539) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 539): v12
v_lshl_add_u32 v12, s17, 5, v15 // (op 539) 
 // (op 539) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 539): v15

 // (op 539) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 539) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 539)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 539)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 539): v20, v21
v_mul_lo_u32 v21, v12, s9 // (op 539) most significant: low of low * high
 // (op 539) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 539): v13
v_mul_hi_u32 v13, v12, s8 // (op 539) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 539) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 539) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 539): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 539): v12

 // (op 539) BEGIN: Split
 // (op 539) {Split: Add({Tile: v[16:17]:U64}, v[20:21]:I64)U64}
 // (op 539) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 539)      lhs (Tile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 539)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated : 2 VGPRs (Value: UInt64) (op 539): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 539): s0, s1
v_add_co_u32 v12, s[0:1], v16, v20 // (op 539) least significant half
v_addc_co_u32 v13, s[0:1], v17, v21, s[0:1] // (op 539) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 539): s0, s1

 // (op 539) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 539): v20, v21

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 539): v16, v17

 // (op 539) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 539) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 539)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 539)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 539) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 539): v16, v17
v_mul_lo_u32 v17, v13, 6 // (op 539) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 539): v15
v_mul_hi_u32 v15, v12, 6 // (op 539) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 539) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 539) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 539): v15

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 539): v12, v13

 // (op 539) LogicalShiftR(v[16:17]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 539): v12, v13
v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 539) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 539): v16, v17

// Allocated DataFlowTag509: 1 VGPR (Value: UInt32) (op 539): v15
v_mov_b32 v15, v12 // (op 539) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 539): v12, v13

 // (op 467) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(539) END
 // (op 467) Assign VGPR 48:U64(540) BEGIN
 // (op 540) Assign dim(510) = 48:U64
 // (op 540) Assign stride expression
 // (op 467) Assign VGPR 48:U64(540) END
 // (op 467) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(541) BEGIN
 // (op 541) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 541) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // (op 541) reg expression
 // (op 541) reg expression
 // (op 541) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 541) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 541)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 541)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 541) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 541): s0, s1
s_mul_i32 s1, s13, 6 // (op 541) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 541) least significant
// Allocated : 1 SGPR (Value: Int32) (op 541): s14
s_mul_hi_u32 s14, s12, 6 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 541) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 541): s14

 // (op 541) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 541) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 541)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 541)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 541): s14, s15
s_mul_i32 s15, s8, s1 // (op 541) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 541): s18
s_mul_i32 s18, s9, s0 // (op 541) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 541) least significant
// Allocated : 1 SGPR (Value: Int32) (op 541): s19
s_mul_hi_u32 s19, s8, s0 // (op 541) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 541) most significant: sum
s_add_u32 s15, s15, s19 // (op 541) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 541): s19

 // Freeing : 1 SGPR (Value: Int32) (op 541): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 541): s0, s1

 // (op 541) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 541): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 541) 
 // Freeing : 2 SGPRs (Value: Int64) (op 541): s14, s15

 // (op 541) reg expression
 // (op 541) reg expression
s_mov_b32 s44, s10 // (op 541) reg expression
s_mov_b32 s45, s11 // (op 541) reg expression
s_mov_b32 s46, s0 // (op 541) reg expression
s_mov_b32 s47, 131072 // (op 541) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 541): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(541) END
 // (op 467) NOP(532) BEGIN
 // (op 532) generate({})
 // (op 532) end: generate({})
 // (op 467) NOP(532) END
 // (op 467) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(546) BEGIN
 // (op 546) Assign dim(514) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 546) Assign stride expression
 // (op 467) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(546) END
 // (op 467) NOP(533) BEGIN
 // (op 533) generate({})
 // (op 533) end: generate({})
 // (op 467) NOP(533) END
 // (op 467) Assign VGPR 0:U64(549) BEGIN
 // (op 549) Assign dim(516) = 0:U64
 // (op 549) Assign stride expression
 // (op 467) Assign VGPR 0:U64(549) END
 // (op 467) NOP(594) BEGIN
 // (op 594) generate({})
 // (op 594) end: generate({})
 // (op 467) NOP(594) END
 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}(603) BEGIN
 // (op 603) Assign dim(528) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}
 // tag 528: v**UNALLOCATED**
 // (op 603) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32} into DataFlowTag528: VGPR Value: UInt32 x 1: (unallocated)
 // (op 603) reg expression
 // (op 603) reg expression
 // (op 603) reg expression
 // (op 603) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 603) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 603): v12
v_lshrrev_b32 v12, 1, v1 // (op 603) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 603): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 603) 
 // (op 603) BEGIN: Tile
 // (op 603) convert
// Allocated : 1 VGPR (Value: UInt32) (op 603): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 603) 
 // (op 603) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 603): v12

 // (op 603) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 603) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 603): s14
s_lshr_b32 s14, s0, 1 // (op 603) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 603): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 603): v12
v_and_b32 v12, -2, v1 // (op 603) 
 // (op 603) reg expression
 // (op 603) BEGIN: Flatten[1]Tile
 // (op 603) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
 // (op 603) convert
// Allocated : 1 VGPR (Value: UInt32) (op 603): v16
v_sub_u32 v16, v1, v12 // (op 603) 
 // (op 603) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 603): v12

 // (op 603) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 603): v12
v_lshl_add_u32 v12, s14, 2, v16 // (op 603) 
 // (op 603) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 603): v16

 // Freeing : 1 SGPR (Value: UInt32) (op 603): s14

 // (op 603) BEGIN: Split
// Allocated DataFlowTag528: 1 VGPR (Value: UInt32) (op 603): v16
v_mul_lo_u32 v16, v13, s20 // (op 603) 
v_add_u32 v16, v16, v12 // (op 603) 
 // (op 603) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 603): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 603): v13

 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}(603) END
 // (op 467) Assign VGPR 2:U64(604) BEGIN
 // (op 604) Assign dim(529) = 2:U64
 // (op 604) Assign stride expression
 // (op 467) Assign VGPR 2:U64(604) END
 // (op 467) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(605) BEGIN
 // (op 605) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 605) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // (op 605) reg expression
 // (op 605) reg expression
 // (op 605) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 605) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 605)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 605)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 605): s0, s1
s_mul_i32 s1, s20, s7 // (op 605) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 605): s14
s_mul_i32 s14, s21, s6 // (op 605) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 605) least significant
// Allocated : 1 SGPR (Value: Int32) (op 605): s15
s_mul_hi_u32 s15, s20, s6 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 605) most significant: sum
s_add_u32 s1, s1, s15 // (op 605) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 605): s15

 // Freeing : 1 SGPR (Value: Int32) (op 605): s14

 // (op 605) reg expression
 // (op 605) reg expression
s_mov_b32 s48, s22 // (op 605) reg expression
s_mov_b32 s49, s23 // (op 605) reg expression
s_mov_b32 s50, s0 // (op 605) reg expression
s_mov_b32 s51, 131072 // (op 605) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 605): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(605) END
 // (op 467) NOP(596) BEGIN
 // (op 596) generate({})
 // (op 596) end: generate({})
 // (op 467) NOP(596) END
 // (op 467) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(610) BEGIN
 // (op 610) Assign dim(533) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 610) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(610) END
 // (op 467) NOP(597) BEGIN
 // (op 597) generate({})
 // (op 597) end: generate({})
 // (op 467) NOP(597) END
 // (op 467) Assign VGPR 1:U64(613) BEGIN
 // (op 613) Assign dim(535) = 1:U64
 // (op 613) Assign stride expression
 // (op 467) Assign VGPR 1:U64(613) END
 // (op 467) NOP(658) BEGIN
 // (op 658) generate({})
 // (op 658) end: generate({})
 // (op 467) NOP(658) END
 // (op 467) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(667) BEGIN
 // (op 667) Assign dim(547) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 547: v**UNALLOCATED**
 // (op 667) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag547: VGPR Value: UInt32 x 1: (unallocated)
 // (op 667) reg expression
 // (op 667) reg expression
 // (op 667) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 667) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 667): v17
v_and_b32 v17, -2, v1 // (op 667) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 667): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 667) 
 // (op 667) BEGIN: Flatten[1]Tile
 // (op 667) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
 // (op 667) convert
// Allocated : 1 VGPR (Value: UInt32) (op 667): v12
v_sub_u32 v12, v1, v17 // (op 667) 
 // (op 667) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 667): v17

 // (op 667) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 667) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 667): s14
s_lshr_b32 s14, s0, 1 // (op 667) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 667): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 667): v13
v_lshrrev_b32 v13, 1, v1 // (op 667) 
 // (op 667) reg expression
 // (op 667) BEGIN: Tile
 // (op 667) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 667): v17
v_lshl_add_u32 v17, s14, 2, v12 // (op 667) 
 // (op 667) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 667): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 667): s14

 // (op 667) convert
 // (op 667) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 667): v12
v_lshl_add_u32 v12, s17, 5, v13 // (op 667) 
 // (op 667) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 667): v13

 // (op 667) Multiply({Tile: v12:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 667): v13
v_mul_lo_u32 v13, v12, s24 // (op 667) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 667): v12

 // (op 667) BEGIN: Split
 // (op 667) {Split: Add({Tile: v17:U32}, v13:U32)U32}
// Allocated DataFlowTag547: 1 VGPR (Value: UInt32) (op 667): v12
v_add_u32 v12, v17, v13 // (op 667) 
 // (op 667) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 667): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 667): v17

 // (op 467) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(667) END
 // (op 467) Assign VGPR 2:U64(668) BEGIN
 // (op 668) Assign dim(548) = 2:U64
 // (op 668) Assign stride expression
 // (op 467) Assign VGPR 2:U64(668) END
 // (op 467) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(669) BEGIN
 // (op 669) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 669) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // (op 669) reg expression
 // (op 669) reg expression
 // (op 669) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 669) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 669)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 669)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 669): s0, s1
s_mul_i32 s1, s24, s13 // (op 669) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 669): s14
s_mul_i32 s14, s25, s12 // (op 669) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 669) least significant
// Allocated : 1 SGPR (Value: Int32) (op 669): s15
s_mul_hi_u32 s15, s24, s12 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 669) most significant: sum
s_add_u32 s1, s1, s15 // (op 669) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 669): s15

 // Freeing : 1 SGPR (Value: Int32) (op 669): s14

 // (op 669) reg expression
 // (op 669) reg expression
s_mov_b32 s52, s26 // (op 669) reg expression
s_mov_b32 s53, s27 // (op 669) reg expression
s_mov_b32 s54, s0 // (op 669) reg expression
s_mov_b32 s55, 131072 // (op 669) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 669): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(669) END
 // (op 467) NOP(660) BEGIN
 // (op 660) generate({})
 // (op 660) end: generate({})
 // (op 467) NOP(660) END
 // (op 467) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(674) BEGIN
 // (op 674) Assign dim(552) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 674) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(674) END
 // (op 467) NOP(661) BEGIN
 // (op 661) generate({})
 // (op 661) end: generate({})
 // (op 467) NOP(661) END
 // (op 467) Assign VGPR 1:U64(677) BEGIN
 // (op 677) Assign dim(554) = 1:U64
 // (op 677) Assign stride expression
 // (op 467) Assign VGPR 1:U64(677) END
 // (op 467) NOP(755) BEGIN
 // (op 755) generate({})
 // (op 755) end: generate({})
 // (op 467) NOP(755) END
 // (op 467) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(760) BEGIN
 // (op 760) Assign dim(576) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 576: s**UNALLOCATED**
 // (op 760) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag576: SGPR Value: UInt32 x 1: (unallocated)
 // (op 760) reg expression
 // (op 760) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 760): v13
v_and_b32 v13, -4, v1 // (op 760) 
 // (op 760) BEGIN: Tile[1]
 // (op 760) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 760): v17
v_sub_u32 v17, v1, v13 // (op 760) 
 // (op 760) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 760): v13

 // (op 760) BEGIN: Flatten
 // (op 760) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 760) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 760)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 760)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 760): v13
v_lshlrev_b32 v13, 4, v17 // (op 760) 
 // (op 760) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 760): v17

 // (op 760) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 760): v17
v_lshrrev_b32 v17, 2, v1 // (op 760) 
 // (op 760) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 760): v20
v_lshl_add_u32 v20, v17, 6, v13 // (op 760) 
 // (op 760) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 760): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 760): v17

// Allocated DataFlowTag576: 1 SGPR (Value: UInt32) (op 760): s0
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 760) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 760): v20

 // (op 467) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(760) END
 // (op 467) Assign VGPR 1024:U32(761) BEGIN
 // (op 761) Assign dim(577) = 1024:U32
 // (op 761) Assign stride expression
 // (op 467) Assign VGPR 1024:U32(761) END
 // (op 467) NOP(756) BEGIN
 // (op 756) generate({})
 // (op 756) end: generate({})
 // (op 467) NOP(756) END
 // (op 467) Assign VGPR 1:U32(765) BEGIN
 // (op 765) Assign dim(579) = 1:U32
 // (op 765) Assign stride expression
 // (op 467) Assign VGPR 1:U32(765) END
 // (op 467) NOP(807) BEGIN
 // (op 807) generate({})
 // (op 807) end: generate({})
 // (op 467) NOP(807) END
 // (op 467) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(812) BEGIN
 // (op 812) Assign dim(592) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 592: v**UNALLOCATED**
 // (op 812) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag592: VGPR Value: UInt32 x 1: (unallocated)
 // (op 812) reg expression
 // (op 812) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 812): v13
v_and_b32 v13, -2, v1 // (op 812) 
 // (op 812) BEGIN: Tile[1]
 // (op 812) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 812): v17
v_sub_u32 v17, v1, v13 // (op 812) 
 // (op 812) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 812): v13

 // (op 812) BEGIN: Flatten
 // (op 812) {Flatten: ShiftL({Tile[1]: v17:U32}, 5:U32)U32}
 // (op 812) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 812)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 812)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 812): v13
v_lshlrev_b32 v13, 5, v17 // (op 812) 
 // (op 812) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 812): v17

 // (op 812) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 812): v17
v_lshrrev_b32 v17, 1, v1 // (op 812) 
 // (op 812) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 812): v20
v_lshl_add_u32 v20, v17, 6, v13 // (op 812) 
 // (op 812) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 812): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 812): v17

 // (op 812) Multiply({Flatten: v20:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 812): v13
v_mul_lo_u32 v13, v20, 6 // (op 812) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 812): v20

 // (op 812) LogicalShiftR(v13:U32, 3:U32)U32
// Allocated DataFlowTag592: 1 VGPR (Value: UInt32) (op 812): v17
v_lshrrev_b32 v17, 3, v13 // (op 812) 
 // Freeing : 1 VGPR (Value: UInt32) (op 812): v13

 // (op 467) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(812) END
 // (op 467) Assign VGPR 48:U32(813) BEGIN
 // (op 813) Assign dim(593) = 48:U32
 // (op 813) Assign stride expression
 // (op 467) Assign VGPR 48:U32(813) END
 // (op 467) NOP(808) BEGIN
 // (op 808) generate({})
 // (op 808) end: generate({})
 // (op 467) NOP(808) END
 // (op 467) Assign VGPR 0:U32(817) BEGIN
 // (op 817) Assign dim(595) = 0:U32
 // (op 817) Assign stride expression
 // (op 467) Assign VGPR 0:U32(817) END
 // (op 467) NOP(860) BEGIN
 // (op 860) generate({})
 // (op 860) end: generate({})
 // (op 467) NOP(860) END
 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(865) BEGIN
 // (op 865) Assign dim(608) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 608: v**UNALLOCATED**
 // (op 865) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag608: VGPR Value: UInt32 x 1: (unallocated)
 // (op 865) reg expression
 // (op 865) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 865): v13
v_and_b32 v13, -2, v1 // (op 865) 
 // (op 865) BEGIN: Tile[1]Flatten
 // (op 865) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 865): v20
v_sub_u32 v20, v1, v13 // (op 865) 
 // (op 865) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 865): v13

 // (op 865) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 865): v13
v_and_b32 v13, 4294967294, v1 // (op 865) 
 // (op 865) BEGIN: Flatten
 // (op 865) {Flatten: Add(v13:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag608: 1 VGPR (Value: UInt32) (op 865): v21
v_add_u32 v21, v13, v20 // (op 865) 
 // (op 865) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 865): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 865): v13

 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(865) END
 // (op 467) Assign VGPR 2:U32(866) BEGIN
 // (op 866) Assign dim(609) = 2:U32
 // (op 866) Assign stride expression
 // (op 467) Assign VGPR 2:U32(866) END
 // (op 467) NOP(861) BEGIN
 // (op 861) generate({})
 // (op 861) end: generate({})
 // (op 467) NOP(861) END
 // (op 467) Assign VGPR 1:U32(870) BEGIN
 // (op 870) Assign dim(611) = 1:U32
 // (op 870) Assign stride expression
 // (op 467) Assign VGPR 1:U32(870) END
 // (op 467) NOP(913) BEGIN
 // (op 913) generate({})
 // (op 913) end: generate({})
 // (op 467) NOP(913) END
 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(918) BEGIN
 // (op 918) Assign dim(624) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 624: v**UNALLOCATED**
 // (op 918) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag624: VGPR Value: UInt32 x 1: (unallocated)
 // (op 918) reg expression
 // (op 918) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 918): v13
v_and_b32 v13, -2, v1 // (op 918) 
 // (op 918) BEGIN: Tile[1]Flatten
 // (op 918) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 918): v20
v_sub_u32 v20, v1, v13 // (op 918) 
 // (op 918) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 918): v13

 // (op 918) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 918): v13
v_and_b32 v13, 4294967294, v1 // (op 918) 
 // (op 918) BEGIN: Flatten
 // (op 918) {Flatten: Add(v13:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag624: 1 VGPR (Value: UInt32) (op 918): v23
v_add_u32 v23, v13, v20 // (op 918) 
 // (op 918) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 918): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 918): v13

 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(918) END
 // (op 467) Assign VGPR 2:U32(919) BEGIN
 // (op 919) Assign dim(625) = 2:U32
 // (op 919) Assign stride expression
 // (op 467) Assign VGPR 2:U32(919) END
 // (op 467) NOP(914) BEGIN
 // (op 914) generate({})
 // (op 914) end: generate({})
 // (op 467) NOP(914) END
 // (op 467) Assign VGPR 1:U32(923) BEGIN
 // (op 923) Assign dim(627) = 1:U32
 // (op 923) Assign stride expression
 // (op 467) Assign VGPR 1:U32(923) END
 // (op 467) ForLoopOp KLoopTail: LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(171) BEGIN
 // (op 171) Initialize For Loop
 // (op 171) generate({172})
 // (op 171) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(172) BEGIN
 // (op 172) Assign dim(448) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32
 // tag 448: s**UNALLOCATED**
 // (op 172) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag448: SGPR Value: UInt32 x 1: (unallocated)
 // (op 172) reg expression
 // (op 172) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 172): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 172) 
 // (op 172) convert
 // (op 172) BitwiseAnd({ convertForwardSubset: s14:U32}, 4294967294:U32)U32
// Allocated DataFlowTag448: 1 SGPR (Value: UInt32) (op 172): s1
s_and_b32 s1, s14, 4294967294 // (op 172) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 172): s14, s15

 // (op 171) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(172) END
 // (op 171) end: generate({172})
 // (op 171) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 171) 
 // (op 171) Generate LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 171) LessThan({DataFlowTag448: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 171) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 171): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
// Allocated : 1 SGPR (Value: UInt32) (op 171): s18
s_mov_b32 s18, s14 // (op 171) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 171): s14, s15

s_cmp_lt_u32 s1, s18 // (op 171) 
 // Freeing : 1 SGPR (Value: UInt32) (op 171): s18

 // Freeing : 6 VGPRs (Value: BF6x16) (op 378): v30, v31, v32, v33, v34, v35

 // Freeing : 1 VGPR (Value: E8M0) (op 381): v14

 // Freeing : 1 VGPR (Value: E8M0) (op 387): v22

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 171) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 171) 
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 171) 
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:
 // (op 171) 
 // (op 171) generate({173, 1134})
 // (op 171) Assign SGPR Add(DataFlowTag(448)U32, 1:U32)U32(173) BEGIN
 // (op 173) Assign dim(448) = Add(DataFlowTag(448)U32, 1:U32)U32
 // (op 173) Generate Add(DataFlowTag(448)U32, 1:U32)U32 into DataFlowTag448: SGPR Value: UInt32 x 1: s1
 // (op 173) reg expression
 // (op 173) Add({DataFlowTag448: s1:U32}, 1:U32)U32
 // (op 171) Barrier(1134) BEGIN
s_barrier  // (op 1134) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 171) Barrier(1134) END
s_add_u32 s1, s1, 1 // (op 173) 
 // (op 171) Assign SGPR Add(DataFlowTag(448)U32, 1:U32)U32(173) END
 // (op 171) SetCoordinate(222) BEGIN
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({443})
 // (op 222) LoadTileDirect2LDS Value: BF8x4(443) BEGIN
 // (op 443) GEN: LoadTileDirect2LDS
 // (op 443) GEN: loadMacroTileDirect2LDS OP 443 LDS 17 MacroTile 450
 // (op 443) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 443) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride495: VGPR Value: UInt64 x 1: (unallocated)
 // (op 443) reg expression
 // (op 443) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 443) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 443)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 443)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 443): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 443) 
// Allocated Stride495: 2 VGPRs (Value: UInt64) (op 443): v24, v25
v_mov_b64 v[24:25], s[14:15] // (op 443) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 443): s14, s15

 // (op 443) Generate 1:U64 into nullptr
 // (op 443) Generate 1024:U32 into nullptr
 // (op 443) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride495: VGPR Value: UInt64 x 1: v[24:25]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 494: v**UNALLOCATED**
// Allocated Offset494: 2 VGPRs (Value: UInt64) (op 443): v26, v27
v_mov_b64 v[26:27], v[18:19] // (op 443) 
// Allocated : 2 VGPRs (Value: UInt64) (op 443): v28, v29
v_mov_b64 v[28:29], v[26:27] // (op 443) 
 // (op 443) Generate Add(0:U32, {DataFlowTag576Flatten: s0:U32})U32 into nullptr
 // (op 443)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride495: v[24:25]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 443) Generate {DataFlowTag576Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 443) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 443) Load value direct to lds
// VMEM: Expected complete at 4678 (current 4198)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 443) Generate Add(v[28:29]:U64, v24:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 443) reg expression
 // (op 443) reg expression
 // (op 443) Add(v[28:29]:U64, v24:R)U64
 // (op 443) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 443)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 443)      rhs (VGPR Value: Raw32 x 1: v24)
// Allocated : 2 SGPRs (Value: Bool64) (op 443): s14, s15
v_add_co_u32 v28, s[14:15], v28, v24 // (op 443) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 443) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 443): s14, s15

 // (op 443) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 443) reg expression
 // (op 443) Add(m0:U32, 1024:U32)U32
 // (op 443) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 443) 
 // (op 443) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 443) Load value direct to lds
// VMEM: Expected complete at 4682 (current 4202)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride495: 2 VGPRs (Value: UInt64) (op 443): v24, v25

 // Freeing : 2 VGPRs (Value: UInt64) (op 443): v28, v29

 // (op 222) LoadTileDirect2LDS Value: BF8x4(443) END
 // (op 222) Deallocate{}(1307) BEGIN
 // (op 1307) Deallocate 578
 // Deleting tag 578
 // (op 1307) Deallocate 494
 // Freeing Offset494: 2 VGPRs (Value: UInt64) (op 443): v26, v27

 // Deleting tag 494
 // (op 1307) Deallocate 496
 // Deleting tag 496
 // (op 1307) Deallocate 450
 // Deleting tag 450
 // (op 222) Deallocate{}(1307) END
 // (op 222) Assign VGPR Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64(462) BEGIN
 // (op 462) Assign dim(490) = Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64
 // (op 462) Generate Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64 into DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]
 // (op 462) reg expression
 // (op 462) Add({DataFlowTag490: v[18:19]:U64}, 64:U64)U64
 // (op 462) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 462)      lhs (DataFlowTag490: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 462)      rhs (Literal Value: UInt64 x 0: 64)
// Allocated : 2 SGPRs (Value: Bool64) (op 462): s14, s15
v_add_co_u32 v18, s[14:15], 64, v18 // (op 462) least significant half
v_addc_co_u32 v19, s[14:15], 0, v19, s[14:15] // (op 462) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 462): s14, s15

 // (op 222) Assign VGPR Add(DataFlowTag(490)U64, DataFlowTag(491)U64)U64(462) END
 // (op 222) end: generate({443})
 // (op 171) SetCoordinate(222) END
 // (op 171) SetCoordinate(227) BEGIN
 // (op 227) generate({})
 // (op 227) end: generate({})
 // (op 227) generate({190})
 // (op 227) LoadTiled Value: BF6x16(190) BEGIN
 // (op 190) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 190) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 190) Generate 0:U64 into nullptr
 // tag 452: v**UNALLOCATED**
 // (op 190) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 513: v**UNALLOCATED**
 // (op 171) NOP(445) BEGIN
 // (op 445) generate({})
 // (op 445) end: generate({})
 // (op 171) NOP(445) END
// Allocated Offset513: 1 VGPR (Value: UInt32) (op 190): v20
v_mov_b32 v20, v15 // (op 190) 
 // (op 190) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = Offset513: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 6:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 190) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: BF6x16) (op 190): v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[24:27], v20, s[44:47], 0 offen // (op 190) Load value
// VMEM: Expected complete at 4686 (current 4206)
buffer_load_dwordx2 v[28:29], v20, s[44:47], 0 offen offset:16 // (op 190) Load value
// VMEM: Expected stall of 472. CBNW: 0, Inc: 3
// VMEM: Expected complete at 5159 (current 4679)
 // (op 227) LoadTiled Value: BF6x16(190) END
 // (op 227) Deallocate{}(1241) BEGIN
 // (op 1241) Deallocate 515
 // Deleting tag 515
 // (op 1241) Deallocate 513
 // Freeing Offset513: 1 VGPR (Value: UInt32) (op 190): v20

 // Deleting tag 513
 // (op 227) Deallocate{}(1241) END
 // (op 227) Assign VGPR Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32(531) BEGIN
 // (op 531) Assign dim(509) = Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32
 // (op 531) Generate Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32 into DataFlowTag509: VGPR Value: UInt32 x 1: v15
 // (op 531) reg expression
 // (op 531) Add({DataFlowTag509: v15:U32}, 48:U32)U32
v_add_u32 v15, 48, v15 // (op 531) 
 // (op 227) Assign VGPR Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32(531) END
 // (op 227) end: generate({190})
 // (op 171) SetCoordinate(227) END
 // (op 171) SetCoordinate(232) BEGIN
 // (op 232) generate({})
 // (op 232) end: generate({})
 // (op 232) generate({191})
 // (op 232) LoadTiled Value: E8M0(191) BEGIN
 // (op 191) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 191) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 191) Generate 1:U64 into nullptr
 // tag 454: v**UNALLOCATED**
 // (op 191) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 532: v**UNALLOCATED**
 // (op 171) Barrier(1147) BEGIN
s_barrier  // (op 1147) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 171) Barrier(1147) END
// Allocated Offset532: 1 VGPR (Value: UInt32) (op 191): v20
v_mov_b32 v20, v16 // (op 191) 
 // (op 191) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset532: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 191) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 171) StoreLDSTile Value: BF6x16(187) BEGIN
 // (op 187) GEN: StoreLDSTile
 // (op 187) GEN: storeMacroTileLDS OP 187 LDS 25 MacroTile 452
 // (op 187) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 187) Generate 0:U32 into nullptr
 // (op 187) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 187) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: v[24:29]
 // 	info.rowOffsetReg = DataFlowTag592: VGPR Value: UInt32 x 1: v17
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 187) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 191): v22
buffer_load_ubyte v22, v20, s[48:51], 0 offen offset: 0 // (op 191) Load value
// VMEM: Expected complete at 5163 (current 4683)
 // (op 232) LoadTiled Value: E8M0(191) END
 // (op 232) Deallocate{}(1243) BEGIN
 // (op 1243) Deallocate 532
 // Freeing Offset532: 1 VGPR (Value: UInt32) (op 191): v20

 // Deleting tag 532
 // (op 1243) Deallocate 534
 // Deleting tag 534
 // (op 232) Deallocate{}(1243) END
 // (op 232) Assign VGPR Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32(595) BEGIN
 // (op 595) Assign dim(528) = Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32
 // (op 595) Generate Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32 into DataFlowTag528: VGPR Value: UInt32 x 1: v16
 // (op 595) reg expression
 // (op 595) Add({DataFlowTag528: v16:U32}, 2:U32)U32
v_add_u32 v16, 2, v16 // (op 595) 
 // (op 232) Assign VGPR Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32(595) END
 // (op 232) end: generate({191})
 // (op 171) SetCoordinate(232) END
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] offset:4096 // (op 187) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 4627 (current 4532)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 171) SetCoordinate(237) BEGIN
 // (op 237) generate({})
 // (op 237) end: generate({})
 // (op 237) generate({192})
 // (op 237) LoadTiled Value: E8M0(192) BEGIN
 // (op 192) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 192) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 192) Generate 1:U64 into nullptr
 // tag 456: v**UNALLOCATED**
 // (op 192) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 551: v**UNALLOCATED**
// Allocated Offset551: 1 VGPR (Value: UInt32) (op 192): v20
v_mov_b32 v20, v12 // (op 192) 
 // (op 192) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset551: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 192) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 192): v13
buffer_load_ubyte v13, v20, s[52:55], 0 offen offset: 0 // (op 192) Load value
// VMEM: Expected complete at 5170 (current 4690)
 // (op 237) LoadTiled Value: E8M0(192) END
 // (op 237) Deallocate{}(1245) BEGIN
 // (op 1245) Deallocate 553
 // Deleting tag 553
 // (op 1245) Deallocate 551
 // Freeing Offset551: 1 VGPR (Value: UInt32) (op 192): v20

 // Deleting tag 551
 // (op 237) Deallocate{}(1245) END
 // (op 237) Assign VGPR Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32(659) BEGIN
 // (op 659) Assign dim(547) = Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32
 // (op 659) Generate Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32 into DataFlowTag547: VGPR Value: UInt32 x 1: v12
 // (op 659) reg expression
 // (op 659) Add({DataFlowTag547: v12:U32}, 2:U32)U32
v_add_u32 v12, 2, v12 // (op 659) 
 // (op 237) Assign VGPR Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32(659) END
 // (op 237) end: generate({192})
 // (op 171) SetCoordinate(237) END
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[28:29] offset:4112 // (op 187) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 4632 (current 4537)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 171) StoreLDSTile Value: BF6x16(187) END
 // (op 171) Barrier(1137) BEGIN
 // (op 171) Deallocate{}(1235) BEGIN
 // (op 1235) Deallocate 594
 // Deleting tag 594
 // (op 1235) Deallocate 452
 // Freeing : 6 VGPRs (Value: BF6x16) (op 190): v24, v25, v26, v27, v28, v29

 // Deleting tag 452
 // (op 171) Deallocate{}(1235) END
s_barrier  // (op 1137) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 171) Barrier(1137) END
 // (op 171) Barrier(1161) BEGIN
 // (op 171) SetCoordinate(178) BEGIN
 // (op 178) generate({})
 // (op 178) end: generate({})
 // (op 178) generate({179})
 // (op 178) LoadLDSTile{Value: BF8x4}(179) BEGIN
 // (op 179) GEN: loadMacroTileWAVELDS OP 179 LDS 17 WaveTile 103
 // (op 179) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 179) Generate 1:U32 into nullptr
 // tag 458: v**UNALLOCATED**
 // (op 179) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 179) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(32:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 179) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

s_barrier  // (op 1161) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 171) Barrier(1161) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 179): v24, v25, v26, v27, v28, v29, v30, v31
ds_read_b128 v[24:27], v3 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4635 (current 4540)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 171) StoreLDSTile Value: E8M0(188) BEGIN
 // (op 188) GEN: StoreLDSTile
 // (op 188) GEN: storeMacroTileLDS OP 188 LDS 33 MacroTile 454
 // (op 188) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 188) Generate 1:U32 into nullptr
 // (op 188) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 188) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag608: VGPR Value: UInt32 x 1: v21
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 188) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_b128 v[28:31], v3 offset:32 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 87. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4723 (current 4628)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 178) LoadLDSTile{Value: BF8x4}(179) END
 // (op 178) end: generate({179})
 // (op 171) SetCoordinate(178) END
s_waitcnt vmcnt(1)
ds_write_b8 v21, v22 offset:5696 // (op 188) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 4. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 4729 (current 4634)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 171) StoreLDSTile Value: E8M0(188) END
 // (op 171) Barrier(1151) BEGIN
 // (op 171) Deallocate{}(1237) BEGIN
 // (op 1237) Deallocate 454
 // Freeing : 1 VGPR (Value: E8M0) (op 191): v22

 // Deleting tag 454
 // (op 1237) Deallocate 610
 // Deleting tag 610
 // (op 171) Deallocate{}(1237) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1151) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 171) Barrier(1151) END
 // (op 171) Barrier(1175) BEGIN
 // (op 171) SetCoordinate(180) BEGIN
 // (op 180) generate({})
 // (op 180) end: generate({})
 // (op 180) generate({181})
 // (op 180) LoadLDSTile{Value: BF6x16}(181) BEGIN
 // (op 181) GEN: loadMacroTileWAVELDS OP 181 LDS 25 WaveTile 169
 // (op 181) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 181) Generate 0:U32 into nullptr
 // tag 460: v**UNALLOCATED**
 // (op 181) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 181) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 32
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4096
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 181) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1175) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 171) Barrier(1175) END
// Allocated : 6 VGPRs (Value: BF6x16) (op 181): v32, v33, v34, v35, v36, v37
ds_read_b128 v[32:35], v4 offset:4096 // (op 181) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 4733 (current 4638)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 171) StoreLDSTile Value: E8M0(189) BEGIN
 // (op 189) GEN: StoreLDSTile
 // (op 189) GEN: storeMacroTileLDS OP 189 LDS 41 MacroTile 456
 // (op 189) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 189) Generate 1:U32 into nullptr
 // (op 189) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 189) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = DataFlowTag624: VGPR Value: UInt32 x 1: v23
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 189) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_b64 v[36:37], v4 offset:4112 // (op 181) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected stall of 85. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 4819 (current 4724)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 4096, Size: 1536)
 // (op 180) LoadLDSTile{Value: BF6x16}(181) END
 // (op 180) end: generate({181})
 // (op 171) SetCoordinate(180) END
s_waitcnt vmcnt(0)
ds_write_b8 v23, v13 offset:5632 // (op 189) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 5. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 4826 (current 4731)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 171) StoreLDSTile Value: E8M0(189) END
 // (op 171) Barrier(1165) BEGIN
 // (op 171) Deallocate{}(1239) BEGIN
 // (op 1239) Deallocate 456
 // Freeing : 1 VGPR (Value: E8M0) (op 192): v13

 // Deleting tag 456
 // (op 1239) Deallocate 626
 // Deleting tag 626
 // (op 171) Deallocate{}(1239) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1165) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 171) Barrier(1165) END
 // (op 171) SetCoordinate(182) BEGIN
 // (op 182) generate({})
 // (op 182) end: generate({})
 // (op 182) generate({183})
 // (op 182) LoadLDSTile{Value: E8M0}(183) BEGIN
 // (op 183) GEN: loadMacroTileWAVELDS OP 183 LDS 33 WaveTile 235
 // (op 183) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 183) Generate 1:U32 into nullptr
 // tag 462: v**UNALLOCATED**
 // (op 183) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 183) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 183) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 183): v20
ds_read_u8 v20, v0 offset:5696 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4829 (current 4734)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5696, Size: 64)
 // (op 182) LoadLDSTile{Value: E8M0}(183) END
 // (op 182) end: generate({183})
 // (op 171) SetCoordinate(182) END
 // (op 171) Barrier(1179) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1179) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 171) Barrier(1179) END
 // (op 171) SetCoordinate(184) BEGIN
 // (op 184) generate({})
 // (op 184) end: generate({})
 // (op 184) generate({185})
 // (op 184) LoadLDSTile{Value: E8M0}(185) BEGIN
 // (op 185) GEN: loadMacroTileWAVELDS OP 185 LDS 41 WaveTile 295
 // (op 185) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 185) Generate 1:U32 into nullptr
 // tag 464: v**UNALLOCATED**
 // (op 185) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 185) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v6
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(8:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5632
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 185) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 185): v22
ds_read_u8 v22, v6 offset:5632 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4924 (current 4829)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5632, Size: 64)
 // (op 184) LoadLDSTile{Value: E8M0}(185) END
 // (op 184) end: generate({185})
 // (op 171) SetCoordinate(184) END
 // (op 171) Multiply(186) BEGIN
 // (op 186) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile16xBF6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xBF6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[24:31], v[32:37], a[0:15], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 186) 
 // (op 171) Multiply(186) END
 // (op 171) Deallocate{}(1233) BEGIN
 // (op 1233) Deallocate 458
 // Freeing : 8 VGPRs (Value: BF8x4) (op 179): v24, v25, v26, v27, v28, v29, v30, v31

 // Deleting tag 458
 // (op 1233) Deallocate 460
 // Freeing : 6 VGPRs (Value: BF6x16) (op 181): v32, v33, v34, v35, v36, v37

 // Deleting tag 460
 // (op 1233) Deallocate 462
 // Freeing : 1 VGPR (Value: E8M0) (op 183): v20

 // Deleting tag 462
 // (op 1233) Deallocate 464
 // Freeing : 1 VGPR (Value: E8M0) (op 185): v22

 // Deleting tag 464
 // (op 171) Deallocate{}(1233) END
 // (op 171) end: generate({173, 1134})
 // (op 171) For Loop Increment
 // (op 171) generate({})
 // (op 171) end: generate({})
 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
 // (op 171) Generate LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 171) LessThan({DataFlowTag448: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 171) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 171): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 171) 
// Allocated : 1 SGPR (Value: UInt32) (op 171): s18
s_mov_b32 s18, s14 // (op 171) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 171): s14, s15

s_cmp_lt_u32 s1, s18 // (op 171) 
 // Freeing : 1 SGPR (Value: UInt32) (op 171): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 171) 
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:
 // (op 171) 
 // (op 171) Unlock For Loop
 // Unlock None
 // (op 467) ForLoopOp KLoopTail: LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(171) END
 // (op 467) Deallocate{}(1231) BEGIN
 // (op 1231) Deallocate 529
 // Deleting tag 529
 // (op 1231) Deallocate 593
 // Deleting tag 593
 // (op 1231) Deallocate 535
 // Deleting tag 535
 // (op 1231) Deallocate 576
 // Freeing DataFlowTag576: 1 SGPR (Value: UInt32) (op 760): s0

 // Deleting tag 576
 // (op 1231) Deallocate 448
 // Freeing DataFlowTag448: 1 SGPR (Value: UInt32) (op 172): s1

 // Deleting tag 448
 // (op 1231) Deallocate 625
 // Deleting tag 625
 // (op 1231) Deallocate 577
 // Deleting tag 577
 // (op 1231) Deallocate 627
 // Deleting tag 627
 // (op 1231) Deallocate 509
 // Freeing DataFlowTag509: 1 VGPR (Value: UInt32) (op 539): v15

 // Deleting tag 509
 // (op 1231) Deallocate 611
 // Deleting tag 611
 // (op 1231) Deallocate 552
 // Deleting tag 552
 // (op 1231) Deallocate 609
 // Deleting tag 609
 // (op 1231) Deallocate 608
 // Freeing DataFlowTag608: 1 VGPR (Value: UInt32) (op 865): v21

 // Deleting tag 608
 // (op 1231) Deallocate 490
 // Freeing DataFlowTag490: 2 VGPRs (Value: UInt64) (op 472): v18, v19

 // Deleting tag 490
 // (op 1231) Deallocate 595
 // Deleting tag 595
 // (op 1231) Deallocate 592
 // Freeing DataFlowTag592: 1 VGPR (Value: UInt32) (op 812): v17

 // Deleting tag 592
 // (op 1231) Deallocate 516
 // Deleting tag 516
 // (op 1231) Deallocate 547
 // Freeing DataFlowTag547: 1 VGPR (Value: UInt32) (op 667): v12

 // Deleting tag 547
 // (op 1231) Deallocate 491
 // Deleting tag 491
 // (op 1231) Deallocate 497
 // Deleting tag 497
 // (op 1231) Deallocate 624
 // Freeing DataFlowTag624: 1 VGPR (Value: UInt32) (op 918): v23

 // Deleting tag 624
 // (op 1231) Deallocate 510
 // Deleting tag 510
 // (op 1231) Deallocate 514
 // Deleting tag 514
 // (op 1231) Deallocate 528
 // Freeing DataFlowTag528: 1 VGPR (Value: UInt32) (op 603): v16

 // Deleting tag 528
 // (op 1231) Deallocate 548
 // Deleting tag 548
 // (op 1231) Deallocate 495
 // Deleting tag 495
 // (op 1231) Deallocate 554
 // Deleting tag 554
 // (op 1231) Deallocate 579
 // Deleting tag 579
 // (op 1231) Deallocate 533
 // Deleting tag 533
 // (op 467) Deallocate{}(1231) END
 // (op 467) end: generate({461})
 // (op 467) Unlock Scope 467
 // Unlock None
 // (op 218) Scope(467) END
 // (op 218) end: generate({467})
 // (op 217) SetCoordinate(218) END
 // (op 217) end: generate({218})
 // (op 340) SetCoordinate(217) END
 // (op 340) Deallocate{}(1247) BEGIN
 // (op 1247) Deallocate 431
 // Deleting tag 431
 // (op 1247) Deallocate 142
 // Deleting tag 142
 // (op 1247) Deallocate 586
 // Deleting tag 586
 // (op 1247) Deallocate 602
 // Deleting tag 602
 // (op 1247) Deallocate 618
 // Deleting tag 618
 // (op 1247) Deallocate 634
 // Deleting tag 634
 // (op 1247) Deallocate 4
 // Deleting tag 4
 // (op 1247) Deallocate 512
 // Freeing DataFlowTag512: 4 SGPRs (Buffer: None) (op 564): s44, s45, s46, s47

 // Deleting tag 512
 // (op 1247) Deallocate 208
 // Deleting tag 208
 // (op 1247) Deallocate 76
 // Deleting tag 76
 // (op 1247) Deallocate 493
 // Freeing DataFlowTag493: 4 SGPRs (Buffer: None) (op 500): s40, s41, s42, s43

 // Deleting tag 493
 // (op 1247) Deallocate 268
 // Deleting tag 268
 // (op 1247) Deallocate 531
 // Freeing DataFlowTag531: 4 SGPRs (Buffer: None) (op 628): s48, s49, s50, s51

 // Deleting tag 531
 // (op 1247) Deallocate 17
 // Deleting tag 17
 // (op 1247) Deallocate 223
 // Deleting tag 223
 // (op 1247) Deallocate 25
 // Deleting tag 25
 // (op 1247) Deallocate 91
 // Deleting tag 91
 // (op 1247) Deallocate 3
 // Deleting tag 3
 // (op 1247) Deallocate 2
 // Deleting tag 2
 // (op 1247) Deallocate 550
 // Freeing DataFlowTag550: 4 SGPRs (Buffer: None) (op 692): s52, s53, s54, s55

 // Deleting tag 550
 // (op 1247) Deallocate 433
 // Deleting tag 433
 // (op 1247) Deallocate 41
 // Deleting tag 41
 // (op 1247) Deallocate 33
 // Deleting tag 33
 // (op 1247) Deallocate 283
 // Deleting tag 283
 // (op 1247) Deallocate 429
 // Deleting tag 429
 // (op 1247) Deallocate 427
 // Deleting tag 427
 // (op 1247) Deallocate 157
 // Deleting tag 157
 // (op 340) Deallocate{}(1247) END
 // (op 340) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer, Tensor_2_stride_1_3, Tensor_4_pointer}(1381) BEGIN
 // (op 1381) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1381) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1381) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1381) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1381) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1381) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1381) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1381) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 1381) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1381) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 340) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer, Tensor_2_stride_1_3, Tensor_4_pointer}(1381) END
 // (op 340) Barrier(1249) BEGIN
s_barrier  // (op 1249) 
 // (op 340) Barrier(1249) END
 // (op 340) end: generate({345})
 // (op 340) Unlock Scope 340
 // Unlock None
 // (op 1) Scope(340) END
 // (op 1) Deallocate{}(1287) BEGIN
 // (op 1287) Deallocate 635
 // Deleting tag 635
 // (op 1287) Deallocate 665
 // Deleting tag 665
 // (op 1287) Deallocate 633
 // Deleting tag 633
 // (op 1287) Deallocate 584
 // Freeing DataFlowTag584: 1 VGPR (Value: UInt32) (op 787): v3

 // Deleting tag 584
 // (op 1287) Deallocate 654
 // Freeing DataFlowTag654: 1 VGPR (Value: UInt32) (op 1019): v7

 // Deleting tag 654
 // (op 1287) Deallocate 670
 // Freeing DataFlowTag670: 1 VGPR (Value: UInt32) (op 1067): v5

 // Deleting tag 670
 // (op 1287) Deallocate 663
 // Deleting tag 663
 // (op 1287) Deallocate 632
 // Freeing DataFlowTag632: 1 VGPR (Value: UInt32) (op 946): v6

 // Deleting tag 632
 // (op 1287) Deallocate 671
 // Deleting tag 671
 // (op 1287) Deallocate 587
 // Deleting tag 587
 // (op 1287) Deallocate 585
 // Deleting tag 585
 // (op 1287) Deallocate 673
 // Deleting tag 673
 // (op 1287) Deallocate 657
 // Deleting tag 657
 // (op 1287) Deallocate 619
 // Deleting tag 619
 // (op 1287) Deallocate 678
 // Freeing DataFlowTag678: 1 VGPR (Value: UInt32) (op 1091): v10

 // Deleting tag 678
 // (op 1287) Deallocate 681
 // Deleting tag 681
 // (op 1287) Deallocate 655
 // Deleting tag 655
 // (op 1287) Deallocate 617
 // Deleting tag 617
 // (op 1287) Deallocate 601
 // Deleting tag 601
 // (op 1287) Deallocate 679
 // Deleting tag 679
 // (op 1287) Deallocate 616
 // Freeing DataFlowTag616: 1 VGPR (Value: UInt32) (op 893): v0

 // Deleting tag 616
 // (op 1287) Deallocate 662
 // Freeing DataFlowTag662: 1 VGPR (Value: UInt32) (op 1043): v8

 // Deleting tag 662
 // (op 1287) Deallocate 603
 // Deleting tag 603
 // (op 1287) Deallocate 600
 // Freeing DataFlowTag600: 1 VGPR (Value: UInt32) (op 840): v4

 // Deleting tag 600
 // (op 1) Deallocate{}(1287) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(133) BEGIN
 // (op 133) Lock for Conditional Branch
 // Lock Branch
 // (op 133) Generate Equal(0.00000:S, DataFlowTag(44)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 133) reg expression
 // (op 133) Equal(0.00000:S, v9:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v9 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
 // (op 133) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 133) 
s_cbranch_vccz RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
 // (op 133) generate({348})
 // (op 133) Scope(348) BEGIN
 // (op 348) Lock Scope 348
 // Lock Branch
 // (op 348) generate({351})
 // (op 348) NOP(351) BEGIN
 // (op 351) generate({})
 // (op 351) end: generate({})
 // (op 348) NOP(351) END
 // (op 348) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) BEGIN
 // (op 50) Assign dim(59) = Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA
 // (op 50) Generate Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA into nullptr
 // (op 50) reg expression
 // (op 50) reg expression
 // (op 50) Multiply(v11:S, {DataFlowTag55: a[0:15]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 50): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
s_nop 4
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
// Allocated : 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
v_mul_f32 v28, v11, v12 // (op 50) 
v_mul_f32 v29, v11, v13 // (op 50) 
v_mul_f32 v30, v11, v14 // (op 50) 
v_mul_f32 v31, v11, v15 // (op 50) 
v_mul_f32 v32, v11, v16 // (op 50) 
v_mul_f32 v33, v11, v17 // (op 50) 
v_mul_f32 v34, v11, v18 // (op 50) 
v_mul_f32 v35, v11, v19 // (op 50) 
v_mul_f32 v36, v11, v20 // (op 50) 
v_mul_f32 v37, v11, v21 // (op 50) 
v_mul_f32 v38, v11, v22 // (op 50) 
v_mul_f32 v39, v11, v23 // (op 50) 
v_mul_f32 v40, v11, v24 // (op 50) 
v_mul_f32 v41, v11, v25 // (op 50) 
v_mul_f32 v42, v11, v26 // (op 50) 
v_mul_f32 v43, v11, v27 // (op 50) 
 // Freeing : 16 VGPRs (Value: Float) (op 50): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // (op 348) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) END
 // (op 348) NOP(956) BEGIN
 // (op 956) generate({})
 // (op 956) end: generate({})
 // (op 348) NOP(956) END
 // (op 348) NOP(953) BEGIN
 // (op 953) generate({})
 // (op 953) end: generate({})
 // (op 348) NOP(953) END
 // (op 348) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(960) BEGIN
 // (op 960) Assign dim(636) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 636: v**UNALLOCATED**
 // (op 960) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag636: VGPR Value: UInt32 x 1: (unallocated)
 // (op 960) reg expression
 // (op 960) reg expression
 // (op 960) reg expression
 // (op 960) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 960) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 960): v0
v_lshrrev_b32 v0, 6, v1 // (op 960) 
 // (op 960) convert
 // (op 960) BEGIN: Tile[1]
 // (op 960) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 960): v10
v_and_b32 v10, 63, v1 // (op 960) 
 // (op 960) END: Tile[1]
 // (op 960) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 960): v3
v_lshrrev_b32 v3, 5, v10 // (op 960) 
 // (op 960) BEGIN: Flatten
 // (op 960) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 960) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 960)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 960)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 960): v4
v_lshlrev_b32 v4, 2, v3 // (op 960) 
 // (op 960) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 960): v3

 // (op 960) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 960): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 960) 
 // (op 960) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 960): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 960): v0

 // (op 960) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 960): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 960) 
 // (op 960) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 960): v3

 // (op 960) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 960): v3
v_lshrrev_b32 v3, 2, v10 // (op 960) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 960): v10

 // (op 960) BEGIN: Tile[1]
 // (op 960) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 960): v10
v_and_b32 v10, 7, v3 // (op 960) 
 // (op 960) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 960): v3

 // (op 960) BEGIN: Tile[1]
 // (op 960) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 960): v3
v_and_b32 v3, 3, v1 // (op 960) 
 // (op 960) END: Tile[1]
 // (op 960) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 960): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 960) 
 // (op 960) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 960): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 960): v10

 // (op 960) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 960): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 960) 
 // (op 960) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 960): v4

 // (op 960) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 960): v10
v_mul_lo_u32 v10, v3, s30 // (op 960) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 960): v3

 // (op 960) BEGIN: Join
// Allocated DataFlowTag636: 1 VGPR (Value: UInt32) (op 960): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 960) 
 // (op 960) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 960): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 960): v0

 // (op 348) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(960) END
 // (op 348) Assign VGPR 32:U64(961) BEGIN
 // (op 961) Assign dim(637) = 32:U64
 // (op 961) Assign stride expression
 // (op 348) Assign VGPR 32:U64(961) END
 // (op 348) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(962) BEGIN
 // (op 962) Assign dim(639) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 639: s**UNALLOCATED**
 // (op 962) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag639: SGPR Buffer: None x 1: (unallocated)
 // (op 962) reg expression
 // (op 962) reg expression
 // (op 962) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 962) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 962)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 962)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 962): s14, s15
s_mul_i32 s15, s30, s13 // (op 962) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 962): s0
s_mul_i32 s0, s31, s12 // (op 962) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 962) least significant
// Allocated : 1 SGPR (Value: Int32) (op 962): s1
s_mul_hi_u32 s1, s30, s12 // (op 962) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 962) most significant: sum
s_add_u32 s15, s15, s1 // (op 962) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 962): s1

 // Freeing : 1 SGPR (Value: Int32) (op 962): s0

 // (op 962) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 962) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 962)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 962)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 962): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 962) 
 // Freeing : 2 SGPRs (Value: Int64) (op 962): s14, s15

 // (op 962) reg expression
 // (op 962) reg expression
// Allocated DataFlowTag639: 4 SGPRs (Buffer: None) (op 962): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 962) reg expression
s_mov_b32 s9, s33 // (op 962) reg expression
s_mov_b32 s10, s0 // (op 962) reg expression
s_mov_b32 s11, 131072 // (op 962) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 962): s0, s1

 // (op 348) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(962) END
 // (op 348) NOP(954) BEGIN
 // (op 954) generate({})
 // (op 954) end: generate({})
 // (op 348) NOP(954) END
 // (op 348) Assign VGPR 4:U64(967) BEGIN
 // (op 967) Assign dim(641) = 4:U64
 // (op 967) Assign stride expression
 // (op 348) Assign VGPR 4:U64(967) END
 // (op 348) StoreTiled Value: Float(56) BEGIN
 // (op 56) GEN: storeMacroTileWAVE OP 56 MacroTile 59 WaveTile 365
 // (op 56) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 56) Generate 32:U64 into nullptr
 // (op 56) Generate 4:U64 into nullptr
 // (op 56) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Float x 16: v[28:43]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 56) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Float x 16: v[28:43]
 // 	info.rowOffsetReg = DataFlowTag636: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 56) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx4 v[28:31], v3, s[8:11], 0 offen // (op 56) Store value
// VMEM: Expected complete at 5915 (current 5435)
buffer_store_dwordx4 v[32:35], v3, s[8:11], 0 offen offset:32 // (op 56) Store value
// VMEM: Expected complete at 5916 (current 5436)
buffer_store_dwordx4 v[36:39], v3, s[8:11], 0 offen offset:64 // (op 56) Store value
// VMEM: Expected complete at 5917 (current 5437)
buffer_store_dwordx4 v[40:43], v3, s[8:11], 0 offen offset:96 // (op 56) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6396 (current 5916)
 // (op 348) StoreTiled Value: Float(56) END
 // (op 348) Deallocate{}(1213) BEGIN
 // (op 1213) Deallocate 640
 // Deleting tag 640
 // (op 1213) Deallocate 636
 // Freeing DataFlowTag636: 1 VGPR (Value: UInt32) (op 960): v3

 // Deleting tag 636
 // (op 1213) Deallocate 641
 // Deleting tag 641
 // (op 1213) Deallocate 637
 // Deleting tag 637
 // (op 1213) Deallocate 59
 // Freeing DataFlowTag59: 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 59
 // (op 348) Deallocate{}(1213) END
 // (op 348) end: generate({351})
 // (op 348) Unlock Scope 348
 // Unlock None
 // (op 133) Scope(348) END
 // (op 133) end: generate({348})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 133) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 133) 
s_branch RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
 // (op 133) 
 // (op 133) generate({354})
 // (op 133) Scope(354) BEGIN
 // (op 354) Lock Scope 354
 // Lock Branch
 // (op 354) generate({357})
 // (op 354) NOP(357) BEGIN
 // (op 357) generate({})
 // (op 357) end: generate({})
 // (op 354) NOP(357) END
 // (op 354) NOP(141) BEGIN
 // (op 141) generate({})
 // (op 141) end: generate({})
 // (op 354) NOP(141) END
 // (op 354) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(142) BEGIN
 // (op 142) Assign dim(423) = Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA
 // (op 142) Generate Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA into nullptr
 // (op 142) reg expression
 // (op 142) reg expression
 // (op 142) Multiply(v11:S, {DataFlowTag55: a[0:15]:S})S
 // (op 354) Scope(725) BEGIN
 // (op 725) Lock Scope 725
 // Lock Branch
 // (op 725) generate({722})
 // (op 725) NOP(722) BEGIN
 // (op 722) generate({})
 // (op 722) end: generate({})
 // (op 725) NOP(722) END
 // (op 725) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(730) BEGIN
 // (op 730) Assign dim(566) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}
 // tag 566: v**UNALLOCATED**
 // (op 730) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32} into DataFlowTag566: VGPR Value: UInt32 x 1: (unallocated)
 // (op 730) reg expression
 // (op 730) reg expression
 // (op 730) reg expression
 // (op 730) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 730) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 730): v0
v_lshrrev_b32 v0, 6, v1 // (op 730) 
 // (op 730) convert
 // (op 730) BEGIN: Flatten[1]
 // (op 730) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 730): v10
v_and_b32 v10, 63, v1 // (op 730) 
 // (op 730) END: Flatten[1]
 // (op 730) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v3
v_lshrrev_b32 v3, 5, v10 // (op 730) 
 // (op 730) BEGIN: Tile
 // (op 730) {Tile: ShiftL(v3:U32, 2:U32)U32}
 // (op 730) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 730)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 730)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 730): v4
v_lshlrev_b32 v4, 2, v3 // (op 730) 
 // (op 730) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v3

 // (op 730) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 730): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 730) 
 // (op 730) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 730): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 730): v0

 // (op 730) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 730): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 730) 
 // (op 730) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 730): v3

 // (op 730) LogicalShiftR({Flatten[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v3
v_lshrrev_b32 v3, 2, v10 // (op 730) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 730): v10

 // (op 730) BEGIN: Flatten[1]
 // (op 730) {Flatten[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 730): v10
v_and_b32 v10, 7, v3 // (op 730) 
 // (op 730) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v3

 // (op 730) BEGIN: Flatten[1]
 // (op 730) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 730): v3
v_and_b32 v3, 3, v1 // (op 730) 
 // (op 730) END: Flatten[1]
 // (op 730) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 730): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 730) 
 // (op 730) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 730): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 730): v10

 // (op 730) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 730): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 730) 
 // (op 730) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 730): v4

 // (op 730) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v10
v_mul_lo_u32 v10, v3, s34 // (op 730) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 730): v3

 // (op 730) BEGIN: Split
// Allocated DataFlowTag566: 1 VGPR (Value: UInt32) (op 730): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 730) 
 // (op 730) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 730): v0

 // (op 725) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(730) END
 // (op 725) Deallocate{}(1315) BEGIN
 // (op 1315) Deallocate 435
 // Deleting tag 435
 // (op 1315) Deallocate 437
 // Deleting tag 437
 // (op 1315) Deallocate 324
 // Deleting tag 324
 // (op 725) Deallocate{}(1315) END
 // (op 725) Assign VGPR 32:U64(731) BEGIN
 // (op 731) Assign dim(567) = 32:U64
 // (op 731) Assign stride expression
 // (op 725) Assign VGPR 32:U64(731) END
 // (op 725) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(732) BEGIN
 // (op 732) Assign dim(569) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 569: s**UNALLOCATED**
 // (op 732) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag569: SGPR Buffer: None x 1: (unallocated)
 // (op 732) reg expression
 // (op 732) reg expression
 // (op 732) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 732) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 732)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 732)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 732): s14, s15
s_mul_i32 s15, s34, s13 // (op 732) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 732): s0
s_mul_i32 s0, s35, s12 // (op 732) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 732) least significant
// Allocated : 1 SGPR (Value: Int32) (op 732): s1
s_mul_hi_u32 s1, s34, s12 // (op 732) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 732) most significant: sum
s_add_u32 s15, s15, s1 // (op 732) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 732): s1

 // Freeing : 1 SGPR (Value: Int32) (op 732): s0

 // (op 732) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 732) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 732)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 732)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 732): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 732) 
 // Freeing : 2 SGPRs (Value: Int64) (op 732): s14, s15

 // (op 732) reg expression
 // (op 732) reg expression
// Allocated DataFlowTag569: 4 SGPRs (Buffer: None) (op 732): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 732) reg expression
s_mov_b32 s5, s37 // (op 732) reg expression
s_mov_b32 s6, s0 // (op 732) reg expression
s_mov_b32 s7, 131072 // (op 732) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 732): s0, s1

 // (op 725) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(732) END
 // (op 725) Deallocate{Tensor_10_pointer}(1385) BEGIN
 // (op 1385) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 725) Deallocate{Tensor_10_pointer}(1385) END
 // (op 725) NOP(723) BEGIN
 // (op 723) generate({})
 // (op 723) end: generate({})
 // (op 725) NOP(723) END
 // (op 725) Assign VGPR 4:U64(737) BEGIN
 // (op 737) Assign dim(571) = 4:U64
 // (op 737) Assign stride expression
 // (op 725) Assign VGPR 4:U64(737) END
 // (op 725) LoadTiled Value: Float(145) BEGIN
 // (op 145) GEN: loadMacroTileWAVECIACCUM OP 145 WaveTile 325
 // (op 145) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 145) Generate 32:U64 into nullptr
 // (op 145) Generate 4:U64 into nullptr
 // tag 419: v**UNALLOCATED**
 // (op 145) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag569: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 145) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag566: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag569: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 145) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 145) Load value
// VMEM: Expected complete at 6904 (current 6424)
buffer_load_dwordx4 v[16:19], v3, s[4:7], 0 offen offset:32 // (op 145) Load value
// VMEM: Expected complete at 6905 (current 6425)
buffer_load_dwordx4 v[20:23], v3, s[4:7], 0 offen offset:64 // (op 145) Load value
// VMEM: Expected complete at 6906 (current 6426)
buffer_load_dwordx4 v[24:27], v3, s[4:7], 0 offen offset:96 // (op 145) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 7385 (current 6905)
 // (op 725) LoadTiled Value: Float(145) END
 // (op 725) Deallocate{Tensor_10_stride_1_10}(1227) BEGIN
 // (op 1227) Deallocate 5
 // Deleting tag 5
 // (op 1227) Deallocate 566
 // Freeing DataFlowTag566: 1 VGPR (Value: UInt32) (op 730): v3

 // Deleting tag 566
 // (op 1227) Deallocate 570
 // Deleting tag 570
 // (op 1227) Deallocate 571
 // Deleting tag 571
 // (op 1227) Deallocate 569
 // Freeing DataFlowTag569: 4 SGPRs (Buffer: None) (op 732): s4, s5, s6, s7

 // Deleting tag 569
 // (op 1227) Deallocate 567
 // Deleting tag 567
 // (op 1227) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 725) Deallocate{Tensor_10_stride_1_10}(1227) END
 // (op 725) end: generate({722})
 // (op 725) Unlock Scope 725
 // Unlock None
 // (op 354) Scope(725) END
// Allocated : 16 VGPRs (Value: Float) (op 142): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
v_accvgpr_read v28, a0 // (op 142) 
 // (op 354) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA(146) BEGIN
 // (op 146) Assign dim(421) = Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA
 // (op 146) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA into nullptr
 // (op 146) reg expression
 // (op 146) reg expression
 // (op 146) Multiply(v9:S, v[12:27]:S)S
v_accvgpr_read v29, a1 // (op 142) 
v_accvgpr_read v30, a2 // (op 142) 
v_accvgpr_read v31, a3 // (op 142) 
v_accvgpr_read v32, a4 // (op 142) 
v_accvgpr_read v33, a5 // (op 142) 
v_accvgpr_read v34, a6 // (op 142) 
v_accvgpr_read v35, a7 // (op 142) 
v_accvgpr_read v36, a8 // (op 142) 
v_accvgpr_read v37, a9 // (op 142) 
v_accvgpr_read v38, a10 // (op 142) 
v_accvgpr_read v39, a11 // (op 142) 
v_accvgpr_read v40, a12 // (op 142) 
v_accvgpr_read v41, a13 // (op 142) 
v_accvgpr_read v42, a14 // (op 142) 
v_accvgpr_read v43, a15 // (op 142) 
// Allocated : 16 VGPRs (Value: Float) (op 142): v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59
v_mul_f32 v44, v11, v28 // (op 142) 
v_mul_f32 v45, v11, v29 // (op 142) 
v_mul_f32 v46, v11, v30 // (op 142) 
v_mul_f32 v47, v11, v31 // (op 142) 
v_mul_f32 v48, v11, v32 // (op 142) 
v_mul_f32 v49, v11, v33 // (op 142) 
v_mul_f32 v50, v11, v34 // (op 142) 
v_mul_f32 v51, v11, v35 // (op 142) 
v_mul_f32 v52, v11, v36 // (op 142) 
v_mul_f32 v53, v11, v37 // (op 142) 
v_mul_f32 v54, v11, v38 // (op 142) 
v_mul_f32 v55, v11, v39 // (op 142) 
v_mul_f32 v56, v11, v40 // (op 142) 
v_mul_f32 v57, v11, v41 // (op 142) 
v_mul_f32 v58, v11, v42 // (op 142) 
v_mul_f32 v59, v11, v43 // (op 142) 
 // Freeing : 16 VGPRs (Value: Float) (op 142): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // (op 354) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(142) END
// Allocated : 16 VGPRs (Value: Float) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
s_waitcnt vmcnt(3)
v_mul_f32 v28, v9, v12 // (op 146) 
v_mul_f32 v29, v9, v13 // (op 146) 
v_mul_f32 v30, v9, v14 // (op 146) 
v_mul_f32 v31, v9, v15 // (op 146) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v9, v16 // (op 146) 
v_mul_f32 v33, v9, v17 // (op 146) 
v_mul_f32 v34, v9, v18 // (op 146) 
v_mul_f32 v35, v9, v19 // (op 146) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v9, v20 // (op 146) 
v_mul_f32 v37, v9, v21 // (op 146) 
v_mul_f32 v38, v9, v22 // (op 146) 
v_mul_f32 v39, v9, v23 // (op 146) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v9, v24 // (op 146) 
v_mul_f32 v41, v9, v25 // (op 146) 
v_mul_f32 v42, v9, v26 // (op 146) 
v_mul_f32 v43, v9, v27 // (op 146) 
 // (op 354) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(419)NA)NA(146) END
 // (op 354) Deallocate{}(1229) BEGIN
 // (op 1229) Deallocate 419
 // Freeing : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 419
 // (op 354) Deallocate{}(1229) END
 // (op 354) Assign VGPR Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA(143) BEGIN
 // (op 143) Assign dim(425) = Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA
 // (op 143) Generate Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA into nullptr
 // (op 143) reg expression
 // (op 143) reg expression
 // (op 143) Add({DataFlowTag421: v[28:43]:S}, {DataFlowTag423: v[44:59]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 143): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
v_add_f32 v12, v28, v44 // (op 143) 
v_add_f32 v13, v29, v45 // (op 143) 
v_add_f32 v14, v30, v46 // (op 143) 
v_add_f32 v15, v31, v47 // (op 143) 
v_add_f32 v16, v32, v48 // (op 143) 
v_add_f32 v17, v33, v49 // (op 143) 
v_add_f32 v18, v34, v50 // (op 143) 
v_add_f32 v19, v35, v51 // (op 143) 
v_add_f32 v20, v36, v52 // (op 143) 
v_add_f32 v21, v37, v53 // (op 143) 
v_add_f32 v22, v38, v54 // (op 143) 
v_add_f32 v23, v39, v55 // (op 143) 
v_add_f32 v24, v40, v56 // (op 143) 
v_add_f32 v25, v41, v57 // (op 143) 
v_add_f32 v26, v42, v58 // (op 143) 
v_add_f32 v27, v43, v59 // (op 143) 
 // (op 354) Assign VGPR Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA(143) END
 // (op 354) Deallocate{}(1223) BEGIN
 // (op 1223) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 142): v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59

 // Deleting tag 423
 // (op 1223) Deallocate 421
 // Freeing DataFlowTag421: 16 VGPRs (Value: Float) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 421
 // (op 354) Deallocate{}(1223) END
 // (op 354) NOP(973) BEGIN
 // (op 973) generate({})
 // (op 973) end: generate({})
 // (op 354) NOP(973) END
 // (op 354) NOP(970) BEGIN
 // (op 970) generate({})
 // (op 970) end: generate({})
 // (op 354) NOP(970) END
 // (op 354) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(977) BEGIN
 // (op 977) Assign dim(642) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 642: v**UNALLOCATED**
 // (op 977) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag642: VGPR Value: UInt32 x 1: (unallocated)
 // (op 977) reg expression
 // (op 977) reg expression
 // (op 977) reg expression
 // (op 977) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 977) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 977): v0
v_lshrrev_b32 v0, 6, v1 // (op 977) 
 // (op 977) convert
 // (op 977) BEGIN: Tile[1]
 // (op 977) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 977): v10
v_and_b32 v10, 63, v1 // (op 977) 
 // (op 977) END: Tile[1]
 // (op 977) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 977): v3
v_lshrrev_b32 v3, 5, v10 // (op 977) 
 // (op 977) BEGIN: Flatten
 // (op 977) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 977) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 977)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 977)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 977): v4
v_lshlrev_b32 v4, 2, v3 // (op 977) 
 // (op 977) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 977): v3

 // (op 977) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 977): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 977) 
 // (op 977) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 977): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 977): v0

 // (op 977) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 977): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 977) 
 // (op 977) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 977): v3

 // (op 977) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 977): v3
v_lshrrev_b32 v3, 2, v10 // (op 977) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 977): v10

 // (op 977) BEGIN: Tile[1]
 // (op 977) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 977): v10
v_and_b32 v10, 7, v3 // (op 977) 
 // (op 977) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 977): v3

 // (op 977) BEGIN: Tile[1]
 // (op 977) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 977): v3
v_and_b32 v3, 3, v1 // (op 977) 
 // (op 977) END: Tile[1]
 // (op 977) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 977): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 977) 
 // (op 977) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 977): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 977): v10

 // (op 977) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 977): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 977) 
 // (op 977) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 977): v4

 // (op 977) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 977): v10
v_mul_lo_u32 v10, v3, s30 // (op 977) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 977): v3

 // (op 977) BEGIN: Join
// Allocated DataFlowTag642: 1 VGPR (Value: UInt32) (op 977): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 977) 
 // (op 977) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 977): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 977): v0

 // (op 354) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(977) END
 // (op 354) Assign VGPR 32:U64(978) BEGIN
 // (op 978) Assign dim(643) = 32:U64
 // (op 978) Assign stride expression
 // (op 354) Assign VGPR 32:U64(978) END
 // (op 354) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(979) BEGIN
 // (op 979) Assign dim(639) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 979) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // (op 979) reg expression
 // (op 979) reg expression
 // (op 979) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 979) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 979)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 979)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 979): s14, s15
s_mul_i32 s15, s30, s13 // (op 979) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 979): s0
s_mul_i32 s0, s31, s12 // (op 979) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 979) least significant
// Allocated : 1 SGPR (Value: Int32) (op 979): s1
s_mul_hi_u32 s1, s30, s12 // (op 979) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 979) most significant: sum
s_add_u32 s15, s15, s1 // (op 979) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 979): s1

 // Freeing : 1 SGPR (Value: Int32) (op 979): s0

 // (op 979) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 979) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 979)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 979)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 979): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 979) 
 // Freeing : 2 SGPRs (Value: Int64) (op 979): s14, s15

 // (op 979) reg expression
 // (op 979) reg expression
s_mov_b32 s8, s32 // (op 979) reg expression
s_mov_b32 s9, s33 // (op 979) reg expression
s_mov_b32 s10, s0 // (op 979) reg expression
s_mov_b32 s11, 131072 // (op 979) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 979): s0, s1

 // (op 354) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(979) END
 // (op 354) NOP(971) BEGIN
 // (op 971) generate({})
 // (op 971) end: generate({})
 // (op 354) NOP(971) END
 // (op 354) Assign VGPR 4:U64(984) BEGIN
 // (op 984) Assign dim(645) = 4:U64
 // (op 984) Assign stride expression
 // (op 354) Assign VGPR 4:U64(984) END
 // (op 354) StoreTiled Value: Float(144) BEGIN
 // (op 144) GEN: storeMacroTileWAVE OP 144 MacroTile 425 WaveTile 365
 // (op 144) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 144) Generate 32:U64 into nullptr
 // (op 144) Generate 4:U64 into nullptr
 // (op 144) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag425: VGPR Value: Float x 16: v[12:27]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 144) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag425: VGPR Value: Float x 16: v[12:27]
 // 	info.rowOffsetReg = DataFlowTag642: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(0:U32, 4:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 144) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 144) Store value
// VMEM: Expected complete at 7911 (current 7431)
buffer_store_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 144) Store value
// VMEM: Expected complete at 7912 (current 7432)
buffer_store_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 144) Store value
// VMEM: Expected complete at 7913 (current 7433)
buffer_store_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 144) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 8392 (current 7912)
 // (op 354) StoreTiled Value: Float(144) END
 // (op 354) Deallocate{}(1225) BEGIN
 // (op 1225) Deallocate 644
 // Deleting tag 644
 // (op 1225) Deallocate 645
 // Deleting tag 645
 // (op 1225) Deallocate 643
 // Deleting tag 643
 // (op 1225) Deallocate 642
 // Freeing DataFlowTag642: 1 VGPR (Value: UInt32) (op 977): v3

 // Deleting tag 642
 // (op 1225) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 143): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 425
 // (op 354) Deallocate{}(1225) END
 // (op 354) end: generate({357})
 // (op 354) Unlock Scope 354
 // Unlock None
 // (op 133) Scope(354) END
 // (op 133) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1289) BEGIN
 // (op 1289) Deallocate 364
 // Deleting tag 364
 // (op 1289) Deallocate 639
 // Freeing DataFlowTag639: 4 SGPRs (Buffer: None) (op 962): s8, s9, s10, s11

 // Deleting tag 639
 // (op 1289) Deallocate 441
 // Deleting tag 441
 // (op 1289) Deallocate 439
 // Deleting tag 439
 // (op 1289) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1289) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 133) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1289) END
 // (op 133) end: generate({354})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 133) 
RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:
 // (op 133) 
 // (op 133) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(133) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1221) BEGIN
 // (op 1221) Deallocate 55
 // Freeing DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 63): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 55
 // (op 1221) Deallocate 53
 // Freeing : 1 VGPR (Value: Float) (op 43): v11

 // Deleting tag 53
 // (op 1221) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 2): v9

 // Deleting tag 44
 // (op 1221) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1221) END
 // (op 1) end: generate({783, 836, 889, 942, 1015, 1039, 1063, 1087})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 7424 // lds bytes
  .amdhsa_accum_offset 64
 // Initial kernel state
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 14
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 1
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocroller
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_BF8_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 7424
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 63
    .agpr_count: 16
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

