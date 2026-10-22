// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.FP4.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.FP6.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
 // T_LOAD_TILED 3 Source 2
 // Tensor.E8M0.d2 4, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
 // T_LOAD_TILED 5 Source 4
 // BlockScale(Separate, {1, 32}): Data: 1, Scale: 5
 // Tensor.E8M0.d2 7, (base=&120, sizes={&128 &136 }, strides={&144 &152 })
 // T_LOAD_TILED 8 Source 7
 // BlockScale(Separate, {32, 1}): Data: 3, Scale: 8
 // Tensor.Half.d2 10, (base=&160, sizes={&168 &176 }, strides={&184 &192 })
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
 // Tensor.Half.d2 21, (base=&208, sizes={&216 &224 }, strides={&232 &240 })
 // T_EXECUTE 19
 //   E_Cvt 22, 19
 // T_STORE_TILED 24 Source 22 Dest 21

 // Command arguments:
 // Tensor_0_pointer: PointerGlobal: FP4(offset: 0, size: 8, read_write)
 // Tensor_0_size_0: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: FP6(offset: 40, size: 8, read_write)
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
 // Tensor_10_pointer: PointerGlobal: Half(offset: 160, size: 8, read_write)
 // Tensor_10_size_0: Value: Int64(offset: 168, size: 8, read_only)
 // Tensor_10_size_1: Value: Int64(offset: 176, size: 8, read_only)
 // Tensor_10_stride_0: Value: Int64(offset: 184, size: 8, read_only)
 // Tensor_10_stride_1: Value: Int64(offset: 192, size: 8, read_only)
 // user_Float_Value_12: Value: Float(offset: 200, size: 4, read_only)
 // user_Float_Value_14: Value: Float(offset: 204, size: 4, read_only)
 // Tensor_21_pointer: PointerGlobal: Half(offset: 208, size: 8, read_write)
 // Tensor_21_size_0: Value: Int64(offset: 216, size: 8, read_only)
 // Tensor_21_size_1: Value: Int64(offset: 224, size: 8, read_only)
 // Tensor_21_stride_0: Value: Int64(offset: 232, size: 8, read_only)
 // Tensor_21_stride_1: Value: Int64(offset: 240, size: 8, read_only)

 // Kernel Arguments:
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: FP4, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP6, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}
.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
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
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:
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
 // KernelArg{Tensor_0_pointer, PointerGlobal: FP4, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP6, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: Half, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}

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
 // Tensor_0_pointer: SGPR PointerGlobal: FP4 x 1: s[4:5]
 // Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]
 // Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]
 // Tensor_2_pointer: SGPR PointerGlobal: FP6 x 1: s[10:11]
 // Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]
 // user_Float_Value_14: SGPR Value: Float x 1: s14
 // user_Float_Value_12: SGPR Value: Float x 1: s15
 // Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]
 // Tensor_4_pointer: SGPR PointerGlobal: E8M0 x 1: s[22:23]
 // Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]
 // Tensor_7_pointer: SGPR PointerGlobal: E8M0 x 1: s[26:27]
 // Tensor_0_size_1_14: SGPR Value: Int64 x 1: s[28:29]
 // Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]
 // Tensor_21_pointer: SGPR PointerGlobal: Half x 1: s[32:33]
 // Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]
 // Tensor_10_pointer: SGPR PointerGlobal: Half x 1: s[36:37]
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
 // (op 1) generate({765, 818, 871, 924, 982, 1006, 1030, 1054})
 // (op 1) NOP(765) BEGIN
 // (op 765) generate({})
 // (op 765) end: generate({})
 // (op 1) NOP(765) END
 // (op 1) NOP(818) BEGIN
 // (op 818) generate({})
 // (op 818) end: generate({})
 // (op 1) NOP(818) END
 // (op 1) NOP(871) BEGIN
 // (op 871) generate({})
 // (op 871) end: generate({})
 // (op 1) NOP(871) END
 // (op 1) NOP(924) BEGIN
 // (op 924) generate({})
 // (op 924) end: generate({})
 // (op 1) NOP(924) END
 // (op 1) NOP(982) BEGIN
 // (op 982) generate({})
 // (op 982) end: generate({})
 // (op 1) NOP(982) END
 // (op 1) NOP(1006) BEGIN
 // (op 1006) generate({})
 // (op 1006) end: generate({})
 // (op 1) NOP(1006) END
 // (op 1) NOP(1030) BEGIN
 // (op 1030) generate({})
 // (op 1030) end: generate({})
 // (op 1) NOP(1030) END
 // (op 1) NOP(1054) BEGIN
 // (op 1054) generate({})
 // (op 1054) end: generate({})
 // (op 1) NOP(1054) END
 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(769) BEGIN
 // (op 769) Assign dim(588) = LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 588: v**UNALLOCATED**
 // (op 769) Generate LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag588: VGPR Value: UInt32 x 1: (unallocated)
 // (op 769) reg expression
 // (op 769) BEGIN: Flatten[1]
 // (op 769) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(822) BEGIN
 // (op 822) Assign dim(604) = LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 604: v**UNALLOCATED**
 // (op 822) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag604: VGPR Value: UInt32 x 1: (unallocated)
 // (op 822) reg expression
 // (op 822) BEGIN: Flatten[1]
 // (op 822) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(875) BEGIN
 // (op 875) Assign dim(620) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 620: v**UNALLOCATED**
 // (op 875) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag620: VGPR Value: UInt32 x 1: (unallocated)
 // (op 875) reg expression
 // (op 875) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(928) BEGIN
 // (op 928) Assign dim(636) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 636: v**UNALLOCATED**
 // (op 928) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag636: VGPR Value: UInt32 x 1: (unallocated)
 // (op 928) reg expression
 // (op 928) BEGIN: Flatten[1]Tile
 // (op 928) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(986) BEGIN
 // (op 986) Assign dim(654) = LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 654: v**UNALLOCATED**
 // (op 986) Generate LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag654: VGPR Value: UInt32 x 1: (unallocated)
 // (op 986) reg expression
 // (op 986) BEGIN: Flatten[1]
 // (op 986) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1010) BEGIN
 // (op 1010) Assign dim(662) = LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 662: v**UNALLOCATED**
 // (op 1010) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag662: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1010) reg expression
 // (op 1010) BEGIN: Flatten[1]
 // (op 1010) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(1034) BEGIN
 // (op 1034) Assign dim(670) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 670: v**UNALLOCATED**
 // (op 1034) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag670: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1034) reg expression
 // (op 1034) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(1058) BEGIN
 // (op 1058) Assign dim(678) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 678: v**UNALLOCATED**
 // (op 1058) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag678: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1058) reg expression
 // (op 1058) BEGIN: Flatten[1]Tile
 // (op 1058) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 769): v0
v_and_b32 v0, 63, v1 // (op 769) 
 // (op 769) END: Flatten[1]
 // (op 769) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 769): v3
v_lshrrev_b32 v3, 4, v0 // (op 769) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 769): v0

 // (op 769) BEGIN: Flatten[1]
 // (op 769) {Flatten[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 769): v0
v_and_b32 v0, 1, v3 // (op 769) 
 // (op 769) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 769): v3

 // (op 769) BEGIN: Flatten[1]
 // (op 769) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 769): v3
v_and_b32 v3, 15, v1 // (op 769) 
 // (op 769) END: Flatten[1]
 // (op 769) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 769): v4
v_lshl_add_u32 v4, v0, 4, v3 // (op 769) 
 // (op 769) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 769): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 769): v0

 // (op 769) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 769): v0
v_lshrrev_b32 v0, 6, v1 // (op 769) 
 // (op 769) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 769): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 769) 
 // (op 769) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 769): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 769): v0

 // (op 769) BEGIN: TileTile
 // (op 769) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 769): v0
v_and_b32 v0, 32, v1 // (op 769) 
 // (op 769) END: TileTile
 // (op 769) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 769): v4
v_lshl_add_u32 v4, v3, 6, v0 // (op 769) 
 // (op 769) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 769): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 769): v3

 // (op 769) ShiftL({Tile: v4:U32}, 2:U32)U32
 // (op 769) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 769)         value (Tile: VGPR Value: UInt32 x 1: v4) 
 // (op 769)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 769): v0
v_lshlrev_b32 v0, 2, v4 // (op 769) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 769): v4

 // (op 769) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag588: 1 VGPR (Value: UInt32) (op 769): v3
v_lshrrev_b32 v3, 3, v0 // (op 769) 
 // Freeing : 1 VGPR (Value: UInt32) (op 769): v0

 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(769) END
// Allocated : 1 VGPR (Value: UInt32) (op 822): v0
v_and_b32 v0, 63, v1 // (op 822) 
 // (op 822) END: Flatten[1]
 // (op 822) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 822): v4
v_lshrrev_b32 v4, 4, v0 // (op 822) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 822): v0

 // (op 822) BEGIN: Flatten[1]
 // (op 822) {Flatten[1]: BitwiseAnd(v4:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 822): v0
v_and_b32 v0, 1, v4 // (op 822) 
 // (op 822) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 822): v4

 // (op 822) BEGIN: Flatten[1]
 // (op 822) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 822): v4
v_and_b32 v4, 15, v1 // (op 822) 
 // (op 822) END: Flatten[1]
 // (op 822) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 822): v5
v_lshl_add_u32 v5, v0, 4, v4 // (op 822) 
 // (op 822) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 822): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 822): v0

 // (op 822) BEGIN: TileTile
 // (op 822) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 822): v0
v_and_b32 v0, 32, v1 // (op 822) 
 // (op 822) END: TileTile
 // (op 822) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 822): v4
v_lshl_add_u32 v4, v5, 6, v0 // (op 822) 
 // (op 822) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 822): v0

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 822): v5

 // (op 822) Multiply({Tile: v4:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 822): v0
v_mul_lo_u32 v0, v4, 6 // (op 822) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 822): v4

 // (op 822) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag604: 1 VGPR (Value: UInt32) (op 822): v4
v_lshrrev_b32 v4, 3, v0 // (op 822) 
 // Freeing : 1 VGPR (Value: UInt32) (op 822): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(822) END
// Allocated : 1 VGPR (Value: UInt32) (op 875): v0
v_lshrrev_b32 v0, 6, v1 // (op 875) 
 // (op 875) BEGIN: Flatten[1]
 // (op 875) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 875): v5
v_and_b32 v5, 31, v1 // (op 875) 
 // (op 875) END: Flatten[1]
 // (op 875) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 875): v6
v_lshl_add_u32 v6, v0, 5, v5 // (op 875) 
 // (op 875) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 875): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 875): v0

 // (op 875) BEGIN: Flatten[1]
 // (op 875) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 875): v0
v_and_b32 v0, 63, v1 // (op 875) 
 // (op 875) END: Flatten[1]
 // (op 875) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 875): v5
v_lshrrev_b32 v5, 5, v0 // (op 875) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 875): v0

 // (op 875) BEGIN: Tile
// Allocated DataFlowTag620: 1 VGPR (Value: UInt32) (op 875): v0
v_lshl_add_u32 v0, v6, 1, v5 // (op 875) 
 // (op 875) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 875): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 875): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(875) END
// Allocated : 1 VGPR (Value: UInt32) (op 928): v5
v_and_b32 v5, 31, v1 // (op 928) 
 // (op 928) END: Flatten[1]Tile
 // (op 928) BEGIN: Flatten[1]
 // (op 928) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 928): v6
v_and_b32 v6, 63, v1 // (op 928) 
 // (op 928) END: Flatten[1]
 // (op 928) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 928): v7
v_lshrrev_b32 v7, 5, v6 // (op 928) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 928): v6

 // (op 928) BEGIN: Tile
// Allocated DataFlowTag636: 1 VGPR (Value: UInt32) (op 928): v6
v_lshl_add_u32 v6, v5, 1, v7 // (op 928) 
 // (op 928) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 928): v7

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 928): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(928) END
// Allocated : 1 VGPR (Value: UInt32) (op 986): v5
v_and_b32 v5, 63, v1 // (op 986) 
 // (op 986) END: Flatten[1]
 // (op 986) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 986): v7
v_lshrrev_b32 v7, 4, v5 // (op 986) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 986): v5

 // (op 986) BEGIN: Flatten[1]
 // (op 986) {Flatten[1]: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 986): v5
v_and_b32 v5, 1, v7 // (op 986) 
 // (op 986) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 986): v7

 // (op 986) BEGIN: Flatten[1]
 // (op 986) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 986): v7
v_and_b32 v7, 15, v1 // (op 986) 
 // (op 986) END: Flatten[1]
 // (op 986) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 986): v8
v_lshl_add_u32 v8, v5, 4, v7 // (op 986) 
 // (op 986) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 986): v7

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 986): v5

 // (op 986) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 986): v5
v_lshrrev_b32 v5, 6, v1 // (op 986) 
 // (op 986) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 986): v7
v_lshl_add_u32 v7, v5, 5, v8 // (op 986) 
 // (op 986) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 986): v8

 // Freeing : 1 VGPR (Value: UInt32) (op 986): v5

 // (op 986) BEGIN: TileTile
 // (op 986) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 986): v5
v_and_b32 v5, 32, v1 // (op 986) 
 // (op 986) END: TileTile
 // (op 986) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 986): v8
v_lshl_add_u32 v8, v7, 6, v5 // (op 986) 
 // (op 986) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 986): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 986): v7

 // (op 986) ShiftL({Tile: v8:U32}, 2:U32)U32
 // (op 986) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 986)         value (Tile: VGPR Value: UInt32 x 1: v8) 
 // (op 986)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 986): v5
v_lshlrev_b32 v5, 2, v8 // (op 986) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 986): v8

 // (op 986) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag654: 1 VGPR (Value: UInt32) (op 986): v7
v_lshrrev_b32 v7, 3, v5 // (op 986) 
 // Freeing : 1 VGPR (Value: UInt32) (op 986): v5

 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(986) END
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v5
v_and_b32 v5, 63, v1 // (op 1010) 
 // (op 1010) END: Flatten[1]
 // (op 1010) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
 // (op 1) Deallocate{}(1290) BEGIN
 // (op 1290) Deallocate 104
 // Deleting tag 104
 // (op 1) Deallocate{}(1290) END
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v8
v_lshrrev_b32 v8, 4, v5 // (op 1010) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1010): v5

 // (op 1010) BEGIN: Flatten[1]
 // (op 1010) {Flatten[1]: BitwiseAnd(v8:U32, 1:U32)U32}
 // (op 1) Assign VGPR 32:U32(770) BEGIN
 // (op 770) Assign dim(589) = 32:U32
 // (op 770) Assign stride expression
 // (op 1) Assign VGPR 32:U32(770) END
 // (op 1) Assign VGPR 32:U32(987) BEGIN
 // (op 987) Assign dim(655) = 32:U32
 // (op 987) Assign stride expression
 // (op 1) Assign VGPR 32:U32(987) END
 // (op 1) NOP(766) BEGIN
 // (op 766) generate({})
 // (op 766) end: generate({})
 // (op 1) NOP(766) END
 // (op 1) NOP(983) BEGIN
 // (op 983) generate({})
 // (op 983) end: generate({})
 // (op 1) NOP(983) END
 // (op 1) Assign VGPR 0:U32(774) BEGIN
 // (op 774) Assign dim(591) = 0:U32
 // (op 774) Assign stride expression
 // (op 1) Assign VGPR 0:U32(774) END
 // (op 1) Assign VGPR 0:U32(991) BEGIN
 // (op 991) Assign dim(657) = 0:U32
 // (op 991) Assign stride expression
 // (op 1) Assign VGPR 0:U32(991) END
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v5
v_and_b32 v5, 1, v8 // (op 1010) 
 // (op 1010) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1010): v8

 // (op 1010) BEGIN: Flatten[1]
 // (op 1010) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v8
v_and_b32 v8, 15, v1 // (op 1010) 
 // (op 1010) END: Flatten[1]
 // (op 1010) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v9
v_lshl_add_u32 v9, v5, 4, v8 // (op 1010) 
 // (op 1010) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1010): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1010): v5

 // (op 1010) BEGIN: TileTile
 // (op 1010) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v5
v_and_b32 v5, 32, v1 // (op 1010) 
 // (op 1010) END: TileTile
 // (op 1010) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v8
v_lshl_add_u32 v8, v9, 6, v5 // (op 1010) 
 // (op 1010) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1010): v5

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1010): v9

 // (op 1010) Multiply({Tile: v8:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1010): v5
v_mul_lo_u32 v5, v8, 6 // (op 1010) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1010): v8

 // (op 1010) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag662: 1 VGPR (Value: UInt32) (op 1010): v8
v_lshrrev_b32 v8, 3, v5 // (op 1010) 
 // Freeing : 1 VGPR (Value: UInt32) (op 1010): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 6:U32)U32, 3:U32)U32(1010) END
// Allocated : 1 VGPR (Value: UInt32) (op 1034): v5
v_lshrrev_b32 v5, 6, v1 // (op 1034) 
 // (op 1034) BEGIN: Flatten[1]
 // (op 1034) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Deallocate{}(1293) BEGIN
 // (op 1293) Deallocate 170
 // Deleting tag 170
 // (op 1) Deallocate{}(1293) END
// Allocated : 1 VGPR (Value: UInt32) (op 1034): v9
v_and_b32 v9, 31, v1 // (op 1034) 
 // (op 1034) END: Flatten[1]
 // (op 1034) BEGIN: Tile
 // (op 1) Assign VGPR 48:U32(823) BEGIN
 // (op 823) Assign dim(605) = 48:U32
 // (op 823) Assign stride expression
 // (op 1) Assign VGPR 48:U32(823) END
 // (op 1) Assign VGPR 48:U32(1011) BEGIN
 // (op 1011) Assign dim(663) = 48:U32
 // (op 1011) Assign stride expression
 // (op 1) Assign VGPR 48:U32(1011) END
 // (op 1) NOP(819) BEGIN
 // (op 819) generate({})
 // (op 819) end: generate({})
 // (op 1) NOP(819) END
 // (op 1) NOP(1007) BEGIN
 // (op 1007) generate({})
 // (op 1007) end: generate({})
 // (op 1) NOP(1007) END
 // (op 1) Assign VGPR 0:U32(827) BEGIN
 // (op 827) Assign dim(607) = 0:U32
 // (op 827) Assign stride expression
 // (op 1) Assign VGPR 0:U32(827) END
 // (op 1) Assign VGPR 0:U32(1015) BEGIN
 // (op 1015) Assign dim(665) = 0:U32
 // (op 1015) Assign stride expression
 // (op 1) Assign VGPR 0:U32(1015) END
// Allocated : 1 VGPR (Value: UInt32) (op 1034): v10
v_lshl_add_u32 v10, v5, 5, v9 // (op 1034) 
 // (op 1034) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1034): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1034): v5

 // (op 1034) BEGIN: Flatten[1]
 // (op 1034) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1034): v5
v_and_b32 v5, 63, v1 // (op 1034) 
 // (op 1034) END: Flatten[1]
 // (op 1034) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1034): v9
v_lshrrev_b32 v9, 5, v5 // (op 1034) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1034): v5

 // (op 1034) BEGIN: Tile
// Allocated DataFlowTag670: 1 VGPR (Value: UInt32) (op 1034): v5
v_lshl_add_u32 v5, v10, 1, v9 // (op 1034) 
 // (op 1034) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1034): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1034): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(1034) END
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v9
v_and_b32 v9, 31, v1 // (op 1058) 
 // (op 1058) END: Flatten[1]Tile
 // (op 1058) BEGIN: Flatten[1]
 // (op 1058) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1296) BEGIN
 // (op 1296) Deallocate 236
 // Deleting tag 236
 // (op 1) Deallocate{}(1296) END
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v10
v_and_b32 v10, 63, v1 // (op 1058) 
 // (op 1058) END: Flatten[1]
 // (op 1058) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
 // (op 1) Assign VGPR 2:U32(876) BEGIN
 // (op 876) Assign dim(621) = 2:U32
 // (op 876) Assign stride expression
 // (op 1) Assign VGPR 2:U32(876) END
 // (op 1) Assign VGPR 2:U32(1035) BEGIN
 // (op 1035) Assign dim(671) = 2:U32
 // (op 1035) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1035) END
 // (op 1) NOP(872) BEGIN
 // (op 872) generate({})
 // (op 872) end: generate({})
 // (op 1) NOP(872) END
 // (op 1) NOP(1031) BEGIN
 // (op 1031) generate({})
 // (op 1031) end: generate({})
 // (op 1) NOP(1031) END
 // (op 1) Assign VGPR 1:U32(880) BEGIN
 // (op 880) Assign dim(623) = 1:U32
 // (op 880) Assign stride expression
 // (op 1) Assign VGPR 1:U32(880) END
 // (op 1) Assign VGPR 1:U32(1039) BEGIN
 // (op 1039) Assign dim(673) = 1:U32
 // (op 1039) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1039) END
// Allocated : 1 VGPR (Value: UInt32) (op 1058): v11
v_lshrrev_b32 v11, 5, v10 // (op 1058) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1058): v10

 // (op 1058) BEGIN: Tile
// Allocated DataFlowTag678: 1 VGPR (Value: UInt32) (op 1058): v10
v_lshl_add_u32 v10, v9, 1, v11 // (op 1058) 
 // (op 1058) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1058): v11

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 1058): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(1058) END
 // (op 1) Deallocate{}(1299) BEGIN
 // (op 1299) Deallocate 296
 // Deleting tag 296
 // (op 1) Deallocate{}(1299) END
 // (op 1) Assign VGPR 2:U32(929) BEGIN
 // (op 929) Assign dim(637) = 2:U32
 // (op 929) Assign stride expression
 // (op 1) Assign VGPR 2:U32(929) END
 // (op 1) Assign VGPR 2:U32(1059) BEGIN
 // (op 1059) Assign dim(679) = 2:U32
 // (op 1059) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1059) END
 // (op 1) NOP(925) BEGIN
 // (op 925) generate({})
 // (op 925) end: generate({})
 // (op 1) NOP(925) END
 // (op 1) NOP(1055) BEGIN
 // (op 1055) generate({})
 // (op 1055) end: generate({})
 // (op 1) NOP(1055) END
 // (op 1) Assign VGPR 1:U32(933) BEGIN
 // (op 933) Assign dim(639) = 1:U32
 // (op 933) Assign stride expression
 // (op 1) Assign VGPR 1:U32(933) END
 // (op 1) Assign VGPR 1:U32(1063) BEGIN
 // (op 1063) Assign dim(681) = 1:U32
 // (op 1063) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1063) END
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
 // (op 1) Deallocate{user_Float_Value_14}(1353) BEGIN
 // (op 1353) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1353) END
 // (op 1) Deallocate{user_Float_Value_12}(1355) BEGIN
 // (op 1355) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1355) END
 // (op 1) Scope(344) BEGIN
 // (op 344) Lock Scope 344
 // Lock Branch
 // (op 344) generate({349})
 // (op 344) NOP(349) BEGIN
 // (op 349) generate({})
 // (op 349) end: generate({})
 // (op 344) NOP(349) END
 // (op 344) Assign ACCVGPR 0.00000:S(65) BEGIN
 // (op 65) Assign dim(55) = 0.00000:S
 // tag 55: a**UNALLOCATED**
 // (op 65) Generate 0.00000:S into DataFlowTag55: ACCVGPR Value: Float x 16: (unallocated)
// Allocated DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 65): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
 // (op 344) Assign ACCVGPR 0.00000:S(65) END
 // (op 344) Scope(471) BEGIN
 // (op 471) Lock Scope 471
 // Lock Branch
 // (op 471) generate({466})
 // (op 471) NOP(466) BEGIN
 // (op 466) generate({})
 // (op 466) end: generate({})
 // (op 471) NOP(466) END
 // (op 471) Assign VGPR 32:U64(476) BEGIN
 // (op 476) Assign dim(502) = 32:U64
 // (op 476) Assign stride expression
 // (op 471) Assign VGPR 32:U64(476) END
 // (op 471) NOP(467) BEGIN
 // (op 467) generate({})
 // (op 467) end: generate({})
 // (op 471) NOP(467) END
 // (op 471) Assign VGPR Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32(479) BEGIN
 // (op 479) Assign dim(503) = Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32
 // tag 503: v**UNALLOCATED**
 // (op 479) Generate Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32 into DataFlowTag503: VGPR Value: UInt32 x 1: (unallocated)
 // (op 479) reg expression
 // (op 479) reg expression
 // (op 479) reg expression
 // (op 479) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 479): v12
v_lshrrev_b32 v12, 1, v1 // (op 479) 
 // (op 479) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 479): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 479) 
 // (op 479) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 479): v12

 // (op 479) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 479): v12
v_and_b32 v12, -2, v1 // (op 479) 
 // (op 479) BEGIN: Flatten[1]
 // (op 479) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 479): v14
v_sub_u32 v14, v1, v12 // (op 479) 
 // (op 479) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 479): v12

 // (op 479) BEGIN: TileTile
 // (op 479) {TileTile: ShiftL({Flatten[1]: v14:U32}, 5:U32)U32}
 // (op 479) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 479)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 479)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 479): v12
v_lshlrev_b32 v12, 5, v14 // (op 479) 
 // (op 479) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 479): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 479): v14, v15
v_mov_b32 v15, 0 // (op 479) convert
v_mov_b32 v14, v12 // (op 479) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 479): v12

 // (op 479) BEGIN: Split
 // (op 479) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 479)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 479)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 479): v16, v17
v_mul_lo_u32 v17, v13, s3 // (op 479) most significant: low of low * high
 // (op 479) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 479): v12
v_mul_hi_u32 v12, v13, s2 // (op 479) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 479) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 479) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 479): v12

 // (op 479) Add: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 479)      lhs (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 479)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 479): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 479) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 479) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 479): s0, s1

 // (op 479) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 479): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 479): v13

 // (op 479) ShiftL({Split: v[16:17]:U64}, 2:U32)U64
 // (op 479) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 479)         value (Split: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 479)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 VGPRs (Value: UInt64) (op 479): v12, v13
v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 479) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 479): v16, v17

 // (op 479) LogicalShiftR(v[12:13]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 479): v14, v15
v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 479) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 479): v12, v13

// Allocated DataFlowTag503: 1 VGPR (Value: UInt32) (op 479): v12
v_mov_b32 v12, v14 // (op 479) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 479): v14, v15

 // (op 471) Assign VGPR Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32(479) END
 // (op 471) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64(480) BEGIN
 // (op 480) Assign dim(504) = LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64
 // (op 480) Assign stride expression
 // (op 471) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64(480) END
 // (op 471) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(481) BEGIN
 // (op 481) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 497: s**UNALLOCATED**
 // (op 481) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: (unallocated)
 // (op 481) reg expression
 // (op 481) reg expression
 // (op 481) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 481) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 481)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 481)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 481): s0, s1
s_mul_i32 s1, s2, s7 // (op 481) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 481): s14
s_mul_i32 s14, s3, s6 // (op 481) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 481) least significant
// Allocated : 1 SGPR (Value: Int32) (op 481): s15
s_mul_hi_u32 s15, s2, s6 // (op 481) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 481) most significant: sum
s_add_u32 s1, s1, s15 // (op 481) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 481): s15

 // Freeing : 1 SGPR (Value: Int32) (op 481): s14

 // (op 481) ShiftL(s[0:1]:I64, 2:U32)I64
 // (op 481) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 481)         value (SGPR Value: Int64 x 1: s[0:1]) 
 // (op 481)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 481): s14, s15
s_lshl_b64 s[14:15], s[0:1], 2 // (op 481) 
 // Freeing : 2 SGPRs (Value: Int64) (op 481): s0, s1

 // (op 481) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 481): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 481) 
 // Freeing : 2 SGPRs (Value: Int64) (op 481): s14, s15

 // (op 481) reg expression
 // (op 481) reg expression
// Allocated DataFlowTag497: 4 SGPRs (Buffer: None) (op 481): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 481) reg expression
s_mov_b32 s41, s5 // (op 481) reg expression
s_mov_b32 s42, s0 // (op 481) reg expression
s_mov_b32 s43, 131072 // (op 481) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 481): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(481) END
 // (op 471) NOP(468) BEGIN
 // (op 468) generate({})
 // (op 468) end: generate({})
 // (op 471) NOP(468) END
 // (op 471) Assign VGPR 0:U64(486) BEGIN
 // (op 486) Assign dim(506) = 0:U64
 // (op 486) Assign stride expression
 // (op 471) Assign VGPR 0:U64(486) END
 // (op 471) NOP(489) BEGIN
 // (op 489) generate({})
 // (op 489) end: generate({})
 // (op 471) NOP(489) END
 // (op 471) Assign VGPR Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32(498) BEGIN
 // (op 498) Assign dim(507) = Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32
 // tag 507: v**UNALLOCATED**
 // (op 498) Generate Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32 into DataFlowTag507: VGPR Value: UInt32 x 1: (unallocated)
 // (op 498) reg expression
 // (op 498) reg expression
 // (op 498) reg expression
 // (op 498) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 498): v13
v_lshrrev_b32 v13, 1, v1 // (op 498) 
 // (op 498) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 498): v14
v_lshl_add_u32 v14, s16, 5, v13 // (op 498) 
 // (op 498) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 498): v13

 // (op 498) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 498): v13
v_and_b32 v13, -2, v1 // (op 498) 
 // (op 498) BEGIN: Flatten[1]
 // (op 498) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 498): v15
v_sub_u32 v15, v1, v13 // (op 498) 
 // (op 498) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 498): v13

 // (op 498) BEGIN: TileTile
 // (op 498) {TileTile: ShiftL({Flatten[1]: v15:U32}, 5:U32)U32}
 // (op 498) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 498)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 498)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 498): v13
v_lshlrev_b32 v13, 5, v15 // (op 498) 
 // (op 498) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 498): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 498): v16, v17
v_mov_b32 v17, 0 // (op 498) convert
v_mov_b32 v16, v13 // (op 498) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 498): v13

 // (op 498) BEGIN: Split
 // (op 498) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 498)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 498)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 498): v18, v19
v_mul_lo_u32 v19, v14, s3 // (op 498) most significant: low of low * high
 // (op 498) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 498): v13
v_mul_hi_u32 v13, v14, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 498) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 498) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 498): v13

 // (op 498) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 498)      lhs (VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 498)      rhs (TileTile: VGPR Value: UInt64 x 1: v[16:17])
// Allocated : 2 SGPRs (Value: Bool64) (op 498): s0, s1
v_add_co_u32 v18, s[0:1], v18, v16 // (op 498) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 498) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 498): s0, s1

 // (op 498) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 498): v16, v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 498): v14

 // (op 498) ShiftL({Split: v[18:19]:U64}, 2:U32)U64
 // (op 498) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 498)         value (Split: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 498)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 VGPRs (Value: UInt64) (op 498): v16, v17
v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 498) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 498): v18, v19

 // (op 498) LogicalShiftR(v[16:17]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 498): v14, v15
v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 498) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 498): v16, v17

// Allocated DataFlowTag507: 1 VGPR (Value: UInt32) (op 498): v13
v_mov_b32 v13, v14 // (op 498) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 498): v14, v15

 // (op 471) Assign VGPR Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32(498) END
 // (op 471) Assign VGPR 32:U64(499) BEGIN
 // (op 499) Assign dim(508) = 32:U64
 // (op 499) Assign stride expression
 // (op 471) Assign VGPR 32:U64(499) END
 // (op 471) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(500) BEGIN
 // (op 500) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 500) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
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

 // (op 500) ShiftL(s[0:1]:I64, 2:U32)I64
 // (op 500) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 500)         value (SGPR Value: Int64 x 1: s[0:1]) 
 // (op 500)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 500): s14, s15
s_lshl_b64 s[14:15], s[0:1], 2 // (op 500) 
 // Freeing : 2 SGPRs (Value: Int64) (op 500): s0, s1

 // (op 500) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 500): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 500) 
 // Freeing : 2 SGPRs (Value: Int64) (op 500): s14, s15

 // (op 500) reg expression
 // (op 500) reg expression
s_mov_b32 s40, s4 // (op 500) reg expression
s_mov_b32 s41, s5 // (op 500) reg expression
s_mov_b32 s42, s0 // (op 500) reg expression
s_mov_b32 s43, 131072 // (op 500) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 500): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(500) END
 // (op 471) NOP(491) BEGIN
 // (op 491) generate({})
 // (op 491) end: generate({})
 // (op 471) NOP(491) END
 // (op 471) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64(505) BEGIN
 // (op 505) Assign dim(510) = LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64
 // (op 505) Assign stride expression
 // (op 471) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64(505) END
 // (op 471) NOP(492) BEGIN
 // (op 492) generate({})
 // (op 492) end: generate({})
 // (op 471) NOP(492) END
 // (op 471) Assign VGPR 0:U64(508) BEGIN
 // (op 508) Assign dim(512) = 0:U64
 // (op 508) Assign stride expression
 // (op 471) Assign VGPR 0:U64(508) END
 // (op 471) NOP(533) BEGIN
 // (op 533) generate({})
 // (op 533) end: generate({})
 // (op 471) NOP(533) END
 // (op 471) Assign VGPR 48:U64(540) BEGIN
 // (op 540) Assign dim(521) = 48:U64
 // (op 540) Assign stride expression
 // (op 471) Assign VGPR 48:U64(540) END
 // (op 471) NOP(534) BEGIN
 // (op 534) generate({})
 // (op 534) end: generate({})
 // (op 471) NOP(534) END
 // (op 471) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(543) BEGIN
 // (op 543) Assign dim(522) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 522: v**UNALLOCATED**
 // (op 543) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag522: VGPR Value: UInt32 x 1: (unallocated)
 // (op 543) reg expression
 // (op 543) reg expression
 // (op 543) reg expression
 // (op 543) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 543): v14
v_and_b32 v14, -2, v1 // (op 543) 
 // (op 543) BEGIN: Flatten[1]
 // (op 543) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 543): v15
v_sub_u32 v15, v1, v14 // (op 543) 
 // (op 543) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 543): v14

 // (op 543) BEGIN: TileTile
 // (op 543) {TileTile: ShiftL({Flatten[1]: v15:U32}, 5:U32)U32}
 // (op 543) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 543)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 543)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 543): v14
v_lshlrev_b32 v14, 5, v15 // (op 543) 
 // (op 543) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 543): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 543): v16, v17
v_mov_b32 v17, 0 // (op 543) convert
v_mov_b32 v16, v14 // (op 543) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 543): v14

 // (op 543) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 543): v14
v_lshrrev_b32 v14, 1, v1 // (op 543) 
 // (op 543) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 543): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 543) 
 // (op 543) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 543): v14

 // (op 543) Multiply({Tile: v15:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 543) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 543)           lhs (Tile: VGPR Value: UInt32 x 1: v15) 
 // (op 543)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 543): v18, v19
v_mul_lo_u32 v19, v15, s9 // (op 543) most significant: low of low * high
 // (op 543) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 543): v14
v_mul_hi_u32 v14, v15, s8 // (op 543) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 543) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 543) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 543): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 543): v15

 // (op 543) BEGIN: Split
 // (op 543) {Split: Add({TileTile: v[16:17]:U64}, v[18:19]:I64)U64}
 // (op 543) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 543)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 543)      rhs (VGPR Value: Int64 x 1: v[18:19])
// Allocated : 2 VGPRs (Value: UInt64) (op 543): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 543): s0, s1
v_add_co_u32 v14, s[0:1], v16, v18 // (op 543) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 543) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 543): s0, s1

 // (op 543) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 543): v18, v19

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 543): v16, v17

 // (op 543) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 543) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 543)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 543)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 543) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 543): v16, v17
v_mul_lo_u32 v17, v15, 6 // (op 543) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 543): v18
v_mul_hi_u32 v18, v14, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 543) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 543): v18

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 543): v14, v15

 // (op 543) LogicalShiftR(v[16:17]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 543): v14, v15
v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 543) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 543): v16, v17

// Allocated DataFlowTag522: 1 VGPR (Value: UInt32) (op 543): v16
v_mov_b32 v16, v14 // (op 543) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 543): v14, v15

 // (op 471) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(543) END
 // (op 471) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(544) BEGIN
 // (op 544) Assign dim(523) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 544) Assign stride expression
 // (op 471) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(544) END
 // (op 471) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(545) BEGIN
 // (op 545) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 516: s**UNALLOCATED**
 // (op 545) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: (unallocated)
 // (op 545) reg expression
 // (op 545) reg expression
 // (op 545) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 545) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 545)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 545)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 545) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 545): s0, s1
s_mul_i32 s1, s13, 6 // (op 545) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 545) least significant
// Allocated : 1 SGPR (Value: Int32) (op 545): s14
s_mul_hi_u32 s14, s12, 6 // (op 545) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 545) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 545): s14

 // (op 545) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 545) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 545)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 545)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 545): s14, s15
s_mul_i32 s15, s8, s1 // (op 545) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 545): s18
s_mul_i32 s18, s9, s0 // (op 545) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 545) least significant
// Allocated : 1 SGPR (Value: Int32) (op 545): s19
s_mul_hi_u32 s19, s8, s0 // (op 545) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 545) most significant: sum
s_add_u32 s15, s15, s19 // (op 545) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 545): s19

 // Freeing : 1 SGPR (Value: Int32) (op 545): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 545): s0, s1

 // (op 545) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 545): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 545) 
 // Freeing : 2 SGPRs (Value: Int64) (op 545): s14, s15

 // (op 545) reg expression
 // (op 545) reg expression
// Allocated DataFlowTag516: 4 SGPRs (Buffer: None) (op 545): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 545) reg expression
s_mov_b32 s45, s11 // (op 545) reg expression
s_mov_b32 s46, s0 // (op 545) reg expression
s_mov_b32 s47, 131072 // (op 545) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 545): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(545) END
 // (op 471) NOP(535) BEGIN
 // (op 535) generate({})
 // (op 535) end: generate({})
 // (op 471) NOP(535) END
 // (op 471) Assign VGPR 0:U64(550) BEGIN
 // (op 550) Assign dim(525) = 0:U64
 // (op 550) Assign stride expression
 // (op 471) Assign VGPR 0:U64(550) END
 // (op 471) NOP(553) BEGIN
 // (op 553) generate({})
 // (op 553) end: generate({})
 // (op 471) NOP(553) END
 // (op 471) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(562) BEGIN
 // (op 562) Assign dim(526) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 526: v**UNALLOCATED**
 // (op 562) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag526: VGPR Value: UInt32 x 1: (unallocated)
 // (op 562) reg expression
 // (op 562) reg expression
 // (op 562) reg expression
 // (op 562) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 562): v14
v_and_b32 v14, -2, v1 // (op 562) 
 // (op 562) BEGIN: Flatten[1]
 // (op 562) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 562): v15
v_sub_u32 v15, v1, v14 // (op 562) 
 // (op 562) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 562): v14

 // (op 562) BEGIN: TileTile
 // (op 562) {TileTile: ShiftL({Flatten[1]: v15:U32}, 5:U32)U32}
 // (op 562) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 562)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 562)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 562): v14
v_lshlrev_b32 v14, 5, v15 // (op 562) 
 // (op 562) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 562): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 562): v18, v19
v_mov_b32 v19, 0 // (op 562) convert
v_mov_b32 v18, v14 // (op 562) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 562): v14

 // (op 562) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 562): v17
v_lshrrev_b32 v17, 1, v1 // (op 562) 
 // (op 562) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 562): v14
v_lshl_add_u32 v14, s17, 5, v17 // (op 562) 
 // (op 562) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 562): v17

 // (op 562) Multiply({Tile: v14:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 562) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 562)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 562)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 562): v20, v21
v_mul_lo_u32 v21, v14, s9 // (op 562) most significant: low of low * high
 // (op 562) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 562): v15
v_mul_hi_u32 v15, v14, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v20, v14, s8 // (op 562) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 562) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 562): v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 562): v14

 // (op 562) BEGIN: Split
 // (op 562) {Split: Add({TileTile: v[18:19]:U64}, v[20:21]:I64)U64}
 // (op 562) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 562)      lhs (TileTile: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 562)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated : 2 VGPRs (Value: UInt64) (op 562): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 562): s0, s1
v_add_co_u32 v14, s[0:1], v18, v20 // (op 562) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 562) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 562): s0, s1

 // (op 562) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 562): v20, v21

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 562): v18, v19

 // (op 562) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 562) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 562)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 562)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 562) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 562): v18, v19
v_mul_lo_u32 v19, v15, 6 // (op 562) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 562): v17
v_mul_hi_u32 v17, v14, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 562) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 562) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 562): v17

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 562): v14, v15

 // (op 562) LogicalShiftR(v[18:19]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 562): v14, v15
v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 562) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 562): v18, v19

// Allocated DataFlowTag526: 1 VGPR (Value: UInt32) (op 562): v17
v_mov_b32 v17, v14 // (op 562) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 562): v14, v15

 // (op 471) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(562) END
 // (op 471) Assign VGPR 48:U64(563) BEGIN
 // (op 563) Assign dim(527) = 48:U64
 // (op 563) Assign stride expression
 // (op 471) Assign VGPR 48:U64(563) END
 // (op 471) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(564) BEGIN
 // (op 564) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 564) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
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
s_mov_b32 s44, s10 // (op 564) reg expression
s_mov_b32 s45, s11 // (op 564) reg expression
s_mov_b32 s46, s0 // (op 564) reg expression
s_mov_b32 s47, 131072 // (op 564) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 564): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(564) END
 // (op 471) NOP(555) BEGIN
 // (op 555) generate({})
 // (op 555) end: generate({})
 // (op 471) NOP(555) END
 // (op 471) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(569) BEGIN
 // (op 569) Assign dim(529) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 569) Assign stride expression
 // (op 471) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(569) END
 // (op 471) NOP(556) BEGIN
 // (op 556) generate({})
 // (op 556) end: generate({})
 // (op 471) NOP(556) END
 // (op 471) Assign VGPR 0:U64(572) BEGIN
 // (op 572) Assign dim(531) = 0:U64
 // (op 572) Assign stride expression
 // (op 471) Assign VGPR 0:U64(572) END
 // (op 471) NOP(597) BEGIN
 // (op 597) generate({})
 // (op 597) end: generate({})
 // (op 471) NOP(597) END
 // (op 471) Assign VGPR 2:U64(604) BEGIN
 // (op 604) Assign dim(540) = 2:U64
 // (op 604) Assign stride expression
 // (op 471) Assign VGPR 2:U64(604) END
 // (op 471) NOP(598) BEGIN
 // (op 598) generate({})
 // (op 598) end: generate({})
 // (op 471) NOP(598) END
 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(607) BEGIN
 // (op 607) Assign dim(541) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}
 // tag 541: v**UNALLOCATED**
 // (op 607) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32} into DataFlowTag541: VGPR Value: UInt32 x 1: (unallocated)
 // (op 607) reg expression
 // (op 607) reg expression
 // (op 607) reg expression
 // (op 607) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 607) convert
// Allocated : 1 VGPR (Value: UInt32) (op 607): v14
v_lshrrev_b32 v14, 1, v1 // (op 607) 
 // (op 607) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 607): v15
v_lshl_add_u32 v15, s16, 5, v14 // (op 607) 
 // (op 607) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 607): v14

 // (op 607) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 607): v14
v_and_b32 v14, -2, v1 // (op 607) 
 // (op 607) BEGIN: Flatten[1]TileTile
 // (op 607) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 607): v18
v_sub_u32 v18, v1, v14 // (op 607) 
 // (op 607) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 607): v14

 // (op 607) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag541: 1 VGPR (Value: UInt32) (op 607): v14
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v14, v15, s20 // (op 607) 
v_add_u32 v14, v14, v18 // (op 607) 
 // (op 607) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 607): v18

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 607): v15

 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(607) END
 // (op 471) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(608) BEGIN
 // (op 608) Assign dim(542) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 608) Assign stride expression
 // (op 471) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(608) END
 // (op 471) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(609) BEGIN
 // (op 609) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 535: s**UNALLOCATED**
 // (op 609) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: (unallocated)
 // (op 609) reg expression
 // (op 609) reg expression
 // (op 609) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 609) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 609)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 609)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 609): s0, s1
s_mul_i32 s1, s20, s7 // (op 609) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 609): s14
s_mul_i32 s14, s21, s6 // (op 609) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 609) least significant
// Allocated : 1 SGPR (Value: Int32) (op 609): s15
s_mul_hi_u32 s15, s20, s6 // (op 609) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 609) most significant: sum
s_add_u32 s1, s1, s15 // (op 609) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 609): s15

 // Freeing : 1 SGPR (Value: Int32) (op 609): s14

 // (op 609) reg expression
 // (op 609) reg expression
// Allocated DataFlowTag535: 4 SGPRs (Buffer: None) (op 609): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 609) reg expression
s_mov_b32 s49, s23 // (op 609) reg expression
s_mov_b32 s50, s0 // (op 609) reg expression
s_mov_b32 s51, 131072 // (op 609) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 609): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(609) END
 // (op 471) NOP(599) BEGIN
 // (op 599) generate({})
 // (op 599) end: generate({})
 // (op 471) NOP(599) END
 // (op 471) Assign VGPR 1:U64(614) BEGIN
 // (op 614) Assign dim(544) = 1:U64
 // (op 614) Assign stride expression
 // (op 471) Assign VGPR 1:U64(614) END
 // (op 471) NOP(617) BEGIN
 // (op 617) generate({})
 // (op 617) end: generate({})
 // (op 471) NOP(617) END
 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(626) BEGIN
 // (op 626) Assign dim(545) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}
 // tag 545: v**UNALLOCATED**
 // (op 626) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32} into DataFlowTag545: VGPR Value: UInt32 x 1: (unallocated)
 // (op 626) reg expression
 // (op 626) reg expression
 // (op 626) reg expression
 // (op 626) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 626) convert
// Allocated : 1 VGPR (Value: UInt32) (op 626): v15
v_lshrrev_b32 v15, 1, v1 // (op 626) 
 // (op 626) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 626): v18
v_lshl_add_u32 v18, s16, 5, v15 // (op 626) 
 // (op 626) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 626): v15

 // (op 626) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 626): v15
v_and_b32 v15, -2, v1 // (op 626) 
 // (op 626) BEGIN: Flatten[1]TileTile
 // (op 626) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 626): v19
v_sub_u32 v19, v1, v15 // (op 626) 
 // (op 626) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 626): v15

 // (op 626) BEGIN: Split
// Allocated DataFlowTag545: 1 VGPR (Value: UInt32) (op 626): v15
v_mul_lo_u32 v15, v18, s20 // (op 626) 
v_add_u32 v15, v15, v19 // (op 626) 
 // (op 626) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 626): v19

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 626): v18

 // (op 471) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(626) END
 // (op 471) Assign VGPR 2:U64(627) BEGIN
 // (op 627) Assign dim(546) = 2:U64
 // (op 627) Assign stride expression
 // (op 471) Assign VGPR 2:U64(627) END
 // (op 471) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(628) BEGIN
 // (op 628) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 628) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
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
s_mov_b32 s48, s22 // (op 628) reg expression
s_mov_b32 s49, s23 // (op 628) reg expression
s_mov_b32 s50, s0 // (op 628) reg expression
s_mov_b32 s51, 131072 // (op 628) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 628): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(628) END
 // (op 471) NOP(619) BEGIN
 // (op 619) generate({})
 // (op 619) end: generate({})
 // (op 471) NOP(619) END
 // (op 471) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(633) BEGIN
 // (op 633) Assign dim(548) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 633) Assign stride expression
 // (op 471) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(633) END
 // (op 471) NOP(620) BEGIN
 // (op 620) generate({})
 // (op 620) end: generate({})
 // (op 471) NOP(620) END
 // (op 471) Assign VGPR 1:U64(636) BEGIN
 // (op 636) Assign dim(550) = 1:U64
 // (op 636) Assign stride expression
 // (op 471) Assign VGPR 1:U64(636) END
 // (op 471) NOP(661) BEGIN
 // (op 661) generate({})
 // (op 661) end: generate({})
 // (op 471) NOP(661) END
 // (op 471) Assign VGPR 2:U64(668) BEGIN
 // (op 668) Assign dim(559) = 2:U64
 // (op 668) Assign stride expression
 // (op 471) Assign VGPR 2:U64(668) END
 // (op 471) NOP(662) BEGIN
 // (op 662) generate({})
 // (op 662) end: generate({})
 // (op 471) NOP(662) END
 // (op 471) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(671) BEGIN
 // (op 671) Assign dim(560) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 560: v**UNALLOCATED**
 // (op 671) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 671) reg expression
 // (op 671) reg expression
 // (op 671) reg expression
 // (op 671) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 671) convert
// Allocated : 1 VGPR (Value: UInt32) (op 671): v18
v_and_b32 v18, -2, v1 // (op 671) 
 // (op 671) BEGIN: Flatten[1]TileTile
 // (op 671) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v18:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 671): v19
v_sub_u32 v19, v1, v18 // (op 671) 
 // (op 671) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 671): v18

 // (op 671) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 671): v18
v_lshrrev_b32 v18, 1, v1 // (op 671) 
 // (op 671) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 671): v20
v_lshl_add_u32 v20, s17, 5, v18 // (op 671) 
 // (op 671) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 671): v18

 // (op 671) Multiply({Tile: v20:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 671): v18
v_mul_lo_u32 v18, v20, s24 // (op 671) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 671): v20

 // (op 671) BEGIN: Split
 // (op 671) {Split: Add({Flatten[1]TileTile: v19:U32}, v18:U32)U32}
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 671): v20
v_add_u32 v20, v19, v18 // (op 671) 
 // (op 671) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 671): v18

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 671): v19

 // (op 471) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(671) END
 // (op 471) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(672) BEGIN
 // (op 672) Assign dim(561) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 672) Assign stride expression
 // (op 471) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(672) END
 // (op 471) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(673) BEGIN
 // (op 673) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 554: s**UNALLOCATED**
 // (op 673) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: (unallocated)
 // (op 673) reg expression
 // (op 673) reg expression
 // (op 673) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 673) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 673)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 673)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 673): s0, s1
s_mul_i32 s1, s24, s13 // (op 673) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 673): s14
s_mul_i32 s14, s25, s12 // (op 673) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 673) least significant
// Allocated : 1 SGPR (Value: Int32) (op 673): s15
s_mul_hi_u32 s15, s24, s12 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 673) most significant: sum
s_add_u32 s1, s1, s15 // (op 673) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 673): s15

 // Freeing : 1 SGPR (Value: Int32) (op 673): s14

 // (op 673) reg expression
 // (op 673) reg expression
// Allocated DataFlowTag554: 4 SGPRs (Buffer: None) (op 673): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 673) reg expression
s_mov_b32 s53, s27 // (op 673) reg expression
s_mov_b32 s54, s0 // (op 673) reg expression
s_mov_b32 s55, 131072 // (op 673) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 673): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(673) END
 // (op 471) NOP(663) BEGIN
 // (op 663) generate({})
 // (op 663) end: generate({})
 // (op 471) NOP(663) END
 // (op 471) Assign VGPR 1:U64(678) BEGIN
 // (op 678) Assign dim(563) = 1:U64
 // (op 678) Assign stride expression
 // (op 471) Assign VGPR 1:U64(678) END
 // (op 471) NOP(681) BEGIN
 // (op 681) generate({})
 // (op 681) end: generate({})
 // (op 471) NOP(681) END
 // (op 471) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(690) BEGIN
 // (op 690) Assign dim(564) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 564: v**UNALLOCATED**
 // (op 690) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 690) reg expression
 // (op 690) reg expression
 // (op 690) reg expression
 // (op 690) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 690) convert
// Allocated : 1 VGPR (Value: UInt32) (op 690): v18
v_and_b32 v18, -2, v1 // (op 690) 
 // (op 690) BEGIN: Flatten[1]TileTile
 // (op 690) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v18:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 690): v19
v_sub_u32 v19, v1, v18 // (op 690) 
 // (op 690) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 690): v18

 // (op 690) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 690): v18
v_lshrrev_b32 v18, 1, v1 // (op 690) 
 // (op 690) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 690): v21
v_lshl_add_u32 v21, s17, 5, v18 // (op 690) 
 // (op 690) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 690): v18

 // (op 690) Multiply({Tile: v21:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 690): v18
v_mul_lo_u32 v18, v21, s24 // (op 690) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 690): v21

 // (op 690) BEGIN: Split
 // (op 690) {Split: Add({Flatten[1]TileTile: v19:U32}, v18:U32)U32}
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 690): v21
v_add_u32 v21, v19, v18 // (op 690) 
 // (op 690) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 690): v18

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 690): v19

 // (op 471) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(690) END
 // (op 471) Assign VGPR 2:U64(691) BEGIN
 // (op 691) Assign dim(565) = 2:U64
 // (op 691) Assign stride expression
 // (op 471) Assign VGPR 2:U64(691) END
 // (op 471) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(692) BEGIN
 // (op 692) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 692) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
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
s_mov_b32 s52, s26 // (op 692) reg expression
s_mov_b32 s53, s27 // (op 692) reg expression
s_mov_b32 s54, s0 // (op 692) reg expression
s_mov_b32 s55, 131072 // (op 692) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 692): s0, s1

 // (op 471) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(692) END
 // (op 471) NOP(683) BEGIN
 // (op 683) generate({})
 // (op 683) end: generate({})
 // (op 471) NOP(683) END
 // (op 471) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(697) BEGIN
 // (op 697) Assign dim(567) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 697) Assign stride expression
 // (op 471) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(697) END
 // (op 471) NOP(684) BEGIN
 // (op 684) generate({})
 // (op 684) end: generate({})
 // (op 471) NOP(684) END
 // (op 471) Assign VGPR 1:U64(700) BEGIN
 // (op 700) Assign dim(569) = 1:U64
 // (op 700) Assign stride expression
 // (op 471) Assign VGPR 1:U64(700) END
 // (op 471) NOP(364) BEGIN
 // (op 364) generate({})
 // (op 364) end: generate({})
 // (op 471) NOP(364) END
 // (op 471) SetCoordinate(369) BEGIN
 // (op 369) generate({})
 // (op 369) end: generate({})
 // (op 369) generate({370})
 // (op 369) LoadTiled Value: FP6x16(370) BEGIN
 // (op 370) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 370) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 370) Generate 0:U64 into nullptr
 // tag 9: v**UNALLOCATED**
 // (op 370) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 370) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag522: VGPR Value: UInt32 x 1: v16
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 370) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 370): v22, v23, v24, v25, v26, v27
buffer_load_dwordx4 v[22:25], v16, s[44:47], 0 offen // (op 370) Load value
// VMEM: Expected complete at 823 (current 343)
buffer_load_dwordx2 v[26:27], v16, s[44:47], 0 offen offset:16 // (op 370) Load value
// VMEM: Expected complete at 824 (current 344)
 // (op 369) LoadTiled Value: FP6x16(370) END
 // (op 369) end: generate({370})
 // (op 471) SetCoordinate(369) END
 // (op 471) SetCoordinate(372) BEGIN
 // (op 372) generate({})
 // (op 372) end: generate({})
 // (op 372) generate({373})
 // (op 372) LoadTiled Value: E8M0(373) BEGIN
 // (op 373) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 373) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 373) Generate 1:U64 into nullptr
 // tag 11: v**UNALLOCATED**
 // (op 373) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 373) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v20
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 373) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 373): v18
buffer_load_ubyte v18, v20, s[52:55], 0 offen offset: 0 // (op 373) Load value
// VMEM: Expected complete at 825 (current 345)
 // (op 372) LoadTiled Value: E8M0(373) END
 // (op 372) end: generate({373})
 // (op 471) SetCoordinate(372) END
 // (op 471) SetCoordinate(375) BEGIN
 // (op 375) generate({})
 // (op 375) end: generate({})
 // (op 375) generate({376})
 // (op 375) LoadTiled Value: FP4x8(376) BEGIN
 // (op 376) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 376) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 376) Generate 0:U64 into nullptr
 // tag 8: v**UNALLOCATED**
 // (op 376) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 376) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag503: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 376) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 376): v28, v29, v30, v31
buffer_load_dwordx4 v[28:31], v12, s[40:43], 0 offen // (op 376) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1304 (current 824)
 // (op 375) LoadTiled Value: FP4x8(376) END
 // (op 375) end: generate({376})
 // (op 471) SetCoordinate(375) END
 // (op 471) SetCoordinate(378) BEGIN
 // (op 378) generate({})
 // (op 378) end: generate({})
 // (op 378) generate({379})
 // (op 378) LoadTiled Value: E8M0(379) BEGIN
 // (op 379) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 379) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 379) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 379) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 379) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag541: VGPR Value: UInt32 x 1: v14
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 379) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 379): v19
buffer_load_ubyte v19, v14, s[48:51], 0 offen offset: 0 // (op 379) Load value
// VMEM: Expected complete at 1305 (current 825)
 // (op 378) LoadTiled Value: E8M0(379) END
 // (op 378) end: generate({379})
 // (op 471) SetCoordinate(378) END
 // (op 471) SetCoordinate(381) BEGIN
 // (op 381) generate({})
 // (op 381) end: generate({})
 // (op 381) generate({382})
 // (op 381) LoadTiled Value: FP6x16(382) BEGIN
 // (op 382) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 382) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 382) Generate 0:U64 into nullptr
 // tag 474: v**UNALLOCATED**
 // (op 382) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 382) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag522: VGPR Value: UInt32 x 1: v16
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 382) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 382): v32, v33, v34, v35, v36, v37
buffer_load_dwordx4 v[32:35], v16, s[44:47], 0 offen offset:48 // (op 382) Load value
// VMEM: Expected complete at 1306 (current 826)
buffer_load_dwordx2 v[36:37], v16, s[44:47], 0 offen offset:64 // (op 382) Load value
// VMEM: Expected stall of 478. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1785 (current 1305)
 // (op 381) LoadTiled Value: FP6x16(382) END
 // (op 381) end: generate({382})
 // (op 471) SetCoordinate(381) END
 // (op 471) Deallocate{}(1272) BEGIN
 // (op 1272) Deallocate 521
 // Deleting tag 521
 // (op 1272) Deallocate 522
 // Freeing DataFlowTag522: 1 VGPR (Value: UInt32) (op 543): v16

 // Deleting tag 522
 // (op 1272) Deallocate 523
 // Deleting tag 523
 // (op 1272) Deallocate 525
 // Deleting tag 525
 // (op 1272) Deallocate 524
 // Deleting tag 524
 // (op 471) Deallocate{}(1272) END
 // (op 471) SetCoordinate(384) BEGIN
 // (op 384) generate({})
 // (op 384) end: generate({})
 // (op 384) generate({385})
 // (op 384) LoadTiled Value: E8M0(385) BEGIN
 // (op 385) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 385) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 385) Generate 1:U64 into nullptr
 // tag 482: v**UNALLOCATED**
 // (op 385) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 385) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v20
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 385) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 385): v16
buffer_load_ubyte v16, v20, s[52:55], 0 offen offset:2 // (op 385) Load value
// VMEM: Expected complete at 1786 (current 1306)
 // (op 384) LoadTiled Value: E8M0(385) END
 // (op 384) end: generate({385})
 // (op 471) SetCoordinate(384) END
 // (op 471) Deallocate{}(1274) BEGIN
 // (op 1274) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 671): v20

 // Deleting tag 560
 // (op 1274) Deallocate 563
 // Deleting tag 563
 // (op 1274) Deallocate 561
 // Deleting tag 561
 // (op 1274) Deallocate 562
 // Deleting tag 562
 // (op 1274) Deallocate 559
 // Deleting tag 559
 // (op 471) Deallocate{}(1274) END
 // (op 471) SetCoordinate(387) BEGIN
 // (op 387) generate({})
 // (op 387) end: generate({})
 // (op 387) generate({388})
 // (op 387) LoadTiled Value: FP4x8(388) BEGIN
 // (op 388) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 388) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 388) Generate 0:U64 into nullptr
 // tag 470: v**UNALLOCATED**
 // (op 388) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 388) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag503: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 32
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 388) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 388): v38, v39, v40, v41
buffer_load_dwordx4 v[38:41], v12, s[40:43], 0 offen offset:32 // (op 388) Load value
// VMEM: Expected complete at 1787 (current 1307)
 // (op 387) LoadTiled Value: FP4x8(388) END
 // (op 387) end: generate({388})
 // (op 471) SetCoordinate(387) END
 // (op 471) Deallocate{}(1276) BEGIN
 // (op 1276) Deallocate 506
 // Deleting tag 506
 // (op 1276) Deallocate 503
 // Freeing DataFlowTag503: 1 VGPR (Value: UInt32) (op 479): v12

 // Deleting tag 503
 // (op 1276) Deallocate 502
 // Deleting tag 502
 // (op 1276) Deallocate 504
 // Deleting tag 504
 // (op 1276) Deallocate 505
 // Deleting tag 505
 // (op 471) Deallocate{}(1276) END
 // (op 471) SetCoordinate(390) BEGIN
 // (op 390) generate({})
 // (op 390) end: generate({})
 // (op 390) generate({391})
 // (op 390) LoadTiled Value: E8M0(391) BEGIN
 // (op 391) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 391) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 391) Generate 1:U64 into nullptr
 // tag 478: v**UNALLOCATED**
 // (op 391) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 391) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag541: VGPR Value: UInt32 x 1: v14
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 391) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 391): v12
buffer_load_ubyte v12, v14, s[48:51], 0 offen offset:2 // (op 391) Load value
// VMEM: Expected stall of 478. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2266 (current 1786)
 // (op 390) LoadTiled Value: E8M0(391) END
 // (op 390) end: generate({391})
 // (op 471) SetCoordinate(390) END
 // (op 471) Deallocate{}(1278) BEGIN
 // (op 1278) Deallocate 540
 // Deleting tag 540
 // (op 1278) Deallocate 541
 // Freeing DataFlowTag541: 1 VGPR (Value: UInt32) (op 607): v14

 // Deleting tag 541
 // (op 1278) Deallocate 543
 // Deleting tag 543
 // (op 1278) Deallocate 544
 // Deleting tag 544
 // (op 1278) Deallocate 542
 // Deleting tag 542
 // (op 471) Deallocate{}(1278) END
 // (op 471) Scope(805) BEGIN
 // (op 805) Lock Scope 805
 // Lock Branch
 // (op 805) generate({802})
 // (op 805) NOP(802) BEGIN
 // (op 802) generate({})
 // (op 802) end: generate({})
 // (op 805) NOP(802) END
 // (op 805) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(810) BEGIN
 // (op 810) Assign dim(600) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 600: v**UNALLOCATED**
 // (op 810) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag600: VGPR Value: UInt32 x 1: (unallocated)
 // (op 810) reg expression
 // (op 810) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 810): v14
v_and_b32 v14, -2, v1 // (op 810) 
 // (op 810) BEGIN: Tile[1]
 // (op 810) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 810): v20
v_sub_u32 v20, v1, v14 // (op 810) 
 // (op 810) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 810): v14

 // (op 810) BEGIN: Flatten
 // (op 810) {Flatten: ShiftL({Tile[1]: v20:U32}, 5:U32)U32}
 // (op 810) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 810)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 810)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 810): v14
v_lshlrev_b32 v14, 5, v20 // (op 810) 
 // (op 810) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 810): v20

 // (op 810) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 810): v20
v_lshrrev_b32 v20, 1, v1 // (op 810) 
 // (op 810) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 810): v42
v_lshl_add_u32 v42, v20, 6, v14 // (op 810) 
 // (op 810) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 810): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 810): v20

 // (op 810) Multiply({Flatten: v42:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 810): v14
v_mul_lo_u32 v14, v42, 6 // (op 810) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 810): v42

 // (op 810) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag600: 1 VGPR (Value: UInt32) (op 810): v20
v_lshrrev_b32 v20, 3, v14 // (op 810) 
 // Freeing : 1 VGPR (Value: UInt32) (op 810): v14

 // (op 805) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(810) END
 // (op 805) Assign VGPR 48:U32(811) BEGIN
 // (op 811) Assign dim(601) = 48:U32
 // (op 811) Assign stride expression
 // (op 805) Assign VGPR 48:U32(811) END
 // (op 805) NOP(803) BEGIN
 // (op 803) generate({})
 // (op 803) end: generate({})
 // (op 805) NOP(803) END
 // (op 805) Assign VGPR 0:U32(815) BEGIN
 // (op 815) Assign dim(603) = 0:U32
 // (op 815) Assign stride expression
 // (op 805) Assign VGPR 0:U32(815) END
 // (op 805) StoreLDSTile Value: FP6x16(393) BEGIN
 // (op 393) GEN: StoreLDSTile
 // (op 393) GEN: storeMacroTileLDS OP 393 LDS 25 MacroTile 9
 // (op 393) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 393) Generate 0:U32 into nullptr
 // (op 393) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[22:27]
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

 // (op 393) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[22:27]
 // 	info.rowOffsetReg = DataFlowTag600: VGPR Value: UInt32 x 1: v20
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
 // (op 393) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b128 v20, v[22:25] // (op 393) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 1890 (current 1795)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
s_waitcnt vmcnt(8)
ds_write_b64 v20, v[26:27] offset:16 // (op 393) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 1892 (current 1797)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 805) StoreLDSTile Value: FP6x16(393) END
 // (op 805) Deallocate{}(1280) BEGIN
 // (op 1280) Deallocate 601
 // Deleting tag 601
 // (op 1280) Deallocate 600
 // Freeing DataFlowTag600: 1 VGPR (Value: UInt32) (op 810): v20

 // Deleting tag 600
 // (op 1280) Deallocate 603
 // Deleting tag 603
 // (op 1280) Deallocate 602
 // Deleting tag 602
 // (op 805) Deallocate{}(1280) END
 // (op 805) end: generate({802})
 // (op 805) Unlock Scope 805
 // Unlock None
 // (op 471) Scope(805) END
 // (op 471) Scope(911) BEGIN
 // (op 911) Lock Scope 911
 // Lock Branch
 // (op 911) generate({908})
 // (op 911) NOP(908) BEGIN
 // (op 908) generate({})
 // (op 908) end: generate({})
 // (op 911) NOP(908) END
 // (op 911) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(916) BEGIN
 // (op 916) Assign dim(632) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 632: v**UNALLOCATED**
 // (op 916) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag632: VGPR Value: UInt32 x 1: (unallocated)
 // (op 916) reg expression
 // (op 916) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 916): v14
v_and_b32 v14, -2, v1 // (op 916) 
 // (op 916) BEGIN: Tile[1]Flatten
 // (op 916) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 916): v20
v_sub_u32 v20, v1, v14 // (op 916) 
 // (op 916) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 916): v14

 // (op 916) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 916): v14
v_and_b32 v14, 4294967294, v1 // (op 916) 
 // (op 916) BEGIN: Flatten
 // (op 916) {Flatten: Add(v14:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag632: 1 VGPR (Value: UInt32) (op 916): v42
v_add_u32 v42, v14, v20 // (op 916) 
 // (op 916) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 916): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 916): v14

 // (op 911) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(916) END
 // (op 911) Assign VGPR 2:U32(917) BEGIN
 // (op 917) Assign dim(633) = 2:U32
 // (op 917) Assign stride expression
 // (op 911) Assign VGPR 2:U32(917) END
 // (op 911) NOP(909) BEGIN
 // (op 909) generate({})
 // (op 909) end: generate({})
 // (op 911) NOP(909) END
 // (op 911) Assign VGPR 1:U32(921) BEGIN
 // (op 921) Assign dim(635) = 1:U32
 // (op 921) Assign stride expression
 // (op 911) Assign VGPR 1:U32(921) END
 // (op 911) StoreLDSTile Value: E8M0(394) BEGIN
 // (op 394) GEN: StoreLDSTile
 // (op 394) GEN: storeMacroTileLDS OP 394 LDS 41 MacroTile 11
 // (op 394) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 394) Generate 1:U32 into nullptr
 // (op 394) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v18
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 394) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v18
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v42
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 394) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b8 v42, v18 offset:1536 // (op 394) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1898 (current 1803)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 911) StoreLDSTile Value: E8M0(394) END
 // (op 911) Deallocate{}(1282) BEGIN
 // (op 1282) Deallocate 635
 // Deleting tag 635
 // (op 1282) Deallocate 634
 // Deleting tag 634
 // (op 1282) Deallocate 633
 // Deleting tag 633
 // (op 1282) Deallocate 632
 // Freeing DataFlowTag632: 1 VGPR (Value: UInt32) (op 916): v42

 // Deleting tag 632
 // (op 911) Deallocate{}(1282) END
 // (op 911) end: generate({908})
 // (op 911) Unlock Scope 911
 // Unlock None
 // (op 471) Scope(911) END
 // (op 471) Scope(752) BEGIN
 // (op 752) Lock Scope 752
 // Lock Branch
 // (op 752) generate({749})
 // (op 752) NOP(749) BEGIN
 // (op 749) generate({})
 // (op 749) end: generate({})
 // (op 752) NOP(749) END
 // (op 752) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(757) BEGIN
 // (op 757) Assign dim(584) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 584: v**UNALLOCATED**
 // (op 757) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag584: VGPR Value: UInt32 x 1: (unallocated)
 // (op 757) reg expression
 // (op 757) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 757): v14
v_and_b32 v14, -2, v1 // (op 757) 
 // (op 757) BEGIN: Tile[1]
 // (op 757) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 757): v20
v_sub_u32 v20, v1, v14 // (op 757) 
 // (op 757) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 757): v14

 // (op 757) BEGIN: Flatten
 // (op 757) {Flatten: ShiftL({Tile[1]: v20:U32}, 5:U32)U32}
 // (op 757) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 757)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 757)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 757): v14
v_lshlrev_b32 v14, 5, v20 // (op 757) 
 // (op 757) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 757): v20

 // (op 757) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 757): v20
v_lshrrev_b32 v20, 1, v1 // (op 757) 
 // (op 757) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 757): v42
v_lshl_add_u32 v42, v20, 6, v14 // (op 757) 
 // (op 757) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 757): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 757): v20

 // (op 757) ShiftL({Flatten: v42:U32}, 2:U32)U32
 // (op 757) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 757)         value (Flatten: VGPR Value: UInt32 x 1: v42) 
 // (op 757)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 757): v14
v_lshlrev_b32 v14, 2, v42 // (op 757) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 757): v42

 // (op 757) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag584: 1 VGPR (Value: UInt32) (op 757): v20
v_lshrrev_b32 v20, 3, v14 // (op 757) 
 // Freeing : 1 VGPR (Value: UInt32) (op 757): v14

 // (op 752) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(757) END
 // (op 752) Assign VGPR 32:U32(758) BEGIN
 // (op 758) Assign dim(585) = 32:U32
 // (op 758) Assign stride expression
 // (op 752) Assign VGPR 32:U32(758) END
 // (op 752) NOP(750) BEGIN
 // (op 750) generate({})
 // (op 750) end: generate({})
 // (op 752) NOP(750) END
 // (op 752) Assign VGPR 0:U32(762) BEGIN
 // (op 762) Assign dim(587) = 0:U32
 // (op 762) Assign stride expression
 // (op 752) Assign VGPR 0:U32(762) END
 // (op 752) StoreLDSTile Value: FP4x8(395) BEGIN
 // (op 395) GEN: StoreLDSTile
 // (op 395) GEN: storeMacroTileLDS OP 395 LDS 17 MacroTile 8
 // (op 395) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 395) Generate 0:U32 into nullptr
 // (op 395) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[28:31]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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

 // (op 395) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[28:31]
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v20
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 395) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b128 v20, v[28:31] offset:1600 // (op 395) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected stall of 80. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1987 (current 1892)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 752) StoreLDSTile Value: FP4x8(395) END
 // (op 752) Deallocate{}(1284) BEGIN
 // (op 1284) Deallocate 586
 // Deleting tag 586
 // (op 1284) Deallocate 584
 // Freeing DataFlowTag584: 1 VGPR (Value: UInt32) (op 757): v20

 // Deleting tag 584
 // (op 1284) Deallocate 585
 // Deleting tag 585
 // (op 1284) Deallocate 587
 // Deleting tag 587
 // (op 752) Deallocate{}(1284) END
 // (op 752) end: generate({749})
 // (op 752) Unlock Scope 752
 // Unlock None
 // (op 471) Scope(752) END
 // (op 471) Scope(858) BEGIN
 // (op 858) Lock Scope 858
 // Lock Branch
 // (op 858) generate({855})
 // (op 858) NOP(855) BEGIN
 // (op 855) generate({})
 // (op 855) end: generate({})
 // (op 858) NOP(855) END
 // (op 858) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(863) BEGIN
 // (op 863) Assign dim(616) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 616: v**UNALLOCATED**
 // (op 863) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag616: VGPR Value: UInt32 x 1: (unallocated)
 // (op 863) reg expression
 // (op 863) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 863): v14
v_and_b32 v14, -2, v1 // (op 863) 
 // (op 863) BEGIN: Tile[1]Flatten
 // (op 863) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 863): v20
v_sub_u32 v20, v1, v14 // (op 863) 
 // (op 863) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 863): v14

 // (op 863) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 863): v14
v_and_b32 v14, 4294967294, v1 // (op 863) 
 // (op 863) BEGIN: Flatten
 // (op 863) {Flatten: Add(v14:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag616: 1 VGPR (Value: UInt32) (op 863): v42
v_add_u32 v42, v14, v20 // (op 863) 
 // (op 863) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 863): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 863): v14

 // (op 858) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(863) END
 // (op 858) Assign VGPR 2:U32(864) BEGIN
 // (op 864) Assign dim(617) = 2:U32
 // (op 864) Assign stride expression
 // (op 858) Assign VGPR 2:U32(864) END
 // (op 858) NOP(856) BEGIN
 // (op 856) generate({})
 // (op 856) end: generate({})
 // (op 858) NOP(856) END
 // (op 858) Assign VGPR 1:U32(868) BEGIN
 // (op 868) Assign dim(619) = 1:U32
 // (op 868) Assign stride expression
 // (op 858) Assign VGPR 1:U32(868) END
 // (op 858) StoreLDSTile Value: E8M0(396) BEGIN
 // (op 396) GEN: StoreLDSTile
 // (op 396) GEN: storeMacroTileLDS OP 396 LDS 33 MacroTile 10
 // (op 396) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 396) Generate 1:U32 into nullptr
 // (op 396) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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

 // (op 396) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v42
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 396) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b8 v42, v19 offset:2624 // (op 396) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1993 (current 1898)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 858) StoreLDSTile Value: E8M0(396) END
 // (op 858) Deallocate{}(1286) BEGIN
 // (op 1286) Deallocate 617
 // Deleting tag 617
 // (op 1286) Deallocate 616
 // Freeing DataFlowTag616: 1 VGPR (Value: UInt32) (op 863): v42

 // Deleting tag 616
 // (op 1286) Deallocate 619
 // Deleting tag 619
 // (op 1286) Deallocate 618
 // Deleting tag 618
 // (op 858) Deallocate{}(1286) END
 // (op 858) end: generate({855})
 // (op 858) Unlock Scope 858
 // Unlock None
 // (op 471) Scope(858) END
 // (op 471) NOP(367) BEGIN
 // (op 367) generate({})
 // (op 367) end: generate({})
 // (op 471) NOP(367) END
 // (op 471) NOP(724) BEGIN
 // (op 724) generate({})
 // (op 724) end: generate({})
 // (op 471) NOP(724) END
 // (op 471) NOP(721) BEGIN
 // (op 721) generate({})
 // (op 721) end: generate({})
 // (op 471) NOP(721) END
 // (op 471) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(728) BEGIN
 // (op 728) Assign dim(576) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 576: v**UNALLOCATED**
 // (op 728) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag576: VGPR Value: UInt32 x 1: (unallocated)
 // (op 728) reg expression
 // (op 728) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 728): v14
v_and_b32 v14, -2, v1 // (op 728) 
 // (op 728) BEGIN: Tile[1]
 // (op 728) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 728): v20
v_sub_u32 v20, v1, v14 // (op 728) 
 // (op 728) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 728): v14

 // (op 728) BEGIN: Flatten
 // (op 728) {Flatten: ShiftL({Tile[1]: v20:U32}, 5:U32)U32}
 // (op 728) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 728)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 728)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 728): v14
v_lshlrev_b32 v14, 5, v20 // (op 728) 
 // (op 728) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 728): v20

 // (op 728) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 728): v20
v_lshrrev_b32 v20, 1, v1 // (op 728) 
 // (op 728) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 728): v42
v_lshl_add_u32 v42, v20, 6, v14 // (op 728) 
 // (op 728) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 728): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 728): v20

 // (op 728) ShiftL({Flatten: v42:U32}, 2:U32)U32
 // (op 728) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 728)         value (Flatten: VGPR Value: UInt32 x 1: v42) 
 // (op 728)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 728): v14
v_lshlrev_b32 v14, 2, v42 // (op 728) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 728): v42

 // (op 728) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag576: 1 VGPR (Value: UInt32) (op 728): v20
v_lshrrev_b32 v20, 3, v14 // (op 728) 
 // Freeing : 1 VGPR (Value: UInt32) (op 728): v14

 // (op 471) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(728) END
 // (op 471) Assign VGPR 32:U32(729) BEGIN
 // (op 729) Assign dim(577) = 32:U32
 // (op 729) Assign stride expression
 // (op 471) Assign VGPR 32:U32(729) END
 // (op 471) NOP(722) BEGIN
 // (op 722) generate({})
 // (op 722) end: generate({})
 // (op 471) NOP(722) END
 // (op 471) Assign VGPR 0:U32(733) BEGIN
 // (op 733) Assign dim(579) = 0:U32
 // (op 733) Assign stride expression
 // (op 471) Assign VGPR 0:U32(733) END
 // (op 471) NOP(776) BEGIN
 // (op 776) generate({})
 // (op 776) end: generate({})
 // (op 471) NOP(776) END
 // (op 471) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(781) BEGIN
 // (op 781) Assign dim(592) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 592: v**UNALLOCATED**
 // (op 781) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag592: VGPR Value: UInt32 x 1: (unallocated)
 // (op 781) reg expression
 // (op 781) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 781): v14
v_and_b32 v14, -2, v1 // (op 781) 
 // (op 781) BEGIN: Tile[1]
 // (op 781) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 781): v42
v_sub_u32 v42, v1, v14 // (op 781) 
 // (op 781) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 781): v14

 // (op 781) BEGIN: Flatten
 // (op 781) {Flatten: ShiftL({Tile[1]: v42:U32}, 5:U32)U32}
 // (op 781) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 781)         value (Tile[1]: VGPR Value: UInt32 x 1: v42) 
 // (op 781)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 781): v14
v_lshlrev_b32 v14, 5, v42 // (op 781) 
 // (op 781) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 781): v42

 // (op 781) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 781): v42
v_lshrrev_b32 v42, 1, v1 // (op 781) 
 // (op 781) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 781): v43
v_lshl_add_u32 v43, v42, 6, v14 // (op 781) 
 // (op 781) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 781): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 781): v42

 // (op 781) Multiply({Flatten: v43:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 781): v14
v_mul_lo_u32 v14, v43, 6 // (op 781) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 781): v43

 // (op 781) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag592: 1 VGPR (Value: UInt32) (op 781): v42
v_lshrrev_b32 v42, 3, v14 // (op 781) 
 // Freeing : 1 VGPR (Value: UInt32) (op 781): v14

 // (op 471) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(781) END
 // (op 471) Assign VGPR 48:U32(782) BEGIN
 // (op 782) Assign dim(593) = 48:U32
 // (op 782) Assign stride expression
 // (op 471) Assign VGPR 48:U32(782) END
 // (op 471) NOP(777) BEGIN
 // (op 777) generate({})
 // (op 777) end: generate({})
 // (op 471) NOP(777) END
 // (op 471) Assign VGPR 0:U32(786) BEGIN
 // (op 786) Assign dim(595) = 0:U32
 // (op 786) Assign stride expression
 // (op 471) Assign VGPR 0:U32(786) END
 // (op 471) NOP(829) BEGIN
 // (op 829) generate({})
 // (op 829) end: generate({})
 // (op 471) NOP(829) END
 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(834) BEGIN
 // (op 834) Assign dim(608) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 608: v**UNALLOCATED**
 // (op 834) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag608: VGPR Value: UInt32 x 1: (unallocated)
 // (op 834) reg expression
 // (op 834) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 834): v14
v_and_b32 v14, -2, v1 // (op 834) 
 // (op 834) BEGIN: Tile[1]Flatten
 // (op 834) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 834): v43
v_sub_u32 v43, v1, v14 // (op 834) 
 // (op 834) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 834): v14

 // (op 834) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 834): v14
v_and_b32 v14, 4294967294, v1 // (op 834) 
 // (op 834) BEGIN: Flatten
 // (op 834) {Flatten: Add(v14:U32, {Tile[1]Flatten: v43:U32})U32}
// Allocated DataFlowTag608: 1 VGPR (Value: UInt32) (op 834): v44
v_add_u32 v44, v14, v43 // (op 834) 
 // (op 834) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 834): v43

 // Freeing : 1 VGPR (Value: UInt32) (op 834): v14

 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(834) END
 // (op 471) Assign VGPR 2:U32(835) BEGIN
 // (op 835) Assign dim(609) = 2:U32
 // (op 835) Assign stride expression
 // (op 471) Assign VGPR 2:U32(835) END
 // (op 471) NOP(830) BEGIN
 // (op 830) generate({})
 // (op 830) end: generate({})
 // (op 471) NOP(830) END
 // (op 471) Assign VGPR 1:U32(839) BEGIN
 // (op 839) Assign dim(611) = 1:U32
 // (op 839) Assign stride expression
 // (op 471) Assign VGPR 1:U32(839) END
 // (op 471) NOP(882) BEGIN
 // (op 882) generate({})
 // (op 882) end: generate({})
 // (op 471) NOP(882) END
 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(887) BEGIN
 // (op 887) Assign dim(624) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 624: v**UNALLOCATED**
 // (op 887) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag624: VGPR Value: UInt32 x 1: (unallocated)
 // (op 887) reg expression
 // (op 887) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 887): v14
v_and_b32 v14, -2, v1 // (op 887) 
 // (op 887) BEGIN: Tile[1]Flatten
 // (op 887) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 887): v43
v_sub_u32 v43, v1, v14 // (op 887) 
 // (op 887) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 887): v14

 // (op 887) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 887): v14
v_and_b32 v14, 4294967294, v1 // (op 887) 
 // (op 887) BEGIN: Flatten
 // (op 887) {Flatten: Add(v14:U32, {Tile[1]Flatten: v43:U32})U32}
// Allocated DataFlowTag624: 1 VGPR (Value: UInt32) (op 887): v45
v_add_u32 v45, v14, v43 // (op 887) 
 // (op 887) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 887): v43

 // Freeing : 1 VGPR (Value: UInt32) (op 887): v14

 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(887) END
 // (op 471) Assign VGPR 2:U32(888) BEGIN
 // (op 888) Assign dim(625) = 2:U32
 // (op 888) Assign stride expression
 // (op 471) Assign VGPR 2:U32(888) END
 // (op 471) NOP(883) BEGIN
 // (op 883) generate({})
 // (op 883) end: generate({})
 // (op 471) NOP(883) END
 // (op 471) Assign VGPR 1:U32(892) BEGIN
 // (op 892) Assign dim(627) = 1:U32
 // (op 892) Assign stride expression
 // (op 471) Assign VGPR 1:U32(892) END
 // (op 471) NOP(969) BEGIN
 // (op 969) generate({})
 // (op 969) end: generate({})
 // (op 471) NOP(969) END
 // (op 471) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(974) BEGIN
 // (op 974) Assign dim(650) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 650: v**UNALLOCATED**
 // (op 974) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag650: VGPR Value: UInt32 x 1: (unallocated)
 // (op 974) reg expression
 // (op 974) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 974): v14
v_and_b32 v14, -2, v1 // (op 974) 
 // (op 974) BEGIN: Tile[1]
 // (op 974) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 974): v43
v_sub_u32 v43, v1, v14 // (op 974) 
 // (op 974) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 974): v14

 // (op 974) BEGIN: Flatten
 // (op 974) {Flatten: ShiftL({Tile[1]: v43:U32}, 5:U32)U32}
 // (op 974) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 974)         value (Tile[1]: VGPR Value: UInt32 x 1: v43) 
 // (op 974)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 974): v14
v_lshlrev_b32 v14, 5, v43 // (op 974) 
 // (op 974) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 974): v43

 // (op 974) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 974): v43
v_lshrrev_b32 v43, 1, v1 // (op 974) 
 // (op 974) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 974): v46
v_lshl_add_u32 v46, v43, 6, v14 // (op 974) 
 // (op 974) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 974): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 974): v43

 // (op 974) ShiftL({Flatten: v46:U32}, 2:U32)U32
 // (op 974) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 974)         value (Flatten: VGPR Value: UInt32 x 1: v46) 
 // (op 974)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 974): v14
v_lshlrev_b32 v14, 2, v46 // (op 974) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 974): v46

 // (op 974) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag650: 1 VGPR (Value: UInt32) (op 974): v43
v_lshrrev_b32 v43, 3, v14 // (op 974) 
 // Freeing : 1 VGPR (Value: UInt32) (op 974): v14

 // (op 471) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(974) END
 // (op 471) Assign VGPR 32:U32(975) BEGIN
 // (op 975) Assign dim(651) = 32:U32
 // (op 975) Assign stride expression
 // (op 471) Assign VGPR 32:U32(975) END
 // (op 471) NOP(970) BEGIN
 // (op 970) generate({})
 // (op 970) end: generate({})
 // (op 471) NOP(970) END
 // (op 471) Assign VGPR 0:U32(979) BEGIN
 // (op 979) Assign dim(653) = 0:U32
 // (op 979) Assign stride expression
 // (op 471) Assign VGPR 0:U32(979) END
 // (op 471) NOP(993) BEGIN
 // (op 993) generate({})
 // (op 993) end: generate({})
 // (op 471) NOP(993) END
 // (op 471) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(998) BEGIN
 // (op 998) Assign dim(658) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 658: v**UNALLOCATED**
 // (op 998) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag658: VGPR Value: UInt32 x 1: (unallocated)
 // (op 998) reg expression
 // (op 998) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 998): v14
v_and_b32 v14, -2, v1 // (op 998) 
 // (op 998) BEGIN: Tile[1]
 // (op 998) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 998): v46
v_sub_u32 v46, v1, v14 // (op 998) 
 // (op 998) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 998): v14

 // (op 998) BEGIN: Flatten
 // (op 998) {Flatten: ShiftL({Tile[1]: v46:U32}, 5:U32)U32}
 // (op 998) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 998)         value (Tile[1]: VGPR Value: UInt32 x 1: v46) 
 // (op 998)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 998): v14
v_lshlrev_b32 v14, 5, v46 // (op 998) 
 // (op 998) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 998): v46

 // (op 998) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 998): v46
v_lshrrev_b32 v46, 1, v1 // (op 998) 
 // (op 998) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 998): v47
v_lshl_add_u32 v47, v46, 6, v14 // (op 998) 
 // (op 998) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 998): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 998): v46

 // (op 998) Multiply({Flatten: v47:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 998): v14
v_mul_lo_u32 v14, v47, 6 // (op 998) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 998): v47

 // (op 998) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag658: 1 VGPR (Value: UInt32) (op 998): v46
v_lshrrev_b32 v46, 3, v14 // (op 998) 
 // Freeing : 1 VGPR (Value: UInt32) (op 998): v14

 // (op 471) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(998) END
 // (op 471) Assign VGPR 48:U32(999) BEGIN
 // (op 999) Assign dim(659) = 48:U32
 // (op 999) Assign stride expression
 // (op 471) Assign VGPR 48:U32(999) END
 // (op 471) NOP(994) BEGIN
 // (op 994) generate({})
 // (op 994) end: generate({})
 // (op 471) NOP(994) END
 // (op 471) Assign VGPR 0:U32(1003) BEGIN
 // (op 1003) Assign dim(661) = 0:U32
 // (op 1003) Assign stride expression
 // (op 471) Assign VGPR 0:U32(1003) END
 // (op 471) NOP(1017) BEGIN
 // (op 1017) generate({})
 // (op 1017) end: generate({})
 // (op 471) NOP(1017) END
 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(1022) BEGIN
 // (op 1022) Assign dim(666) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 666: v**UNALLOCATED**
 // (op 1022) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag666: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1022) reg expression
 // (op 1022) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1022): v14
v_and_b32 v14, -2, v1 // (op 1022) 
 // (op 1022) BEGIN: Tile[1]Flatten
 // (op 1022) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1022): v47
v_sub_u32 v47, v1, v14 // (op 1022) 
 // (op 1022) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1022): v14

 // (op 1022) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1022): v14
v_and_b32 v14, 4294967294, v1 // (op 1022) 
 // (op 1022) BEGIN: Flatten
 // (op 1022) {Flatten: Add(v14:U32, {Tile[1]Flatten: v47:U32})U32}
// Allocated DataFlowTag666: 1 VGPR (Value: UInt32) (op 1022): v48
v_add_u32 v48, v14, v47 // (op 1022) 
 // (op 1022) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1022): v47

 // Freeing : 1 VGPR (Value: UInt32) (op 1022): v14

 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(1022) END
 // (op 471) Assign VGPR 2:U32(1023) BEGIN
 // (op 1023) Assign dim(667) = 2:U32
 // (op 1023) Assign stride expression
 // (op 471) Assign VGPR 2:U32(1023) END
 // (op 471) NOP(1018) BEGIN
 // (op 1018) generate({})
 // (op 1018) end: generate({})
 // (op 471) NOP(1018) END
 // (op 471) Assign VGPR 1:U32(1027) BEGIN
 // (op 1027) Assign dim(669) = 1:U32
 // (op 1027) Assign stride expression
 // (op 471) Assign VGPR 1:U32(1027) END
 // (op 471) NOP(1041) BEGIN
 // (op 1041) generate({})
 // (op 1041) end: generate({})
 // (op 471) NOP(1041) END
 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(1046) BEGIN
 // (op 1046) Assign dim(674) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 674: v**UNALLOCATED**
 // (op 1046) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag674: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1046) reg expression
 // (op 1046) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v14
v_and_b32 v14, -2, v1 // (op 1046) 
 // (op 1046) BEGIN: Tile[1]Flatten
 // (op 1046) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v47
v_sub_u32 v47, v1, v14 // (op 1046) 
 // (op 1046) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1046): v14

 // (op 1046) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1046): v14
v_and_b32 v14, 4294967294, v1 // (op 1046) 
 // (op 1046) BEGIN: Flatten
 // (op 1046) {Flatten: Add(v14:U32, {Tile[1]Flatten: v47:U32})U32}
// Allocated DataFlowTag674: 1 VGPR (Value: UInt32) (op 1046): v49
v_add_u32 v49, v14, v47 // (op 1046) 
 // (op 1046) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1046): v47

 // Freeing : 1 VGPR (Value: UInt32) (op 1046): v14

 // (op 471) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(1046) END
 // (op 471) Assign VGPR 2:U32(1047) BEGIN
 // (op 1047) Assign dim(675) = 2:U32
 // (op 1047) Assign stride expression
 // (op 471) Assign VGPR 2:U32(1047) END
 // (op 471) NOP(1042) BEGIN
 // (op 1042) generate({})
 // (op 1042) end: generate({})
 // (op 471) NOP(1042) END
 // (op 471) Assign VGPR 1:U32(1051) BEGIN
 // (op 1051) Assign dim(677) = 1:U32
 // (op 1051) Assign stride expression
 // (op 471) Assign VGPR 1:U32(1051) END
 // (op 471) ForLoopOp KLoop: LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(60) BEGIN
 // (op 60) Initialize For Loop
 // (op 60) generate({61})
 // (op 60) Assign SGPR 0:U32(61) BEGIN
 // (op 61) Assign dim(397) = 0:U32
 // tag 397: s**UNALLOCATED**
 // (op 61) Generate 0:U32 into DataFlowTag397: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag397: 1 SGPR (Value: UInt32) (op 61): s0
s_mov_b32 s0, 0 // (op 61) call()
 // (op 60) Assign SGPR 0:U32(61) END
 // (op 60) end: generate({61})
 // (op 60) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 60) 
 // (op 60) Generate LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 60) LessThan({DataFlowTag397: s0:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 60) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 60) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 60): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 60) 
// Allocated : 1 SGPR (Value: UInt32) (op 60): s1
s_mov_b32 s1, s14 // (op 60) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 60): s14, s15

// Allocated : 1 SGPR (Value: UInt32) (op 60): s14
s_and_b32 s14, s1, 4294967294 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s1

s_cmp_lt_u32 s0, s14 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s14

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 60) 
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60:
 // (op 60) 
 // (op 60) generate({62, 292, 1095})
 // (op 60) Assign SGPR Add(DataFlowTag(397)U32, 2:U32)U32(62) BEGIN
 // (op 62) Assign dim(397) = Add(DataFlowTag(397)U32, 2:U32)U32
 // (op 62) Generate Add(DataFlowTag(397)U32, 2:U32)U32 into DataFlowTag397: SGPR Value: UInt32 x 1: s0
 // (op 62) reg expression
 // (op 62) Add({DataFlowTag397: s0:U32}, 2:U32)U32
 // (op 60) SetCoordinate(292) BEGIN
 // (op 292) generate({})
 // (op 292) end: generate({})
 // (op 292) generate({10})
 // (op 292) LoadTiled Value: FP6x16(10) BEGIN
 // (op 10) GEN: loadMacroTileVGPRCI Value: FP6x16
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
 // 	info.data = VGPR Value: FP6x16 x 2: v[22:27]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 528: v**UNALLOCATED**
 // (op 60) Barrier(1095) BEGIN
s_add_u32 s0, s0, 2 // (op 62) 
 // (op 60) Assign SGPR Add(DataFlowTag(397)U32, 2:U32)U32(62) END
// Allocated Offset528: 1 VGPR (Value: UInt32) (op 10): v14
v_mov_b32 v14, v17 // (op 10) 
 // (op 10) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[22:27]
 // 	info.rowOffsetReg = Offset528: VGPR Value: UInt32 x 1: v14
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
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

buffer_load_dwordx4 v[22:25], v14, s[44:47], 0 offen offset:96 // (op 10) Load value
// VMEM: Expected complete at 2432 (current 1952)
buffer_load_dwordx2 v[26:27], v14, s[44:47], 0 offen offset:112 // (op 10) Load value
// VMEM: Expected complete at 2433 (current 1953)
 // (op 292) LoadTiled Value: FP6x16(10) END
 // (op 292) end: generate({10})
 // (op 60) SetCoordinate(292) END
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1095) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 60) Barrier(1095) END
 // (op 60) SetCoordinate(302) BEGIN
 // (op 302) generate({})
 // (op 302) end: generate({})
 // (op 302) generate({25})
 // (op 302) LoadTiled Value: E8M0(25) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v18
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 566: v**UNALLOCATED**
 // (op 60) SetCoordinate(70) BEGIN
 // (op 70) generate({})
 // (op 70) end: generate({})
 // (op 70) generate({72})
 // (op 70) LoadLDSTile{Value: FP4x8}(72) BEGIN
 // (op 72) GEN: loadMacroTileWAVELDS OP 72 LDS 17 WaveTile 105
 // (op 72) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 72) Generate 0:U32 into nullptr
 // tag 15: v**UNALLOCATED**
 // (op 72) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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

 // (op 72) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 72) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated Offset566: 1 VGPR (Value: UInt32) (op 25): v47
v_mov_b32 v47, v21 // (op 25) 
 // (op 25) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v18
 // 	info.rowOffsetReg = Offset566: VGPR Value: UInt32 x 1: v47
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
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

// Allocated : 4 VGPRs (Value: FP4x8) (op 72): v50, v51, v52, v53
ds_read_b128 v[50:53], v3 offset:1600 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848, 880, 912, 944, 976, 1008]
// DSMEM: Expected complete at 2086 (current 1991)
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 70) LoadLDSTile{Value: FP4x8}(72) END
 // (op 70) end: generate({72})
 // (op 60) SetCoordinate(70) END
buffer_load_ubyte v18, v47, s[52:55], 0 offen offset:4 // (op 25) Load value
// VMEM: Expected stall of 309. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2747 (current 2267)
 // (op 302) LoadTiled Value: E8M0(25) END
 // (op 302) end: generate({25})
 // (op 60) SetCoordinate(302) END
 // (op 60) Barrier(1108) BEGIN
 // (op 60) SetCoordinate(287) BEGIN
 // (op 287) generate({})
 // (op 287) end: generate({})
 // (op 287) generate({4})
 // (op 287) LoadTiled Value: FP4x8(4) BEGIN
 // (op 4) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 4) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 4) Generate 0:U64 into nullptr
 // (op 4) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[28:31]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 509: v**UNALLOCATED**
s_barrier  // (op 1108) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 60) Barrier(1108) END
// Allocated Offset509: 1 VGPR (Value: UInt32) (op 4): v54
v_mov_b32 v54, v13 // (op 4) 
 // (op 4) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[28:31]
 // 	info.rowOffsetReg = Offset509: VGPR Value: UInt32 x 1: v54
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 64
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 4) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) SetCoordinate(75) BEGIN
 // (op 75) generate({})
 // (op 75) end: generate({})
 // (op 75) generate({77})
 // (op 75) LoadLDSTile{Value: FP6x16}(77) BEGIN
 // (op 77) GEN: loadMacroTileWAVELDS OP 77 LDS 25 WaveTile 171
 // (op 77) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 77) Generate 0:U32 into nullptr
 // tag 23: v**UNALLOCATED**
 // (op 77) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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

 // (op 77) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
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
 // (op 77) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 77): v56, v57, v58, v59, v60, v61
ds_read_b128 v[56:59], v4 // (op 77) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 2399 (current 2304)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
ds_read_b64 v[60:61], v4 offset:16 // (op 77) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 2400 (current 2305)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 75) LoadLDSTile{Value: FP6x16}(77) END
 // (op 75) end: generate({77})
 // (op 60) SetCoordinate(75) END
buffer_load_dwordx4 v[28:31], v54, s[40:43], 0 offen offset:64 // (op 4) Load value
// VMEM: Expected stall of 161. CBNW: 5, Inc: 3
// VMEM: Expected complete at 2913 (current 2433)
 // (op 287) LoadTiled Value: FP4x8(4) END
 // (op 287) end: generate({4})
 // (op 60) SetCoordinate(287) END
 // (op 60) Barrier(1122) BEGIN
 // (op 60) SetCoordinate(297) BEGIN
 // (op 297) generate({})
 // (op 297) end: generate({})
 // (op 297) generate({17})
 // (op 297) LoadTiled Value: E8M0(17) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 547: v**UNALLOCATED**
// Allocated Offset547: 1 VGPR (Value: UInt32) (op 17): v55
v_mov_b32 v55, v15 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = Offset547: VGPR Value: UInt32 x 1: v55
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
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

buffer_load_ubyte v19, v55, s[48:51], 0 offen offset:4 // (op 17) Load value
// VMEM: Expected complete at 2915 (current 2435)
 // (op 297) LoadTiled Value: E8M0(17) END
 // (op 297) end: generate({17})
 // (op 60) SetCoordinate(297) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1122) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 60) Barrier(1122) END
 // (op 60) Barrier(1159) BEGIN
 // (op 60) SetCoordinate(80) BEGIN
 // (op 80) generate({})
 // (op 80) end: generate({})
 // (op 80) generate({82})
 // (op 80) LoadLDSTile{Value: E8M0}(82) BEGIN
 // (op 82) GEN: loadMacroTileWAVELDS OP 82 LDS 33 WaveTile 237
 // (op 82) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 82) Generate 1:U32 into nullptr
 // tag 31: v**UNALLOCATED**
 // (op 82) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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

 // (op 82) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v0
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 82) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1159) 
 // (op 60) Barrier(1159) END
// Allocated : 1 VGPR (Value: E8M0) (op 82): v62
ds_read_u8 v62, v0 offset:2624 // (op 82) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2568 (current 2473)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 80) LoadLDSTile{Value: E8M0}(82) END
 // (op 80) end: generate({82})
 // (op 60) SetCoordinate(80) END
 // (op 60) StoreLDSTile Value: FP6x16(257) BEGIN
 // (op 257) GEN: StoreLDSTile
 // (op 257) GEN: storeMacroTileLDS OP 257 LDS 476 MacroTile 474
 // (op 257) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 257) Generate 0:U32 into nullptr
 // (op 257) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[32:37]
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
 // 	info.offset = Literal Value: UInt32 x 0: 2688
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

 // (op 257) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[32:37]
 // 	info.rowOffsetReg = DataFlowTag658: VGPR Value: UInt32 x 1: v46
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
 // 	info.offset = Literal Value: UInt32 x 0: 2688
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 257) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Barrier(1136) BEGIN
s_barrier  // (op 1136) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 60) Barrier(1136) END
s_waitcnt vmcnt(9)
ds_write_b128 v46, v[32:35] offset:2688 // (op 257) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 2571 (current 2476)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 2688, Size: 1536)
 // (op 60) SetCoordinate(85) BEGIN
 // (op 85) generate({})
 // (op 85) end: generate({})
 // (op 85) generate({87})
 // (op 85) LoadLDSTile{Value: E8M0}(87) BEGIN
 // (op 87) GEN: loadMacroTileWAVELDS OP 87 LDS 41 WaveTile 297
 // (op 87) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 87) Generate 1:U32 into nullptr
 // tag 39: v**UNALLOCATED**
 // (op 87) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 87) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag636: VGPR Value: UInt32 x 1: v6
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 87) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 87): v63
ds_read_u8 v63, v6 offset:1536 // (op 87) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2572 (current 2477)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 85) LoadLDSTile{Value: E8M0}(87) END
 // (op 85) end: generate({87})
 // (op 60) SetCoordinate(85) END
s_waitcnt vmcnt(8)
ds_write_b64 v46, v[36:37] offset:2704 // (op 257) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected stall of 91. CBNW: 3, Inc: 3
// DSMEM: Expected complete at 2665 (current 2570)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 2688, Size: 1536)
 // (op 60) StoreLDSTile Value: FP6x16(257) END
 // (op 60) Multiply(90) BEGIN
 // (op 90) Generate ScaledMatrixMultiply(WaveTile8xFP4, WaveTile16xFP6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 60) Deallocate{}(1254) BEGIN
 // (op 1254) Deallocate 660
 // Deleting tag 660
 // (op 60) Deallocate{}(1254) END
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[50:53], v[56:61], a[0:15], v62, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 90) 
 // (op 60) Multiply(90) END
 // (op 60) Barrier(1171) BEGIN
 // (op 60) Deallocate{}(1188) BEGIN
 // (op 1188) Deallocate 23
 // Freeing : 6 VGPRs (Value: FP6x16) (op 77): v56, v57, v58, v59, v60, v61

 // Deleting tag 23
 // (op 1188) Deallocate 39
 // Freeing : 1 VGPR (Value: E8M0) (op 87): v63

 // Deleting tag 39
 // (op 1188) Deallocate 15
 // Freeing : 4 VGPRs (Value: FP4x8) (op 72): v50, v51, v52, v53

 // Deleting tag 15
 // (op 1188) Deallocate 31
 // Freeing : 1 VGPR (Value: E8M0) (op 82): v62

 // Deleting tag 31
 // (op 60) Deallocate{}(1188) END
s_barrier  // (op 1171) 
 // (op 60) Barrier(1171) END
 // (op 60) StoreLDSTile Value: E8M0(259) BEGIN
 // (op 259) GEN: StoreLDSTile
 // (op 259) GEN: storeMacroTileLDS OP 259 LDS 484 MacroTile 482
 // (op 259) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 259) Generate 1:U32 into nullptr
 // (op 259) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 4224
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

 // (op 259) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
 // 	info.rowOffsetReg = DataFlowTag674: VGPR Value: UInt32 x 1: v49
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
 // 	info.offset = Literal Value: UInt32 x 0: 4224
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 259) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b8 v49, v16 offset:4224 // (op 259) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2672 (current 2577)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 4224, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(259) END
 // (op 60) Deallocate{}(1258) BEGIN
 // (op 1258) Deallocate 676
 // Deleting tag 676
 // (op 60) Deallocate{}(1258) END
 // (op 60) Barrier(1153) BEGIN
s_barrier  // (op 1153) 
 // (op 60) Barrier(1153) END
 // (op 60) StoreLDSTile Value: FP4x8(247) BEGIN
 // (op 247) GEN: StoreLDSTile
 // (op 247) GEN: storeMacroTileLDS OP 247 LDS 472 MacroTile 470
 // (op 247) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 247) Generate 0:U32 into nullptr
 // (op 247) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[38:41]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4288
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[38:41]
 // 	info.rowOffsetReg = DataFlowTag650: VGPR Value: UInt32 x 1: v43
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4288
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b128 v43, v[38:41] offset:4288 // (op 247) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 2675 (current 2580)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 4288, Size: 1024)
 // (op 60) StoreLDSTile Value: FP4x8(247) END
 // (op 60) Deallocate{}(1226) BEGIN
 // (op 1226) Deallocate 652
 // Deleting tag 652
 // (op 60) Deallocate{}(1226) END
 // (op 60) Barrier(1165) BEGIN
s_barrier  // (op 1165) 
 // (op 60) Barrier(1165) END
 // (op 60) StoreLDSTile Value: E8M0(258) BEGIN
 // (op 258) GEN: StoreLDSTile
 // (op 258) GEN: storeMacroTileLDS OP 258 LDS 480 MacroTile 478
 // (op 258) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 258) Generate 1:U32 into nullptr
 // (op 258) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
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
 // 	info.offset = Literal Value: UInt32 x 0: 5312
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

 // (op 258) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = DataFlowTag666: VGPR Value: UInt32 x 1: v48
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
 // 	info.offset = Literal Value: UInt32 x 0: 5312
 // 	info.bufDesc = (nullptr)
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

s_waitcnt vmcnt(5)
ds_write_b8 v48, v12 offset:5312 // (op 258) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 84. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2762 (current 2667)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 5312, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(258) END
 // (op 60) Deallocate{}(1256) BEGIN
 // (op 1256) Deallocate 668
 // Deleting tag 668
 // (op 60) Deallocate{}(1256) END
 // (op 60) NOP(410) BEGIN
 // (op 410) generate({})
 // (op 410) end: generate({})
 // (op 60) NOP(410) END
 // (op 60) SetCoordinate(313) BEGIN
 // (op 313) generate({})
 // (op 313) end: generate({})
 // (op 313) generate({260})
 // (op 313) LoadTiled Value: FP6x16(260) BEGIN
 // (op 260) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 260) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 260) Generate 0:U64 into nullptr
 // (op 260) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[32:37]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 260) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[32:37]
 // 	info.rowOffsetReg = Offset528: VGPR Value: UInt32 x 1: v14
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 260) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Barrier(1150) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 4288, Size: 1024)
 // (op 60) Barrier(1150) END
buffer_load_dwordx4 v[32:35], v14, s[44:47], 0 offen offset:144 // (op 260) Load value
// VMEM: Expected stall of 114. CBNW: 2, Inc: 3
// VMEM: Expected complete at 3228 (current 2748)
 // (op 60) SetCoordinate(248) BEGIN
 // (op 248) generate({})
 // (op 248) end: generate({})
 // (op 248) generate({249})
 // (op 248) LoadLDSTile{Value: FP4x8}(249) BEGIN
 // (op 249) GEN: loadMacroTileWAVELDS OP 249 LDS 472 WaveTile 105
 // (op 249) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 249) Generate 0:U32 into nullptr
 // tag 486: v**UNALLOCATED**
 // (op 249) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4288
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag654: VGPR Value: UInt32 x 1: v7
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4288
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 249): v50, v51, v52, v53
ds_read_b128 v[50:53], v7 offset:4288 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848, 880, 912, 944, 976, 1008]
// DSMEM: Expected complete at 2888 (current 2793)
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 4288, Size: 1024)
 // (op 248) LoadLDSTile{Value: FP4x8}(249) END
 // (op 248) Deallocate{}(1232) BEGIN
 // (op 1232) Deallocate 656
 // Deleting tag 656
 // (op 248) Deallocate{}(1232) END
 // (op 248) end: generate({249})
 // (op 60) SetCoordinate(248) END
buffer_load_dwordx2 v[36:37], v14, s[44:47], 0 offen offset:160 // (op 260) Load value
// VMEM: Expected stall of 164. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3394 (current 2914)
 // (op 313) LoadTiled Value: FP6x16(260) END
 // (op 313) end: generate({260})
 // (op 60) SetCoordinate(313) END
 // (op 60) Deallocate{}(1228) BEGIN
 // (op 1228) Deallocate 472
 // Deleting tag 472
 // (op 60) Deallocate{}(1228) END
 // (op 60) Deallocate{}(1262) BEGIN
 // (op 1262) Deallocate 528
 // Freeing Offset528: 1 VGPR (Value: UInt32) (op 10): v14

 // Deleting tag 528
 // (op 1262) Deallocate 530
 // Deleting tag 530
 // (op 60) Deallocate{}(1262) END
 // (op 60) Barrier(1156) BEGIN
 // (op 60) Barrier(1230) BEGIN
 // (op 60) SetCoordinate(323) BEGIN
 // (op 323) generate({})
 // (op 323) end: generate({})
 // (op 323) generate({262})
 // (op 323) LoadTiled Value: E8M0(262) BEGIN
 // (op 262) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 262) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 262) Generate 1:U64 into nullptr
 // (op 262) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 262) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
 // 	info.rowOffsetReg = Offset566: VGPR Value: UInt32 x 1: v47
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 262) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Assign VGPR Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32(554) BEGIN
 // (op 554) Assign dim(526) = Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32
 // (op 554) Generate Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32 into DataFlowTag526: VGPR Value: UInt32 x 1: v17
 // (op 554) reg expression
 // (op 554) Add({DataFlowTag526: v17:U32}, 96:U32)U32
s_barrier  // (op 1156) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 2688, Size: 1536)
 // (op 60) Barrier(1156) END
s_barrier  // (op 1230) 
 // (op 60) Barrier(1230) END
 // (op 60) SetCoordinate(250) BEGIN
 // (op 250) generate({})
 // (op 250) end: generate({})
 // (op 250) generate({251})
 // (op 250) LoadLDSTile{Value: FP6x16}(251) BEGIN
 // (op 251) GEN: loadMacroTileWAVELDS OP 251 LDS 476 WaveTile 171
 // (op 251) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 251) Generate 0:U32 into nullptr
 // tag 488: v**UNALLOCATED**
 // (op 251) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.offset = Literal Value: UInt32 x 0: 2688
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
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.offset = Literal Value: UInt32 x 0: 2688
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
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v16, v47, s[52:55], 0 offen offset:6 // (op 262) Load value
// VMEM: Expected complete at 3397 (current 2917)
 // (op 323) LoadTiled Value: E8M0(262) END
 // (op 323) end: generate({262})
 // (op 60) SetCoordinate(323) END
v_add_u32 v17, 96, v17 // (op 554) 
 // (op 60) Assign VGPR Add(DataFlowTag(526)U32, ShiftL(Convert(DataFlowTag(527)U64)U32, 1:U32)U32)U32(554) END
 // (op 60) Deallocate{}(1266) BEGIN
 // (op 1266) Deallocate 566
 // Freeing Offset566: 1 VGPR (Value: UInt32) (op 25): v47

 // Deleting tag 566
 // (op 1266) Deallocate 568
 // Deleting tag 568
 // (op 60) Deallocate{}(1266) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 251): v56, v57, v58, v59, v60, v61
ds_read_b128 v[56:59], v8 offset:2688 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 3058 (current 2963)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 2688, Size: 1536)
 // (op 60) SetCoordinate(308) BEGIN
 // (op 308) generate({})
 // (op 308) end: generate({})
 // (op 308) generate({246})
 // (op 308) LoadTiled Value: FP4x8(246) BEGIN
 // (op 246) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 246) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 246) Generate 0:U64 into nullptr
 // (op 246) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[38:41]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 246) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[38:41]
 // 	info.rowOffsetReg = Offset509: VGPR Value: UInt32 x 1: v54
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 96
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 246) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Assign VGPR Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32(682) BEGIN
 // (op 682) Assign dim(564) = Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32
 // (op 682) Generate Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32 into DataFlowTag564: VGPR Value: UInt32 x 1: v21
 // (op 682) reg expression
 // (op 682) Add({DataFlowTag564: v21:U32}, 4:U32)U32
ds_read_b64 v[60:61], v8 offset:2704 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 3059 (current 2964)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 2688, Size: 1536)
 // (op 250) LoadLDSTile{Value: FP6x16}(251) END
 // (op 250) Deallocate{}(1238) BEGIN
 // (op 1238) Deallocate 664
 // Deleting tag 664
 // (op 250) Deallocate{}(1238) END
 // (op 250) end: generate({251})
 // (op 60) SetCoordinate(250) END
v_add_u32 v21, 4, v21 // (op 682) 
 // (op 60) Assign VGPR Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32(682) END
 // (op 60) Deallocate{}(1234) BEGIN
 // (op 1234) Deallocate 476
 // Deleting tag 476
 // (op 60) Deallocate{}(1234) END
buffer_load_dwordx4 v[38:41], v54, s[40:43], 0 offen offset:96 // (op 246) Load value
// VMEM: Expected stall of 307. CBNW: 5, Inc: 3
// VMEM: Expected complete at 3709 (current 3229)
 // (op 308) LoadTiled Value: FP4x8(246) END
 // (op 308) end: generate({246})
 // (op 60) SetCoordinate(308) END
 // (op 60) Barrier(1162) BEGIN
 // (op 60) Barrier(1236) BEGIN
 // (op 60) Deallocate{}(1260) BEGIN
 // (op 1260) Deallocate 509
 // Freeing Offset509: 1 VGPR (Value: UInt32) (op 4): v54

 // Deleting tag 509
 // (op 1260) Deallocate 511
 // Deleting tag 511
 // (op 60) Deallocate{}(1260) END
s_barrier  // (op 1236) 
 // (op 60) Barrier(1236) END
 // (op 60) SetCoordinate(318) BEGIN
 // (op 318) generate({})
 // (op 318) end: generate({})
 // (op 318) generate({261})
 // (op 318) LoadTiled Value: E8M0(261) BEGIN
 // (op 261) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 261) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 261) Generate 1:U64 into nullptr
 // (op 261) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 261) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = Offset547: VGPR Value: UInt32 x 1: v55
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 261) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Assign VGPR Add(DataFlowTag(507)U32, ShiftL(Convert(DataFlowTag(508)U64)U32, 1:U32)U32)U32(490) BEGIN
 // (op 490) Assign dim(507) = Add(DataFlowTag(507)U32, ShiftL(Convert(DataFlowTag(508)U64)U32, 1:U32)U32)U32
 // (op 490) Generate Add(DataFlowTag(507)U32, ShiftL(Convert(DataFlowTag(508)U64)U32, 1:U32)U32)U32 into DataFlowTag507: VGPR Value: UInt32 x 1: v13
 // (op 490) reg expression
 // (op 490) Add({DataFlowTag507: v13:U32}, 64:U32)U32
v_add_u32 v13, 64, v13 // (op 490) 
 // (op 60) Assign VGPR Add(DataFlowTag(507)U32, ShiftL(Convert(DataFlowTag(508)U64)U32, 1:U32)U32)U32(490) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1162) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5312, Size: 64)
 // (op 60) Barrier(1162) END
buffer_load_ubyte v12, v55, s[48:51], 0 offen offset:6 // (op 261) Load value
// VMEM: Expected stall of 161. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3875 (current 3395)
 // (op 318) LoadTiled Value: E8M0(261) END
 // (op 318) end: generate({261})
 // (op 60) SetCoordinate(318) END
 // (op 60) SetCoordinate(252) BEGIN
 // (op 252) generate({})
 // (op 252) end: generate({})
 // (op 252) generate({253})
 // (op 252) LoadLDSTile{Value: E8M0}(253) BEGIN
 // (op 253) GEN: loadMacroTileWAVELDS OP 253 LDS 480 WaveTile 237
 // (op 253) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 253) Generate 1:U32 into nullptr
 // tag 490: v**UNALLOCATED**
 // (op 253) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 5312
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
 // 	info.offset = Literal Value: UInt32 x 0: 5312
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 60) Deallocate{}(1264) BEGIN
 // (op 1264) Deallocate 547
 // Freeing Offset547: 1 VGPR (Value: UInt32) (op 17): v55

 // Deleting tag 547
 // (op 1264) Deallocate 549
 // Deleting tag 549
 // (op 60) Deallocate{}(1264) END
// Allocated : 1 VGPR (Value: E8M0) (op 253): v14
ds_read_u8 v14, v5 offset:5312 // (op 253) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3535 (current 3440)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 5312, Size: 64)
 // (op 252) LoadLDSTile{Value: E8M0}(253) END
 // (op 252) Deallocate{}(1244) BEGIN
 // (op 1244) Deallocate 672
 // Deleting tag 672
 // (op 252) Deallocate{}(1244) END
 // (op 252) end: generate({253})
 // (op 60) SetCoordinate(252) END
 // (op 60) Assign VGPR Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32(618) BEGIN
 // (op 618) Assign dim(545) = Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32
 // (op 618) Generate Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32 into DataFlowTag545: VGPR Value: UInt32 x 1: v15
 // (op 618) reg expression
 // (op 618) Add({DataFlowTag545: v15:U32}, 4:U32)U32
 // (op 60) Barrier(1111) BEGIN
 // (op 60) Deallocate{}(1240) BEGIN
 // (op 1240) Deallocate 480
 // Deleting tag 480
 // (op 60) Deallocate{}(1240) END
s_barrier  // (op 1111) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 60) Barrier(1111) END
 // (op 60) Barrier(1168) BEGIN
 // (op 60) Barrier(1242) BEGIN
 // (op 60) StoreLDSTile Value: FP6x16(13) BEGIN
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
 // 	info.data = VGPR Value: FP6x16 x 2: v[22:27]
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

 // (op 13) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[22:27]
 // 	info.rowOffsetReg = DataFlowTag592: VGPR Value: UInt32 x 1: v42
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
 // (op 13) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1168) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4224, Size: 64)
 // (op 60) Barrier(1168) END
s_barrier  // (op 1242) 
 // (op 60) Barrier(1242) END
 // (op 60) SetCoordinate(254) BEGIN
 // (op 254) generate({})
 // (op 254) end: generate({})
 // (op 254) generate({255})
 // (op 254) LoadLDSTile{Value: E8M0}(255) BEGIN
 // (op 255) GEN: loadMacroTileWAVELDS OP 255 LDS 484 WaveTile 297
 // (op 255) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 255) Generate 1:U32 into nullptr
 // tag 492: v**UNALLOCATED**
 // (op 255) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 4224
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
 // 	info.offset = Literal Value: UInt32 x 0: 4224
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

v_add_u32 v15, 4, v15 // (op 618) 
 // (op 60) Assign VGPR Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32(618) END
// Allocated : 1 VGPR (Value: E8M0) (op 255): v47
ds_read_u8 v47, v10 offset:4224 // (op 255) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3540 (current 3445)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4224, Size: 64)
 // (op 254) LoadLDSTile{Value: E8M0}(255) END
 // (op 254) Deallocate{}(1250) BEGIN
 // (op 1250) Deallocate 680
 // Deleting tag 680
 // (op 254) Deallocate{}(1250) END
 // (op 254) end: generate({255})
 // (op 60) SetCoordinate(254) END
s_waitcnt vmcnt(9)
ds_write_b128 v42, v[22:25] // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3542 (current 3447)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 60) Deallocate{}(1246) BEGIN
 // (op 1246) Deallocate 484
 // Deleting tag 484
 // (op 60) Deallocate{}(1246) END
s_waitcnt vmcnt(8)
ds_write_b64 v42, v[26:27] offset:16 // (op 13) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected stall of 88. CBNW: 3, Inc: 3
// DSMEM: Expected complete at 3632 (current 3537)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 60) StoreLDSTile Value: FP6x16(13) END
 // (op 60) Multiply(256) BEGIN
 // (op 256) Generate ScaledMatrixMultiply(WaveTile8xFP4, WaveTile16xFP6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 60) Barrier(1248) BEGIN
 // (op 60) Deallocate{}(1176) BEGIN
 // (op 1176) Deallocate 594
 // Deleting tag 594
 // (op 60) Deallocate{}(1176) END
s_barrier  // (op 1248) 
 // (op 60) Barrier(1248) END
 // (op 60) Barrier(1139) BEGIN
s_barrier  // (op 1139) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 60) Barrier(1139) END
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[50:53], v[56:61], a[0:15], v14, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 256) 
 // (op 60) Multiply(256) END
 // (op 60) StoreLDSTile Value: E8M0(30) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v18
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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
 // 	info.data = VGPR Value: E8M0 x 1: v18
 // 	info.rowOffsetReg = DataFlowTag624: VGPR Value: UInt32 x 1: v45
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 60) Deallocate{}(1252) BEGIN
 // (op 1252) Deallocate 490
 // Freeing : 1 VGPR (Value: E8M0) (op 253): v14

 // Deleting tag 490
 // (op 1252) Deallocate 492
 // Freeing : 1 VGPR (Value: E8M0) (op 255): v47

 // Deleting tag 492
 // (op 1252) Deallocate 488
 // Freeing : 6 VGPRs (Value: FP6x16) (op 251): v56, v57, v58, v59, v60, v61

 // Deleting tag 488
 // (op 1252) Deallocate 486
 // Freeing : 4 VGPRs (Value: FP4x8) (op 249): v50, v51, v52, v53

 // Deleting tag 486
 // (op 60) Deallocate{}(1252) END
s_waitcnt vmcnt(7)
ds_write_b8 v45, v18 offset:1536 // (op 30) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3639 (current 3544)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(30) END
 // (op 60) Deallocate{}(1180) BEGIN
 // (op 1180) Deallocate 626
 // Deleting tag 626
 // (op 60) Deallocate{}(1180) END
 // (op 60) Barrier(1098) BEGIN
s_barrier  // (op 1098) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 60) Barrier(1098) END
 // (op 60) StoreLDSTile Value: FP4x8(6) BEGIN
 // (op 6) GEN: StoreLDSTile
 // (op 6) GEN: storeMacroTileLDS OP 6 LDS 17 MacroTile 8
 // (op 6) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 6) Generate 0:U32 into nullptr
 // (op 6) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[28:31]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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

 // (op 6) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[28:31]
 // 	info.rowOffsetReg = DataFlowTag576: VGPR Value: UInt32 x 1: v20
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 6) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b128 v20, v[28:31] offset:1600 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 3642 (current 3547)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 60) StoreLDSTile Value: FP4x8(6) END
 // (op 60) Deallocate{}(1174) BEGIN
 // (op 1174) Deallocate 578
 // Deleting tag 578
 // (op 60) Deallocate{}(1174) END
 // (op 60) Barrier(1125) BEGIN
s_barrier  // (op 1125) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 60) Barrier(1125) END
 // (op 60) StoreLDSTile Value: E8M0(21) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag608: VGPR Value: UInt32 x 1: v44
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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

s_waitcnt vmcnt(5)
ds_write_b8 v44, v19 offset:2624 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 84. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 3729 (current 3634)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(21) END
 // (op 60) Deallocate{}(1178) BEGIN
 // (op 1178) Deallocate 610
 // Deleting tag 610
 // (op 60) Deallocate{}(1178) END
 // (op 60) NOP(411) BEGIN
 // (op 411) generate({})
 // (op 411) end: generate({})
 // (op 60) NOP(411) END
 // (op 60) end: generate({62, 292, 1095})
 // (op 60) For Loop Increment
 // (op 60) generate({})
 // (op 60) end: generate({})
 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
 // (op 60) Generate LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 60) LessThan({DataFlowTag397: s0:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 60) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 60) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 60): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 60) 
// Allocated : 1 SGPR (Value: UInt32) (op 60): s1
s_mov_b32 s1, s14 // (op 60) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 60): s14, s15

// Allocated : 1 SGPR (Value: UInt32) (op 60): s14
s_and_b32 s14, s1, 4294967294 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s1

s_cmp_lt_u32 s0, s14 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s14

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 60) 
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60:
 // (op 60) 
 // (op 60) Unlock For Loop
 // Unlock None
 // (op 471) ForLoopOp KLoop: LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(60) END
 // (op 471) Deallocate{}(1186) BEGIN
 // (op 1186) Deallocate 512
 // Deleting tag 512
 // (op 1186) Deallocate 650
 // Freeing DataFlowTag650: 1 VGPR (Value: UInt32) (op 974): v43

 // Deleting tag 650
 // (op 1186) Deallocate 651
 // Deleting tag 651
 // (op 1186) Deallocate 526
 // Freeing DataFlowTag526: 1 VGPR (Value: UInt32) (op 562): v17

 // Deleting tag 526
 // (op 1186) Deallocate 653
 // Deleting tag 653
 // (op 1186) Deallocate 661
 // Deleting tag 661
 // (op 1186) Deallocate 666
 // Freeing DataFlowTag666: 1 VGPR (Value: UInt32) (op 1022): v48

 // Deleting tag 666
 // (op 1186) Deallocate 667
 // Deleting tag 667
 // (op 1186) Deallocate 669
 // Deleting tag 669
 // (op 1186) Deallocate 674
 // Freeing DataFlowTag674: 1 VGPR (Value: UInt32) (op 1046): v49

 // Deleting tag 674
 // (op 1186) Deallocate 675
 // Deleting tag 675
 // (op 1186) Deallocate 658
 // Freeing DataFlowTag658: 1 VGPR (Value: UInt32) (op 998): v46

 // Deleting tag 658
 // (op 1186) Deallocate 677
 // Deleting tag 677
 // (op 1186) Deallocate 659
 // Deleting tag 659
 // (op 1186) Deallocate 510
 // Deleting tag 510
 // (op 1186) Deallocate 508
 // Deleting tag 508
 // (op 1186) Deallocate 507
 // Freeing DataFlowTag507: 1 VGPR (Value: UInt32) (op 498): v13

 // Deleting tag 507
 // (op 1186) Deallocate 624
 // Freeing DataFlowTag624: 1 VGPR (Value: UInt32) (op 887): v45

 // Deleting tag 624
 // (op 1186) Deallocate 397
 // Freeing DataFlowTag397: 1 SGPR (Value: UInt32) (op 61): s0

 // Deleting tag 397
 // (op 1186) Deallocate 546
 // Deleting tag 546
 // (op 1186) Deallocate 609
 // Deleting tag 609
 // (op 1186) Deallocate 470
 // Deleting tag 470
 // (op 1186) Deallocate 482
 // Deleting tag 482
 // (op 1186) Deallocate 576
 // Freeing DataFlowTag576: 1 VGPR (Value: UInt32) (op 728): v20

 // Deleting tag 576
 // (op 1186) Deallocate 577
 // Deleting tag 577
 // (op 1186) Deallocate 579
 // Deleting tag 579
 // (op 1186) Deallocate 592
 // Freeing DataFlowTag592: 1 VGPR (Value: UInt32) (op 781): v42

 // Deleting tag 592
 // (op 1186) Deallocate 593
 // Deleting tag 593
 // (op 1186) Deallocate 595
 // Deleting tag 595
 // (op 1186) Deallocate 608
 // Freeing DataFlowTag608: 1 VGPR (Value: UInt32) (op 834): v44

 // Deleting tag 608
 // (op 1186) Deallocate 611
 // Deleting tag 611
 // (op 1186) Deallocate 625
 // Deleting tag 625
 // (op 1186) Deallocate 627
 // Deleting tag 627
 // (op 1186) Deallocate 474
 // Deleting tag 474
 // (op 1186) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 379): v19

 // Deleting tag 10
 // (op 1186) Deallocate 11
 // Freeing : 1 VGPR (Value: E8M0) (op 373): v18

 // Deleting tag 11
 // (op 1186) Deallocate 9
 // Freeing : 6 VGPRs (Value: FP6x16) (op 370): v22, v23, v24, v25, v26, v27

 // Deleting tag 9
 // (op 1186) Deallocate 569
 // Deleting tag 569
 // (op 1186) Deallocate 567
 // Deleting tag 567
 // (op 1186) Deallocate 565
 // Deleting tag 565
 // (op 1186) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 690): v21

 // Deleting tag 564
 // (op 1186) Deallocate 527
 // Deleting tag 527
 // (op 1186) Deallocate 548
 // Deleting tag 548
 // (op 1186) Deallocate 545
 // Freeing DataFlowTag545: 1 VGPR (Value: UInt32) (op 626): v15

 // Deleting tag 545
 // (op 1186) Deallocate 8
 // Freeing : 4 VGPRs (Value: FP4x8) (op 376): v28, v29, v30, v31

 // Deleting tag 8
 // (op 1186) Deallocate 529
 // Deleting tag 529
 // (op 1186) Deallocate 478
 // Deleting tag 478
 // (op 1186) Deallocate 531
 // Deleting tag 531
 // (op 1186) Deallocate 550
 // Deleting tag 550
 // (op 471) Deallocate{}(1186) END
 // (op 471) end: generate({466})
 // (op 471) Unlock Scope 471
 // Unlock None
 // (op 344) Scope(471) END
 // (op 344) SetCoordinate(221) BEGIN
 // (op 221) generate({})
 // (op 221) end: generate({})
 // (op 221) generate({222})
 // (op 221) SetCoordinate(222) BEGIN
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({448})
 // (op 222) Scope(448) BEGIN
 // (op 448) Lock Scope 448
 // Lock Branch
 // (op 448) generate({442})
 // (op 448) NOP(442) BEGIN
 // (op 442) generate({})
 // (op 442) end: generate({})
 // (op 448) NOP(442) END
 // (op 448) Assign VGPR Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32(453) BEGIN
 // (op 453) Assign dim(494) = Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32
 // tag 494: v**UNALLOCATED**
 // (op 453) Generate Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32 into DataFlowTag494: VGPR Value: UInt32 x 1: (unallocated)
 // (op 453) reg expression
 // (op 453) reg expression
 // (op 453) reg expression
 // (op 453) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 453) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 453): v13
v_lshrrev_b32 v13, 1, v1 // (op 453) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 453): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 453) 
 // (op 453) BEGIN: Tile
 // (op 453) convert
// Allocated : 1 VGPR (Value: UInt32) (op 453): v14
v_lshl_add_u32 v14, s16, 5, v13 // (op 453) 
 // (op 453) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 453): v13

 // (op 453) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 453) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 453): s14
s_lshr_b32 s14, s0, 1 // (op 453) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 453): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 453): v13
v_and_b32 v13, -2, v1 // (op 453) 
 // (op 453) reg expression
 // (op 453) BEGIN: Flatten[1]
 // (op 453) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 453): v15
v_sub_u32 v15, v1, v13 // (op 453) 
 // (op 453) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 453): v13

 // (op 453) BEGIN: Tile
 // (op 453) {Tile: ShiftL({Flatten[1]: v15:U32}, 5:U32)U32}
 // (op 453) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 453)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 453)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 453): v13
v_lshlrev_b32 v13, 5, v15 // (op 453) 
 // (op 453) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 453): v15

 // (op 453) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 453): v15
v_lshl_add_u32 v15, s14, 7, v13 // (op 453) 
 // (op 453) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 453): v13

 // Freeing : 1 SGPR (Value: UInt32) (op 453): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 453): v30, v31
v_mov_b32 v31, 0 // (op 453) convert
v_mov_b32 v30, v15 // (op 453) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 453): v15

 // (op 453) BEGIN: Split
 // (op 453) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 453)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 453)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 453): v28, v29
v_mul_lo_u32 v29, v14, s3 // (op 453) most significant: low of low * high
 // (op 453) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 453): v13
v_mul_hi_u32 v13, v14, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v28, v14, s2 // (op 453) least significant: low of low * low
v_add_u32 v29, v29, v13 // (op 453) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 453): v13

 // (op 453) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 453)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 453)      rhs (Tile: VGPR Value: UInt64 x 1: v[30:31])
// Allocated : 2 SGPRs (Value: Bool64) (op 453): s0, s1
v_add_co_u32 v28, s[0:1], v28, v30 // (op 453) least significant half
v_addc_co_u32 v29, s[0:1], v29, v31, s[0:1] // (op 453) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 453): s0, s1

 // (op 453) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 453): v30, v31

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 453): v14

 // (op 453) ShiftL({Split: v[28:29]:U64}, 2:U32)U64
 // (op 453) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 453)         value (Split: VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 453)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 VGPRs (Value: UInt64) (op 453): v14, v15
v_lshlrev_b64 v[14:15], 2, v[28:29] // (op 453) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 453): v28, v29

 // (op 453) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 453): v30, v31
v_lshrrev_b64 v[30:31], 3, v[14:15] // (op 453) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 453): v14, v15

// Allocated DataFlowTag494: 1 VGPR (Value: UInt32) (op 453): v13
v_mov_b32 v13, v30 // (op 453) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 453): v30, v31

 // (op 448) Assign VGPR Convert(LogicalShiftR(ShiftL({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 1:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64)U64}, 2:U32)U64, 3:U32)U64)U32(453) END
 // (op 448) Assign VGPR 32:U64(454) BEGIN
 // (op 454) Assign dim(495) = 32:U64
 // (op 454) Assign stride expression
 // (op 448) Assign VGPR 32:U64(454) END
 // (op 448) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(455) BEGIN
 // (op 455) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 455) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // (op 455) reg expression
 // (op 455) reg expression
 // (op 455) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 455) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 455)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 455)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 455): s0, s1
s_mul_i32 s1, s2, s7 // (op 455) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 455): s14
s_mul_i32 s14, s3, s6 // (op 455) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 455) least significant
// Allocated : 1 SGPR (Value: Int32) (op 455): s15
s_mul_hi_u32 s15, s2, s6 // (op 455) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 455) most significant: sum
s_add_u32 s1, s1, s15 // (op 455) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 455): s15

 // Freeing : 1 SGPR (Value: Int32) (op 455): s14

 // (op 455) ShiftL(s[0:1]:I64, 2:U32)I64
 // (op 455) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 455)         value (SGPR Value: Int64 x 1: s[0:1]) 
 // (op 455)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 455): s14, s15
s_lshl_b64 s[14:15], s[0:1], 2 // (op 455) 
 // Freeing : 2 SGPRs (Value: Int64) (op 455): s0, s1

 // (op 455) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 455): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 455) 
 // Freeing : 2 SGPRs (Value: Int64) (op 455): s14, s15

 // (op 455) reg expression
 // (op 455) reg expression
s_mov_b32 s40, s4 // (op 455) reg expression
s_mov_b32 s41, s5 // (op 455) reg expression
s_mov_b32 s42, s0 // (op 455) reg expression
s_mov_b32 s43, 131072 // (op 455) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 455): s0, s1

 // (op 448) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(455) END
 // (op 448) NOP(444) BEGIN
 // (op 444) generate({})
 // (op 444) end: generate({})
 // (op 448) NOP(444) END
 // (op 448) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64(460) BEGIN
 // (op 460) Assign dim(499) = LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64
 // (op 460) Assign stride expression
 // (op 448) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_0_stride_0_0:I64)U64, 2:U32)U64, 3:U32)U64(460) END
 // (op 448) NOP(445) BEGIN
 // (op 445) generate({})
 // (op 445) end: generate({})
 // (op 448) NOP(445) END
 // (op 448) Assign VGPR 0:U64(463) BEGIN
 // (op 463) Assign dim(501) = 0:U64
 // (op 463) Assign stride expression
 // (op 448) Assign VGPR 0:U64(463) END
 // (op 448) NOP(511) BEGIN
 // (op 511) generate({})
 // (op 511) end: generate({})
 // (op 448) NOP(511) END
 // (op 448) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(520) BEGIN
 // (op 520) Assign dim(513) = Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 513: v**UNALLOCATED**
 // (op 520) Generate Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag513: VGPR Value: UInt32 x 1: (unallocated)
 // (op 520) reg expression
 // (op 520) reg expression
 // (op 520) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 520) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 520): v14
v_and_b32 v14, -2, v1 // (op 520) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 520): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 520) 
 // (op 520) BEGIN: Flatten[1]
 // (op 520) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
 // (op 520) convert
// Allocated : 1 VGPR (Value: UInt32) (op 520): v15
v_sub_u32 v15, v1, v14 // (op 520) 
 // (op 520) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 520): v14

 // (op 520) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 520) BEGIN: Tile
 // (op 520) {Tile: ShiftL({Flatten[1]: v15:U32}, 5:U32)U32}
 // (op 520) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 520)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 520)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 SGPR (Value: UInt32) (op 520): s14
s_lshr_b32 s14, s0, 1 // (op 520) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 520): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 520): v14
v_lshlrev_b32 v14, 5, v15 // (op 520) 
 // (op 520) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 520): v15

 // (op 520) reg expression
 // (op 520) BEGIN: Tile
 // (op 520) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 520): v15
v_lshl_add_u32 v15, s14, 7, v14 // (op 520) 
 // (op 520) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 520): v14

 // Freeing : 1 SGPR (Value: UInt32) (op 520): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 520): v30, v31
v_mov_b32 v31, 0 // (op 520) convert
v_mov_b32 v30, v15 // (op 520) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 520): v15

 // (op 520) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 520): v14
v_lshrrev_b32 v14, 1, v1 // (op 520) 
 // (op 520) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 520): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 520) 
 // (op 520) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 520): v14

 // (op 520) Multiply({Tile: v15:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 520) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 520)           lhs (Tile: VGPR Value: UInt32 x 1: v15) 
 // (op 520)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 520): v28, v29
v_mul_lo_u32 v29, v15, s9 // (op 520) most significant: low of low * high
 // (op 520) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 520): v14
v_mul_hi_u32 v14, v15, s8 // (op 520) most significant: high of low * low
v_mul_lo_u32 v28, v15, s8 // (op 520) least significant: low of low * low
v_add_u32 v29, v29, v14 // (op 520) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 520): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 520): v15

 // (op 520) BEGIN: Split
 // (op 520) {Split: Add({Tile: v[30:31]:U64}, v[28:29]:I64)U64}
 // (op 520) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 520)      lhs (Tile: VGPR Value: UInt64 x 1: v[30:31]) 
 // (op 520)      rhs (VGPR Value: Int64 x 1: v[28:29])
// Allocated : 2 VGPRs (Value: UInt64) (op 520): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 520): s0, s1
v_add_co_u32 v14, s[0:1], v30, v28 // (op 520) least significant half
v_addc_co_u32 v15, s[0:1], v31, v29, s[0:1] // (op 520) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 520): s0, s1

 // (op 520) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 520): v28, v29

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 520): v30, v31

 // (op 520) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 520) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 520)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 520)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 520) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 520): v30, v31
v_mul_lo_u32 v31, v15, 6 // (op 520) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 520): v17
v_mul_hi_u32 v17, v14, 6 // (op 520) most significant: high of low * low
v_mul_lo_u32 v30, v14, 6 // (op 520) least significant: low of low * low
v_add_u32 v31, v31, v17 // (op 520) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 520): v17

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 520): v14, v15

 // (op 520) LogicalShiftR(v[30:31]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 520): v14, v15
v_lshrrev_b64 v[14:15], 3, v[30:31] // (op 520) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 520): v30, v31

// Allocated DataFlowTag513: 1 VGPR (Value: UInt32) (op 520): v17
v_mov_b32 v17, v14 // (op 520) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 520): v14, v15

 // (op 448) Assign VGPR Convert(LogicalShiftR(Multiply({Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}, 6:U32)U64, 3:U32)U64)U32(520) END
 // (op 448) Assign VGPR 48:U64(521) BEGIN
 // (op 521) Assign dim(514) = 48:U64
 // (op 521) Assign stride expression
 // (op 448) Assign VGPR 48:U64(521) END
 // (op 448) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(522) BEGIN
 // (op 522) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 522) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // (op 522) reg expression
 // (op 522) reg expression
 // (op 522) Multiply({Tensor_2_size_1_4: s[12:13]:I64}, 6:U32)I64
 // (op 522) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 522)           lhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 522)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 522) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 522): s0, s1
s_mul_i32 s1, s13, 6 // (op 522) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 522) least significant
// Allocated : 1 SGPR (Value: Int32) (op 522): s14
s_mul_hi_u32 s14, s12, 6 // (op 522) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 522) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 522): s14

 // (op 522) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, s[0:1]:I64)I64
 // (op 522) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 522)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 522)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 522): s14, s15
s_mul_i32 s15, s8, s1 // (op 522) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 522): s18
s_mul_i32 s18, s9, s0 // (op 522) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 522) least significant
// Allocated : 1 SGPR (Value: Int32) (op 522): s19
s_mul_hi_u32 s19, s8, s0 // (op 522) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 522) most significant: sum
s_add_u32 s15, s15, s19 // (op 522) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 522): s19

 // Freeing : 1 SGPR (Value: Int32) (op 522): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 522): s0, s1

 // (op 522) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 522): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 522) 
 // Freeing : 2 SGPRs (Value: Int64) (op 522): s14, s15

 // (op 522) reg expression
 // (op 522) reg expression
s_mov_b32 s44, s10 // (op 522) reg expression
s_mov_b32 s45, s11 // (op 522) reg expression
s_mov_b32 s46, s0 // (op 522) reg expression
s_mov_b32 s47, 131072 // (op 522) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 522): s0, s1

 // (op 448) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_2_stride_1_3:I64, Multiply(Tensor_2_size_1_4:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(522) END
 // (op 448) NOP(513) BEGIN
 // (op 513) generate({})
 // (op 513) end: generate({})
 // (op 448) NOP(513) END
 // (op 448) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(527) BEGIN
 // (op 527) Assign dim(518) = LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 527) Assign stride expression
 // (op 448) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_2_stride_1_3:I64)U64, 6:U32)U64, 3:U32)U64(527) END
 // (op 448) NOP(514) BEGIN
 // (op 514) generate({})
 // (op 514) end: generate({})
 // (op 448) NOP(514) END
 // (op 448) Assign VGPR 0:U64(530) BEGIN
 // (op 530) Assign dim(520) = 0:U64
 // (op 530) Assign stride expression
 // (op 448) Assign VGPR 0:U64(530) END
 // (op 448) NOP(575) BEGIN
 // (op 575) generate({})
 // (op 575) end: generate({})
 // (op 448) NOP(575) END
 // (op 448) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}(584) BEGIN
 // (op 584) Assign dim(532) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}
 // tag 532: v**UNALLOCATED**
 // (op 584) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32} into DataFlowTag532: VGPR Value: UInt32 x 1: (unallocated)
 // (op 584) reg expression
 // (op 584) reg expression
 // (op 584) reg expression
 // (op 584) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 584) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 584): v14
v_lshrrev_b32 v14, 1, v1 // (op 584) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 584): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 584) 
 // (op 584) BEGIN: Tile
 // (op 584) convert
// Allocated : 1 VGPR (Value: UInt32) (op 584): v15
v_lshl_add_u32 v15, s16, 5, v14 // (op 584) 
 // (op 584) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 584): v14

 // (op 584) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 584) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 584): s14
s_lshr_b32 s14, s0, 1 // (op 584) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 584): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 584): v14
v_and_b32 v14, -2, v1 // (op 584) 
 // (op 584) reg expression
 // (op 584) BEGIN: Flatten[1]Tile
 // (op 584) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
 // (op 584) convert
// Allocated : 1 VGPR (Value: UInt32) (op 584): v18
v_sub_u32 v18, v1, v14 // (op 584) 
 // (op 584) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 584): v14

 // (op 584) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 584): v14
v_lshl_add_u32 v14, s14, 2, v18 // (op 584) 
 // (op 584) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 584): v18

 // Freeing : 1 SGPR (Value: UInt32) (op 584): s14

 // (op 584) BEGIN: Split
// Allocated DataFlowTag532: 1 VGPR (Value: UInt32) (op 584): v18
v_mul_lo_u32 v18, v15, s20 // (op 584) 
v_add_u32 v18, v18, v14 // (op 584) 
 // (op 584) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 584): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 584): v15

 // (op 448) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}(584) END
 // (op 448) Assign VGPR 2:U64(585) BEGIN
 // (op 585) Assign dim(533) = 2:U64
 // (op 585) Assign stride expression
 // (op 448) Assign VGPR 2:U64(585) END
 // (op 448) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(586) BEGIN
 // (op 586) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 586) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // (op 586) reg expression
 // (op 586) reg expression
 // (op 586) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 586) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 586)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 586)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 586): s0, s1
s_mul_i32 s1, s20, s7 // (op 586) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 586): s14
s_mul_i32 s14, s21, s6 // (op 586) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 586) least significant
// Allocated : 1 SGPR (Value: Int32) (op 586): s15
s_mul_hi_u32 s15, s20, s6 // (op 586) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 586) most significant: sum
s_add_u32 s1, s1, s15 // (op 586) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 586): s15

 // Freeing : 1 SGPR (Value: Int32) (op 586): s14

 // (op 586) reg expression
 // (op 586) reg expression
s_mov_b32 s48, s22 // (op 586) reg expression
s_mov_b32 s49, s23 // (op 586) reg expression
s_mov_b32 s50, s0 // (op 586) reg expression
s_mov_b32 s51, 131072 // (op 586) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 586): s0, s1

 // (op 448) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(586) END
 // (op 448) NOP(577) BEGIN
 // (op 577) generate({})
 // (op 577) end: generate({})
 // (op 448) NOP(577) END
 // (op 448) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(591) BEGIN
 // (op 591) Assign dim(537) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 591) Assign stride expression
 // (op 448) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(591) END
 // (op 448) NOP(578) BEGIN
 // (op 578) generate({})
 // (op 578) end: generate({})
 // (op 448) NOP(578) END
 // (op 448) Assign VGPR 1:U64(594) BEGIN
 // (op 594) Assign dim(539) = 1:U64
 // (op 594) Assign stride expression
 // (op 448) Assign VGPR 1:U64(594) END
 // (op 448) NOP(639) BEGIN
 // (op 639) generate({})
 // (op 639) end: generate({})
 // (op 448) NOP(639) END
 // (op 448) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(648) BEGIN
 // (op 648) Assign dim(551) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 551: v**UNALLOCATED**
 // (op 648) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag551: VGPR Value: UInt32 x 1: (unallocated)
 // (op 648) reg expression
 // (op 648) reg expression
 // (op 648) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 648) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 648): v14
v_and_b32 v14, -2, v1 // (op 648) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 648): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 648) 
 // (op 648) BEGIN: Flatten[1]Tile
 // (op 648) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
 // (op 648) convert
// Allocated : 1 VGPR (Value: UInt32) (op 648): v15
v_sub_u32 v15, v1, v14 // (op 648) 
 // (op 648) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 648): v14

 // (op 648) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 648) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 648): s14
s_lshr_b32 s14, s0, 1 // (op 648) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 648): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 648): v14
v_lshrrev_b32 v14, 1, v1 // (op 648) 
 // (op 648) reg expression
 // (op 648) BEGIN: Tile
 // (op 648) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 648): v19
v_lshl_add_u32 v19, s14, 2, v15 // (op 648) 
 // (op 648) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 648): v15

 // Freeing : 1 SGPR (Value: UInt32) (op 648): s14

 // (op 648) convert
 // (op 648) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 648): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 648) 
 // (op 648) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 648): v14

 // (op 648) Multiply({Tile: v15:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 648): v14
v_mul_lo_u32 v14, v15, s24 // (op 648) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 648): v15

 // (op 648) BEGIN: Split
 // (op 648) {Split: Add({Tile: v19:U32}, v14:U32)U32}
// Allocated DataFlowTag551: 1 VGPR (Value: UInt32) (op 648): v15
v_add_u32 v15, v19, v14 // (op 648) 
 // (op 648) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 648): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 648): v19

 // (op 448) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(648) END
 // (op 448) Assign VGPR 2:U64(649) BEGIN
 // (op 649) Assign dim(552) = 2:U64
 // (op 649) Assign stride expression
 // (op 448) Assign VGPR 2:U64(649) END
 // (op 448) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(650) BEGIN
 // (op 650) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 650) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // (op 650) reg expression
 // (op 650) reg expression
 // (op 650) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 650) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 650)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 650)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 650): s0, s1
s_mul_i32 s1, s24, s13 // (op 650) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 650): s14
s_mul_i32 s14, s25, s12 // (op 650) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 650) least significant
// Allocated : 1 SGPR (Value: Int32) (op 650): s15
s_mul_hi_u32 s15, s24, s12 // (op 650) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 650) most significant: sum
s_add_u32 s1, s1, s15 // (op 650) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 650): s15

 // Freeing : 1 SGPR (Value: Int32) (op 650): s14

 // (op 650) reg expression
 // (op 650) reg expression
s_mov_b32 s52, s26 // (op 650) reg expression
s_mov_b32 s53, s27 // (op 650) reg expression
s_mov_b32 s54, s0 // (op 650) reg expression
s_mov_b32 s55, 131072 // (op 650) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 650): s0, s1

 // (op 448) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(650) END
 // (op 448) NOP(641) BEGIN
 // (op 641) generate({})
 // (op 641) end: generate({})
 // (op 448) NOP(641) END
 // (op 448) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(655) BEGIN
 // (op 655) Assign dim(556) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 655) Assign stride expression
 // (op 448) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(655) END
 // (op 448) NOP(642) BEGIN
 // (op 642) generate({})
 // (op 642) end: generate({})
 // (op 448) NOP(642) END
 // (op 448) Assign VGPR 1:U64(658) BEGIN
 // (op 658) Assign dim(558) = 1:U64
 // (op 658) Assign stride expression
 // (op 448) Assign VGPR 1:U64(658) END
 // (op 448) NOP(736) BEGIN
 // (op 736) generate({})
 // (op 736) end: generate({})
 // (op 448) NOP(736) END
 // (op 448) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(741) BEGIN
 // (op 741) Assign dim(580) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 580: v**UNALLOCATED**
 // (op 741) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag580: VGPR Value: UInt32 x 1: (unallocated)
 // (op 741) reg expression
 // (op 741) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 741): v14
v_and_b32 v14, -2, v1 // (op 741) 
 // (op 741) BEGIN: Tile[1]
 // (op 741) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 741): v19
v_sub_u32 v19, v1, v14 // (op 741) 
 // (op 741) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 741): v14

 // (op 741) BEGIN: Flatten
 // (op 741) {Flatten: ShiftL({Tile[1]: v19:U32}, 5:U32)U32}
 // (op 741) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 741)         value (Tile[1]: VGPR Value: UInt32 x 1: v19) 
 // (op 741)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 741): v14
v_lshlrev_b32 v14, 5, v19 // (op 741) 
 // (op 741) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 741): v19

 // (op 741) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 741): v19
v_lshrrev_b32 v19, 1, v1 // (op 741) 
 // (op 741) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 741): v20
v_lshl_add_u32 v20, v19, 6, v14 // (op 741) 
 // (op 741) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 741): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 741): v19

 // (op 741) ShiftL({Flatten: v20:U32}, 2:U32)U32
 // (op 741) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 741)         value (Flatten: VGPR Value: UInt32 x 1: v20) 
 // (op 741)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 741): v14
v_lshlrev_b32 v14, 2, v20 // (op 741) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 741): v20

 // (op 741) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag580: 1 VGPR (Value: UInt32) (op 741): v19
v_lshrrev_b32 v19, 3, v14 // (op 741) 
 // Freeing : 1 VGPR (Value: UInt32) (op 741): v14

 // (op 448) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(741) END
 // (op 448) Assign VGPR 32:U32(742) BEGIN
 // (op 742) Assign dim(581) = 32:U32
 // (op 742) Assign stride expression
 // (op 448) Assign VGPR 32:U32(742) END
 // (op 448) NOP(737) BEGIN
 // (op 737) generate({})
 // (op 737) end: generate({})
 // (op 448) NOP(737) END
 // (op 448) Assign VGPR 0:U32(746) BEGIN
 // (op 746) Assign dim(583) = 0:U32
 // (op 746) Assign stride expression
 // (op 448) Assign VGPR 0:U32(746) END
 // (op 448) NOP(789) BEGIN
 // (op 789) generate({})
 // (op 789) end: generate({})
 // (op 448) NOP(789) END
 // (op 448) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(794) BEGIN
 // (op 794) Assign dim(596) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 596: v**UNALLOCATED**
 // (op 794) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag596: VGPR Value: UInt32 x 1: (unallocated)
 // (op 794) reg expression
 // (op 794) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 794): v14
v_and_b32 v14, -2, v1 // (op 794) 
 // (op 794) BEGIN: Tile[1]
 // (op 794) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 794): v20
v_sub_u32 v20, v1, v14 // (op 794) 
 // (op 794) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 794): v14

 // (op 794) BEGIN: Flatten
 // (op 794) {Flatten: ShiftL({Tile[1]: v20:U32}, 5:U32)U32}
 // (op 794) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 794)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 794)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 794): v14
v_lshlrev_b32 v14, 5, v20 // (op 794) 
 // (op 794) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 794): v20

 // (op 794) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 794): v20
v_lshrrev_b32 v20, 1, v1 // (op 794) 
 // (op 794) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 794): v21
v_lshl_add_u32 v21, v20, 6, v14 // (op 794) 
 // (op 794) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 794): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 794): v20

 // (op 794) Multiply({Flatten: v21:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 794): v14
v_mul_lo_u32 v14, v21, 6 // (op 794) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 794): v21

 // (op 794) LogicalShiftR(v14:U32, 3:U32)U32
// Allocated DataFlowTag596: 1 VGPR (Value: UInt32) (op 794): v20
v_lshrrev_b32 v20, 3, v14 // (op 794) 
 // Freeing : 1 VGPR (Value: UInt32) (op 794): v14

 // (op 448) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(794) END
 // (op 448) Assign VGPR 48:U32(795) BEGIN
 // (op 795) Assign dim(597) = 48:U32
 // (op 795) Assign stride expression
 // (op 448) Assign VGPR 48:U32(795) END
 // (op 448) NOP(790) BEGIN
 // (op 790) generate({})
 // (op 790) end: generate({})
 // (op 448) NOP(790) END
 // (op 448) Assign VGPR 0:U32(799) BEGIN
 // (op 799) Assign dim(599) = 0:U32
 // (op 799) Assign stride expression
 // (op 448) Assign VGPR 0:U32(799) END
 // (op 448) NOP(842) BEGIN
 // (op 842) generate({})
 // (op 842) end: generate({})
 // (op 448) NOP(842) END
 // (op 448) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(847) BEGIN
 // (op 847) Assign dim(612) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 612: v**UNALLOCATED**
 // (op 847) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag612: VGPR Value: UInt32 x 1: (unallocated)
 // (op 847) reg expression
 // (op 847) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 847): v14
v_and_b32 v14, -2, v1 // (op 847) 
 // (op 847) BEGIN: Tile[1]Flatten
 // (op 847) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 847): v21
v_sub_u32 v21, v1, v14 // (op 847) 
 // (op 847) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 847): v14

 // (op 847) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 847): v14
v_and_b32 v14, 4294967294, v1 // (op 847) 
 // (op 847) BEGIN: Flatten
 // (op 847) {Flatten: Add(v14:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag612: 1 VGPR (Value: UInt32) (op 847): v22
v_add_u32 v22, v14, v21 // (op 847) 
 // (op 847) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 847): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 847): v14

 // (op 448) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(847) END
 // (op 448) Assign VGPR 2:U32(848) BEGIN
 // (op 848) Assign dim(613) = 2:U32
 // (op 848) Assign stride expression
 // (op 448) Assign VGPR 2:U32(848) END
 // (op 448) NOP(843) BEGIN
 // (op 843) generate({})
 // (op 843) end: generate({})
 // (op 448) NOP(843) END
 // (op 448) Assign VGPR 1:U32(852) BEGIN
 // (op 852) Assign dim(615) = 1:U32
 // (op 852) Assign stride expression
 // (op 448) Assign VGPR 1:U32(852) END
 // (op 448) NOP(895) BEGIN
 // (op 895) generate({})
 // (op 895) end: generate({})
 // (op 448) NOP(895) END
 // (op 448) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(900) BEGIN
 // (op 900) Assign dim(628) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 628: v**UNALLOCATED**
 // (op 900) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag628: VGPR Value: UInt32 x 1: (unallocated)
 // (op 900) reg expression
 // (op 900) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 900): v14
v_and_b32 v14, -2, v1 // (op 900) 
 // (op 900) BEGIN: Tile[1]Flatten
 // (op 900) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 900): v21
v_sub_u32 v21, v1, v14 // (op 900) 
 // (op 900) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 900): v14

 // (op 900) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 900): v14
v_and_b32 v14, 4294967294, v1 // (op 900) 
 // (op 900) BEGIN: Flatten
 // (op 900) {Flatten: Add(v14:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag628: 1 VGPR (Value: UInt32) (op 900): v23
v_add_u32 v23, v14, v21 // (op 900) 
 // (op 900) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 900): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 900): v14

 // (op 448) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(900) END
 // (op 448) Assign VGPR 2:U32(901) BEGIN
 // (op 901) Assign dim(629) = 2:U32
 // (op 901) Assign stride expression
 // (op 448) Assign VGPR 2:U32(901) END
 // (op 448) NOP(896) BEGIN
 // (op 896) generate({})
 // (op 896) end: generate({})
 // (op 448) NOP(896) END
 // (op 448) Assign VGPR 1:U32(905) BEGIN
 // (op 905) Assign dim(631) = 1:U32
 // (op 905) Assign stride expression
 // (op 448) Assign VGPR 1:U32(905) END
 // (op 448) ForLoopOp KLoopTail: LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(175) BEGIN
 // (op 175) Initialize For Loop
 // (op 175) generate({176})
 // (op 175) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(176) BEGIN
 // (op 176) Assign dim(452) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32
 // tag 452: s**UNALLOCATED**
 // (op 176) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag452: SGPR Value: UInt32 x 1: (unallocated)
 // (op 176) reg expression
 // (op 176) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 176): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 176) 
 // (op 176) convert
 // (op 176) BitwiseAnd({ convertForwardSubset: s0:U32}, 4294967294:U32)U32
// Allocated DataFlowTag452: 1 SGPR (Value: UInt32) (op 176): s14
s_and_b32 s14, s0, 4294967294 // (op 176) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 176): s0, s1

 // (op 175) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(176) END
 // (op 175) end: generate({176})
 // (op 175) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 175) 
 // (op 175) Generate LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 175) LessThan({DataFlowTag452: s14:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 175) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 175): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 175) 
// Allocated : 1 SGPR (Value: UInt32) (op 175): s15
s_mov_b32 s15, s0 // (op 175) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 175): s0, s1

s_cmp_lt_u32 s14, s15 // (op 175) 
 // Freeing : 1 SGPR (Value: UInt32) (op 175): s15

 // Freeing : 6 VGPRs (Value: FP6x16) (op 382): v32, v33, v34, v35, v36, v37

 // Freeing : 1 VGPR (Value: E8M0) (op 385): v16

 // Freeing : 4 VGPRs (Value: FP4x8) (op 388): v38, v39, v40, v41

 // Freeing : 1 VGPR (Value: E8M0) (op 391): v12

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 175) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 175) 
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 175) 
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175:
 // (op 175) 
 // (op 175) generate({177, 226})
 // (op 175) Assign SGPR Add(DataFlowTag(452)U32, 1:U32)U32(177) BEGIN
 // (op 177) Assign dim(452) = Add(DataFlowTag(452)U32, 1:U32)U32
 // (op 177) Generate Add(DataFlowTag(452)U32, 1:U32)U32 into DataFlowTag452: SGPR Value: UInt32 x 1: s14
 // (op 177) reg expression
 // (op 177) Add({DataFlowTag452: s14:U32}, 1:U32)U32
 // (op 175) SetCoordinate(226) BEGIN
 // (op 226) generate({})
 // (op 226) end: generate({})
 // (op 226) generate({180})
 // (op 226) LoadTiled Value: FP4x8(180) BEGIN
 // (op 180) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 180) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 180) Generate 0:U64 into nullptr
 // tag 454: v**UNALLOCATED**
 // (op 180) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 498: v**UNALLOCATED**
s_add_u32 s14, s14, 1 // (op 177) 
 // (op 175) Assign SGPR Add(DataFlowTag(452)U32, 1:U32)U32(177) END
// Allocated Offset498: 1 VGPR (Value: UInt32) (op 180): v12
v_mov_b32 v12, v13 // (op 180) 
 // (op 180) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = Offset498: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U64, 4:U32)U64, 8:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 180) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 180): v24, v25, v26, v27
buffer_load_dwordx4 v[24:27], v12, s[40:43], 0 offen // (op 180) Load value
// VMEM: Expected complete at 4362 (current 3882)
 // (op 226) LoadTiled Value: FP4x8(180) END
 // (op 226) Deallocate{}(1204) BEGIN
 // (op 1204) Deallocate 498
 // Freeing Offset498: 1 VGPR (Value: UInt32) (op 180): v12

 // Deleting tag 498
 // (op 1204) Deallocate 500
 // Deleting tag 500
 // (op 226) Deallocate{}(1204) END
 // (op 226) Assign VGPR Add(DataFlowTag(494)U32, Convert(DataFlowTag(495)U64)U32)U32(443) BEGIN
 // (op 443) Assign dim(494) = Add(DataFlowTag(494)U32, Convert(DataFlowTag(495)U64)U32)U32
 // (op 443) Generate Add(DataFlowTag(494)U32, Convert(DataFlowTag(495)U64)U32)U32 into DataFlowTag494: VGPR Value: UInt32 x 1: v13
 // (op 443) reg expression
 // (op 443) Add({DataFlowTag494: v13:U32}, 32:U32)U32
v_add_u32 v13, 32, v13 // (op 443) 
 // (op 226) Assign VGPR Add(DataFlowTag(494)U32, Convert(DataFlowTag(495)U64)U32)U32(443) END
 // (op 226) end: generate({180})
 // (op 175) SetCoordinate(226) END
 // (op 175) SetCoordinate(231) BEGIN
 // (op 231) generate({})
 // (op 231) end: generate({})
 // (op 231) generate({194})
 // (op 231) LoadTiled Value: FP6x16(194) BEGIN
 // (op 194) GEN: loadMacroTileVGPRCI Value: FP6x16
 // (op 194) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 194) Generate 0:U64 into nullptr
 // tag 456: v**UNALLOCATED**
 // (op 194) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 517: v**UNALLOCATED**
 // (op 175) Barrier(1101) BEGIN
s_barrier  // (op 1101) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 175) Barrier(1101) END
// Allocated Offset517: 1 VGPR (Value: UInt32) (op 194): v12
v_mov_b32 v12, v17 // (op 194) 
 // (op 194) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = Offset517: VGPR Value: UInt32 x 1: v12
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 194) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 175) StoreLDSTile Value: FP4x8(181) BEGIN
 // (op 181) GEN: StoreLDSTile
 // (op 181) GEN: storeMacroTileLDS OP 181 LDS 17 MacroTile 454
 // (op 181) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 181) Generate 0:U32 into nullptr
 // (op 181) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[24:27]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[24:27]
 // 	info.rowOffsetReg = DataFlowTag580: VGPR Value: UInt32 x 1: v19
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: FP6x16) (op 194): v28, v29, v30, v31, v32, v33
buffer_load_dwordx4 v[28:31], v12, s[44:47], 0 offen // (op 194) Load value
// VMEM: Expected complete at 4366 (current 3886)
buffer_load_dwordx2 v[32:33], v12, s[44:47], 0 offen offset:16 // (op 194) Load value
// VMEM: Expected complete at 4367 (current 3887)
 // (op 231) LoadTiled Value: FP6x16(194) END
 // (op 231) Deallocate{}(1216) BEGIN
 // (op 1216) Deallocate 519
 // Deleting tag 519
 // (op 1216) Deallocate 517
 // Freeing Offset517: 1 VGPR (Value: UInt32) (op 194): v12

 // Deleting tag 517
 // (op 231) Deallocate{}(1216) END
 // (op 231) Assign VGPR Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32(512) BEGIN
 // (op 512) Assign dim(513) = Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32
 // (op 512) Generate Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32 into DataFlowTag513: VGPR Value: UInt32 x 1: v17
 // (op 512) reg expression
 // (op 512) Add({DataFlowTag513: v17:U32}, 48:U32)U32
v_add_u32 v17, 48, v17 // (op 512) 
 // (op 231) Assign VGPR Add(DataFlowTag(513)U32, Convert(DataFlowTag(514)U64)U32)U32(512) END
 // (op 231) end: generate({194})
 // (op 175) SetCoordinate(231) END
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[24:27] offset:1600 // (op 181) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 3883 (current 3788)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 175) StoreLDSTile Value: FP4x8(181) END
 // (op 175) SetCoordinate(236) BEGIN
 // (op 236) generate({})
 // (op 236) end: generate({})
 // (op 236) generate({195})
 // (op 236) LoadTiled Value: E8M0(195) BEGIN
 // (op 195) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 195) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 195) Generate 1:U64 into nullptr
 // tag 458: v**UNALLOCATED**
 // (op 195) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 536: v**UNALLOCATED**
 // (op 175) Deallocate{}(1206) BEGIN
 // (op 1206) Deallocate 582
 // Deleting tag 582
 // (op 1206) Deallocate 454
 // Freeing : 4 VGPRs (Value: FP4x8) (op 180): v24, v25, v26, v27

 // Deleting tag 454
 // (op 175) Deallocate{}(1206) END
// Allocated Offset536: 1 VGPR (Value: UInt32) (op 195): v12
v_mov_b32 v12, v18 // (op 195) 
 // (op 195) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset536: VGPR Value: UInt32 x 1: v12
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
 // 	info.bufDesc = DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 195) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 175) Barrier(1114) BEGIN
s_barrier  // (op 1114) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 175) Barrier(1114) END
// Allocated : 1 VGPR (Value: E8M0) (op 195): v14
buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 195) Load value
// VMEM: Expected complete at 4847 (current 4367)
 // (op 236) LoadTiled Value: E8M0(195) END
 // (op 236) Deallocate{}(1218) BEGIN
 // (op 1218) Deallocate 538
 // Deleting tag 538
 // (op 1218) Deallocate 536
 // Freeing Offset536: 1 VGPR (Value: UInt32) (op 195): v12

 // Deleting tag 536
 // (op 236) Deallocate{}(1218) END
 // (op 236) Assign VGPR Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32(576) BEGIN
 // (op 576) Assign dim(532) = Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32
 // (op 576) Generate Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32 into DataFlowTag532: VGPR Value: UInt32 x 1: v18
 // (op 576) reg expression
 // (op 576) Add({DataFlowTag532: v18:U32}, 2:U32)U32
 // (op 175) StoreLDSTile Value: FP6x16(191) BEGIN
 // (op 191) GEN: StoreLDSTile
 // (op 191) GEN: storeMacroTileLDS OP 191 LDS 25 MacroTile 456
 // (op 191) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 191) Generate 0:U32 into nullptr
 // (op 191) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[28:33]
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

 // (op 191) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: v[28:33]
 // 	info.rowOffsetReg = DataFlowTag596: VGPR Value: UInt32 x 1: v20
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
 // (op 191) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

v_add_u32 v18, 2, v18 // (op 576) 
 // (op 236) Assign VGPR Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32(576) END
 // (op 236) end: generate({195})
 // (op 175) SetCoordinate(236) END
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[28:31] // (op 191) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3889 (current 3794)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 175) SetCoordinate(241) BEGIN
 // (op 241) generate({})
 // (op 241) end: generate({})
 // (op 241) generate({196})
 // (op 241) LoadTiled Value: E8M0(196) BEGIN
 // (op 196) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 196) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 196) Generate 1:U64 into nullptr
 // tag 460: v**UNALLOCATED**
 // (op 196) 	Dir = Load
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 555: v**UNALLOCATED**
// Allocated Offset555: 1 VGPR (Value: UInt32) (op 196): v12
v_mov_b32 v12, v15 // (op 196) 
 // (op 196) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset555: VGPR Value: UInt32 x 1: v12
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
 // 	info.bufDesc = DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 196) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 196): v16
buffer_load_ubyte v16, v12, s[52:55], 0 offen offset: 0 // (op 196) Load value
// VMEM: Expected complete at 4852 (current 4372)
 // (op 241) LoadTiled Value: E8M0(196) END
 // (op 241) Deallocate{}(1220) BEGIN
 // (op 1220) Deallocate 557
 // Deleting tag 557
 // (op 1220) Deallocate 555
 // Freeing Offset555: 1 VGPR (Value: UInt32) (op 196): v12

 // Deleting tag 555
 // (op 241) Deallocate{}(1220) END
 // (op 241) Assign VGPR Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32(640) BEGIN
 // (op 640) Assign dim(551) = Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32
 // (op 640) Generate Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32 into DataFlowTag551: VGPR Value: UInt32 x 1: v15
 // (op 640) reg expression
 // (op 640) Add({DataFlowTag551: v15:U32}, 2:U32)U32
v_add_u32 v15, 2, v15 // (op 640) 
 // (op 241) Assign VGPR Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32(640) END
 // (op 241) end: generate({196})
 // (op 175) SetCoordinate(241) END
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[32:33] offset:16 // (op 191) Store local data 
// Modelled addresses (normalized): [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720, 744, 768, 792, 816, 840, 864, 888, 912, 936, 960, 984, 1008, 1032, 1056, 1080, 1104, 1128, 1152, 1176, 1200, 1224, 1248, 1272, 1296, 1320, 1344, 1368, 1392, 1416, 1440, 1464, 1488, 1512]
// DSMEM: Expected complete at 3894 (current 3799)
// Extra dsts: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 175) StoreLDSTile Value: FP6x16(191) END
 // (op 175) Barrier(1104) BEGIN
 // (op 175) Deallocate{}(1210) BEGIN
 // (op 1210) Deallocate 598
 // Deleting tag 598
 // (op 1210) Deallocate 456
 // Freeing : 6 VGPRs (Value: FP6x16) (op 194): v28, v29, v30, v31, v32, v33

 // Deleting tag 456
 // (op 175) Deallocate{}(1210) END
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 175) Barrier(1104) END
 // (op 175) Barrier(1128) BEGIN
 // (op 175) SetCoordinate(182) BEGIN
 // (op 182) generate({})
 // (op 182) end: generate({})
 // (op 182) generate({183})
 // (op 182) LoadLDSTile{Value: FP4x8}(183) BEGIN
 // (op 183) GEN: loadMacroTileWAVELDS OP 183 LDS 17 WaveTile 105
 // (op 183) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 183) Generate 0:U32 into nullptr
 // tag 462: v**UNALLOCATED**
 // (op 183) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1600
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1128) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 175) Barrier(1128) END
// Allocated : 4 VGPRs (Value: FP4x8) (op 183): v24, v25, v26, v27
ds_read_b128 v[24:27], v3 offset:1600 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848, 880, 912, 944, 976, 1008]
// DSMEM: Expected complete at 3982 (current 3887)
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 1600, Size: 1024)
 // (op 182) LoadLDSTile{Value: FP4x8}(183) END
 // (op 182) end: generate({183})
 // (op 175) SetCoordinate(182) END
 // (op 175) StoreLDSTile Value: E8M0(192) BEGIN
 // (op 192) GEN: StoreLDSTile
 // (op 192) GEN: storeMacroTileLDS OP 192 LDS 33 MacroTile 458
 // (op 192) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 192) Generate 1:U32 into nullptr
 // (op 192) 	Dir = Store
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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

 // (op 192) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag612: VGPR Value: UInt32 x 1: v22
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
 // 	info.bufDesc = (nullptr)
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

 // (op 175) Barrier(1118) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1118) 
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 175) Barrier(1118) END
s_waitcnt vmcnt(1)
ds_write_b8 v22, v14 offset:2624 // (op 192) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3993 (current 3898)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 175) StoreLDSTile Value: E8M0(192) END
 // (op 175) SetCoordinate(184) BEGIN
 // (op 184) generate({})
 // (op 184) end: generate({})
 // (op 184) generate({185})
 // (op 184) LoadLDSTile{Value: FP6x16}(185) BEGIN
 // (op 185) GEN: loadMacroTileWAVELDS OP 185 LDS 25 WaveTile 171
 // (op 185) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 185) Generate 0:U32 into nullptr
 // tag 464: v**UNALLOCATED**
 // (op 185) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
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

 // (op 185) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
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
 // (op 185) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 175) Deallocate{}(1212) BEGIN
 // (op 1212) Deallocate 614
 // Deleting tag 614
 // (op 1212) Deallocate 458
 // Freeing : 1 VGPR (Value: E8M0) (op 195): v14

 // Deleting tag 458
 // (op 175) Deallocate{}(1212) END
// Allocated : 6 VGPRs (Value: FP6x16) (op 185): v28, v29, v30, v31, v32, v33
ds_read_b128 v[28:31], v4 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected complete at 3994 (current 3899)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 175) Barrier(1142) BEGIN
s_barrier  // (op 1142) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 175) Barrier(1142) END
ds_read_b64 v[32:33], v4 offset:16 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816, 864, 912, 960, 1008, 1056, 1104, 1152, 1200, 1248, 1296, 1344, 1392, 1440, 1488, 24, 72, 120, 168, 216, 264, 312, 360, 408, 456, 504, 552, 600, 648, 696, 744, 792, 840, 888, 936, 984, 1032, 1080, 1128, 1176, 1224, 1272, 1320, 1368, 1416, 1464, 1512]
// DSMEM: Expected stall of 82. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4078 (current 3983)
// Extra srcs: LDS Value: FP6x16 x 0: LDS:(Offset: 0, Size: 1536)
 // (op 184) LoadLDSTile{Value: FP6x16}(185) END
 // (op 184) end: generate({185})
 // (op 175) SetCoordinate(184) END
 // (op 175) StoreLDSTile Value: E8M0(193) BEGIN
 // (op 193) GEN: StoreLDSTile
 // (op 193) GEN: storeMacroTileLDS OP 193 LDS 41 MacroTile 460
 // (op 193) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 193) Generate 1:U32 into nullptr
 // (op 193) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

 // (op 193) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
 // 	info.rowOffsetReg = DataFlowTag628: VGPR Value: UInt32 x 1: v23
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 193) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 175) Barrier(1132) BEGIN
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1132) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 175) Barrier(1132) END
s_waitcnt vmcnt(0)
ds_write_b8 v23, v16 offset:1536 // (op 193) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 4092 (current 3997)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 175) StoreLDSTile Value: E8M0(193) END
 // (op 175) SetCoordinate(186) BEGIN
 // (op 186) generate({})
 // (op 186) end: generate({})
 // (op 186) generate({187})
 // (op 186) LoadLDSTile{Value: E8M0}(187) BEGIN
 // (op 187) GEN: loadMacroTileWAVELDS OP 187 LDS 33 WaveTile 237
 // (op 187) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 187) Generate 1:U32 into nullptr
 // tag 466: v**UNALLOCATED**
 // (op 187) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v0
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
 // 	info.offset = Literal Value: UInt32 x 0: 2624
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 175) Deallocate{}(1214) BEGIN
 // (op 1214) Deallocate 630
 // Deleting tag 630
 // (op 1214) Deallocate 460
 // Freeing : 1 VGPR (Value: E8M0) (op 196): v16

 // Deleting tag 460
 // (op 175) Deallocate{}(1214) END
// Allocated : 1 VGPR (Value: E8M0) (op 187): v12
ds_read_u8 v12, v0 offset:2624 // (op 187) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4093 (current 3998)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 2624, Size: 64)
 // (op 186) LoadLDSTile{Value: E8M0}(187) END
 // (op 186) end: generate({187})
 // (op 175) SetCoordinate(186) END
 // (op 175) Barrier(1146) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 175) Barrier(1146) END
 // (op 175) SetCoordinate(188) BEGIN
 // (op 188) generate({})
 // (op 188) end: generate({})
 // (op 188) generate({189})
 // (op 188) LoadLDSTile{Value: E8M0}(189) BEGIN
 // (op 189) GEN: loadMacroTileWAVELDS OP 189 LDS 41 WaveTile 297
 // (op 189) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 189) Generate 1:U32 into nullptr
 // tag 468: v**UNALLOCATED**
 // (op 189) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag636: VGPR Value: UInt32 x 1: v6
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
 // 	info.offset = Literal Value: UInt32 x 0: 1536
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

// Allocated : 1 VGPR (Value: E8M0) (op 189): v14
ds_read_u8 v14, v6 offset:1536 // (op 189) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4190 (current 4095)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1536, Size: 64)
 // (op 188) LoadLDSTile{Value: E8M0}(189) END
 // (op 188) end: generate({189})
 // (op 175) SetCoordinate(188) END
 // (op 175) Multiply(190) BEGIN
 // (op 190) Generate ScaledMatrixMultiply(WaveTile8xFP4, WaveTile16xFP6, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)16xFP6 into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[24:27], v[28:33], a[0:15], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 190) 
 // (op 175) Multiply(190) END
 // (op 175) Deallocate{}(1208) BEGIN
 // (op 1208) Deallocate 462
 // Freeing : 4 VGPRs (Value: FP4x8) (op 183): v24, v25, v26, v27

 // Deleting tag 462
 // (op 1208) Deallocate 464
 // Freeing : 6 VGPRs (Value: FP6x16) (op 185): v28, v29, v30, v31, v32, v33

 // Deleting tag 464
 // (op 1208) Deallocate 466
 // Freeing : 1 VGPR (Value: E8M0) (op 187): v12

 // Deleting tag 466
 // (op 1208) Deallocate 468
 // Freeing : 1 VGPR (Value: E8M0) (op 189): v14

 // Deleting tag 468
 // (op 175) Deallocate{}(1208) END
 // (op 175) end: generate({177, 226})
 // (op 175) For Loop Increment
 // (op 175) generate({})
 // (op 175) end: generate({})
 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
 // (op 175) Generate LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 175) LessThan({DataFlowTag452: s14:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 175) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 175): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 175) 
// Allocated : 1 SGPR (Value: UInt32) (op 175): s15
s_mov_b32 s15, s0 // (op 175) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 175): s0, s1

s_cmp_lt_u32 s14, s15 // (op 175) 
 // Freeing : 1 SGPR (Value: UInt32) (op 175): s15

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 175) 
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175:
 // (op 175) 
 // (op 175) Unlock For Loop
 // Unlock None
 // (op 448) ForLoopOp KLoopTail: LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(175) END
 // (op 448) Deallocate{}(1202) BEGIN
 // (op 1202) Deallocate 518
 // Deleting tag 518
 // (op 1202) Deallocate 537
 // Deleting tag 537
 // (op 1202) Deallocate 494
 // Freeing DataFlowTag494: 1 VGPR (Value: UInt32) (op 453): v13

 // Deleting tag 494
 // (op 1202) Deallocate 495
 // Deleting tag 495
 // (op 1202) Deallocate 499
 // Deleting tag 499
 // (op 1202) Deallocate 532
 // Freeing DataFlowTag532: 1 VGPR (Value: UInt32) (op 584): v18

 // Deleting tag 532
 // (op 1202) Deallocate 514
 // Deleting tag 514
 // (op 1202) Deallocate 533
 // Deleting tag 533
 // (op 1202) Deallocate 520
 // Deleting tag 520
 // (op 1202) Deallocate 452
 // Freeing DataFlowTag452: 1 SGPR (Value: UInt32) (op 176): s14

 // Deleting tag 452
 // (op 1202) Deallocate 629
 // Deleting tag 629
 // (op 1202) Deallocate 597
 // Deleting tag 597
 // (op 1202) Deallocate 513
 // Freeing DataFlowTag513: 1 VGPR (Value: UInt32) (op 520): v17

 // Deleting tag 513
 // (op 1202) Deallocate 581
 // Deleting tag 581
 // (op 1202) Deallocate 631
 // Deleting tag 631
 // (op 1202) Deallocate 501
 // Deleting tag 501
 // (op 1202) Deallocate 539
 // Deleting tag 539
 // (op 1202) Deallocate 558
 // Deleting tag 558
 // (op 1202) Deallocate 551
 // Freeing DataFlowTag551: 1 VGPR (Value: UInt32) (op 648): v15

 // Deleting tag 551
 // (op 1202) Deallocate 552
 // Deleting tag 552
 // (op 1202) Deallocate 583
 // Deleting tag 583
 // (op 1202) Deallocate 580
 // Freeing DataFlowTag580: 1 VGPR (Value: UInt32) (op 741): v19

 // Deleting tag 580
 // (op 1202) Deallocate 596
 // Freeing DataFlowTag596: 1 VGPR (Value: UInt32) (op 794): v20

 // Deleting tag 596
 // (op 1202) Deallocate 599
 // Deleting tag 599
 // (op 1202) Deallocate 612
 // Freeing DataFlowTag612: 1 VGPR (Value: UInt32) (op 847): v22

 // Deleting tag 612
 // (op 1202) Deallocate 613
 // Deleting tag 613
 // (op 1202) Deallocate 556
 // Deleting tag 556
 // (op 1202) Deallocate 615
 // Deleting tag 615
 // (op 1202) Deallocate 628
 // Freeing DataFlowTag628: 1 VGPR (Value: UInt32) (op 900): v23

 // Deleting tag 628
 // (op 448) Deallocate{}(1202) END
 // (op 448) end: generate({442})
 // (op 448) Unlock Scope 448
 // Unlock None
 // (op 222) Scope(448) END
 // (op 222) end: generate({448})
 // (op 221) SetCoordinate(222) END
 // (op 221) end: generate({222})
 // (op 344) SetCoordinate(221) END
 // (op 344) Deallocate{}(1222) BEGIN
 // (op 1222) Deallocate 4
 // Deleting tag 4
 // (op 1222) Deallocate 433
 // Deleting tag 433
 // (op 1222) Deallocate 535
 // Freeing DataFlowTag535: 4 SGPRs (Buffer: None) (op 609): s48, s49, s50, s51

 // Deleting tag 535
 // (op 1222) Deallocate 33
 // Deleting tag 33
 // (op 1222) Deallocate 590
 // Deleting tag 590
 // (op 1222) Deallocate 144
 // Deleting tag 144
 // (op 1222) Deallocate 606
 // Deleting tag 606
 // (op 1222) Deallocate 638
 // Deleting tag 638
 // (op 1222) Deallocate 622
 // Deleting tag 622
 // (op 1222) Deallocate 497
 // Freeing DataFlowTag497: 4 SGPRs (Buffer: None) (op 481): s40, s41, s42, s43

 // Deleting tag 497
 // (op 1222) Deallocate 431
 // Deleting tag 431
 // (op 1222) Deallocate 1
 // Deleting tag 1
 // (op 1222) Deallocate 159
 // Deleting tag 159
 // (op 1222) Deallocate 210
 // Deleting tag 210
 // (op 1222) Deallocate 17
 // Deleting tag 17
 // (op 1222) Deallocate 78
 // Deleting tag 78
 // (op 1222) Deallocate 225
 // Deleting tag 225
 // (op 1222) Deallocate 41
 // Deleting tag 41
 // (op 1222) Deallocate 285
 // Deleting tag 285
 // (op 1222) Deallocate 25
 // Deleting tag 25
 // (op 1222) Deallocate 93
 // Deleting tag 93
 // (op 1222) Deallocate 3
 // Deleting tag 3
 // (op 1222) Deallocate 437
 // Deleting tag 437
 // (op 1222) Deallocate 270
 // Deleting tag 270
 // (op 1222) Deallocate 516
 // Freeing DataFlowTag516: 4 SGPRs (Buffer: None) (op 545): s44, s45, s46, s47

 // Deleting tag 516
 // (op 1222) Deallocate 2
 // Deleting tag 2
 // (op 1222) Deallocate 554
 // Freeing DataFlowTag554: 4 SGPRs (Buffer: None) (op 673): s52, s53, s54, s55

 // Deleting tag 554
 // (op 1222) Deallocate 435
 // Deleting tag 435
 // (op 344) Deallocate{}(1222) END
 // (op 344) Deallocate{Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_0_pointer, Tensor_2_pointer, Tensor_0_size_1_14, Tensor_7_stride_1_8, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_4_pointer, Tensor_2_stride_1_3}(1361) BEGIN
 // (op 1361) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1361) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1361) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1361) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 1361) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1361) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1361) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1361) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1361) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 1361) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 344) Deallocate{Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_0_pointer, Tensor_2_pointer, Tensor_0_size_1_14, Tensor_7_stride_1_8, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_4_pointer, Tensor_2_stride_1_3}(1361) END
 // (op 344) Barrier(1224) BEGIN
s_barrier  // (op 1224) 
 // (op 344) Barrier(1224) END
 // (op 344) end: generate({349})
 // (op 344) Unlock Scope 344
 // Unlock None
 // (op 1) Scope(344) END
 // (op 1) Deallocate{}(1268) BEGIN
 // (op 1268) Deallocate 639
 // Deleting tag 639
 // (op 1268) Deallocate 662
 // Freeing DataFlowTag662: 1 VGPR (Value: UInt32) (op 1010): v8

 // Deleting tag 662
 // (op 1268) Deallocate 637
 // Deleting tag 637
 // (op 1268) Deallocate 654
 // Freeing DataFlowTag654: 1 VGPR (Value: UInt32) (op 986): v7

 // Deleting tag 654
 // (op 1268) Deallocate 670
 // Freeing DataFlowTag670: 1 VGPR (Value: UInt32) (op 1034): v5

 // Deleting tag 670
 // (op 1268) Deallocate 589
 // Deleting tag 589
 // (op 1268) Deallocate 636
 // Freeing DataFlowTag636: 1 VGPR (Value: UInt32) (op 928): v6

 // Deleting tag 636
 // (op 1268) Deallocate 665
 // Deleting tag 665
 // (op 1268) Deallocate 673
 // Deleting tag 673
 // (op 1268) Deallocate 678
 // Freeing DataFlowTag678: 1 VGPR (Value: UInt32) (op 1058): v10

 // Deleting tag 678
 // (op 1268) Deallocate 591
 // Deleting tag 591
 // (op 1268) Deallocate 607
 // Deleting tag 607
 // (op 1268) Deallocate 657
 // Deleting tag 657
 // (op 1268) Deallocate 681
 // Deleting tag 681
 // (op 1268) Deallocate 623
 // Deleting tag 623
 // (op 1268) Deallocate 679
 // Deleting tag 679
 // (op 1268) Deallocate 620
 // Freeing DataFlowTag620: 1 VGPR (Value: UInt32) (op 875): v0

 // Deleting tag 620
 // (op 1268) Deallocate 621
 // Deleting tag 621
 // (op 1268) Deallocate 655
 // Deleting tag 655
 // (op 1268) Deallocate 588
 // Freeing DataFlowTag588: 1 VGPR (Value: UInt32) (op 769): v3

 // Deleting tag 588
 // (op 1268) Deallocate 663
 // Deleting tag 663
 // (op 1268) Deallocate 604
 // Freeing DataFlowTag604: 1 VGPR (Value: UInt32) (op 822): v4

 // Deleting tag 604
 // (op 1268) Deallocate 605
 // Deleting tag 605
 // (op 1268) Deallocate 671
 // Deleting tag 671
 // (op 1) Deallocate{}(1268) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(135) BEGIN
 // (op 135) Lock for Conditional Branch
 // Lock Branch
 // (op 135) Generate Equal(0.00000:S, DataFlowTag(44)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 135) reg expression
 // (op 135) Equal(0.00000:S, v9:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v9 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
 // (op 135) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 135) 
s_cbranch_vccz RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135
 // (op 135) generate({352})
 // (op 135) Scope(352) BEGIN
 // (op 352) Lock Scope 352
 // Lock Branch
 // (op 352) generate({355})
 // (op 352) NOP(355) BEGIN
 // (op 355) generate({})
 // (op 355) end: generate({})
 // (op 352) NOP(355) END
 // (op 352) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) BEGIN
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

 // (op 352) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(50) END
 // (op 352) Assign VGPR Convert(DataFlowTag(59)NA)H(56) BEGIN
 // (op 56) Assign dim(61) = Convert(DataFlowTag(59)NA)H
 // (op 56) Generate Convert(DataFlowTag(59)NA)H into nullptr
 // (op 56) reg expression
// Allocated DataFlowTag59: 16 VGPRs (Value: Half) (op 56): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
v_cvt_f16_f32 v12, v28 // (op 56) 
v_cvt_f16_f32 v13, v29 // (op 56) 
v_cvt_f16_f32 v14, v30 // (op 56) 
v_cvt_f16_f32 v15, v31 // (op 56) 
v_cvt_f16_f32 v16, v32 // (op 56) 
v_cvt_f16_f32 v17, v33 // (op 56) 
v_cvt_f16_f32 v18, v34 // (op 56) 
v_cvt_f16_f32 v19, v35 // (op 56) 
v_cvt_f16_f32 v20, v36 // (op 56) 
v_cvt_f16_f32 v21, v37 // (op 56) 
v_cvt_f16_f32 v22, v38 // (op 56) 
v_cvt_f16_f32 v23, v39 // (op 56) 
v_cvt_f16_f32 v24, v40 // (op 56) 
v_cvt_f16_f32 v25, v41 // (op 56) 
v_cvt_f16_f32 v26, v42 // (op 56) 
v_cvt_f16_f32 v27, v43 // (op 56) 
 // (op 352) Assign VGPR Convert(DataFlowTag(59)NA)H(56) END
 // (op 352) Deallocate{}(1182) BEGIN
 // (op 1182) Deallocate 59
 // Freeing DataFlowTag59: 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 59
 // (op 352) Deallocate{}(1182) END
 // (op 352) NOP(938) BEGIN
 // (op 938) generate({})
 // (op 938) end: generate({})
 // (op 352) NOP(938) END
 // (op 352) NOP(935) BEGIN
 // (op 935) generate({})
 // (op 935) end: generate({})
 // (op 352) NOP(935) END
 // (op 352) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(942) BEGIN
 // (op 942) Assign dim(640) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 640: v**UNALLOCATED**
 // (op 942) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag640: VGPR Value: UInt32 x 1: (unallocated)
 // (op 942) reg expression
 // (op 942) reg expression
 // (op 942) reg expression
 // (op 942) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 942) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 942): v0
v_lshrrev_b32 v0, 6, v1 // (op 942) 
 // (op 942) convert
 // (op 942) BEGIN: Tile[1]
 // (op 942) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 942): v10
v_and_b32 v10, 63, v1 // (op 942) 
 // (op 942) END: Tile[1]
 // (op 942) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 942): v3
v_lshrrev_b32 v3, 5, v10 // (op 942) 
 // (op 942) BEGIN: Flatten
 // (op 942) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 942) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 942)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 942)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 942): v4
v_lshlrev_b32 v4, 2, v3 // (op 942) 
 // (op 942) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 942): v3

 // (op 942) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 942): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 942) 
 // (op 942) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 942): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 942): v0

 // (op 942) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 942): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 942) 
 // (op 942) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 942): v3

 // (op 942) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 942): v3
v_lshrrev_b32 v3, 2, v10 // (op 942) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 942): v10

 // (op 942) BEGIN: Tile[1]
 // (op 942) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 942): v10
v_and_b32 v10, 7, v3 // (op 942) 
 // (op 942) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 942): v3

 // (op 942) BEGIN: Tile[1]
 // (op 942) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 942): v3
v_and_b32 v3, 3, v1 // (op 942) 
 // (op 942) END: Tile[1]
 // (op 942) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 942): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 942) 
 // (op 942) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 942): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 942): v10

 // (op 942) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 942): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 942) 
 // (op 942) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 942): v4

 // (op 942) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 942): v10
v_mul_lo_u32 v10, v3, s30 // (op 942) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 942): v3

 // (op 942) BEGIN: Join
// Allocated DataFlowTag640: 1 VGPR (Value: UInt32) (op 942): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 942) 
 // (op 942) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 942): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 942): v0

 // (op 352) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(942) END
 // (op 352) Assign VGPR 16:U64(943) BEGIN
 // (op 943) Assign dim(641) = 16:U64
 // (op 943) Assign stride expression
 // (op 352) Assign VGPR 16:U64(943) END
 // (op 352) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(944) BEGIN
 // (op 944) Assign dim(643) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 643: s**UNALLOCATED**
 // (op 944) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag643: SGPR Buffer: None x 1: (unallocated)
 // (op 944) reg expression
 // (op 944) reg expression
 // (op 944) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 944) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 944)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 944)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 944): s14, s15
s_mul_i32 s15, s30, s13 // (op 944) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 944): s0
s_mul_i32 s0, s31, s12 // (op 944) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 944) least significant
// Allocated : 1 SGPR (Value: Int32) (op 944): s1
s_mul_hi_u32 s1, s30, s12 // (op 944) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 944) most significant: sum
s_add_u32 s15, s15, s1 // (op 944) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 944): s1

 // Freeing : 1 SGPR (Value: Int32) (op 944): s0

 // (op 944) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 944) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 944)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 944)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 944): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 944) 
 // Freeing : 2 SGPRs (Value: Int64) (op 944): s14, s15

 // (op 944) reg expression
 // (op 944) reg expression
// Allocated DataFlowTag643: 4 SGPRs (Buffer: None) (op 944): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 944) reg expression
s_mov_b32 s9, s33 // (op 944) reg expression
s_mov_b32 s10, s0 // (op 944) reg expression
s_mov_b32 s11, 131072 // (op 944) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 944): s0, s1

 // (op 352) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(944) END
 // (op 352) NOP(936) BEGIN
 // (op 936) generate({})
 // (op 936) end: generate({})
 // (op 352) NOP(936) END
 // (op 352) Assign VGPR 2:U64(949) BEGIN
 // (op 949) Assign dim(645) = 2:U64
 // (op 949) Assign stride expression
 // (op 352) Assign VGPR 2:U64(949) END
 // (op 352) StoreTiled Value: Halfx2(58) BEGIN
 // (op 58) GEN: storeMacroTileWAVE OP 58 MacroTile 61 WaveTile 367
 // (op 58) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 58) Generate 16:U64 into nullptr
 // (op 58) Generate 2:U64 into nullptr
 // (op 58) Convert in LSTGen
 // (op 58) Generate Convert({DataFlowTag59: v[12:27]:H})2xH into nullptr
 // (op 58) reg expression
// Allocated DataFlowTag59: 8 VGPRs (Value: Halfx2) (op 58): v28, v29, v30, v31, v32, v33, v34, v35
v_pack_b32_f16 v28, v12, v13 // (op 58) 
v_pack_b32_f16 v29, v14, v15 // (op 58) 
v_pack_b32_f16 v30, v16, v17 // (op 58) 
v_pack_b32_f16 v31, v18, v19 // (op 58) 
v_pack_b32_f16 v32, v20, v21 // (op 58) 
v_pack_b32_f16 v33, v22, v23 // (op 58) 
v_pack_b32_f16 v34, v24, v25 // (op 58) 
v_pack_b32_f16 v35, v26, v27 // (op 58) 
 // (op 58) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Halfx2 x 8: v[28:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 58) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag59: VGPR Value: Halfx2 x 8: v[28:35]
 // 	info.rowOffsetReg = DataFlowTag640: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 58) 	info.m = 4
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 16
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx2 v[28:29], v3, s[8:11], 0 offen // (op 58) Store value
// VMEM: Expected complete at 5520 (current 5040)
buffer_store_dwordx2 v[30:31], v3, s[8:11], 0 offen offset:16 // (op 58) Store value
// VMEM: Expected complete at 5521 (current 5041)
buffer_store_dwordx2 v[32:33], v3, s[8:11], 0 offen offset:32 // (op 58) Store value
// VMEM: Expected complete at 5522 (current 5042)
buffer_store_dwordx2 v[34:35], v3, s[8:11], 0 offen offset:48 // (op 58) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6001 (current 5521)
 // Freeing DataFlowTag59: 8 VGPRs (Value: Halfx2) (op 58): v28, v29, v30, v31, v32, v33, v34, v35

 // (op 352) StoreTiled Value: Halfx2(58) END
 // (op 352) Deallocate{}(1184) BEGIN
 // (op 1184) Deallocate 641
 // Deleting tag 641
 // (op 1184) Deallocate 61
 // Freeing DataFlowTag59: 16 VGPRs (Value: Half) (op 56): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 61
 // (op 1184) Deallocate 645
 // Deleting tag 645
 // (op 1184) Deallocate 644
 // Deleting tag 644
 // (op 1184) Deallocate 640
 // Freeing DataFlowTag640: 1 VGPR (Value: UInt32) (op 942): v3

 // Deleting tag 640
 // (op 352) Deallocate{}(1184) END
 // (op 352) end: generate({355})
 // (op 352) Unlock Scope 352
 // Unlock None
 // (op 135) Scope(352) END
 // (op 135) end: generate({352})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 135) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 135) 
s_branch RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135:
 // (op 135) 
 // (op 135) generate({358})
 // (op 135) Scope(358) BEGIN
 // (op 358) Lock Scope 358
 // Lock Branch
 // (op 358) generate({361})
 // (op 358) NOP(361) BEGIN
 // (op 361) generate({})
 // (op 361) end: generate({})
 // (op 358) NOP(361) END
 // (op 358) NOP(143) BEGIN
 // (op 143) generate({})
 // (op 143) end: generate({})
 // (op 358) NOP(143) END
 // (op 358) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(144) BEGIN
 // (op 144) Assign dim(425) = Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA
 // (op 144) Generate Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA into nullptr
 // (op 144) reg expression
 // (op 144) reg expression
 // (op 144) Multiply(v11:S, {DataFlowTag55: a[0:15]:S})S
 // (op 358) Scope(706) BEGIN
 // (op 706) Lock Scope 706
 // Lock Branch
 // (op 706) generate({703})
 // (op 706) NOP(703) BEGIN
 // (op 703) generate({})
 // (op 703) end: generate({})
 // (op 706) NOP(703) END
 // (op 706) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(711) BEGIN
 // (op 711) Assign dim(570) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}
 // tag 570: v**UNALLOCATED**
 // (op 711) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32} into DataFlowTag570: VGPR Value: UInt32 x 1: (unallocated)
 // (op 711) reg expression
 // (op 711) reg expression
 // (op 711) reg expression
 // (op 711) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 711) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 711): v0
v_lshrrev_b32 v0, 6, v1 // (op 711) 
 // (op 711) convert
 // (op 711) BEGIN: Flatten[1]
 // (op 711) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 711): v10
v_and_b32 v10, 63, v1 // (op 711) 
 // (op 711) END: Flatten[1]
 // (op 711) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v3
v_lshrrev_b32 v3, 5, v10 // (op 711) 
 // (op 711) BEGIN: Tile
 // (op 711) {Tile: ShiftL(v3:U32, 2:U32)U32}
 // (op 711) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 711)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 711)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 711): v4
v_lshlrev_b32 v4, 2, v3 // (op 711) 
 // (op 711) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v3

 // (op 711) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 711): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 711) 
 // (op 711) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 711): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 711): v0

 // (op 711) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 711): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 711) 
 // (op 711) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 711): v3

 // (op 711) LogicalShiftR({Flatten[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v3
v_lshrrev_b32 v3, 2, v10 // (op 711) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 711): v10

 // (op 711) BEGIN: Flatten[1]
 // (op 711) {Flatten[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 711): v10
v_and_b32 v10, 7, v3 // (op 711) 
 // (op 711) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v3

 // (op 711) BEGIN: Flatten[1]
 // (op 711) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 711): v3
v_and_b32 v3, 3, v1 // (op 711) 
 // (op 711) END: Flatten[1]
 // (op 711) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 711): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 711) 
 // (op 711) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 711): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 711): v10

 // (op 711) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 711): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 711) 
 // (op 711) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 711): v4

 // (op 711) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v10
v_mul_lo_u32 v10, v3, s34 // (op 711) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 711): v3

 // (op 711) BEGIN: Split
// Allocated DataFlowTag570: 1 VGPR (Value: UInt32) (op 711): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 711) 
 // (op 711) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 711): v0

 // (op 706) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(711) END
 // (op 706) Deallocate{}(1288) BEGIN
 // (op 1288) Deallocate 439
 // Deleting tag 439
 // (op 1288) Deallocate 441
 // Deleting tag 441
 // (op 1288) Deallocate 326
 // Deleting tag 326
 // (op 706) Deallocate{}(1288) END
 // (op 706) Assign VGPR 16:U64(712) BEGIN
 // (op 712) Assign dim(571) = 16:U64
 // (op 712) Assign stride expression
 // (op 706) Assign VGPR 16:U64(712) END
 // (op 706) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(713) BEGIN
 // (op 713) Assign dim(573) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 573: s**UNALLOCATED**
 // (op 713) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag573: SGPR Buffer: None x 1: (unallocated)
 // (op 713) reg expression
 // (op 713) reg expression
 // (op 713) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 713) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 713)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 713)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 713): s14, s15
s_mul_i32 s15, s34, s13 // (op 713) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 713): s0
s_mul_i32 s0, s35, s12 // (op 713) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 713) least significant
// Allocated : 1 SGPR (Value: Int32) (op 713): s1
s_mul_hi_u32 s1, s34, s12 // (op 713) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 713) most significant: sum
s_add_u32 s15, s15, s1 // (op 713) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 713): s1

 // Freeing : 1 SGPR (Value: Int32) (op 713): s0

 // (op 713) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 713) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 713)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 713)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 713): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 713) 
 // Freeing : 2 SGPRs (Value: Int64) (op 713): s14, s15

 // (op 713) reg expression
 // (op 713) reg expression
// Allocated DataFlowTag573: 4 SGPRs (Buffer: None) (op 713): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 713) reg expression
s_mov_b32 s5, s37 // (op 713) reg expression
s_mov_b32 s6, s0 // (op 713) reg expression
s_mov_b32 s7, 131072 // (op 713) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 713): s0, s1

 // (op 706) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(713) END
 // (op 706) Deallocate{Tensor_10_pointer}(1365) BEGIN
 // (op 1365) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 706) Deallocate{Tensor_10_pointer}(1365) END
 // (op 706) NOP(704) BEGIN
 // (op 704) generate({})
 // (op 704) end: generate({})
 // (op 706) NOP(704) END
 // (op 706) Assign VGPR 2:U64(718) BEGIN
 // (op 718) Assign dim(575) = 2:U64
 // (op 718) Assign stride expression
 // (op 706) Assign VGPR 2:U64(718) END
 // (op 706) LoadTiled Value: Halfx2(148) BEGIN
 // (op 148) GEN: loadMacroTileWAVECIACCUM OP 148 WaveTile 327
 // (op 148) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 148) Generate 16:U64 into nullptr
 // (op 148) Generate 2:U64 into nullptr
 // tag 421: v**UNALLOCATED**
 // (op 148) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Halfx2 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag573: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 148) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Halfx2 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag570: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag573: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 148) 	info.m = 4
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 16
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 8 VGPRs (Value: Halfx2) (op 148): v12, v13, v14, v15, v16, v17, v18, v19
buffer_load_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 148) Load value
// VMEM: Expected complete at 6509 (current 6029)
buffer_load_dwordx2 v[14:15], v3, s[4:7], 0 offen offset:16 // (op 148) Load value
// VMEM: Expected complete at 6510 (current 6030)
buffer_load_dwordx2 v[16:17], v3, s[4:7], 0 offen offset:32 // (op 148) Load value
// VMEM: Expected complete at 6511 (current 6031)
buffer_load_dwordx2 v[18:19], v3, s[4:7], 0 offen offset:48 // (op 148) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6990 (current 6510)
 // (op 706) LoadTiled Value: Halfx2(148) END
 // (op 706) Deallocate{Tensor_10_stride_1_10}(1198) BEGIN
 // (op 1198) Deallocate 571
 // Deleting tag 571
 // (op 1198) Deallocate 573
 // Freeing DataFlowTag573: 4 SGPRs (Buffer: None) (op 713): s4, s5, s6, s7

 // Deleting tag 573
 // (op 1198) Deallocate 575
 // Deleting tag 575
 // (op 1198) Deallocate 570
 // Freeing DataFlowTag570: 1 VGPR (Value: UInt32) (op 711): v3

 // Deleting tag 570
 // (op 1198) Deallocate 574
 // Deleting tag 574
 // (op 1198) Deallocate 5
 // Deleting tag 5
 // (op 1198) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 706) Deallocate{Tensor_10_stride_1_10}(1198) END
 // (op 706) end: generate({703})
 // (op 706) Unlock Scope 706
 // Unlock None
 // (op 358) Scope(706) END
// Allocated : 16 VGPRs (Value: Float) (op 144): v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35
v_accvgpr_read v20, a0 // (op 144) 
 // (op 358) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA(149) BEGIN
 // (op 149) Assign dim(423) = Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA
 // (op 149) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA into nullptr
 // (op 149) reg expression
 // (op 149) reg expression
 // (op 149) Multiply(v9:S, v[12:19]:2xH)S
v_accvgpr_read v21, a1 // (op 144) 
v_accvgpr_read v22, a2 // (op 144) 
v_accvgpr_read v23, a3 // (op 144) 
v_accvgpr_read v24, a4 // (op 144) 
v_accvgpr_read v25, a5 // (op 144) 
v_accvgpr_read v26, a6 // (op 144) 
v_accvgpr_read v27, a7 // (op 144) 
v_accvgpr_read v28, a8 // (op 144) 
v_accvgpr_read v29, a9 // (op 144) 
v_accvgpr_read v30, a10 // (op 144) 
v_accvgpr_read v31, a11 // (op 144) 
v_accvgpr_read v32, a12 // (op 144) 
v_accvgpr_read v33, a13 // (op 144) 
v_accvgpr_read v34, a14 // (op 144) 
v_accvgpr_read v35, a15 // (op 144) 
// Allocated : 16 VGPRs (Value: Float) (op 144): v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51
v_mul_f32 v36, v11, v20 // (op 144) 
v_mul_f32 v37, v11, v21 // (op 144) 
v_mul_f32 v38, v11, v22 // (op 144) 
v_mul_f32 v39, v11, v23 // (op 144) 
v_mul_f32 v40, v11, v24 // (op 144) 
v_mul_f32 v41, v11, v25 // (op 144) 
v_mul_f32 v42, v11, v26 // (op 144) 
v_mul_f32 v43, v11, v27 // (op 144) 
v_mul_f32 v44, v11, v28 // (op 144) 
v_mul_f32 v45, v11, v29 // (op 144) 
v_mul_f32 v46, v11, v30 // (op 144) 
v_mul_f32 v47, v11, v31 // (op 144) 
v_mul_f32 v48, v11, v32 // (op 144) 
v_mul_f32 v49, v11, v33 // (op 144) 
v_mul_f32 v50, v11, v34 // (op 144) 
v_mul_f32 v51, v11, v35 // (op 144) 
 // Freeing : 16 VGPRs (Value: Float) (op 144): v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35

 // (op 358) Assign VGPR Multiply(DataFlowTag(53)NA, DataFlowTag(55)NA)NA(144) END
// Allocated : 2 VGPRs (Value: Float) (op 149): v20, v21
s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 149) 
v_lshrrev_b32 v21, 16, v12 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
// Allocated : 16 VGPRs (Value: Float) (op 149): v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67
v_mul_f32 v52, v9, v20 // (op 149) 
v_mul_f32 v53, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v13 // (op 149) 
v_lshrrev_b32 v21, 16, v13 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v54, v9, v20 // (op 149) 
v_mul_f32 v55, v9, v21 // (op 149) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 149) 
v_lshrrev_b32 v21, 16, v14 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v56, v9, v20 // (op 149) 
v_mul_f32 v57, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v15 // (op 149) 
v_lshrrev_b32 v21, 16, v15 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v58, v9, v20 // (op 149) 
v_mul_f32 v59, v9, v21 // (op 149) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 149) 
v_lshrrev_b32 v21, 16, v16 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v60, v9, v20 // (op 149) 
v_mul_f32 v61, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v17 // (op 149) 
v_lshrrev_b32 v21, 16, v17 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v62, v9, v20 // (op 149) 
v_mul_f32 v63, v9, v21 // (op 149) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 149) 
v_lshrrev_b32 v21, 16, v18 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v64, v9, v20 // (op 149) 
v_mul_f32 v65, v9, v21 // (op 149) 
v_cvt_f32_f16 v20, v19 // (op 149) 
v_lshrrev_b32 v21, 16, v19 // (op 149) 
v_cvt_f32_f16 v21, v21 // (op 149) 
v_mul_f32 v66, v9, v20 // (op 149) 
v_mul_f32 v67, v9, v21 // (op 149) 
 // Freeing : 2 VGPRs (Value: Float) (op 149): v20, v21

 // (op 358) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(421)NA)NA(149) END
 // (op 358) Deallocate{}(1200) BEGIN
 // (op 1200) Deallocate 421
 // Freeing : 8 VGPRs (Value: Halfx2) (op 148): v12, v13, v14, v15, v16, v17, v18, v19

 // Deleting tag 421
 // (op 358) Deallocate{}(1200) END
 // (op 358) Assign VGPR Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA(145) BEGIN
 // (op 145) Assign dim(427) = Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA
 // (op 145) Generate Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA into nullptr
 // (op 145) reg expression
 // (op 145) reg expression
 // (op 145) Add({DataFlowTag423: v[52:67]:S}, {DataFlowTag425: v[36:51]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27
v_add_f32 v12, v52, v36 // (op 145) 
v_add_f32 v13, v53, v37 // (op 145) 
v_add_f32 v14, v54, v38 // (op 145) 
v_add_f32 v15, v55, v39 // (op 145) 
v_add_f32 v16, v56, v40 // (op 145) 
v_add_f32 v17, v57, v41 // (op 145) 
v_add_f32 v18, v58, v42 // (op 145) 
v_add_f32 v19, v59, v43 // (op 145) 
v_add_f32 v20, v60, v44 // (op 145) 
v_add_f32 v21, v61, v45 // (op 145) 
v_add_f32 v22, v62, v46 // (op 145) 
v_add_f32 v23, v63, v47 // (op 145) 
v_add_f32 v24, v64, v48 // (op 145) 
v_add_f32 v25, v65, v49 // (op 145) 
v_add_f32 v26, v66, v50 // (op 145) 
v_add_f32 v27, v67, v51 // (op 145) 
 // (op 358) Assign VGPR Add(DataFlowTag(423)NA, DataFlowTag(425)NA)NA(145) END
 // (op 358) Deallocate{}(1192) BEGIN
 // (op 1192) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 149): v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67

 // Deleting tag 423
 // (op 1192) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 144): v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51

 // Deleting tag 425
 // (op 358) Deallocate{}(1192) END
 // (op 358) Assign VGPR Convert(DataFlowTag(427)NA)H(146) BEGIN
 // (op 146) Assign dim(429) = Convert(DataFlowTag(427)NA)H
 // (op 146) Generate Convert(DataFlowTag(427)NA)H into nullptr
 // (op 146) reg expression
// Allocated DataFlowTag427: 16 VGPRs (Value: Half) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43
v_cvt_f16_f32 v28, v12 // (op 146) 
v_cvt_f16_f32 v29, v13 // (op 146) 
v_cvt_f16_f32 v30, v14 // (op 146) 
v_cvt_f16_f32 v31, v15 // (op 146) 
v_cvt_f16_f32 v32, v16 // (op 146) 
v_cvt_f16_f32 v33, v17 // (op 146) 
v_cvt_f16_f32 v34, v18 // (op 146) 
v_cvt_f16_f32 v35, v19 // (op 146) 
v_cvt_f16_f32 v36, v20 // (op 146) 
v_cvt_f16_f32 v37, v21 // (op 146) 
v_cvt_f16_f32 v38, v22 // (op 146) 
v_cvt_f16_f32 v39, v23 // (op 146) 
v_cvt_f16_f32 v40, v24 // (op 146) 
v_cvt_f16_f32 v41, v25 // (op 146) 
v_cvt_f16_f32 v42, v26 // (op 146) 
v_cvt_f16_f32 v43, v27 // (op 146) 
 // (op 358) Assign VGPR Convert(DataFlowTag(427)NA)H(146) END
 // (op 358) Deallocate{}(1194) BEGIN
 // (op 1194) Deallocate 427
 // Freeing DataFlowTag427: 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 427
 // (op 358) Deallocate{}(1194) END
 // (op 358) NOP(955) BEGIN
 // (op 955) generate({})
 // (op 955) end: generate({})
 // (op 358) NOP(955) END
 // (op 358) NOP(952) BEGIN
 // (op 952) generate({})
 // (op 952) end: generate({})
 // (op 358) NOP(952) END
 // (op 358) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(959) BEGIN
 // (op 959) Assign dim(646) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 646: v**UNALLOCATED**
 // (op 959) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag646: VGPR Value: UInt32 x 1: (unallocated)
 // (op 959) reg expression
 // (op 959) reg expression
 // (op 959) reg expression
 // (op 959) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 959) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 959): v0
v_lshrrev_b32 v0, 6, v1 // (op 959) 
 // (op 959) convert
 // (op 959) BEGIN: Tile[1]
 // (op 959) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 959): v10
v_and_b32 v10, 63, v1 // (op 959) 
 // (op 959) END: Tile[1]
 // (op 959) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 959): v3
v_lshrrev_b32 v3, 5, v10 // (op 959) 
 // (op 959) BEGIN: Flatten
 // (op 959) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 959) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 959)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 959)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 959): v4
v_lshlrev_b32 v4, 2, v3 // (op 959) 
 // (op 959) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 959): v3

 // (op 959) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 959): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 959) 
 // (op 959) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 959): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 959): v0

 // (op 959) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 959): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 959) 
 // (op 959) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 959): v3

 // (op 959) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 959): v3
v_lshrrev_b32 v3, 2, v10 // (op 959) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 959): v10

 // (op 959) BEGIN: Tile[1]
 // (op 959) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 959): v10
v_and_b32 v10, 7, v3 // (op 959) 
 // (op 959) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 959): v3

 // (op 959) BEGIN: Tile[1]
 // (op 959) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 959): v3
v_and_b32 v3, 3, v1 // (op 959) 
 // (op 959) END: Tile[1]
 // (op 959) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 959): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 959) 
 // (op 959) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 959): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 959): v10

 // (op 959) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 959): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 959) 
 // (op 959) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 959): v4

 // (op 959) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 959): v10
v_mul_lo_u32 v10, v3, s30 // (op 959) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 959): v3

 // (op 959) BEGIN: Join
// Allocated DataFlowTag646: 1 VGPR (Value: UInt32) (op 959): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 959) 
 // (op 959) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 959): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 959): v0

 // (op 358) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(959) END
 // (op 358) Assign VGPR 16:U64(960) BEGIN
 // (op 960) Assign dim(647) = 16:U64
 // (op 960) Assign stride expression
 // (op 358) Assign VGPR 16:U64(960) END
 // (op 358) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(961) BEGIN
 // (op 961) Assign dim(643) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 961) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // (op 961) reg expression
 // (op 961) reg expression
 // (op 961) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 961) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 961)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 961)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 961): s14, s15
s_mul_i32 s15, s30, s13 // (op 961) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 961): s0
s_mul_i32 s0, s31, s12 // (op 961) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 961) least significant
// Allocated : 1 SGPR (Value: Int32) (op 961): s1
s_mul_hi_u32 s1, s30, s12 // (op 961) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 961) most significant: sum
s_add_u32 s15, s15, s1 // (op 961) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 961): s1

 // Freeing : 1 SGPR (Value: Int32) (op 961): s0

 // (op 961) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 961) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 961)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 961)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 961): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 961) 
 // Freeing : 2 SGPRs (Value: Int64) (op 961): s14, s15

 // (op 961) reg expression
 // (op 961) reg expression
s_mov_b32 s8, s32 // (op 961) reg expression
s_mov_b32 s9, s33 // (op 961) reg expression
s_mov_b32 s10, s0 // (op 961) reg expression
s_mov_b32 s11, 131072 // (op 961) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 961): s0, s1

 // (op 358) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(961) END
 // (op 358) NOP(953) BEGIN
 // (op 953) generate({})
 // (op 953) end: generate({})
 // (op 358) NOP(953) END
 // (op 358) Assign VGPR 2:U64(966) BEGIN
 // (op 966) Assign dim(649) = 2:U64
 // (op 966) Assign stride expression
 // (op 358) Assign VGPR 2:U64(966) END
 // (op 358) StoreTiled Value: Halfx2(147) BEGIN
 // (op 147) GEN: storeMacroTileWAVE OP 147 MacroTile 429 WaveTile 367
 // (op 147) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 147) Generate 16:U64 into nullptr
 // (op 147) Generate 2:U64 into nullptr
 // (op 147) Convert in LSTGen
 // (op 147) Generate Convert({DataFlowTag427: v[28:43]:H})2xH into nullptr
 // (op 147) reg expression
// Allocated DataFlowTag427: 8 VGPRs (Value: Halfx2) (op 147): v12, v13, v14, v15, v16, v17, v18, v19
v_pack_b32_f16 v12, v28, v29 // (op 147) 
v_pack_b32_f16 v13, v30, v31 // (op 147) 
v_pack_b32_f16 v14, v32, v33 // (op 147) 
v_pack_b32_f16 v15, v34, v35 // (op 147) 
v_pack_b32_f16 v16, v36, v37 // (op 147) 
v_pack_b32_f16 v17, v38, v39 // (op 147) 
v_pack_b32_f16 v18, v40, v41 // (op 147) 
v_pack_b32_f16 v19, v42, v43 // (op 147) 
 // (op 147) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag427: VGPR Value: Halfx2 x 8: v[12:19]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 147) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag427: VGPR Value: Halfx2 x 8: v[12:19]
 // 	info.rowOffsetReg = DataFlowTag646: VGPR Value: UInt32 x 1: v3
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 16
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(64:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 2
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 8
 // 	t.elementBlockStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(8:U64, 2:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 147) 	info.m = 4
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 16
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx2 v[12:13], v3, s[8:11], 0 offen // (op 147) Store value
// VMEM: Expected complete at 7546 (current 7066)
buffer_store_dwordx2 v[14:15], v3, s[8:11], 0 offen offset:16 // (op 147) Store value
// VMEM: Expected complete at 7547 (current 7067)
buffer_store_dwordx2 v[16:17], v3, s[8:11], 0 offen offset:32 // (op 147) Store value
// VMEM: Expected complete at 7548 (current 7068)
buffer_store_dwordx2 v[18:19], v3, s[8:11], 0 offen offset:48 // (op 147) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 8027 (current 7547)
 // Freeing DataFlowTag427: 8 VGPRs (Value: Halfx2) (op 147): v12, v13, v14, v15, v16, v17, v18, v19

 // (op 358) StoreTiled Value: Halfx2(147) END
 // (op 358) Deallocate{}(1196) BEGIN
 // (op 1196) Deallocate 649
 // Deleting tag 649
 // (op 1196) Deallocate 648
 // Deleting tag 648
 // (op 1196) Deallocate 647
 // Deleting tag 647
 // (op 1196) Deallocate 646
 // Freeing DataFlowTag646: 1 VGPR (Value: UInt32) (op 959): v3

 // Deleting tag 646
 // (op 1196) Deallocate 429
 // Freeing DataFlowTag427: 16 VGPRs (Value: Half) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 429
 // (op 358) Deallocate{}(1196) END
 // (op 358) end: generate({361})
 // (op 358) Unlock Scope 358
 // Unlock None
 // (op 135) Scope(358) END
 // (op 135) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1270) BEGIN
 // (op 1270) Deallocate 643
 // Freeing DataFlowTag643: 4 SGPRs (Buffer: None) (op 944): s8, s9, s10, s11

 // Deleting tag 643
 // (op 1270) Deallocate 445
 // Deleting tag 445
 // (op 1270) Deallocate 443
 // Deleting tag 443
 // (op 1270) Deallocate 366
 // Deleting tag 366
 // (op 1270) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1270) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 135) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1270) END
 // (op 135) end: generate({358})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 135) 
RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135:
 // (op 135) 
 // (op 135) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(135) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1190) BEGIN
 // (op 1190) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 2): v9

 // Deleting tag 44
 // (op 1190) Deallocate 53
 // Freeing : 1 VGPR (Value: Float) (op 43): v11

 // Deleting tag 53
 // (op 1190) Deallocate 55
 // Freeing DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 65): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 55
 // (op 1190) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1190) END
 // (op 1) end: generate({765, 818, 871, 924, 982, 1006, 1030, 1054})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 5376 // lds bytes
  .amdhsa_accum_offset 68
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F6
    DestDataType: h
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
  - .name: RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_FP4_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 5376
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 68
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

