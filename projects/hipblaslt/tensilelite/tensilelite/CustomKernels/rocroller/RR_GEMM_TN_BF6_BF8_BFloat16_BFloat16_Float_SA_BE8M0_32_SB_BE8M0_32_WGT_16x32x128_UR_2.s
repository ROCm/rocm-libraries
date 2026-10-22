// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.BF6.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.BF8.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
 // T_LOAD_TILED 3 Source 2
 // Tensor.E8M0.d2 4, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
 // T_LOAD_TILED 5 Source 4
 // BlockScale(Separate, {1, 32}): Data: 1, Scale: 5
 // Tensor.E8M0.d2 7, (base=&120, sizes={&128 &136 }, strides={&144 &152 })
 // T_LOAD_TILED 8 Source 7
 // BlockScale(Separate, {32, 1}): Data: 3, Scale: 8
 // Tensor.BFloat16.d2 10, (base=&160, sizes={&168 &176 }, strides={&184 &192 })
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
 // Tensor.BFloat16.d2 21, (base=&208, sizes={&216 &224 }, strides={&232 &240 })
 // T_EXECUTE 19
 //   E_Cvt 22, 19
 // T_STORE_TILED 24 Source 22 Dest 21

 // Command arguments:
 // Tensor_0_pointer: PointerGlobal: BF6(offset: 0, size: 8, read_write)
 // Tensor_0_size_0: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: BF8(offset: 40, size: 8, read_write)
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
 // Tensor_10_pointer: PointerGlobal: BFloat16(offset: 160, size: 8, read_write)
 // Tensor_10_size_0: Value: Int64(offset: 168, size: 8, read_only)
 // Tensor_10_size_1: Value: Int64(offset: 176, size: 8, read_only)
 // Tensor_10_stride_0: Value: Int64(offset: 184, size: 8, read_only)
 // Tensor_10_stride_1: Value: Int64(offset: 192, size: 8, read_only)
 // user_Float_Value_12: Value: Float(offset: 200, size: 4, read_only)
 // user_Float_Value_14: Value: Float(offset: 204, size: 4, read_only)
 // Tensor_21_pointer: PointerGlobal: BFloat16(offset: 208, size: 8, read_write)
 // Tensor_21_size_0: Value: Int64(offset: 216, size: 8, read_only)
 // Tensor_21_size_1: Value: Int64(offset: 224, size: 8, read_only)
 // Tensor_21_stride_0: Value: Int64(offset: 232, size: 8, read_only)
 // Tensor_21_stride_1: Value: Int64(offset: 240, size: 8, read_only)

 // Kernel Arguments:
 // KernelArg{Tensor_0_stride_0_0, Value: Int64, CommandArgument(Tensor_0_stride_0)I64(c 0), o:0, s:8, preloaded}
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF6, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: BFloat16, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: BFloat16, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}
.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
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
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:
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
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF6, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
 // KernelArg{Tensor_2_size_1_4, Value: Int64, CommandArgument(Tensor_2_size_1)I64(c 0), o:40, s:8, preloaded}
 // KernelArg{user_Float_Value_14, Value: Float, CommandArgument(user_Float_Value_14)S(c 0), o:48, s:4, preloaded}
 // KernelArg{user_Float_Value_12, Value: Float, CommandArgument(user_Float_Value_12)S(c 0), o:52, s:4, preloaded}
 // KernelArg{Tensor_4_stride_0_6, Value: Int64, CommandArgument(Tensor_4_stride_0)I64(c 0), o:56, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_4_pointer)PG(c 0), o:64, s:8}
 // KernelArg{Tensor_7_stride_1_8, Value: Int64, CommandArgument(Tensor_7_stride_1)I64(c 0), o:72, s:8}
 // KernelArg{Tensor_7_pointer, PointerGlobal: E8M0, read_write, CommandArgument(Tensor_7_pointer)PG(c 0), o:80, s:8}
 // KernelArg{Tensor_0_size_1_14, Value: Int64, CommandArgument(Tensor_0_size_1)I64(c 0), o:88, s:8}
 // KernelArg{Tensor_21_stride_1_16, Value: Int64, CommandArgument(Tensor_21_stride_1)I64(c 0), o:96, s:8}
 // KernelArg{Tensor_21_pointer, PointerGlobal: BFloat16, read_write, CommandArgument(Tensor_21_pointer)PG(c 0), o:104, s:8}
 // KernelArg{Tensor_10_stride_1_10, Value: Int64, CommandArgument(Tensor_10_stride_1)I64(c 0), o:112, s:8}
 // KernelArg{Tensor_10_pointer, PointerGlobal: BFloat16, read_write, CommandArgument(Tensor_10_pointer)PG(c 0), o:120, s:8}

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
 // Tensor_0_pointer: SGPR PointerGlobal: BF6 x 1: s[4:5]
 // Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]
 // Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]
 // Tensor_2_pointer: SGPR PointerGlobal: BF8 x 1: s[10:11]
 // Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13]
 // user_Float_Value_14: SGPR Value: Float x 1: s14
 // user_Float_Value_12: SGPR Value: Float x 1: s15
 // Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]
 // Tensor_4_pointer: SGPR PointerGlobal: E8M0 x 1: s[22:23]
 // Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]
 // Tensor_7_pointer: SGPR PointerGlobal: E8M0 x 1: s[26:27]
 // Tensor_0_size_1_14: SGPR Value: Int64 x 1: s[28:29]
 // Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]
 // Tensor_21_pointer: SGPR PointerGlobal: BFloat16 x 1: s[32:33]
 // Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]
 // Tensor_10_pointer: SGPR PointerGlobal: BFloat16 x 1: s[36:37]
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
 // (op 1) generate({767, 819, 872, 930, 969, 993})
 // (op 1) NOP(767) BEGIN
 // (op 767) generate({})
 // (op 767) end: generate({})
 // (op 1) NOP(767) END
 // (op 1) NOP(819) BEGIN
 // (op 819) generate({})
 // (op 819) end: generate({})
 // (op 1) NOP(819) END
 // (op 1) NOP(872) BEGIN
 // (op 872) generate({})
 // (op 872) end: generate({})
 // (op 1) NOP(872) END
 // (op 1) NOP(930) BEGIN
 // (op 930) generate({})
 // (op 930) end: generate({})
 // (op 1) NOP(930) END
 // (op 1) NOP(969) BEGIN
 // (op 969) generate({})
 // (op 969) end: generate({})
 // (op 1) NOP(969) END
 // (op 1) NOP(993) BEGIN
 // (op 993) generate({})
 // (op 993) end: generate({})
 // (op 1) NOP(993) END
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(771) BEGIN
 // (op 771) Assign dim(548) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 548: v**UNALLOCATED**
 // (op 771) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag548: VGPR Value: UInt32 x 1: (unallocated)
 // (op 771) reg expression
 // (op 771) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(823) BEGIN
 // (op 823) Assign dim(564) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32}
 // tag 564: v**UNALLOCATED**
 // (op 823) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 823) reg expression
 // (op 823) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32}(876) BEGIN
 // (op 876) Assign dim(580) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 580: v**UNALLOCATED**
 // (op 876) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag580: VGPR Value: UInt32 x 1: (unallocated)
 // (op 876) reg expression
 // (op 876) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(934) BEGIN
 // (op 934) Assign dim(598) = LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 598: v**UNALLOCATED**
 // (op 934) Generate LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag598: VGPR Value: UInt32 x 1: (unallocated)
 // (op 934) reg expression
 // (op 934) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(973) BEGIN
 // (op 973) Assign dim(610) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32}
 // tag 610: v**UNALLOCATED**
 // (op 973) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32} into DataFlowTag610: VGPR Value: UInt32 x 1: (unallocated)
 // (op 973) reg expression
 // (op 973) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32}(997) BEGIN
 // (op 997) Assign dim(618) = {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 618: v**UNALLOCATED**
 // (op 997) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag618: VGPR Value: UInt32 x 1: (unallocated)
 // (op 997) reg expression
 // (op 997) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 771): v0
v_lshrrev_b32 v0, 7, v1 // (op 771) 
 // (op 771) BEGIN: Flatten[1]Tile
 // (op 771) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 771): v3
v_and_b32 v3, 15, v1 // (op 771) 
 // (op 771) END: Flatten[1]Tile
 // (op 771) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 771): v4
v_lshl_add_u32 v4, v0, 4, v3 // (op 771) 
 // (op 771) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 771): v3

 // Freeing : 1 VGPR (Value: UInt32) (op 771): v0

 // (op 771) BEGIN: Flatten[1]
 // (op 771) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 771): v0
v_and_b32 v0, 63, v1 // (op 771) 
 // (op 771) END: Flatten[1]
 // (op 771) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 771): v3
v_lshrrev_b32 v3, 4, v0 // (op 771) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 771): v0

 // (op 771) BEGIN: TileTile
 // (op 771) {TileTile: ShiftL(v3:U32, 5:U32)U32}
 // (op 771) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 771)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 771)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 771): v0
v_lshlrev_b32 v0, 5, v3 // (op 771) 
 // (op 771) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 771): v3

 // (op 771) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 771): v3
v_lshl_add_u32 v3, v4, 7, v0 // (op 771) 
 // (op 771) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 771): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 771): v4

 // (op 771) Multiply({Tile: v3:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 771): v0
v_mul_lo_u32 v0, v3, 6 // (op 771) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 771): v3

 // (op 771) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag548: 1 VGPR (Value: UInt32) (op 771): v3
v_lshrrev_b32 v3, 3, v0 // (op 771) 
 // Freeing : 1 VGPR (Value: UInt32) (op 771): v0

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(771) END
// Allocated : 1 VGPR (Value: UInt32) (op 823): v0
v_lshrrev_b32 v0, 6, v1 // (op 823) 
 // (op 823) BEGIN: Tile
 // (op 823) {Tile: BitwiseAnd(v0:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 823): v4
v_and_b32 v4, 1, v0 // (op 823) 
 // (op 823) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 823): v0

 // (op 823) BEGIN: Flatten[1]Tile
 // (op 823) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 823): v0
v_and_b32 v0, 15, v1 // (op 823) 
 // (op 823) END: Flatten[1]Tile
 // (op 823) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 823): v5
v_lshl_add_u32 v5, v4, 4, v0 // (op 823) 
 // (op 823) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 823): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 823): v4

 // (op 823) BEGIN: TileTile
 // (op 823) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 48:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 823): v0
v_and_b32 v0, 48, v1 // (op 823) 
 // (op 823) END: TileTile
 // (op 823) BEGIN: Tile
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 823): v4
v_lshl_add_u32 v4, v5, 7, v0 // (op 823) 
 // (op 823) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 823): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 823): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(823) END
// Allocated : 1 VGPR (Value: UInt32) (op 876): v0
v_lshrrev_b32 v0, 6, v1 // (op 876) 
 // (op 876) BEGIN: Tile
 // (op 876) {Tile: BitwiseAnd(v0:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 876): v5
v_and_b32 v5, 1, v0 // (op 876) 
 // (op 876) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 876): v0

 // (op 876) BEGIN: Flatten[1]
 // (op 876) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 876): v0
v_and_b32 v0, 15, v1 // (op 876) 
 // (op 876) END: Flatten[1]
 // (op 876) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 876): v6
v_lshl_add_u32 v6, v5, 4, v0 // (op 876) 
 // (op 876) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 876): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 876): v5

 // (op 876) BEGIN: Flatten[1]
 // (op 876) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 876): v0
v_and_b32 v0, 63, v1 // (op 876) 
 // (op 876) END: Flatten[1]
 // (op 876) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 876): v5
v_lshrrev_b32 v5, 4, v0 // (op 876) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 876): v0

 // (op 876) BEGIN: Tile
// Allocated DataFlowTag580: 1 VGPR (Value: UInt32) (op 876): v0
v_lshl_add_u32 v0, v6, 2, v5 // (op 876) 
 // (op 876) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 876): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 876): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32}(876) END
// Allocated : 1 VGPR (Value: UInt32) (op 934): v5
v_lshrrev_b32 v5, 7, v1 // (op 934) 
 // (op 934) BEGIN: Flatten[1]Tile
 // (op 934) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 934): v6
v_and_b32 v6, 15, v1 // (op 934) 
 // (op 934) END: Flatten[1]Tile
 // (op 934) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 934): v7
v_lshl_add_u32 v7, v5, 4, v6 // (op 934) 
 // (op 934) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 934): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 934): v5

 // (op 934) BEGIN: Flatten[1]
 // (op 934) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 934): v5
v_and_b32 v5, 63, v1 // (op 934) 
 // (op 934) END: Flatten[1]
 // (op 934) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 934): v6
v_lshrrev_b32 v6, 4, v5 // (op 934) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 934): v5

 // (op 934) BEGIN: TileTile
 // (op 934) {TileTile: ShiftL(v6:U32, 5:U32)U32}
 // (op 934) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 934)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 934)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 934): v5
v_lshlrev_b32 v5, 5, v6 // (op 934) 
 // (op 934) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 934): v6

 // (op 934) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 934): v6
v_lshl_add_u32 v6, v7, 7, v5 // (op 934) 
 // (op 934) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 934): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 934): v7

 // (op 934) Multiply({Tile: v6:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 934): v5
v_mul_lo_u32 v5, v6, 6 // (op 934) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 934): v6

 // (op 934) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag598: 1 VGPR (Value: UInt32) (op 934): v6
v_lshrrev_b32 v6, 3, v5 // (op 934) 
 // Freeing : 1 VGPR (Value: UInt32) (op 934): v5

 // (op 1) Assign VGPR LogicalShiftR(Multiply({Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(100)U32, 7:U32)U32, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(100)U32, 15:U32)U32})U32}, 7:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(100)U32, 63:U32)U32}, 4:U32)U32, 5:U32)U32})U32}, 6:U32)U32, 3:U32)U32(934) END
// Allocated : 1 VGPR (Value: UInt32) (op 973): v5
v_lshrrev_b32 v5, 6, v1 // (op 973) 
 // (op 973) BEGIN: Tile
 // (op 973) {Tile: BitwiseAnd(v5:U32, 1:U32)U32}
 // (op 1) Deallocate{}(1188) BEGIN
 // (op 1188) Deallocate 100
 // Deleting tag 100
 // (op 1) Deallocate{}(1188) END
// Allocated : 1 VGPR (Value: UInt32) (op 973): v7
v_and_b32 v7, 1, v5 // (op 973) 
 // (op 973) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 973): v5

 // (op 973) BEGIN: Flatten[1]Tile
 // (op 973) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR 96:U32(772) BEGIN
 // (op 772) Assign dim(549) = 96:U32
 // (op 772) Assign stride expression
 // (op 1) Assign VGPR 96:U32(772) END
 // (op 1) Assign VGPR 96:U32(935) BEGIN
 // (op 935) Assign dim(599) = 96:U32
 // (op 935) Assign stride expression
 // (op 1) Assign VGPR 96:U32(935) END
 // (op 1) NOP(768) BEGIN
 // (op 768) generate({})
 // (op 768) end: generate({})
 // (op 1) NOP(768) END
 // (op 1) NOP(931) BEGIN
 // (op 931) generate({})
 // (op 931) end: generate({})
 // (op 1) NOP(931) END
 // (op 1) Assign VGPR 0:U32(776) BEGIN
 // (op 776) Assign dim(551) = 0:U32
 // (op 776) Assign stride expression
 // (op 1) Assign VGPR 0:U32(776) END
 // (op 1) Assign VGPR 0:U32(939) BEGIN
 // (op 939) Assign dim(601) = 0:U32
 // (op 939) Assign stride expression
 // (op 1) Assign VGPR 0:U32(939) END
// Allocated : 1 VGPR (Value: UInt32) (op 973): v5
v_and_b32 v5, 15, v1 // (op 973) 
 // (op 973) END: Flatten[1]Tile
 // (op 973) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 973): v8
v_lshl_add_u32 v8, v7, 4, v5 // (op 973) 
 // (op 973) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 973): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 973): v7

 // (op 973) BEGIN: TileTile
 // (op 973) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 48:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 973): v5
v_and_b32 v5, 48, v1 // (op 973) 
 // (op 973) END: TileTile
 // (op 973) BEGIN: Tile
// Allocated DataFlowTag610: 1 VGPR (Value: UInt32) (op 973): v7
v_lshl_add_u32 v7, v8, 7, v5 // (op 973) 
 // (op 973) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 973): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 973): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(166)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]Tile: BitwiseAnd(DataFlowTag(166)U32, 15:U32)U32})U32}, 7:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(166)U32, 63:U32)U32}, 4294967280:U32)U32})U32}(973) END
// Allocated : 1 VGPR (Value: UInt32) (op 997): v5
v_lshrrev_b32 v5, 6, v1 // (op 997) 
 // (op 997) BEGIN: Tile
 // (op 997) {Tile: BitwiseAnd(v5:U32, 1:U32)U32}
 // (op 1) Deallocate{}(1191) BEGIN
 // (op 1191) Deallocate 166
 // Deleting tag 166
 // (op 1) Deallocate{}(1191) END
// Allocated : 1 VGPR (Value: UInt32) (op 997): v8
v_and_b32 v8, 1, v5 // (op 997) 
 // (op 997) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 997): v5

 // (op 997) BEGIN: Flatten[1]
 // (op 997) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 1) Assign VGPR 128:U32(824) BEGIN
 // (op 824) Assign dim(565) = 128:U32
 // (op 824) Assign stride expression
 // (op 1) Assign VGPR 128:U32(824) END
 // (op 1) Assign VGPR 128:U32(974) BEGIN
 // (op 974) Assign dim(611) = 128:U32
 // (op 974) Assign stride expression
 // (op 1) Assign VGPR 128:U32(974) END
 // (op 1) NOP(820) BEGIN
 // (op 820) generate({})
 // (op 820) end: generate({})
 // (op 1) NOP(820) END
 // (op 1) NOP(970) BEGIN
 // (op 970) generate({})
 // (op 970) end: generate({})
 // (op 1) NOP(970) END
 // (op 1) Assign VGPR 1:U32(828) BEGIN
 // (op 828) Assign dim(567) = 1:U32
 // (op 828) Assign stride expression
 // (op 1) Assign VGPR 1:U32(828) END
 // (op 1) Assign VGPR 1:U32(978) BEGIN
 // (op 978) Assign dim(613) = 1:U32
 // (op 978) Assign stride expression
 // (op 1) Assign VGPR 1:U32(978) END
// Allocated : 1 VGPR (Value: UInt32) (op 997): v5
v_and_b32 v5, 15, v1 // (op 997) 
 // (op 997) END: Flatten[1]
 // (op 997) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 997): v9
v_lshl_add_u32 v9, v8, 4, v5 // (op 997) 
 // (op 997) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 997): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 997): v8

 // (op 997) BEGIN: Flatten[1]
 // (op 997) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 997): v5
v_and_b32 v5, 63, v1 // (op 997) 
 // (op 997) END: Flatten[1]
 // (op 997) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 997): v8
v_lshrrev_b32 v8, 4, v5 // (op 997) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 997): v5

 // (op 997) BEGIN: Tile
// Allocated DataFlowTag618: 1 VGPR (Value: UInt32) (op 997): v5
v_lshl_add_u32 v5, v9, 2, v8 // (op 997) 
 // (op 997) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 997): v8

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 997): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(262)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 15:U32)U32})U32}, 2:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(262)U32, 63:U32)U32}, 4:U32)U32)U32}(997) END
 // (op 1) Deallocate{}(1194) BEGIN
 // (op 1194) Deallocate 262
 // Deleting tag 262
 // (op 1) Deallocate{}(1194) END
 // (op 1) Assign VGPR 4:U32(877) BEGIN
 // (op 877) Assign dim(581) = 4:U32
 // (op 877) Assign stride expression
 // (op 1) Assign VGPR 4:U32(877) END
 // (op 1) Assign VGPR 4:U32(998) BEGIN
 // (op 998) Assign dim(619) = 4:U32
 // (op 998) Assign stride expression
 // (op 1) Assign VGPR 4:U32(998) END
 // (op 1) NOP(873) BEGIN
 // (op 873) generate({})
 // (op 873) end: generate({})
 // (op 1) NOP(873) END
 // (op 1) NOP(994) BEGIN
 // (op 994) generate({})
 // (op 994) end: generate({})
 // (op 1) NOP(994) END
 // (op 1) Assign VGPR 1:U32(881) BEGIN
 // (op 881) Assign dim(583) = 1:U32
 // (op 881) Assign stride expression
 // (op 1) Assign VGPR 1:U32(881) END
 // (op 1) Assign VGPR 1:U32(1002) BEGIN
 // (op 1002) Assign dim(621) = 1:U32
 // (op 1002) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1002) END
 // (op 1) LoadVGPR Value: Float(2) BEGIN
 // tag 35: v**UNALLOCATED**
 // (op 2) GEN: LoadVGPR; scalar value
 // (op 1) LoadVGPR Value: Float(39) BEGIN
 // tag 44: v**UNALLOCATED**
 // (op 39) GEN: LoadVGPR; scalar value
// Allocated : 1 VGPR (Value: Float) (op 2): v8
v_mov_b32 v8, s14 // (op 2) Move value
 // (op 1) LoadVGPR Value: Float(2) END
// Allocated : 1 VGPR (Value: Float) (op 39): v9
v_mov_b32 v9, s15 // (op 39) Move value
 // (op 1) LoadVGPR Value: Float(39) END
 // (op 1) Deallocate{user_Float_Value_14}(1236) BEGIN
 // (op 1236) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1236) END
 // (op 1) Deallocate{user_Float_Value_12}(1238) BEGIN
 // (op 1238) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1238) END
 // (op 1) Scope(314) BEGIN
 // (op 314) Lock Scope 314
 // Lock Branch
 // (op 314) generate({319})
 // (op 314) NOP(319) BEGIN
 // (op 319) generate({})
 // (op 319) end: generate({})
 // (op 314) NOP(319) END
 // (op 314) Assign ACCVGPR 0.00000:S(61) BEGIN
 // (op 61) Assign dim(51) = 0.00000:S
 // tag 51: a**UNALLOCATED**
 // (op 61) Generate 0.00000:S into DataFlowTag51: ACCVGPR Value: Float x 4: (unallocated)
// Allocated DataFlowTag51: 4 ACCVGPRs (Value: Float) (op 61): a0, a1, a2, a3
v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
 // (op 314) Assign ACCVGPR 0.00000:S(61) END
 // (op 314) Scope(473) BEGIN
 // (op 473) Lock Scope 473
 // Lock Branch
 // (op 473) generate({468})
 // (op 473) NOP(468) BEGIN
 // (op 468) generate({})
 // (op 468) end: generate({})
 // (op 473) NOP(468) END
 // (op 473) Assign VGPR 96:U64(478) BEGIN
 // (op 478) Assign dim(462) = 96:U64
 // (op 478) Assign stride expression
 // (op 473) Assign VGPR 96:U64(478) END
 // (op 473) NOP(469) BEGIN
 // (op 469) generate({})
 // (op 469) end: generate({})
 // (op 473) NOP(469) END
 // (op 473) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(481) BEGIN
 // (op 481) Assign dim(463) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 463: v**UNALLOCATED**
 // (op 481) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag463: VGPR Value: UInt32 x 1: (unallocated)
 // (op 481) reg expression
 // (op 481) reg expression
 // (op 481) reg expression
 // (op 481) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 481): v10
v_lshrrev_b32 v10, 3, v1 // (op 481) 
 // (op 481) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 481): v11
v_lshl_add_u32 v11, s16, 4, v10 // (op 481) 
 // (op 481) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 481): v10

 // (op 481) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 481): v10
v_and_b32 v10, -8, v1 // (op 481) 
 // (op 481) BEGIN: Flatten[1]
 // (op 481) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 481): v12
v_sub_u32 v12, v1, v10 // (op 481) 
 // (op 481) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 481): v10

 // (op 481) BEGIN: TileTile
 // (op 481) {TileTile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 481) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 481)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 481)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 481): v10
v_lshlrev_b32 v10, 4, v12 // (op 481) 
 // (op 481) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 481): v12

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 481): v12, v13
v_mov_b32 v13, 0 // (op 481) convert
v_mov_b32 v12, v10 // (op 481) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 481): v10

 // (op 481) BEGIN: Split
 // (op 481) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 481)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 481)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 481): v14, v15
v_mul_lo_u32 v15, v11, s3 // (op 481) most significant: low of low * high
 // (op 481) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 481): v10
v_mul_hi_u32 v10, v11, s2 // (op 481) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 481) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 481) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 481): v10

 // (op 481) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 481)      lhs (VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 481)      rhs (TileTile: VGPR Value: UInt64 x 1: v[12:13])
// Allocated : 2 SGPRs (Value: Bool64) (op 481): s0, s1
v_add_co_u32 v14, s[0:1], v14, v12 // (op 481) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 481) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 481): s0, s1

 // (op 481) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 481): v12, v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 481): v11

 // (op 481) Multiply({Split: v[14:15]:U64}, 6:U32)U64
 // (op 481) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 481)           lhs (Split: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 481)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 481) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 481): v10, v11
v_mul_lo_u32 v11, v15, 6 // (op 481) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 481): v12
v_mul_hi_u32 v12, v14, 6 // (op 481) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 481) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 481) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 481): v12

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 481): v14, v15

 // (op 481) LogicalShiftR(v[10:11]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 481): v12, v13
v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 481) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 481): v10, v11

// Allocated DataFlowTag463: 1 VGPR (Value: UInt32) (op 481): v10
v_mov_b32 v10, v12 // (op 481) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 481): v12, v13

 // (op 473) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(481) END
 // (op 473) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(482) BEGIN
 // (op 482) Assign dim(464) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 482) Assign stride expression
 // (op 473) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(482) END
 // (op 473) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(483) BEGIN
 // (op 483) Assign dim(457) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 457: s**UNALLOCATED**
 // (op 483) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag457: SGPR Buffer: None x 1: (unallocated)
 // (op 483) reg expression
 // (op 483) reg expression
 // (op 483) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 483) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 483)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 483)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 483) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 483): s0, s1
s_mul_i32 s1, s7, 6 // (op 483) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 483) least significant
// Allocated : 1 SGPR (Value: Int32) (op 483): s14
s_mul_hi_u32 s14, s6, 6 // (op 483) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 483) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 483): s14

 // (op 483) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 483) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 483)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 483)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 483): s14, s15
s_mul_i32 s15, s2, s1 // (op 483) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 483): s18
s_mul_i32 s18, s3, s0 // (op 483) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 483) least significant
// Allocated : 1 SGPR (Value: Int32) (op 483): s19
s_mul_hi_u32 s19, s2, s0 // (op 483) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 483) most significant: sum
s_add_u32 s15, s15, s19 // (op 483) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 483): s19

 // Freeing : 1 SGPR (Value: Int32) (op 483): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 483): s0, s1

 // (op 483) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 483): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 483) 
 // Freeing : 2 SGPRs (Value: Int64) (op 483): s14, s15

 // (op 483) reg expression
 // (op 483) reg expression
// Allocated DataFlowTag457: 4 SGPRs (Buffer: None) (op 483): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 483) reg expression
s_mov_b32 s41, s5 // (op 483) reg expression
s_mov_b32 s42, s0 // (op 483) reg expression
s_mov_b32 s43, 131072 // (op 483) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 483): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(483) END
 // (op 473) NOP(470) BEGIN
 // (op 470) generate({})
 // (op 470) end: generate({})
 // (op 473) NOP(470) END
 // (op 473) Assign VGPR 0:U64(488) BEGIN
 // (op 488) Assign dim(466) = 0:U64
 // (op 488) Assign stride expression
 // (op 473) Assign VGPR 0:U64(488) END
 // (op 473) NOP(491) BEGIN
 // (op 491) generate({})
 // (op 491) end: generate({})
 // (op 473) NOP(491) END
 // (op 473) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(500) BEGIN
 // (op 500) Assign dim(467) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 467: v**UNALLOCATED**
 // (op 500) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag467: VGPR Value: UInt32 x 1: (unallocated)
 // (op 500) reg expression
 // (op 500) reg expression
 // (op 500) reg expression
 // (op 500) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 500): v11
v_lshrrev_b32 v11, 3, v1 // (op 500) 
 // (op 500) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 500): v12
v_lshl_add_u32 v12, s16, 4, v11 // (op 500) 
 // (op 500) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 500): v11

 // (op 500) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 500): v11
v_and_b32 v11, -8, v1 // (op 500) 
 // (op 500) BEGIN: Flatten[1]
 // (op 500) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 500): v13
v_sub_u32 v13, v1, v11 // (op 500) 
 // (op 500) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 500): v11

 // (op 500) BEGIN: TileTile
 // (op 500) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 500) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 500)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 500)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 500): v11
v_lshlrev_b32 v11, 4, v13 // (op 500) 
 // (op 500) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 500): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 500): v14, v15
v_mov_b32 v15, 0 // (op 500) convert
v_mov_b32 v14, v11 // (op 500) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 500): v11

 // (op 500) BEGIN: Split
 // (op 500) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 500)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 500)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 500): v16, v17
v_mul_lo_u32 v17, v12, s3 // (op 500) most significant: low of low * high
 // (op 500) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 500): v11
v_mul_hi_u32 v11, v12, s2 // (op 500) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 500) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 500) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 500): v11

 // (op 500) Add: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 500)      lhs (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 500)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 500): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 500) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 500) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 500): s0, s1

 // (op 500) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 500): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 500): v12

 // (op 500) Multiply({Split: v[16:17]:U64}, 6:U32)U64
 // (op 500) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 500)           lhs (Split: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 500)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 500) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 500): v14, v15
v_mul_lo_u32 v15, v17, 6 // (op 500) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 500): v11
v_mul_hi_u32 v11, v16, 6 // (op 500) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 500) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 500) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 500): v11

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 500): v16, v17

 // (op 500) LogicalShiftR(v[14:15]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 500): v12, v13
v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 500) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 500): v14, v15

// Allocated DataFlowTag467: 1 VGPR (Value: UInt32) (op 500): v11
v_mov_b32 v11, v12 // (op 500) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 500): v12, v13

 // (op 473) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(500) END
 // (op 473) Assign VGPR 96:U64(501) BEGIN
 // (op 501) Assign dim(468) = 96:U64
 // (op 501) Assign stride expression
 // (op 473) Assign VGPR 96:U64(501) END
 // (op 473) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(502) BEGIN
 // (op 502) Assign dim(457) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 502) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // (op 502) reg expression
 // (op 502) reg expression
 // (op 502) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 502) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 502)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 502)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 502) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 502): s0, s1
s_mul_i32 s1, s7, 6 // (op 502) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 502) least significant
// Allocated : 1 SGPR (Value: Int32) (op 502): s14
s_mul_hi_u32 s14, s6, 6 // (op 502) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 502) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 502): s14

 // (op 502) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 502) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 502)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 502)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 502): s14, s15
s_mul_i32 s15, s2, s1 // (op 502) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 502): s18
s_mul_i32 s18, s3, s0 // (op 502) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 502) least significant
// Allocated : 1 SGPR (Value: Int32) (op 502): s19
s_mul_hi_u32 s19, s2, s0 // (op 502) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 502) most significant: sum
s_add_u32 s15, s15, s19 // (op 502) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 502): s19

 // Freeing : 1 SGPR (Value: Int32) (op 502): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 502): s0, s1

 // (op 502) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 502): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 502) 
 // Freeing : 2 SGPRs (Value: Int64) (op 502): s14, s15

 // (op 502) reg expression
 // (op 502) reg expression
s_mov_b32 s40, s4 // (op 502) reg expression
s_mov_b32 s41, s5 // (op 502) reg expression
s_mov_b32 s42, s0 // (op 502) reg expression
s_mov_b32 s43, 131072 // (op 502) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 502): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(502) END
 // (op 473) NOP(493) BEGIN
 // (op 493) generate({})
 // (op 493) end: generate({})
 // (op 473) NOP(493) END
 // (op 473) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(507) BEGIN
 // (op 507) Assign dim(470) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 507) Assign stride expression
 // (op 473) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(507) END
 // (op 473) NOP(494) BEGIN
 // (op 494) generate({})
 // (op 494) end: generate({})
 // (op 473) NOP(494) END
 // (op 473) Assign VGPR 0:U64(510) BEGIN
 // (op 510) Assign dim(472) = 0:U64
 // (op 510) Assign stride expression
 // (op 473) Assign VGPR 0:U64(510) END
 // (op 473) NOP(535) BEGIN
 // (op 535) generate({})
 // (op 535) end: generate({})
 // (op 473) NOP(535) END
 // (op 473) Assign VGPR 128:U64(542) BEGIN
 // (op 542) Assign dim(481) = 128:U64
 // (op 542) Assign stride expression
 // (op 473) Assign VGPR 128:U64(542) END
 // (op 473) NOP(536) BEGIN
 // (op 536) generate({})
 // (op 536) end: generate({})
 // (op 473) NOP(536) END
 // (op 473) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(545) BEGIN
 // (op 545) Assign dim(482) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 482: v**UNALLOCATED**
 // (op 545) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag482: VGPR Value: UInt64 x 1: (unallocated)
 // (op 545) reg expression
 // (op 545) reg expression
 // (op 545) reg expression
 // (op 545) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 545): v12
v_and_b32 v12, -8, v1 // (op 545) 
 // (op 545) BEGIN: Flatten[1]
 // (op 545) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 545): v13
v_sub_u32 v13, v1, v12 // (op 545) 
 // (op 545) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 545): v12

 // (op 545) BEGIN: TileTile
 // (op 545) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 545) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 545)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 545)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 545): v12
v_lshlrev_b32 v12, 4, v13 // (op 545) 
 // (op 545) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 545): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 545): v14, v15
v_mov_b32 v15, 0 // (op 545) convert
v_mov_b32 v14, v12 // (op 545) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 545): v12

 // (op 545) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 545): v12
v_lshrrev_b32 v12, 3, v1 // (op 545) 
 // (op 545) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 545): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 545) 
 // (op 545) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 545): v12

 // (op 545) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 545) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 545)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 545)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 545): v16, v17
v_mul_lo_u32 v17, v13, s9 // (op 545) most significant: low of low * high
 // (op 545) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 545): v12
v_mul_hi_u32 v12, v13, s8 // (op 545) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 545) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 545) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 545): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 545): v13

 // (op 545) BEGIN: Split
 // (op 545) {Split: Add({TileTile: v[14:15]:U64}, v[16:17]:I64)U64}
 // (op 545) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 545)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 545)      rhs (VGPR Value: Int64 x 1: v[16:17])
// Allocated DataFlowTag482: 2 VGPRs (Value: UInt64) (op 545): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 545): s0, s1
v_add_co_u32 v12, s[0:1], v14, v16 // (op 545) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 545) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 545): s0, s1

 // (op 545) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 545): v16, v17

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 545): v14, v15

 // (op 473) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(545) END
 // (op 473) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(546) BEGIN
 // (op 546) Assign dim(483) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 546) Assign stride expression
 // (op 473) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(546) END
 // (op 473) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(547) BEGIN
 // (op 547) Assign dim(476) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 476: s**UNALLOCATED**
 // (op 547) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag476: SGPR Buffer: None x 1: (unallocated)
 // (op 547) reg expression
 // (op 547) reg expression
 // (op 547) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 547) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 547)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 547)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 547): s0, s1
s_mul_i32 s1, s8, s13 // (op 547) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 547): s14
s_mul_i32 s14, s9, s12 // (op 547) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 547) least significant
// Allocated : 1 SGPR (Value: Int32) (op 547): s15
s_mul_hi_u32 s15, s8, s12 // (op 547) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 547) most significant: sum
s_add_u32 s1, s1, s15 // (op 547) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 547): s15

 // Freeing : 1 SGPR (Value: Int32) (op 547): s14

 // (op 547) reg expression
 // (op 547) reg expression
// Allocated DataFlowTag476: 4 SGPRs (Buffer: None) (op 547): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 547) reg expression
s_mov_b32 s45, s11 // (op 547) reg expression
s_mov_b32 s46, s0 // (op 547) reg expression
s_mov_b32 s47, 131072 // (op 547) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 547): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(547) END
 // (op 473) NOP(537) BEGIN
 // (op 537) generate({})
 // (op 537) end: generate({})
 // (op 473) NOP(537) END
 // (op 473) Assign VGPR 1:U64(552) BEGIN
 // (op 552) Assign dim(485) = 1:U64
 // (op 552) Assign stride expression
 // (op 473) Assign VGPR 1:U64(552) END
 // (op 473) NOP(555) BEGIN
 // (op 555) generate({})
 // (op 555) end: generate({})
 // (op 473) NOP(555) END
 // (op 473) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(564) BEGIN
 // (op 564) Assign dim(486) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 486: v**UNALLOCATED**
 // (op 564) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag486: VGPR Value: UInt64 x 1: (unallocated)
 // (op 564) reg expression
 // (op 564) reg expression
 // (op 564) reg expression
 // (op 564) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 564): v14
v_and_b32 v14, -8, v1 // (op 564) 
 // (op 564) BEGIN: Flatten[1]
 // (op 564) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 564): v15
v_sub_u32 v15, v1, v14 // (op 564) 
 // (op 564) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 564): v14

 // (op 564) BEGIN: TileTile
 // (op 564) {TileTile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 564) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 564)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 564)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 564): v14
v_lshlrev_b32 v14, 4, v15 // (op 564) 
 // (op 564) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 564): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 564): v16, v17
v_mov_b32 v17, 0 // (op 564) convert
v_mov_b32 v16, v14 // (op 564) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 564): v14

 // (op 564) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 564): v14
v_lshrrev_b32 v14, 3, v1 // (op 564) 
 // (op 564) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 564): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 564) 
 // (op 564) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 564): v14

 // (op 564) Multiply({Tile: v15:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 564) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 564)           lhs (Tile: VGPR Value: UInt32 x 1: v15) 
 // (op 564)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 564): v18, v19
v_mul_lo_u32 v19, v15, s9 // (op 564) most significant: low of low * high
 // (op 564) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 564): v14
v_mul_hi_u32 v14, v15, s8 // (op 564) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 564) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 564) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 564): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 564): v15

 // (op 564) BEGIN: Split
 // (op 564) {Split: Add({TileTile: v[16:17]:U64}, v[18:19]:I64)U64}
 // (op 564) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 564)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 564)      rhs (VGPR Value: Int64 x 1: v[18:19])
// Allocated DataFlowTag486: 2 VGPRs (Value: UInt64) (op 564): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 564): s0, s1
v_add_co_u32 v14, s[0:1], v16, v18 // (op 564) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 564) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 564): s0, s1

 // (op 564) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 564): v18, v19

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 564): v16, v17

 // (op 473) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(564) END
 // (op 473) Assign VGPR 128:U64(565) BEGIN
 // (op 565) Assign dim(487) = 128:U64
 // (op 565) Assign stride expression
 // (op 473) Assign VGPR 128:U64(565) END
 // (op 473) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(566) BEGIN
 // (op 566) Assign dim(476) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 566) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // (op 566) reg expression
 // (op 566) reg expression
 // (op 566) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 566) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 566)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 566)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 566): s0, s1
s_mul_i32 s1, s8, s13 // (op 566) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 566): s14
s_mul_i32 s14, s9, s12 // (op 566) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 566) least significant
// Allocated : 1 SGPR (Value: Int32) (op 566): s15
s_mul_hi_u32 s15, s8, s12 // (op 566) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 566) most significant: sum
s_add_u32 s1, s1, s15 // (op 566) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 566): s15

 // Freeing : 1 SGPR (Value: Int32) (op 566): s14

 // (op 566) reg expression
 // (op 566) reg expression
s_mov_b32 s44, s10 // (op 566) reg expression
s_mov_b32 s45, s11 // (op 566) reg expression
s_mov_b32 s46, s0 // (op 566) reg expression
s_mov_b32 s47, 131072 // (op 566) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 566): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(566) END
 // (op 473) NOP(557) BEGIN
 // (op 557) generate({})
 // (op 557) end: generate({})
 // (op 473) NOP(557) END
 // (op 473) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(571) BEGIN
 // (op 571) Assign dim(489) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 571) Assign stride expression
 // (op 473) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(571) END
 // (op 473) NOP(558) BEGIN
 // (op 558) generate({})
 // (op 558) end: generate({})
 // (op 473) NOP(558) END
 // (op 473) Assign VGPR 1:U64(574) BEGIN
 // (op 574) Assign dim(491) = 1:U64
 // (op 574) Assign stride expression
 // (op 473) Assign VGPR 1:U64(574) END
 // (op 473) NOP(599) BEGIN
 // (op 599) generate({})
 // (op 599) end: generate({})
 // (op 473) NOP(599) END
 // (op 473) Assign VGPR 4:U64(606) BEGIN
 // (op 606) Assign dim(500) = 4:U64
 // (op 606) Assign stride expression
 // (op 473) Assign VGPR 4:U64(606) END
 // (op 473) NOP(600) BEGIN
 // (op 600) generate({})
 // (op 600) end: generate({})
 // (op 473) NOP(600) END
 // (op 473) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32}(609) BEGIN
 // (op 609) Assign dim(501) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 501: v**UNALLOCATED**
 // (op 609) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag501: VGPR Value: UInt32 x 1: (unallocated)
 // (op 609) reg expression
 // (op 609) reg expression
 // (op 609) reg expression
 // (op 609) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 609) convert
// Allocated : 1 VGPR (Value: UInt32) (op 609): v16
v_lshrrev_b32 v16, 7, v1 // (op 609) 
 // (op 609) BEGIN: Flatten[1]
 // (op 609) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 609): v17
v_and_b32 v17, 15, v1 // (op 609) 
 // (op 609) END: Flatten[1]
 // (op 609) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 609): v18
v_lshl_add_u32 v18, v16, 4, v17 // (op 609) 
 // (op 609) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 609): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 609): v16

 // (op 609) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 609): v16
v_lshl_add_u32 v16, s16, 4, v18 // (op 609) 
 // (op 609) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 609): v18

 // (op 609) BEGIN: Flatten[1]
 // (op 609) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 609): v17
v_and_b32 v17, 63, v1 // (op 609) 
 // (op 609) END: Flatten[1]
 // (op 609) LogicalShiftR({Flatten[1]: v17:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 609): v18
v_lshrrev_b32 v18, 4, v17 // (op 609) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 609): v17

 // (op 609) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag501: 1 VGPR (Value: UInt32) (op 609): v17
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v16, s20 // (op 609) 
v_add_u32 v17, v17, v18 // (op 609) 
 // (op 609) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 609): v18

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 609): v16

 // (op 473) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32}(609) END
 // (op 473) Assign VGPR 4:U64(610) BEGIN
 // (op 610) Assign dim(502) = 4:U64
 // (op 610) Assign stride expression
 // (op 473) Assign VGPR 4:U64(610) END
 // (op 473) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(611) BEGIN
 // (op 611) Assign dim(495) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 495: s**UNALLOCATED**
 // (op 611) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag495: SGPR Buffer: None x 1: (unallocated)
 // (op 611) reg expression
 // (op 611) reg expression
 // (op 611) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 611) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 611)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 611)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 611): s0, s1
s_mul_i32 s1, s20, s7 // (op 611) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 611): s14
s_mul_i32 s14, s21, s6 // (op 611) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 611) least significant
// Allocated : 1 SGPR (Value: Int32) (op 611): s15
s_mul_hi_u32 s15, s20, s6 // (op 611) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 611) most significant: sum
s_add_u32 s1, s1, s15 // (op 611) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 611): s15

 // Freeing : 1 SGPR (Value: Int32) (op 611): s14

 // (op 611) reg expression
 // (op 611) reg expression
// Allocated DataFlowTag495: 4 SGPRs (Buffer: None) (op 611): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 611) reg expression
s_mov_b32 s49, s23 // (op 611) reg expression
s_mov_b32 s50, s0 // (op 611) reg expression
s_mov_b32 s51, 131072 // (op 611) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 611): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(611) END
 // (op 473) NOP(601) BEGIN
 // (op 601) generate({})
 // (op 601) end: generate({})
 // (op 473) NOP(601) END
 // (op 473) Assign VGPR 1:U64(616) BEGIN
 // (op 616) Assign dim(504) = 1:U64
 // (op 616) Assign stride expression
 // (op 473) Assign VGPR 1:U64(616) END
 // (op 473) NOP(619) BEGIN
 // (op 619) generate({})
 // (op 619) end: generate({})
 // (op 473) NOP(619) END
 // (op 473) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32}(628) BEGIN
 // (op 628) Assign dim(505) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32}
 // tag 505: v**UNALLOCATED**
 // (op 628) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32} into DataFlowTag505: VGPR Value: UInt32 x 1: (unallocated)
 // (op 628) reg expression
 // (op 628) reg expression
 // (op 628) reg expression
 // (op 628) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 628) convert
// Allocated : 1 VGPR (Value: UInt32) (op 628): v16
v_lshrrev_b32 v16, 7, v1 // (op 628) 
 // (op 628) BEGIN: Flatten[1]
 // (op 628) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 628): v18
v_and_b32 v18, 15, v1 // (op 628) 
 // (op 628) END: Flatten[1]
 // (op 628) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 628): v19
v_lshl_add_u32 v19, v16, 4, v18 // (op 628) 
 // (op 628) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 628): v18

 // Freeing : 1 VGPR (Value: UInt32) (op 628): v16

 // (op 628) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 628): v16
v_lshl_add_u32 v16, s16, 4, v19 // (op 628) 
 // (op 628) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 628): v19

 // (op 628) BEGIN: Flatten[1]
 // (op 628) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 628): v18
v_and_b32 v18, 63, v1 // (op 628) 
 // (op 628) END: Flatten[1]
 // (op 628) LogicalShiftR({Flatten[1]: v18:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 628): v19
v_lshrrev_b32 v19, 4, v18 // (op 628) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 628): v18

 // (op 628) BEGIN: Split
// Allocated DataFlowTag505: 1 VGPR (Value: UInt32) (op 628): v18
v_mul_lo_u32 v18, v16, s20 // (op 628) 
v_add_u32 v18, v18, v19 // (op 628) 
 // (op 628) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 628): v19

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 628): v16

 // (op 473) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32}(628) END
 // (op 473) Assign VGPR 4:U64(629) BEGIN
 // (op 629) Assign dim(506) = 4:U64
 // (op 629) Assign stride expression
 // (op 473) Assign VGPR 4:U64(629) END
 // (op 473) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(630) BEGIN
 // (op 630) Assign dim(495) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 630) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // (op 630) reg expression
 // (op 630) reg expression
 // (op 630) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 630) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 630)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 630)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 630): s0, s1
s_mul_i32 s1, s20, s7 // (op 630) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 630): s14
s_mul_i32 s14, s21, s6 // (op 630) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 630) least significant
// Allocated : 1 SGPR (Value: Int32) (op 630): s15
s_mul_hi_u32 s15, s20, s6 // (op 630) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 630) most significant: sum
s_add_u32 s1, s1, s15 // (op 630) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 630): s15

 // Freeing : 1 SGPR (Value: Int32) (op 630): s14

 // (op 630) reg expression
 // (op 630) reg expression
s_mov_b32 s48, s22 // (op 630) reg expression
s_mov_b32 s49, s23 // (op 630) reg expression
s_mov_b32 s50, s0 // (op 630) reg expression
s_mov_b32 s51, 131072 // (op 630) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 630): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(630) END
 // (op 473) NOP(621) BEGIN
 // (op 621) generate({})
 // (op 621) end: generate({})
 // (op 473) NOP(621) END
 // (op 473) Assign VGPR 4:U64(635) BEGIN
 // (op 635) Assign dim(508) = 4:U64
 // (op 635) Assign stride expression
 // (op 473) Assign VGPR 4:U64(635) END
 // (op 473) NOP(622) BEGIN
 // (op 622) generate({})
 // (op 622) end: generate({})
 // (op 473) NOP(622) END
 // (op 473) Assign VGPR 1:U64(638) BEGIN
 // (op 638) Assign dim(510) = 1:U64
 // (op 638) Assign stride expression
 // (op 473) Assign VGPR 1:U64(638) END
 // (op 473) NOP(663) BEGIN
 // (op 663) generate({})
 // (op 663) end: generate({})
 // (op 473) NOP(663) END
 // (op 473) Assign VGPR 4:U64(670) BEGIN
 // (op 670) Assign dim(519) = 4:U64
 // (op 670) Assign stride expression
 // (op 473) Assign VGPR 4:U64(670) END
 // (op 473) NOP(664) BEGIN
 // (op 664) generate({})
 // (op 664) end: generate({})
 // (op 473) NOP(664) END
 // (op 473) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(673) BEGIN
 // (op 673) Assign dim(520) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 520: v**UNALLOCATED**
 // (op 673) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag520: VGPR Value: UInt32 x 1: (unallocated)
 // (op 673) reg expression
 // (op 673) reg expression
 // (op 673) reg expression
 // (op 673) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 673) convert
// Allocated : 1 VGPR (Value: UInt32) (op 673): v16
v_and_b32 v16, -4, v1 // (op 673) 
 // (op 673) BEGIN: Flatten[1]TileTile
 // (op 673) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 673): v19
v_sub_u32 v19, v1, v16 // (op 673) 
 // (op 673) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 673): v16

 // (op 673) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 673): v16
v_lshrrev_b32 v16, 2, v1 // (op 673) 
 // (op 673) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 673): v20
v_lshl_add_u32 v20, s17, 5, v16 // (op 673) 
 // (op 673) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 673): v16

 // (op 673) Multiply({Tile: v20:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 673): v16
v_mul_lo_u32 v16, v20, s24 // (op 673) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 673): v20

 // (op 673) BEGIN: Split
 // (op 673) {Split: Add({Flatten[1]TileTile: v19:U32}, v16:U32)U32}
// Allocated DataFlowTag520: 1 VGPR (Value: UInt32) (op 673): v20
v_add_u32 v20, v19, v16 // (op 673) 
 // (op 673) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 673): v16

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 673): v19

 // (op 473) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(673) END
 // (op 473) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(674) BEGIN
 // (op 674) Assign dim(521) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 674) Assign stride expression
 // (op 473) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(674) END
 // (op 473) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(675) BEGIN
 // (op 675) Assign dim(514) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 514: s**UNALLOCATED**
 // (op 675) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag514: SGPR Buffer: None x 1: (unallocated)
 // (op 675) reg expression
 // (op 675) reg expression
 // (op 675) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 675) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 675)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 675)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 675): s0, s1
s_mul_i32 s1, s24, s13 // (op 675) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 675): s14
s_mul_i32 s14, s25, s12 // (op 675) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 675) least significant
// Allocated : 1 SGPR (Value: Int32) (op 675): s15
s_mul_hi_u32 s15, s24, s12 // (op 675) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 675) most significant: sum
s_add_u32 s1, s1, s15 // (op 675) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 675): s15

 // Freeing : 1 SGPR (Value: Int32) (op 675): s14

 // (op 675) reg expression
 // (op 675) reg expression
// Allocated DataFlowTag514: 4 SGPRs (Buffer: None) (op 675): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 675) reg expression
s_mov_b32 s53, s27 // (op 675) reg expression
s_mov_b32 s54, s0 // (op 675) reg expression
s_mov_b32 s55, 131072 // (op 675) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 675): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(675) END
 // (op 473) NOP(665) BEGIN
 // (op 665) generate({})
 // (op 665) end: generate({})
 // (op 473) NOP(665) END
 // (op 473) Assign VGPR 1:U64(680) BEGIN
 // (op 680) Assign dim(523) = 1:U64
 // (op 680) Assign stride expression
 // (op 473) Assign VGPR 1:U64(680) END
 // (op 473) NOP(683) BEGIN
 // (op 683) generate({})
 // (op 683) end: generate({})
 // (op 473) NOP(683) END
 // (op 473) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(692) BEGIN
 // (op 692) Assign dim(524) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 524: v**UNALLOCATED**
 // (op 692) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag524: VGPR Value: UInt32 x 1: (unallocated)
 // (op 692) reg expression
 // (op 692) reg expression
 // (op 692) reg expression
 // (op 692) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 692) convert
// Allocated : 1 VGPR (Value: UInt32) (op 692): v16
v_and_b32 v16, -4, v1 // (op 692) 
 // (op 692) BEGIN: Flatten[1]TileTile
 // (op 692) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 692): v19
v_sub_u32 v19, v1, v16 // (op 692) 
 // (op 692) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 692): v16

 // (op 692) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 692): v16
v_lshrrev_b32 v16, 2, v1 // (op 692) 
 // (op 692) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 692): v21
v_lshl_add_u32 v21, s17, 5, v16 // (op 692) 
 // (op 692) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 692): v16

 // (op 692) Multiply({Tile: v21:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 692): v16
v_mul_lo_u32 v16, v21, s24 // (op 692) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 692): v21

 // (op 692) BEGIN: Split
 // (op 692) {Split: Add({Flatten[1]TileTile: v19:U32}, v16:U32)U32}
// Allocated DataFlowTag524: 1 VGPR (Value: UInt32) (op 692): v21
v_add_u32 v21, v19, v16 // (op 692) 
 // (op 692) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 692): v16

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 692): v19

 // (op 473) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(692) END
 // (op 473) Assign VGPR 4:U64(693) BEGIN
 // (op 693) Assign dim(525) = 4:U64
 // (op 693) Assign stride expression
 // (op 473) Assign VGPR 4:U64(693) END
 // (op 473) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(694) BEGIN
 // (op 694) Assign dim(514) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 694) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // (op 694) reg expression
 // (op 694) reg expression
 // (op 694) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 694) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 694)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 694)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 694): s0, s1
s_mul_i32 s1, s24, s13 // (op 694) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 694): s14
s_mul_i32 s14, s25, s12 // (op 694) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 694) least significant
// Allocated : 1 SGPR (Value: Int32) (op 694): s15
s_mul_hi_u32 s15, s24, s12 // (op 694) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 694) most significant: sum
s_add_u32 s1, s1, s15 // (op 694) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 694): s15

 // Freeing : 1 SGPR (Value: Int32) (op 694): s14

 // (op 694) reg expression
 // (op 694) reg expression
s_mov_b32 s52, s26 // (op 694) reg expression
s_mov_b32 s53, s27 // (op 694) reg expression
s_mov_b32 s54, s0 // (op 694) reg expression
s_mov_b32 s55, 131072 // (op 694) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 694): s0, s1

 // (op 473) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(694) END
 // (op 473) NOP(685) BEGIN
 // (op 685) generate({})
 // (op 685) end: generate({})
 // (op 473) NOP(685) END
 // (op 473) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(699) BEGIN
 // (op 699) Assign dim(527) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 699) Assign stride expression
 // (op 473) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(699) END
 // (op 473) NOP(686) BEGIN
 // (op 686) generate({})
 // (op 686) end: generate({})
 // (op 473) NOP(686) END
 // (op 473) Assign VGPR 1:U64(702) BEGIN
 // (op 702) Assign dim(529) = 1:U64
 // (op 702) Assign stride expression
 // (op 473) Assign VGPR 1:U64(702) END
 // (op 473) NOP(334) BEGIN
 // (op 334) generate({})
 // (op 334) end: generate({})
 // (op 473) NOP(334) END
 // (op 473) SetCoordinate(339) BEGIN
 // (op 339) generate({})
 // (op 339) end: generate({})
 // (op 339) generate({340})
 // (op 339) SetCoordinate(340) BEGIN
 // (op 340) generate({})
 // (op 340) end: generate({})
 // (op 340) generate({341})
 // (op 340) LoadTiled Value: E8M0(341) BEGIN
 // (op 341) GEN: loadMacroTileWAVE OP341 WaveTile 203
 // (op 341) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 341) Generate 1:U64 into nullptr
 // tag 49: v**UNALLOCATED**
 // (op 341) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 341) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag501: VGPR Value: UInt32 x 1: v17
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 341) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 341): v16
buffer_load_ubyte v16, v17, s[48:51], 0 offen offset: 0 // (op 341) Load value
// VMEM: Expected complete at 789 (current 309)
 // (op 340) LoadTiled Value: E8M0(341) END
 // (op 340) end: generate({341})
 // (op 339) SetCoordinate(340) END
 // (op 339) end: generate({340})
 // (op 473) SetCoordinate(339) END
 // (op 473) SetCoordinate(344) BEGIN
 // (op 344) generate({})
 // (op 344) end: generate({})
 // (op 344) generate({807})
 // (op 344) Scope(807) BEGIN
 // (op 807) Lock Scope 807
 // Lock Branch
 // (op 807) generate({804})
 // (op 807) NOP(804) BEGIN
 // (op 804) generate({})
 // (op 804) end: generate({})
 // (op 807) NOP(804) END
 // (op 807) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(811) BEGIN
 // (op 811) Assign dim(560) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 560: s**UNALLOCATED**
 // (op 811) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag560: SGPR Value: UInt32 x 1: (unallocated)
 // (op 811) reg expression
 // (op 811) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 811): v19
v_and_b32 v19, -8, v1 // (op 811) 
 // (op 811) BEGIN: Tile[1]
 // (op 811) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v19:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 811): v22
v_sub_u32 v22, v1, v19 // (op 811) 
 // (op 811) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 811): v19

 // (op 811) BEGIN: Flatten
 // (op 811) {Flatten: ShiftL({Tile[1]: v22:U32}, 4:U32)U32}
 // (op 811) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 811)         value (Tile[1]: VGPR Value: UInt32 x 1: v22) 
 // (op 811)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 811): v19
v_lshlrev_b32 v19, 4, v22 // (op 811) 
 // (op 811) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 811): v22

 // (op 811) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 811): v22
v_lshrrev_b32 v22, 3, v1 // (op 811) 
 // (op 811) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 811): v23
v_lshl_add_u32 v23, v22, 7, v19 // (op 811) 
 // (op 811) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 811): v19

 // Freeing : 1 VGPR (Value: UInt32) (op 811): v22

// Allocated DataFlowTag560: 1 SGPR (Value: UInt32) (op 811): s0
s_nop 0
v_readfirstlane_b32 s0, v23 // (op 811) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 811): v23

 // (op 807) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(811) END
 // (op 807) Assign VGPR 2048:U32(812) BEGIN
 // (op 812) Assign dim(561) = 2048:U32
 // (op 812) Assign stride expression
 // (op 807) Assign VGPR 2048:U32(812) END
 // (op 807) NOP(805) BEGIN
 // (op 805) generate({})
 // (op 805) end: generate({})
 // (op 807) NOP(805) END
 // (op 807) Assign VGPR 1:U32(816) BEGIN
 // (op 816) Assign dim(563) = 1:U32
 // (op 816) Assign stride expression
 // (op 807) Assign VGPR 1:U32(816) END
 // (op 807) LoadTileDirect2LDS Value: BF8x4(437) BEGIN
 // (op 437) GEN: LoadTileDirect2LDS
 // (op 437) GEN: loadMacroTileDirect2LDS OP 437 LDS 24 MacroTile 9
 // (op 437) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 437) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride483: VGPR Value: UInt64 x 1: (unallocated)
 // (op 437) reg expression
 // (op 437) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 437) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 437)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 437)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 437): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 437) 
// Allocated Stride483: 2 VGPRs (Value: UInt64) (op 437): v22, v23
v_mov_b64 v[22:23], s[14:15] // (op 437) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 437): s14, s15

 // (op 437) Generate 1:U64 into nullptr
 // (op 437) Generate 2048:U32 into nullptr
 // (op 437) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride483: VGPR Value: UInt64 x 1: v[22:23]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 437): v24, v25
v_mov_b64 v[24:25], v[12:13] // (op 437) 
 // (op 437) Generate Add(0:U32, {DataFlowTag560Flatten: s0:U32})U32 into nullptr
 // (op 437)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride483: v[22:23]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 437) Generate {DataFlowTag560Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 437) reg expression
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 437) Load value direct to lds
// VMEM: Expected complete at 801 (current 321)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 437) Generate Add(v[24:25]:U64, v22:R)U64 into VGPR Value: UInt64 x 1: v[24:25]
 // (op 437) reg expression
 // (op 437) reg expression
 // (op 437) Add(v[24:25]:U64, v22:R)U64
 // (op 437) Add: dest (VGPR Value: UInt64 x 1: v[24:25]) = 
 // (op 437)      lhs (VGPR Value: UInt64 x 1: v[24:25]) 
 // (op 437)      rhs (VGPR Value: Raw32 x 1: v22)
// Allocated : 2 SGPRs (Value: Bool64) (op 437): s14, s15
v_add_co_u32 v24, s[14:15], v24, v22 // (op 437) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 437) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 437): s14, s15

 // (op 437) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 437) reg expression
 // (op 437) Add(m0:U32, 2048:U32)U32
 // (op 437) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 437) 
 // (op 437) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v24, s[44:47], 0 offen offset: 0 lds // (op 437) Load value direct to lds
// VMEM: Expected complete at 805 (current 325)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // Freeing Stride483: 2 VGPRs (Value: UInt64) (op 437): v22, v23

 // Freeing : 2 VGPRs (Value: UInt64) (op 437): v24, v25

 // (op 807) LoadTileDirect2LDS Value: BF8x4(437) END
 // (op 807) Deallocate{}(1182) BEGIN
 // (op 1182) Deallocate 560
 // Freeing DataFlowTag560: 1 SGPR (Value: UInt32) (op 811): s0

 // Deleting tag 560
 // (op 1182) Deallocate 561
 // Deleting tag 561
 // (op 1182) Deallocate 563
 // Deleting tag 563
 // (op 1182) Deallocate 562
 // Deleting tag 562
 // (op 807) Deallocate{}(1182) END
 // (op 807) end: generate({804})
 // (op 807) Unlock Scope 807
 // Unlock None
 // (op 344) Scope(807) END
 // (op 344) end: generate({807})
 // (op 473) SetCoordinate(344) END
 // (op 473) SetCoordinate(347) BEGIN
 // (op 347) generate({})
 // (op 347) end: generate({})
 // (op 347) generate({348})
 // (op 347) LoadTiled Value: E8M0(348) BEGIN
 // (op 348) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 348) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 348) Generate 1:U64 into nullptr
 // tag 10: v**UNALLOCATED**
 // (op 348) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 348) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag520: VGPR Value: UInt32 x 1: v20
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 348) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 348): v19
buffer_load_ubyte v19, v20, s[52:55], 0 offen offset: 0 // (op 348) Load value
// VMEM: Expected stall of 464. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1270 (current 790)
 // (op 347) LoadTiled Value: E8M0(348) END
 // (op 347) end: generate({348})
 // (op 473) SetCoordinate(347) END
 // (op 473) SetCoordinate(350) BEGIN
 // (op 350) generate({})
 // (op 350) end: generate({})
 // (op 350) generate({351})
 // (op 350) LoadTiled Value: BF6x16(351) BEGIN
 // (op 351) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 351) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 351) Generate 0:U64 into nullptr
 // tag 8: v**UNALLOCATED**
 // (op 351) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: (unallocated)
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 351) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag463: VGPR Value: UInt32 x 1: v10
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 351) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: BF6x16) (op 351): v22, v23, v24
buffer_load_dwordx3 v[22:24], v10, s[40:43], 0 offen // (op 351) Load value
// VMEM: Expected stall of 11. CBNW: 1, Inc: 3
// VMEM: Expected complete at 1282 (current 802)
 // (op 350) LoadTiled Value: BF6x16(351) END
 // (op 350) end: generate({351})
 // (op 473) SetCoordinate(350) END
 // (op 473) SetCoordinate(353) BEGIN
 // (op 353) generate({})
 // (op 353) end: generate({})
 // (op 353) generate({354})
 // (op 353) SetCoordinate(354) BEGIN
 // (op 354) generate({})
 // (op 354) end: generate({})
 // (op 354) generate({355})
 // (op 354) LoadTiled Value: E8M0(355) BEGIN
 // (op 355) GEN: loadMacroTileWAVE OP355 WaveTile 203
 // (op 355) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 355) Generate 1:U64 into nullptr
 // tag 450: v**UNALLOCATED**
 // (op 355) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 355) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag501: VGPR Value: UInt32 x 1: v17
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 355) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 355): v25
buffer_load_ubyte v25, v17, s[48:51], 0 offen offset:4 // (op 355) Load value
// VMEM: Expected stall of 3. CBNW: 2, Inc: 3
// VMEM: Expected complete at 1286 (current 806)
 // (op 354) LoadTiled Value: E8M0(355) END
 // (op 354) end: generate({355})
 // (op 353) SetCoordinate(354) END
 // (op 353) end: generate({354})
 // (op 473) SetCoordinate(353) END
 // (op 473) Deallocate{}(1164) BEGIN
 // (op 1164) Deallocate 500
 // Deleting tag 500
 // (op 1164) Deallocate 502
 // Deleting tag 502
 // (op 1164) Deallocate 504
 // Deleting tag 504
 // (op 1164) Deallocate 501
 // Freeing DataFlowTag501: 1 VGPR (Value: UInt32) (op 609): v17

 // Deleting tag 501
 // (op 1164) Deallocate 503
 // Deleting tag 503
 // (op 473) Deallocate{}(1164) END
 // (op 473) SetCoordinate(358) BEGIN
 // (op 358) generate({})
 // (op 358) end: generate({})
 // (op 358) generate({957})
 // (op 358) Scope(957) BEGIN
 // (op 957) Lock Scope 957
 // Lock Branch
 // (op 957) generate({954})
 // (op 957) NOP(954) BEGIN
 // (op 954) generate({})
 // (op 954) end: generate({})
 // (op 957) NOP(954) END
 // (op 957) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(961) BEGIN
 // (op 961) Assign dim(606) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 606: s**UNALLOCATED**
 // (op 961) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag606: SGPR Value: UInt32 x 1: (unallocated)
 // (op 961) reg expression
 // (op 961) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 961): v17
v_and_b32 v17, -8, v1 // (op 961) 
 // (op 961) BEGIN: Tile[1]
 // (op 961) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 961): v26
v_sub_u32 v26, v1, v17 // (op 961) 
 // (op 961) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 961): v17

 // (op 961) BEGIN: Flatten
 // (op 961) {Flatten: ShiftL({Tile[1]: v26:U32}, 4:U32)U32}
 // (op 961) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 961)         value (Tile[1]: VGPR Value: UInt32 x 1: v26) 
 // (op 961)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 961): v17
v_lshlrev_b32 v17, 4, v26 // (op 961) 
 // (op 961) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 961): v26

 // (op 961) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 961): v26
v_lshrrev_b32 v26, 3, v1 // (op 961) 
 // (op 961) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 961): v27
v_lshl_add_u32 v27, v26, 7, v17 // (op 961) 
 // (op 961) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 961): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 961): v26

// Allocated DataFlowTag606: 1 SGPR (Value: UInt32) (op 961): s0
s_nop 0
v_readfirstlane_b32 s0, v27 // (op 961) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 961): v27

 // (op 957) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(961) END
 // (op 957) Assign VGPR 2048:U32(962) BEGIN
 // (op 962) Assign dim(607) = 2048:U32
 // (op 962) Assign stride expression
 // (op 957) Assign VGPR 2048:U32(962) END
 // (op 957) NOP(955) BEGIN
 // (op 955) generate({})
 // (op 955) end: generate({})
 // (op 957) NOP(955) END
 // (op 957) Assign VGPR 1:U32(966) BEGIN
 // (op 966) Assign dim(609) = 1:U32
 // (op 966) Assign stride expression
 // (op 957) Assign VGPR 1:U32(966) END
 // (op 957) LoadTileDirect2LDS Value: BF8x4(442) BEGIN
 // (op 442) GEN: LoadTileDirect2LDS
 // (op 442) GEN: loadMacroTileDirect2LDS OP 442 LDS 440 MacroTile 438
 // (op 442) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 442) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride483: VGPR Value: UInt64 x 1: (unallocated)
 // (op 442) reg expression
 // (op 442) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 442) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 442)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 442)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 442): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 442) 
// Allocated Stride483: 2 VGPRs (Value: UInt64) (op 442): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 442) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 442): s14, s15

 // (op 442) Generate 1:U64 into nullptr
 // (op 442) Generate 2048:U32 into nullptr
 // (op 442) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride483: VGPR Value: UInt64 x 1: v[26:27]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 442): v28, v29
v_mov_b64 v[28:29], v[12:13] // (op 442) 
 // (op 442) Generate Add(4096:U32, {DataFlowTag606Flatten: s0:U32})U32 into nullptr
 // (op 442) reg expression
 // (op 442) Add(4096:U32, {DataFlowTag606Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 442): s1
s_add_u32 s1, 4096, s0 // (op 442) 
 // (op 442)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride483: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 442) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 442) reg expression
 // (op 442) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 442): s14
s_mov_b32 s14, 128 // (op 442) call()
// 128 is not a supported value!
buffer_load_dwordx4 v28, s[44:47], s14 offen offset: 0 lds // (op 442) Load value direct to lds
// VMEM: Expected stall of 451. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1751 (current 1271)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 442) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 442) reg expression
 // (op 442) reg expression
 // (op 442) Add(v[28:29]:U64, v26:R)U64
 // (op 442) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 442)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 442)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 442): s18, s19
v_add_co_u32 v28, s[18:19], v28, v26 // (op 442) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 442) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 442): s18, s19

 // (op 442) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 442) reg expression
 // (op 442) Add(m0:U32, 2048:U32)U32
 // (op 442) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 442) 
 // (op 442) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[44:47], s14 offen offset: 0 lds // (op 442) Load value direct to lds
// VMEM: Expected stall of 8. CBNW: 4, Inc: 3
// VMEM: Expected complete at 1763 (current 1283)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 442): s14

 // Freeing Stride483: 2 VGPRs (Value: UInt64) (op 442): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 442): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 442): s1

 // (op 957) LoadTileDirect2LDS Value: BF8x4(442) END
 // (op 957) Deallocate{}(1184) BEGIN
 // (op 1184) Deallocate 608
 // Deleting tag 608
 // (op 1184) Deallocate 609
 // Deleting tag 609
 // (op 1184) Deallocate 606
 // Freeing DataFlowTag606: 1 SGPR (Value: UInt32) (op 961): s0

 // Deleting tag 606
 // (op 1184) Deallocate 607
 // Deleting tag 607
 // (op 957) Deallocate{}(1184) END
 // (op 957) end: generate({954})
 // (op 957) Unlock Scope 957
 // Unlock None
 // (op 358) Scope(957) END
 // (op 358) end: generate({957})
 // (op 473) SetCoordinate(358) END
 // (op 473) Deallocate{}(1166) BEGIN
 // (op 1166) Deallocate 483
 // Deleting tag 483
 // (op 1166) Deallocate 481
 // Deleting tag 481
 // (op 1166) Deallocate 482
 // Freeing DataFlowTag482: 2 VGPRs (Value: UInt64) (op 545): v12, v13

 // Deleting tag 482
 // (op 1166) Deallocate 485
 // Deleting tag 485
 // (op 1166) Deallocate 484
 // Deleting tag 484
 // (op 473) Deallocate{}(1166) END
 // (op 473) SetCoordinate(361) BEGIN
 // (op 361) generate({})
 // (op 361) end: generate({})
 // (op 361) generate({362})
 // (op 361) LoadTiled Value: E8M0(362) BEGIN
 // (op 362) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 362) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 362) Generate 1:U64 into nullptr
 // tag 442: v**UNALLOCATED**
 // (op 362) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 362) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag520: VGPR Value: UInt32 x 1: v20
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 362) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 362): v17
buffer_load_ubyte v17, v20, s[52:55], 0 offen offset:4 // (op 362) Load value
// VMEM: Expected stall of 3. CBNW: 5, Inc: 3
// VMEM: Expected complete at 1767 (current 1287)
 // (op 361) LoadTiled Value: E8M0(362) END
 // (op 361) end: generate({362})
 // (op 473) SetCoordinate(361) END
 // (op 473) Deallocate{}(1168) BEGIN
 // (op 1168) Deallocate 519
 // Deleting tag 519
 // (op 1168) Deallocate 521
 // Deleting tag 521
 // (op 1168) Deallocate 520
 // Freeing DataFlowTag520: 1 VGPR (Value: UInt32) (op 673): v20

 // Deleting tag 520
 // (op 1168) Deallocate 523
 // Deleting tag 523
 // (op 1168) Deallocate 522
 // Deleting tag 522
 // (op 473) Deallocate{}(1168) END
 // (op 473) SetCoordinate(364) BEGIN
 // (op 364) generate({})
 // (op 364) end: generate({})
 // (op 364) generate({365})
 // (op 364) LoadTiled Value: BF6x16(365) BEGIN
 // (op 365) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 365) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 365) Generate 0:U64 into nullptr
 // tag 434: v**UNALLOCATED**
 // (op 365) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: (unallocated)
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 365) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag463: VGPR Value: UInt32 x 1: v10
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 365) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: BF6x16) (op 365): v26, v27, v28
buffer_load_dwordx3 v[26:28], v10, s[40:43], 0 offen offset:96 // (op 365) Load value
// VMEM: Expected stall of 464. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2232 (current 1752)
 // (op 364) LoadTiled Value: BF6x16(365) END
 // (op 364) end: generate({365})
 // (op 473) SetCoordinate(364) END
 // (op 473) Deallocate{}(1170) BEGIN
 // (op 1170) Deallocate 466
 // Deleting tag 466
 // (op 1170) Deallocate 462
 // Deleting tag 462
 // (op 1170) Deallocate 464
 // Deleting tag 464
 // (op 1170) Deallocate 463
 // Freeing DataFlowTag463: 1 VGPR (Value: UInt32) (op 481): v10

 // Deleting tag 463
 // (op 1170) Deallocate 465
 // Deleting tag 465
 // (op 473) Deallocate{}(1170) END
 // (op 473) NOP(439) BEGIN
 // (op 439) generate({})
 // (op 439) end: generate({})
 // (op 473) NOP(439) END
 // (op 473) Scope(859) BEGIN
 // (op 859) Lock Scope 859
 // Lock Branch
 // (op 859) generate({856})
 // (op 859) NOP(856) BEGIN
 // (op 856) generate({})
 // (op 856) end: generate({})
 // (op 859) NOP(856) END
 // (op 859) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(864) BEGIN
 // (op 864) Assign dim(576) = {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}
 // tag 576: v**UNALLOCATED**
 // (op 864) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32} into DataFlowTag576: VGPR Value: UInt32 x 1: (unallocated)
 // (op 864) reg expression
 // (op 864) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 864): v10
v_and_b32 v10, -4, v1 // (op 864) 
 // (op 864) BEGIN: Tile[1]Flatten
 // (op 864) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 864): v20
v_sub_u32 v20, v1, v10 // (op 864) 
 // (op 864) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 864): v10

 // (op 864) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 864): v10
v_and_b32 v10, 4294967292, v1 // (op 864) 
 // (op 864) BEGIN: Flatten
 // (op 864) {Flatten: Add(v10:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag576: 1 VGPR (Value: UInt32) (op 864): v12
v_add_u32 v12, v10, v20 // (op 864) 
 // (op 864) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 864): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 864): v10

 // (op 859) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(864) END
 // (op 859) Assign VGPR 4:U32(865) BEGIN
 // (op 865) Assign dim(577) = 4:U32
 // (op 865) Assign stride expression
 // (op 859) Assign VGPR 4:U32(865) END
 // (op 859) NOP(857) BEGIN
 // (op 857) generate({})
 // (op 857) end: generate({})
 // (op 859) NOP(857) END
 // (op 859) Assign VGPR 1:U32(869) BEGIN
 // (op 869) Assign dim(579) = 1:U32
 // (op 869) Assign stride expression
 // (op 859) Assign VGPR 1:U32(869) END
 // (op 859) StoreLDSTile Value: E8M0(368) BEGIN
 // (op 368) GEN: StoreLDSTile
 // (op 368) GEN: storeMacroTileLDS OP 368 LDS 32 MacroTile 10
 // (op 368) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 368) Generate 1:U32 into nullptr
 // (op 368) 	Dir = Store
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 368) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag576: VGPR Value: UInt32 x 1: v12
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 368) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v12, v19 offset:8192 // (op 368) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 1853 (current 1758)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 859) StoreLDSTile Value: E8M0(368) END
 // (op 859) Deallocate{}(1172) BEGIN
 // (op 1172) Deallocate 577
 // Deleting tag 577
 // (op 1172) Deallocate 576
 // Freeing DataFlowTag576: 1 VGPR (Value: UInt32) (op 864): v12

 // Deleting tag 576
 // (op 1172) Deallocate 579
 // Deleting tag 579
 // (op 1172) Deallocate 578
 // Deleting tag 578
 // (op 859) Deallocate{}(1172) END
 // (op 859) end: generate({856})
 // (op 859) Unlock Scope 859
 // Unlock None
 // (op 473) Scope(859) END
 // (op 473) Scope(754) BEGIN
 // (op 754) Lock Scope 754
 // Lock Branch
 // (op 754) generate({751})
 // (op 754) NOP(751) BEGIN
 // (op 751) generate({})
 // (op 751) end: generate({})
 // (op 754) NOP(751) END
 // (op 754) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(759) BEGIN
 // (op 759) Assign dim(544) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 544: v**UNALLOCATED**
 // (op 759) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag544: VGPR Value: UInt32 x 1: (unallocated)
 // (op 759) reg expression
 // (op 759) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 759): v10
v_and_b32 v10, -8, v1 // (op 759) 
 // (op 759) BEGIN: Tile[1]
 // (op 759) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 759): v20
v_sub_u32 v20, v1, v10 // (op 759) 
 // (op 759) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 759): v10

 // (op 759) BEGIN: Flatten
 // (op 759) {Flatten: ShiftL({Tile[1]: v20:U32}, 4:U32)U32}
 // (op 759) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 759)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 759)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 759): v10
v_lshlrev_b32 v10, 4, v20 // (op 759) 
 // (op 759) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 759): v20

 // (op 759) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 759): v20
v_lshrrev_b32 v20, 3, v1 // (op 759) 
 // (op 759) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 759): v12
v_lshl_add_u32 v12, v20, 7, v10 // (op 759) 
 // (op 759) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 759): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 759): v20

 // (op 759) Multiply({Flatten: v12:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 759): v10
v_mul_lo_u32 v10, v12, 6 // (op 759) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 759): v12

 // (op 759) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag544: 1 VGPR (Value: UInt32) (op 759): v20
v_lshrrev_b32 v20, 3, v10 // (op 759) 
 // Freeing : 1 VGPR (Value: UInt32) (op 759): v10

 // (op 754) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(759) END
 // (op 754) Assign VGPR 96:U32(760) BEGIN
 // (op 760) Assign dim(545) = 96:U32
 // (op 760) Assign stride expression
 // (op 754) Assign VGPR 96:U32(760) END
 // (op 754) NOP(752) BEGIN
 // (op 752) generate({})
 // (op 752) end: generate({})
 // (op 754) NOP(752) END
 // (op 754) Assign VGPR 0:U32(764) BEGIN
 // (op 764) Assign dim(547) = 0:U32
 // (op 764) Assign stride expression
 // (op 754) Assign VGPR 0:U32(764) END
 // (op 754) StoreLDSTile Value: BF6x16(369) BEGIN
 // (op 369) GEN: StoreLDSTile
 // (op 369) GEN: storeMacroTileLDS OP 369 LDS 16 MacroTile 8
 // (op 369) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 369) Generate 0:U32 into nullptr
 // (op 369) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[22:24]
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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

 // (op 369) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[22:24]
 // 	info.rowOffsetReg = DataFlowTag544: VGPR Value: UInt32 x 1: v20
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
 // 	info.bufDesc = (nullptr)
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
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b96 v20, v[22:24] offset:8320 // (op 369) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 1862 (current 1767)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 754) StoreLDSTile Value: BF6x16(369) END
 // (op 754) Deallocate{}(1174) BEGIN
 // (op 1174) Deallocate 547
 // Deleting tag 547
 // (op 1174) Deallocate 544
 // Freeing DataFlowTag544: 1 VGPR (Value: UInt32) (op 759): v20

 // Deleting tag 544
 // (op 1174) Deallocate 545
 // Deleting tag 545
 // (op 1174) Deallocate 546
 // Deleting tag 546
 // (op 754) Deallocate{}(1174) END
 // (op 754) end: generate({751})
 // (op 754) Unlock Scope 754
 // Unlock None
 // (op 473) Scope(754) END
 // (op 473) NOP(337) BEGIN
 // (op 337) generate({})
 // (op 337) end: generate({})
 // (op 473) NOP(337) END
 // (op 473) NOP(726) BEGIN
 // (op 726) generate({})
 // (op 726) end: generate({})
 // (op 473) NOP(726) END
 // (op 473) NOP(723) BEGIN
 // (op 723) generate({})
 // (op 723) end: generate({})
 // (op 473) NOP(723) END
 // (op 473) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(730) BEGIN
 // (op 730) Assign dim(536) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 536: v**UNALLOCATED**
 // (op 730) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag536: VGPR Value: UInt32 x 1: (unallocated)
 // (op 730) reg expression
 // (op 730) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v10
v_and_b32 v10, -8, v1 // (op 730) 
 // (op 730) BEGIN: Tile[1]
 // (op 730) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 730): v20
v_sub_u32 v20, v1, v10 // (op 730) 
 // (op 730) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v10

 // (op 730) BEGIN: Flatten
 // (op 730) {Flatten: ShiftL({Tile[1]: v20:U32}, 4:U32)U32}
 // (op 730) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 730)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 730)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 730): v10
v_lshlrev_b32 v10, 4, v20 // (op 730) 
 // (op 730) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 730): v20

 // (op 730) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v20
v_lshrrev_b32 v20, 3, v1 // (op 730) 
 // (op 730) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 730): v12
v_lshl_add_u32 v12, v20, 7, v10 // (op 730) 
 // (op 730) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 730): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 730): v20

 // (op 730) Multiply({Flatten: v12:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 730): v10
v_mul_lo_u32 v10, v12, 6 // (op 730) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 730): v12

 // (op 730) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag536: 1 VGPR (Value: UInt32) (op 730): v20
v_lshrrev_b32 v20, 3, v10 // (op 730) 
 // Freeing : 1 VGPR (Value: UInt32) (op 730): v10

 // (op 473) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(730) END
 // (op 473) Assign VGPR 96:U32(731) BEGIN
 // (op 731) Assign dim(537) = 96:U32
 // (op 731) Assign stride expression
 // (op 473) Assign VGPR 96:U32(731) END
 // (op 473) NOP(724) BEGIN
 // (op 724) generate({})
 // (op 724) end: generate({})
 // (op 473) NOP(724) END
 // (op 473) Assign VGPR 0:U32(735) BEGIN
 // (op 735) Assign dim(539) = 0:U32
 // (op 735) Assign stride expression
 // (op 473) Assign VGPR 0:U32(735) END
 // (op 473) NOP(778) BEGIN
 // (op 778) generate({})
 // (op 778) end: generate({})
 // (op 473) NOP(778) END
 // (op 473) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(783) BEGIN
 // (op 783) Assign dim(552) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 552: s**UNALLOCATED**
 // (op 783) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag552: SGPR Value: UInt32 x 1: (unallocated)
 // (op 783) reg expression
 // (op 783) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 783): v10
v_and_b32 v10, -8, v1 // (op 783) 
 // (op 783) BEGIN: Tile[1]
 // (op 783) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 783): v12
v_sub_u32 v12, v1, v10 // (op 783) 
 // (op 783) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 783): v10

 // (op 783) BEGIN: Flatten
 // (op 783) {Flatten: ShiftL({Tile[1]: v12:U32}, 4:U32)U32}
 // (op 783) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 783)         value (Tile[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 783)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 783): v10
v_lshlrev_b32 v10, 4, v12 // (op 783) 
 // (op 783) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 783): v12

 // (op 783) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 783): v12
v_lshrrev_b32 v12, 3, v1 // (op 783) 
 // (op 783) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 783): v13
v_lshl_add_u32 v13, v12, 7, v10 // (op 783) 
 // (op 783) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 783): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 783): v12

// Allocated DataFlowTag552: 1 SGPR (Value: UInt32) (op 783): s0
s_nop 0
v_readfirstlane_b32 s0, v13 // (op 783) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 783): v13

 // (op 473) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(783) END
 // (op 473) Assign VGPR 2048:U32(784) BEGIN
 // (op 784) Assign dim(553) = 2048:U32
 // (op 784) Assign stride expression
 // (op 473) Assign VGPR 2048:U32(784) END
 // (op 473) NOP(779) BEGIN
 // (op 779) generate({})
 // (op 779) end: generate({})
 // (op 473) NOP(779) END
 // (op 473) Assign VGPR 1:U32(788) BEGIN
 // (op 788) Assign dim(555) = 1:U32
 // (op 788) Assign stride expression
 // (op 473) Assign VGPR 1:U32(788) END
 // (op 473) NOP(830) BEGIN
 // (op 830) generate({})
 // (op 830) end: generate({})
 // (op 473) NOP(830) END
 // (op 473) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(835) BEGIN
 // (op 835) Assign dim(568) = {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}
 // tag 568: v**UNALLOCATED**
 // (op 835) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32} into DataFlowTag568: VGPR Value: UInt32 x 1: (unallocated)
 // (op 835) reg expression
 // (op 835) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 835): v10
v_and_b32 v10, -4, v1 // (op 835) 
 // (op 835) BEGIN: Tile[1]Flatten
 // (op 835) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 835): v12
v_sub_u32 v12, v1, v10 // (op 835) 
 // (op 835) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 835): v10

 // (op 835) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 835): v10
v_and_b32 v10, 4294967292, v1 // (op 835) 
 // (op 835) BEGIN: Flatten
 // (op 835) {Flatten: Add(v10:U32, {Tile[1]Flatten: v12:U32})U32}
// Allocated DataFlowTag568: 1 VGPR (Value: UInt32) (op 835): v13
v_add_u32 v13, v10, v12 // (op 835) 
 // (op 835) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 835): v12

 // Freeing : 1 VGPR (Value: UInt32) (op 835): v10

 // (op 473) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(835) END
 // (op 473) Assign VGPR 4:U32(836) BEGIN
 // (op 836) Assign dim(569) = 4:U32
 // (op 836) Assign stride expression
 // (op 473) Assign VGPR 4:U32(836) END
 // (op 473) NOP(831) BEGIN
 // (op 831) generate({})
 // (op 831) end: generate({})
 // (op 473) NOP(831) END
 // (op 473) Assign VGPR 1:U32(840) BEGIN
 // (op 840) Assign dim(571) = 1:U32
 // (op 840) Assign stride expression
 // (op 473) Assign VGPR 1:U32(840) END
 // (op 473) NOP(917) BEGIN
 // (op 917) generate({})
 // (op 917) end: generate({})
 // (op 473) NOP(917) END
 // (op 473) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(922) BEGIN
 // (op 922) Assign dim(594) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 594: v**UNALLOCATED**
 // (op 922) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag594: VGPR Value: UInt32 x 1: (unallocated)
 // (op 922) reg expression
 // (op 922) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 922): v10
v_and_b32 v10, -8, v1 // (op 922) 
 // (op 922) BEGIN: Tile[1]
 // (op 922) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 922): v12
v_sub_u32 v12, v1, v10 // (op 922) 
 // (op 922) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 922): v10

 // (op 922) BEGIN: Flatten
 // (op 922) {Flatten: ShiftL({Tile[1]: v12:U32}, 4:U32)U32}
 // (op 922) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 922)         value (Tile[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 922)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 922): v10
v_lshlrev_b32 v10, 4, v12 // (op 922) 
 // (op 922) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 922): v12

 // (op 922) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 922): v12
v_lshrrev_b32 v12, 3, v1 // (op 922) 
 // (op 922) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 922): v29
v_lshl_add_u32 v29, v12, 7, v10 // (op 922) 
 // (op 922) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 922): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 922): v12

 // (op 922) Multiply({Flatten: v29:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 922): v10
v_mul_lo_u32 v10, v29, 6 // (op 922) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 922): v29

 // (op 922) LogicalShiftR(v10:U32, 3:U32)U32
// Allocated DataFlowTag594: 1 VGPR (Value: UInt32) (op 922): v12
v_lshrrev_b32 v12, 3, v10 // (op 922) 
 // Freeing : 1 VGPR (Value: UInt32) (op 922): v10

 // (op 473) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(922) END
 // (op 473) Assign VGPR 96:U32(923) BEGIN
 // (op 923) Assign dim(595) = 96:U32
 // (op 923) Assign stride expression
 // (op 473) Assign VGPR 96:U32(923) END
 // (op 473) NOP(918) BEGIN
 // (op 918) generate({})
 // (op 918) end: generate({})
 // (op 473) NOP(918) END
 // (op 473) Assign VGPR 0:U32(927) BEGIN
 // (op 927) Assign dim(597) = 0:U32
 // (op 927) Assign stride expression
 // (op 473) Assign VGPR 0:U32(927) END
 // (op 473) NOP(941) BEGIN
 // (op 941) generate({})
 // (op 941) end: generate({})
 // (op 473) NOP(941) END
 // (op 473) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(946) BEGIN
 // (op 946) Assign dim(602) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 602: s**UNALLOCATED**
 // (op 946) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag602: SGPR Value: UInt32 x 1: (unallocated)
 // (op 946) reg expression
 // (op 946) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 946): v10
v_and_b32 v10, -8, v1 // (op 946) 
 // (op 946) BEGIN: Tile[1]
 // (op 946) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 946): v29
v_sub_u32 v29, v1, v10 // (op 946) 
 // (op 946) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 946): v10

 // (op 946) BEGIN: Flatten
 // (op 946) {Flatten: ShiftL({Tile[1]: v29:U32}, 4:U32)U32}
 // (op 946) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 946)         value (Tile[1]: VGPR Value: UInt32 x 1: v29) 
 // (op 946)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 946): v10
v_lshlrev_b32 v10, 4, v29 // (op 946) 
 // (op 946) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 946): v29

 // (op 946) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 946): v29
v_lshrrev_b32 v29, 3, v1 // (op 946) 
 // (op 946) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 946): v30
v_lshl_add_u32 v30, v29, 7, v10 // (op 946) 
 // (op 946) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 946): v10

 // Freeing : 1 VGPR (Value: UInt32) (op 946): v29

// Allocated DataFlowTag602: 1 SGPR (Value: UInt32) (op 946): s1
s_nop 0
v_readfirstlane_b32 s1, v30 // (op 946) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 946): v30

 // (op 473) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(946) END
 // (op 473) Assign VGPR 2048:U32(947) BEGIN
 // (op 947) Assign dim(603) = 2048:U32
 // (op 947) Assign stride expression
 // (op 473) Assign VGPR 2048:U32(947) END
 // (op 473) NOP(942) BEGIN
 // (op 942) generate({})
 // (op 942) end: generate({})
 // (op 473) NOP(942) END
 // (op 473) Assign VGPR 1:U32(951) BEGIN
 // (op 951) Assign dim(605) = 1:U32
 // (op 951) Assign stride expression
 // (op 473) Assign VGPR 1:U32(951) END
 // (op 473) NOP(980) BEGIN
 // (op 980) generate({})
 // (op 980) end: generate({})
 // (op 473) NOP(980) END
 // (op 473) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(985) BEGIN
 // (op 985) Assign dim(614) = {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}
 // tag 614: v**UNALLOCATED**
 // (op 985) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32} into DataFlowTag614: VGPR Value: UInt32 x 1: (unallocated)
 // (op 985) reg expression
 // (op 985) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 985): v10
v_and_b32 v10, -4, v1 // (op 985) 
 // (op 985) BEGIN: Tile[1]Flatten
 // (op 985) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 985): v29
v_sub_u32 v29, v1, v10 // (op 985) 
 // (op 985) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 985): v10

 // (op 985) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 985): v10
v_and_b32 v10, 4294967292, v1 // (op 985) 
 // (op 985) BEGIN: Flatten
 // (op 985) {Flatten: Add(v10:U32, {Tile[1]Flatten: v29:U32})U32}
// Allocated DataFlowTag614: 1 VGPR (Value: UInt32) (op 985): v30
v_add_u32 v30, v10, v29 // (op 985) 
 // (op 985) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 985): v29

 // Freeing : 1 VGPR (Value: UInt32) (op 985): v10

 // (op 473) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(985) END
 // (op 473) Assign VGPR 4:U32(986) BEGIN
 // (op 986) Assign dim(615) = 4:U32
 // (op 986) Assign stride expression
 // (op 473) Assign VGPR 4:U32(986) END
 // (op 473) NOP(981) BEGIN
 // (op 981) generate({})
 // (op 981) end: generate({})
 // (op 473) NOP(981) END
 // (op 473) Assign VGPR 1:U32(990) BEGIN
 // (op 990) Assign dim(617) = 1:U32
 // (op 990) Assign stride expression
 // (op 473) Assign VGPR 1:U32(990) END
 // (op 473) ForLoopOp KLoop: LessThan(DataFlowTag(363)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(56) BEGIN
 // (op 56) Initialize For Loop
 // (op 56) generate({57})
 // (op 56) Assign SGPR 0:U32(57) BEGIN
 // (op 57) Assign dim(363) = 0:U32
 // tag 363: s**UNALLOCATED**
 // (op 57) Generate 0:U32 into DataFlowTag363: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag363: 1 SGPR (Value: UInt32) (op 57): s14
s_mov_b32 s14, 0 // (op 57) call()
 // (op 56) Assign SGPR 0:U32(57) END
 // (op 56) end: generate({57})
 // (op 56) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 56) 
 // (op 56) Generate LessThan(DataFlowTag(363)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 56) LessThan({DataFlowTag363: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 56) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32
 // (op 56) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 56): s18, s19
s_ashr_i64 s[18:19], s[28:29], 7 // (op 56) 
// Allocated : 1 SGPR (Value: UInt32) (op 56): s15
s_mov_b32 s15, s18 // (op 56) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 56): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 56): s18
s_and_b32 s18, s15, 4294967294 // (op 56) 
 // Freeing : 1 SGPR (Value: UInt32) (op 56): s15

s_cmp_lt_u32 s14, s18 // (op 56) 
 // Freeing : 1 SGPR (Value: UInt32) (op 56): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 56) 
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56:
 // (op 56) 
 // (op 56) generate({58, 1030})
 // (op 56) Assign SGPR Add(DataFlowTag(363)U32, 2:U32)U32(58) BEGIN
 // (op 58) Assign dim(363) = Add(DataFlowTag(363)U32, 2:U32)U32
 // (op 58) Generate Add(DataFlowTag(363)U32, 2:U32)U32 into DataFlowTag363: SGPR Value: UInt32 x 1: s14
 // (op 58) reg expression
 // (op 58) Add({DataFlowTag363: s14:U32}, 2:U32)U32
 // (op 56) Barrier(1030) BEGIN
s_add_u32 s14, s14, 2 // (op 58) 
 // (op 56) Assign SGPR Add(DataFlowTag(363)U32, 2:U32)U32(58) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1030) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 56) Barrier(1030) END
 // (op 56) SetCoordinate(65) BEGIN
 // (op 65) generate({})
 // (op 65) end: generate({})
 // (op 65) generate({67})
 // (op 65) LoadLDSTile{Value: BF6x16}(67) BEGIN
 // (op 67) GEN: loadMacroTileWAVELDS OP 67 LDS 16 WaveTile 101
 // (op 67) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 67) Generate 0:U32 into nullptr
 // tag 14: v**UNALLOCATED**
 // (op 67) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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

 // (op 67) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag548: VGPR Value: UInt32 x 1: v3
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8320
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 67) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: BF6x16) (op 67): v32, v33, v34, v35, v36, v37
ds_read_b128 v[32:35], v3 offset:8320 // (op 67) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 1960 (current 1865)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
ds_read_b64 v[36:37], v3 offset:8336 // (op 67) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 1961 (current 1866)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 65) LoadLDSTile{Value: BF6x16}(67) END
 // (op 65) end: generate({67})
 // (op 56) SetCoordinate(65) END
 // (op 56) Barrier(1043) BEGIN
s_barrier  // (op 1043) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 56) Barrier(1043) END
 // (op 56) SetCoordinate(70) BEGIN
 // (op 70) generate({})
 // (op 70) end: generate({})
 // (op 70) generate({72})
 // (op 70) LoadLDSTile{Value: BF8x4}(72) BEGIN
 // (op 72) GEN: loadMacroTileWAVELDS OP 72 LDS 24 WaveTile 167
 // (op 72) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 72) Generate 1:U32 into nullptr
 // tag 22: v**UNALLOCATED**
 // (op 72) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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

 // (op 72) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag564: VGPR Value: UInt32 x 1: v4
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // (op 72) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 64

// Allocated : 8 VGPRs (Value: BF8x4) (op 72): v38, v39, v40, v41, v42, v43, v44, v45
ds_read_b128 v[38:41], v4 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected complete at 1963 (current 1868)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
ds_read_b128 v[42:45], v4 offset:64 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected stall of 92. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2056 (current 1961)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 70) LoadLDSTile{Value: BF8x4}(72) END
 // (op 70) end: generate({72})
 // (op 56) SetCoordinate(70) END
 // (op 56) Barrier(1056) BEGIN
s_barrier  // (op 1056) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 56) Barrier(1056) END
 // (op 56) SetCoordinate(79) BEGIN
 // (op 79) generate({})
 // (op 79) end: generate({})
 // (op 79) generate({81})
 // (op 79) LoadLDSTile{Value: E8M0}(81) BEGIN
 // (op 81) GEN: loadMacroTileWAVELDS OP 81 LDS 32 WaveTile 263
 // (op 81) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 81) Generate 1:U32 into nullptr
 // tag 30: v**UNALLOCATED**
 // (op 81) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 81) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag580: VGPR Value: UInt32 x 1: v0
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 81) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 81): v10
ds_read_u8 v10, v0 offset:8192 // (op 81) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2058 (current 1963)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 79) LoadLDSTile{Value: E8M0}(81) END
 // (op 79) end: generate({81})
 // (op 56) SetCoordinate(79) END
 // (op 56) Multiply(84) BEGIN
 // (op 84) Generate ScaledMatrixMultiply(WaveTile16xBF6, WaveTile4xBF8, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xBF6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[38:45], a[0:3], v16, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 84) 
 // (op 56) Multiply(84) END
 // (op 56) Deallocate{}(1100) BEGIN
 // (op 1100) Deallocate 22
 // Freeing : 8 VGPRs (Value: BF8x4) (op 72): v38, v39, v40, v41, v42, v43, v44, v45

 // Deleting tag 22
 // (op 1100) Deallocate 14
 // Freeing : 6 VGPRs (Value: BF6x16) (op 67): v32, v33, v34, v35, v36, v37

 // Deleting tag 14
 // (op 1100) Deallocate 30
 // Freeing : 1 VGPR (Value: E8M0) (op 81): v10

 // Deleting tag 30
 // (op 56) Deallocate{}(1100) END
 // (op 56) NOP(388) BEGIN
 // (op 388) generate({})
 // (op 388) end: generate({})
 // (op 56) NOP(388) END
 // (op 56) SetCoordinate(75) BEGIN
 // (op 75) generate({})
 // (op 75) end: generate({})
 // (op 75) generate({268})
 // (op 75) SetCoordinate(268) BEGIN
 // (op 268) generate({})
 // (op 268) end: generate({})
 // (op 268) generate({77})
 // (op 268) LoadTiled Value: E8M0(77) BEGIN
 // (op 77) GEN: loadMacroTileWAVE OP77 WaveTile 203
 // (op 77) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 77) Generate 1:U64 into nullptr
 // (op 77) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 507: v**UNALLOCATED**
// Allocated Offset507: 1 VGPR (Value: UInt32) (op 77): v10
v_mov_b32 v10, v18 // (op 77) 
 // (op 77) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v16
 // 	info.rowOffsetReg = Offset507: VGPR Value: UInt32 x 1: v10
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 77) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v16, v10, s[48:51], 0 offen offset:8 // (op 77) Load value
// VMEM: Expected complete at 2396 (current 1916)
 // (op 268) LoadTiled Value: E8M0(77) END
 // (op 268) end: generate({77})
 // (op 75) SetCoordinate(268) END
 // (op 75) end: generate({268})
 // (op 56) SetCoordinate(75) END
 // (op 56) Barrier(1046) BEGIN
s_barrier  // (op 1046) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 56) Barrier(1046) END
 // (op 56) SetCoordinate(271) BEGIN
 // (op 271) generate({})
 // (op 271) end: generate({})
 // (op 271) generate({420})
 // (op 271) LoadTileDirect2LDS Value: BF8x4(420) BEGIN
 // (op 420) GEN: LoadTileDirect2LDS
 // (op 420) GEN: loadMacroTileDirect2LDS OP 420 LDS 24 MacroTile 9
 // (op 420) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 420) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride489: VGPR Value: UInt64 x 1: (unallocated)
 // (op 420) reg expression
 // (op 420) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 420) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 420)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 420)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 420): s18, s19
s_lshl_b64 s[18:19], s[8:9], 4 // (op 420) 
// Allocated Stride489: 2 VGPRs (Value: UInt64) (op 420): v32, v33
v_mov_b64 v[32:33], s[18:19] // (op 420) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 420): s18, s19

 // (op 420) Generate 1:U64 into nullptr
 // (op 420) Generate 2048:U32 into nullptr
 // (op 420) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride489: VGPR Value: UInt64 x 1: v[32:33]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 488: v**UNALLOCATED**
// Allocated Offset488: 2 VGPRs (Value: UInt64) (op 420): v34, v35
v_mov_b64 v[34:35], v[14:15] // (op 420) 
// Allocated : 2 VGPRs (Value: UInt64) (op 420): v36, v37
v_mov_b64 v[36:37], v[34:35] // (op 420) 
 // (op 420) Generate Add(0:U32, {DataFlowTag552Flatten: s0:U32})U32 into nullptr
 // (op 420)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride489: v[32:33]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 420) Generate {DataFlowTag552Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 420) reg expression
 // (op 420) Generate 256:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 256 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 420): s15
s_mov_b32 s15, 256 // (op 420) call()
// 256 is not a supported value!
buffer_load_dwordx4 v36, s[44:47], s15 offen offset: 0 lds // (op 420) Load value direct to lds
// VMEM: Expected complete at 2404 (current 1924)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 420) Generate Add(v[36:37]:U64, v32:R)U64 into VGPR Value: UInt64 x 1: v[36:37]
 // (op 420) reg expression
 // (op 420) reg expression
 // (op 420) Add(v[36:37]:U64, v32:R)U64
 // (op 420) Add: dest (VGPR Value: UInt64 x 1: v[36:37]) = 
 // (op 420)      lhs (VGPR Value: UInt64 x 1: v[36:37]) 
 // (op 420)      rhs (VGPR Value: Raw32 x 1: v32)
// Allocated : 2 SGPRs (Value: Bool64) (op 420): s18, s19
v_add_co_u32 v36, s[18:19], v36, v32 // (op 420) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 420) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 420): s18, s19

 // (op 420) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 420) reg expression
 // (op 420) Add(m0:U32, 2048:U32)U32
 // (op 420) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 420) 
 // (op 420) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v36, s[44:47], s15 offen offset: 0 lds // (op 420) Load value direct to lds
// VMEM: Expected stall of 305. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2713 (current 2233)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 420): s15

 // Freeing Stride489: 2 VGPRs (Value: UInt64) (op 420): v32, v33

 // Freeing : 2 VGPRs (Value: UInt64) (op 420): v36, v37

 // (op 271) LoadTileDirect2LDS Value: BF8x4(420) END
 // (op 271) Deallocate{}(1176) BEGIN
 // (op 1176) Deallocate 554
 // Deleting tag 554
 // (op 271) Deallocate{}(1176) END
 // (op 271) end: generate({420})
 // (op 56) SetCoordinate(271) END
 // (op 56) SetCoordinate(276) BEGIN
 // (op 276) generate({})
 // (op 276) end: generate({})
 // (op 276) generate({21})
 // (op 276) LoadTiled Value: E8M0(21) BEGIN
 // (op 21) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 21) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 21) Generate 1:U64 into nullptr
 // (op 21) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 526: v**UNALLOCATED**
// Allocated Offset526: 1 VGPR (Value: UInt32) (op 21): v29
v_mov_b32 v29, v21 // (op 21) 
 // (op 21) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = Offset526: VGPR Value: UInt32 x 1: v29
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
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
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v19, v29, s[52:55], 0 offen offset:8 // (op 21) Load value
// VMEM: Expected stall of 162. CBNW: 5, Inc: 3
// VMEM: Expected complete at 2877 (current 2397)
 // (op 276) LoadTiled Value: E8M0(21) END
 // (op 276) end: generate({21})
 // (op 56) SetCoordinate(276) END
 // (op 56) SetCoordinate(263) BEGIN
 // (op 263) generate({})
 // (op 263) end: generate({})
 // (op 263) generate({4})
 // (op 263) LoadTiled Value: BF6x16(4) BEGIN
 // (op 4) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 4) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 4) Generate 0:U64 into nullptr
 // (op 4) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[22:24]
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 469: v**UNALLOCATED**
// Allocated Offset469: 1 VGPR (Value: UInt32) (op 4): v31
v_mov_b32 v31, v11 // (op 4) 
 // (op 4) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[22:24]
 // 	info.rowOffsetReg = Offset469: VGPR Value: UInt32 x 1: v31
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
 // 	info.offset = Literal Value: UInt32 x 0: 192
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 4) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_dwordx3 v[22:24], v31, s[40:43], 0 offen offset:192 // (op 4) Load value
// VMEM: Expected stall of 6. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2885 (current 2405)
 // (op 263) LoadTiled Value: BF6x16(4) END
 // (op 263) end: generate({4})
 // (op 56) SetCoordinate(263) END
 // (op 56) NOP(434) BEGIN
 // (op 434) generate({})
 // (op 434) end: generate({})
 // (op 56) NOP(434) END
 // (op 56) Barrier(1085) BEGIN
s_barrier  // (op 1085) 
 // (op 56) Barrier(1085) END
 // (op 56) StoreLDSTile Value: E8M0(240) BEGIN
 // (op 240) GEN: StoreLDSTile
 // (op 240) GEN: storeMacroTileLDS OP 240 LDS 444 MacroTile 442
 // (op 240) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 240) Generate 1:U32 into nullptr
 // (op 240) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9856
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

 // (op 240) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
 // 	info.rowOffsetReg = DataFlowTag614: VGPR Value: UInt32 x 1: v30
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9856
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 240) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v30, v17 offset:9856 // (op 240) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 2649 (current 2554)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 9856, Size: 128)
 // (op 56) StoreLDSTile Value: E8M0(240) END
 // (op 56) Deallocate{}(1152) BEGIN
 // (op 1152) Deallocate 616
 // Deleting tag 616
 // (op 56) Deallocate{}(1152) END
 // (op 56) Barrier(1073) BEGIN
s_barrier  // (op 1073) 
 // (op 56) Barrier(1073) END
 // (op 56) StoreLDSTile Value: BF6x16(229) BEGIN
 // (op 229) GEN: StoreLDSTile
 // (op 229) GEN: storeMacroTileLDS OP 229 LDS 436 MacroTile 434
 // (op 229) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 229) Generate 0:U32 into nullptr
 // (op 229) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[26:28]
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
 // 	info.offset = Literal Value: UInt32 x 0: 9984
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

 // (op 229) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[26:28]
 // 	info.rowOffsetReg = DataFlowTag594: VGPR Value: UInt32 x 1: v12
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
 // 	info.offset = Literal Value: UInt32 x 0: 9984
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 229) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b96 v12, v[26:28] offset:9984 // (op 229) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 2652 (current 2557)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 9984, Size: 1536)
 // (op 56) StoreLDSTile Value: BF6x16(229) END
 // (op 56) Deallocate{}(1132) BEGIN
 // (op 1132) Deallocate 596
 // Deleting tag 596
 // (op 56) Deallocate{}(1132) END
 // (op 56) NOP(382) BEGIN
 // (op 382) generate({})
 // (op 382) end: generate({})
 // (op 56) NOP(382) END
 // (op 56) Barrier(1070) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1070) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 9984, Size: 1536)
 // (op 56) Barrier(1070) END
 // (op 56) SetCoordinate(230) BEGIN
 // (op 230) generate({})
 // (op 230) end: generate({})
 // (op 230) generate({231})
 // (op 230) LoadLDSTile{Value: BF6x16}(231) BEGIN
 // (op 231) GEN: loadMacroTileWAVELDS OP 231 LDS 436 WaveTile 101
 // (op 231) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 231) Generate 0:U32 into nullptr
 // tag 446: v**UNALLOCATED**
 // (op 231) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9984
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

 // (op 231) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag598: VGPR Value: UInt32 x 1: v6
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9984
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 231) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: BF6x16) (op 231): v36, v37, v38, v39, v40, v41
ds_read_b128 v[36:39], v6 offset:9984 // (op 231) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2750 (current 2655)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 9984, Size: 1536)
ds_read_b64 v[40:41], v6 offset:10000 // (op 231) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 2751 (current 2656)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 9984, Size: 1536)
 // (op 230) LoadLDSTile{Value: BF6x16}(231) END
 // (op 230) Deallocate{}(1138) BEGIN
 // (op 1138) Deallocate 600
 // Deleting tag 600
 // (op 230) Deallocate{}(1138) END
 // (op 230) end: generate({231})
 // (op 56) SetCoordinate(230) END
 // (op 56) Deallocate{}(1134) BEGIN
 // (op 1134) Deallocate 436
 // Deleting tag 436
 // (op 56) Deallocate{}(1134) END
 // (op 56) Barrier(1076) BEGIN
 // (op 56) Barrier(1136) BEGIN
s_barrier  // (op 1076) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 56) Barrier(1076) END
s_barrier  // (op 1136) 
 // (op 56) Barrier(1136) END
 // (op 56) SetCoordinate(232) BEGIN
 // (op 232) generate({})
 // (op 232) end: generate({})
 // (op 232) generate({233})
 // (op 232) LoadLDSTile{Value: BF8x4}(233) BEGIN
 // (op 233) GEN: loadMacroTileWAVELDS OP 233 LDS 440 WaveTile 167
 // (op 233) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 233) Generate 1:U32 into nullptr
 // tag 448: v**UNALLOCATED**
 // (op 233) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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

 // (op 233) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag610: VGPR Value: UInt32 x 1: v7
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // (op 233) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 64

// Allocated : 8 VGPRs (Value: BF8x4) (op 233): v42, v43, v44, v45, v46, v47, v48, v49
ds_read_b128 v[42:45], v7 offset:4096 // (op 233) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected complete at 2754 (current 2659)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
ds_read_b128 v[46:49], v7 offset:4160 // (op 233) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2846 (current 2751)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 232) LoadLDSTile{Value: BF8x4}(233) END
 // (op 232) Deallocate{}(1140) BEGIN
 // (op 1140) Deallocate 612
 // Deleting tag 612
 // (op 232) Deallocate{}(1140) END
 // (op 232) end: generate({233})
 // (op 56) SetCoordinate(232) END
 // (op 56) Barrier(1082) BEGIN
s_barrier  // (op 1082) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 9856, Size: 128)
 // (op 56) Barrier(1082) END
 // (op 56) SetCoordinate(236) BEGIN
 // (op 236) generate({})
 // (op 236) end: generate({})
 // (op 236) generate({237})
 // (op 236) LoadLDSTile{Value: E8M0}(237) BEGIN
 // (op 237) GEN: loadMacroTileWAVELDS OP 237 LDS 444 WaveTile 263
 // (op 237) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 237) Generate 1:U32 into nullptr
 // tag 452: v**UNALLOCATED**
 // (op 237) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9856
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

 // (op 237) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag618: VGPR Value: UInt32 x 1: v5
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9856
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 237) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 237): v32
ds_read_u8 v32, v5 offset:9856 // (op 237) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 2848 (current 2753)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 9856, Size: 128)
 // (op 236) LoadLDSTile{Value: E8M0}(237) END
 // (op 236) Deallocate{}(1148) BEGIN
 // (op 1148) Deallocate 620
 // Deleting tag 620
 // (op 236) Deallocate{}(1148) END
 // (op 236) end: generate({237})
 // (op 56) SetCoordinate(236) END
 // (op 56) Deallocate{}(1144) BEGIN
 // (op 1144) Deallocate 444
 // Deleting tag 444
 // (op 56) Deallocate{}(1144) END
 // (op 56) Multiply(238) BEGIN
 // (op 238) Generate ScaledMatrixMultiply(WaveTile16xBF6, WaveTile4xBF8, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xBF6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
 // (op 56) Barrier(1146) BEGIN
s_barrier  // (op 1146) 
 // (op 56) Barrier(1146) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v25, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 238) 
 // (op 56) Multiply(238) END
 // (op 56) Deallocate{}(1150) BEGIN
 // (op 1150) Deallocate 452
 // Freeing : 1 VGPR (Value: E8M0) (op 237): v32

 // Deleting tag 452
 // (op 1150) Deallocate 448
 // Freeing : 8 VGPRs (Value: BF8x4) (op 233): v42, v43, v44, v45, v46, v47, v48, v49

 // Deleting tag 448
 // (op 1150) Deallocate 446
 // Freeing : 6 VGPRs (Value: BF6x16) (op 231): v36, v37, v38, v39, v40, v41

 // Deleting tag 446
 // (op 56) Deallocate{}(1150) END
 // (op 56) NOP(406) BEGIN
 // (op 406) generate({})
 // (op 406) end: generate({})
 // (op 56) NOP(406) END
 // (op 56) SetCoordinate(234) BEGIN
 // (op 234) generate({})
 // (op 234) end: generate({})
 // (op 234) generate({287})
 // (op 234) SetCoordinate(287) BEGIN
 // (op 287) generate({})
 // (op 287) end: generate({})
 // (op 287) generate({235})
 // (op 287) LoadTiled Value: E8M0(235) BEGIN
 // (op 235) GEN: loadMacroTileWAVE OP235 WaveTile 203
 // (op 235) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 235) Generate 1:U64 into nullptr
 // (op 235) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v25
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 235) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v25
 // 	info.rowOffsetReg = Offset507: VGPR Value: UInt32 x 1: v10
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 12
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 235) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_ubyte v25, v10, s[48:51], 0 offen offset:12 // (op 235) Load value
// VMEM: Expected stall of 198. CBNW: 2, Inc: 3
// VMEM: Expected complete at 3194 (current 2714)
 // (op 287) LoadTiled Value: E8M0(235) END
 // (op 287) end: generate({235})
 // (op 234) SetCoordinate(287) END
 // (op 234) end: generate({287})
 // (op 56) SetCoordinate(234) END
 // (op 56) Deallocate{}(1142) BEGIN
 // (op 1142) Deallocate 507
 // Freeing Offset507: 1 VGPR (Value: UInt32) (op 77): v10

 // Deleting tag 507
 // (op 1142) Deallocate 509
 // Deleting tag 509
 // (op 56) Deallocate{}(1142) END
 // (op 56) Assign VGPR Add(DataFlowTag(505)U32, ShiftL(Convert(DataFlowTag(506)U64)U32, 1:U32)U32)U32(620) BEGIN
 // (op 620) Assign dim(505) = Add(DataFlowTag(505)U32, ShiftL(Convert(DataFlowTag(506)U64)U32, 1:U32)U32)U32
 // (op 620) Generate Add(DataFlowTag(505)U32, ShiftL(Convert(DataFlowTag(506)U64)U32, 1:U32)U32)U32 into DataFlowTag505: VGPR Value: UInt32 x 1: v18
 // (op 620) reg expression
 // (op 620) Add({DataFlowTag505: v18:U32}, 8:U32)U32
 // (op 56) Barrier(1079) BEGIN
s_barrier  // (op 1079) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 56) Barrier(1079) END
v_add_u32 v18, 8, v18 // (op 620) 
 // (op 56) Assign VGPR Add(DataFlowTag(505)U32, ShiftL(Convert(DataFlowTag(506)U64)U32, 1:U32)U32)U32(620) END
 // (op 56) SetCoordinate(290) BEGIN
 // (op 290) generate({})
 // (op 290) end: generate({})
 // (op 290) generate({432})
 // (op 290) LoadTileDirect2LDS Value: BF8x4(432) BEGIN
 // (op 432) GEN: LoadTileDirect2LDS
 // (op 432) GEN: loadMacroTileDirect2LDS OP 432 LDS 440 MacroTile 438
 // (op 432) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 432) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride489: VGPR Value: UInt64 x 1: (unallocated)
 // (op 432) reg expression
 // (op 432) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 432) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 432)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 432)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 432): s18, s19
s_lshl_b64 s[18:19], s[8:9], 4 // (op 432) 
// Allocated Stride489: 2 VGPRs (Value: UInt64) (op 432): v32, v33
v_mov_b64 v[32:33], s[18:19] // (op 432) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 432): s18, s19

 // (op 432) Generate 1:U64 into nullptr
 // (op 432) Generate 2048:U32 into nullptr
 // (op 432) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride489: VGPR Value: UInt64 x 1: v[32:33]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 432): v36, v37
v_mov_b64 v[36:37], v[34:35] // (op 432) 
 // (op 432) Generate Add(4096:U32, {DataFlowTag602Flatten: s1:U32})U32 into nullptr
 // (op 432) reg expression
 // (op 432) Add(4096:U32, {DataFlowTag602Flatten: s1:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 432): s15
s_add_u32 s15, 4096, s1 // (op 432) 
 // (op 432)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride489: v[32:33]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 432) Generate s15:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s15 // (op 432) reg expression
 // (op 432) Generate 384:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 384 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 432): s18
s_mov_b32 s18, 384 // (op 432) call()
// 384 is not a supported value!
buffer_load_dwordx4 v36, s[44:47], s18 offen offset: 0 lds // (op 432) Load value direct to lds
// VMEM: Expected stall of 155. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3358 (current 2878)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // (op 432) Generate Add(v[36:37]:U64, v32:R)U64 into VGPR Value: UInt64 x 1: v[36:37]
 // (op 432) reg expression
 // (op 432) reg expression
 // (op 432) Add(v[36:37]:U64, v32:R)U64
 // (op 432) Add: dest (VGPR Value: UInt64 x 1: v[36:37]) = 
 // (op 432)      lhs (VGPR Value: UInt64 x 1: v[36:37]) 
 // (op 432)      rhs (VGPR Value: Raw32 x 1: v32)
// Allocated : 2 SGPRs (Value: Bool64) (op 432): s38, s39
v_add_co_u32 v36, s[38:39], v36, v32 // (op 432) least significant half
v_addc_co_u32 v37, s[38:39], 0, v37, s[38:39] // (op 432) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 432): s38, s39

 // (op 432) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 432) reg expression
 // (op 432) Add(m0:U32, 2048:U32)U32
 // (op 432) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 432) 
 // (op 432) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v36, s[44:47], s18 offen offset: 0 lds // (op 432) Load value direct to lds
// VMEM: Expected stall of 4. CBNW: 4, Inc: 3
// VMEM: Expected complete at 3366 (current 2886)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4096, Size: 4096)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 432): s18

 // Freeing Stride489: 2 VGPRs (Value: UInt64) (op 432): v32, v33

 // Freeing : 2 VGPRs (Value: UInt64) (op 432): v36, v37

 // Freeing : 1 SGPR (Value: UInt32) (op 432): s15

 // (op 290) LoadTileDirect2LDS Value: BF8x4(432) END
 // (op 290) Deallocate{}(1180) BEGIN
 // (op 1180) Deallocate 604
 // Deleting tag 604
 // (op 290) Deallocate{}(1180) END
 // (op 290) end: generate({432})
 // (op 56) SetCoordinate(290) END
 // (op 56) Deallocate{}(1156) BEGIN
 // (op 1156) Deallocate 488
 // Freeing Offset488: 2 VGPRs (Value: UInt64) (op 420): v34, v35

 // Deleting tag 488
 // (op 1156) Deallocate 490
 // Deleting tag 490
 // (op 56) Deallocate{}(1156) END
 // (op 56) SetCoordinate(295) BEGIN
 // (op 295) generate({})
 // (op 295) end: generate({})
 // (op 295) generate({242})
 // (op 295) LoadTiled Value: E8M0(242) BEGIN
 // (op 242) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 242) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 242) Generate 1:U64 into nullptr
 // (op 242) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 242) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v17
 // 	info.rowOffsetReg = Offset526: VGPR Value: UInt32 x 1: v29
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 12
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 242) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 56) Assign VGPR Add(DataFlowTag(486)U64, ShiftL(DataFlowTag(487)U64, 1:U32)U64)U64(556) BEGIN
 // (op 556) Assign dim(486) = Add(DataFlowTag(486)U64, ShiftL(DataFlowTag(487)U64, 1:U32)U64)U64
 // (op 556) Generate Add(DataFlowTag(486)U64, ShiftL(DataFlowTag(487)U64, 1:U32)U64)U64 into DataFlowTag486: VGPR Value: UInt64 x 1: v[14:15]
 // (op 556) reg expression
 // (op 556) Add({DataFlowTag486: v[14:15]:U64}, 256:U64)U64
 // (op 556) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 556)      lhs (DataFlowTag486: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 556)      rhs (Literal Value: UInt64 x 0: 256)
v_add_co_u32 v14, vcc, 256, v14 // (op 556) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 556) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 56) Assign VGPR Add(DataFlowTag(486)U64, ShiftL(DataFlowTag(487)U64, 1:U32)U64)U64(556) END
buffer_load_ubyte v17, v29, s[52:55], 0 offen offset:12 // (op 242) Load value
// VMEM: Expected stall of 306. CBNW: 5, Inc: 3
// VMEM: Expected complete at 3675 (current 3195)
 // (op 295) LoadTiled Value: E8M0(242) END
 // (op 295) end: generate({242})
 // (op 56) SetCoordinate(295) END
 // (op 56) Deallocate{}(1158) BEGIN
 // (op 1158) Deallocate 526
 // Freeing Offset526: 1 VGPR (Value: UInt32) (op 21): v29

 // Deleting tag 526
 // (op 1158) Deallocate 528
 // Deleting tag 528
 // (op 56) Deallocate{}(1158) END
 // (op 56) SetCoordinate(282) BEGIN
 // (op 282) generate({})
 // (op 282) end: generate({})
 // (op 282) generate({228})
 // (op 282) LoadTiled Value: BF6x16(228) BEGIN
 // (op 228) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 228) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 228) Generate 0:U64 into nullptr
 // (op 228) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[26:28]
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 228) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[26:28]
 // 	info.rowOffsetReg = Offset469: VGPR Value: UInt32 x 1: v31
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
 // 	info.offset = Literal Value: UInt32 x 0: 288
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 228) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 56) Assign VGPR Add(DataFlowTag(524)U32, ShiftL(Convert(DataFlowTag(525)U64)U32, 1:U32)U32)U32(684) BEGIN
 // (op 684) Assign dim(524) = Add(DataFlowTag(524)U32, ShiftL(Convert(DataFlowTag(525)U64)U32, 1:U32)U32)U32
 // (op 684) Generate Add(DataFlowTag(524)U32, ShiftL(Convert(DataFlowTag(525)U64)U32, 1:U32)U32)U32 into DataFlowTag524: VGPR Value: UInt32 x 1: v21
 // (op 684) reg expression
 // (op 684) Add({DataFlowTag524: v21:U32}, 8:U32)U32
v_add_u32 v21, 8, v21 // (op 684) 
 // (op 56) Assign VGPR Add(DataFlowTag(524)U32, ShiftL(Convert(DataFlowTag(525)U64)U32, 1:U32)U32)U32(684) END
buffer_load_dwordx3 v[26:28], v31, s[40:43], 0 offen offset:288 // (op 228) Load value
// VMEM: Expected stall of 162. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3839 (current 3359)
 // (op 282) LoadTiled Value: BF6x16(228) END
 // (op 282) end: generate({228})
 // (op 56) SetCoordinate(282) END
 // (op 56) Deallocate{}(1154) BEGIN
 // (op 1154) Deallocate 471
 // Deleting tag 471
 // (op 1154) Deallocate 469
 // Freeing Offset469: 1 VGPR (Value: UInt32) (op 4): v31

 // Deleting tag 469
 // (op 56) Deallocate{}(1154) END
 // (op 56) NOP(422) BEGIN
 // (op 422) generate({})
 // (op 422) end: generate({})
 // (op 56) NOP(422) END
 // (op 56) Assign VGPR Add(DataFlowTag(467)U32, ShiftL(Convert(DataFlowTag(468)U64)U32, 1:U32)U32)U32(492) BEGIN
 // (op 492) Assign dim(467) = Add(DataFlowTag(467)U32, ShiftL(Convert(DataFlowTag(468)U64)U32, 1:U32)U32)U32
 // (op 492) Generate Add(DataFlowTag(467)U32, ShiftL(Convert(DataFlowTag(468)U64)U32, 1:U32)U32)U32 into DataFlowTag467: VGPR Value: UInt32 x 1: v11
 // (op 492) reg expression
 // (op 492) Add({DataFlowTag467: v11:U32}, 192:U32)U32
 // (op 56) Barrier(1059) BEGIN
s_barrier  // (op 1059) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 56) Barrier(1059) END
v_add_u32 v11, 192, v11 // (op 492) 
 // (op 56) Assign VGPR Add(DataFlowTag(467)U32, ShiftL(Convert(DataFlowTag(468)U64)U32, 1:U32)U32)U32(492) END
 // (op 56) StoreLDSTile Value: E8M0(26) BEGIN
 // (op 26) GEN: StoreLDSTile
 // (op 26) GEN: storeMacroTileLDS OP 26 LDS 32 MacroTile 10
 // (op 26) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 26) Generate 1:U32 into nullptr
 // (op 26) 	Dir = Store
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 26) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag568: VGPR Value: UInt32 x 1: v13
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 26) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(6)
ds_write_b8 v13, v19 offset:8192 // (op 26) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 3793 (current 3698)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 56) StoreLDSTile Value: E8M0(26) END
 // (op 56) Deallocate{}(1090) BEGIN
 // (op 1090) Deallocate 570
 // Deleting tag 570
 // (op 56) Deallocate{}(1090) END
 // (op 56) Barrier(1033) BEGIN
s_barrier  // (op 1033) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 56) Barrier(1033) END
 // (op 56) StoreLDSTile Value: BF6x16(6) BEGIN
 // (op 6) GEN: StoreLDSTile
 // (op 6) GEN: storeMacroTileLDS OP 6 LDS 16 MacroTile 8
 // (op 6) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 6) Generate 0:U32 into nullptr
 // (op 6) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[22:24]
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[22:24]
 // 	info.rowOffsetReg = DataFlowTag536: VGPR Value: UInt32 x 1: v20
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(5)
ds_write_b96 v20, v[22:24] offset:8320 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 3796 (current 3701)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 56) StoreLDSTile Value: BF6x16(6) END
 // (op 56) Deallocate{}(1088) BEGIN
 // (op 1088) Deallocate 538
 // Deleting tag 538
 // (op 56) Deallocate{}(1088) END
 // (op 56) NOP(383) BEGIN
 // (op 383) generate({})
 // (op 383) end: generate({})
 // (op 56) NOP(383) END
 // (op 56) end: generate({58, 1030})
 // (op 56) For Loop Increment
 // (op 56) generate({})
 // (op 56) end: generate({})
 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
 // (op 56) Generate LessThan(DataFlowTag(363)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 56) LessThan({DataFlowTag363: s14:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 56) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32, 4294967294:U32)U32
 // (op 56) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 56): s18, s19
s_ashr_i64 s[18:19], s[28:29], 7 // (op 56) 
// Allocated : 1 SGPR (Value: UInt32) (op 56): s15
s_mov_b32 s15, s18 // (op 56) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 56): s18, s19

// Allocated : 1 SGPR (Value: UInt32) (op 56): s18
s_and_b32 s18, s15, 4294967294 // (op 56) 
 // Freeing : 1 SGPR (Value: UInt32) (op 56): s15

s_cmp_lt_u32 s14, s18 // (op 56) 
 // Freeing : 1 SGPR (Value: UInt32) (op 56): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 56) 
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56:
 // (op 56) 
 // (op 56) Unlock For Loop
 // Unlock None
 // (op 473) ForLoopOp KLoop: LessThan(DataFlowTag(363)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32)BL(56) END
 // (op 473) Deallocate{}(1096) BEGIN
 // (op 1096) Deallocate 438
 // Deleting tag 438
 // (op 1096) Deallocate 506
 // Deleting tag 506
 // (op 1096) Deallocate 594
 // Freeing DataFlowTag594: 1 VGPR (Value: UInt32) (op 922): v12

 // Deleting tag 594
 // (op 1096) Deallocate 467
 // Freeing DataFlowTag467: 1 VGPR (Value: UInt32) (op 500): v11

 // Deleting tag 467
 // (op 1096) Deallocate 595
 // Deleting tag 595
 // (op 1096) Deallocate 468
 // Deleting tag 468
 // (op 1096) Deallocate 597
 // Deleting tag 597
 // (op 1096) Deallocate 603
 // Deleting tag 603
 // (op 1096) Deallocate 470
 // Deleting tag 470
 // (op 1096) Deallocate 472
 // Deleting tag 472
 // (op 1096) Deallocate 363
 // Freeing DataFlowTag363: 1 SGPR (Value: UInt32) (op 57): s14

 // Deleting tag 363
 // (op 1096) Deallocate 486
 // Freeing DataFlowTag486: 2 VGPRs (Value: UInt64) (op 564): v14, v15

 // Deleting tag 486
 // (op 1096) Deallocate 491
 // Deleting tag 491
 // (op 1096) Deallocate 510
 // Deleting tag 510
 // (op 1096) Deallocate 617
 // Deleting tag 617
 // (op 1096) Deallocate 8
 // Freeing : 3 VGPRs (Value: BF6x16) (op 351): v22, v23, v24

 // Deleting tag 8
 // (op 1096) Deallocate 450
 // Deleting tag 450
 // (op 1096) Deallocate 442
 // Deleting tag 442
 // (op 1096) Deallocate 569
 // Deleting tag 569
 // (op 1096) Deallocate 537
 // Deleting tag 537
 // (op 1096) Deallocate 539
 // Deleting tag 539
 // (op 1096) Deallocate 553
 // Deleting tag 553
 // (op 1096) Deallocate 555
 // Deleting tag 555
 // (op 1096) Deallocate 568
 // Freeing DataFlowTag568: 1 VGPR (Value: UInt32) (op 835): v13

 // Deleting tag 568
 // (op 1096) Deallocate 571
 // Deleting tag 571
 // (op 1096) Deallocate 602
 // Freeing DataFlowTag602: 1 SGPR (Value: UInt32) (op 946): s1

 // Deleting tag 602
 // (op 1096) Deallocate 605
 // Deleting tag 605
 // (op 1096) Deallocate 615
 // Deleting tag 615
 // (op 1096) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 348): v19

 // Deleting tag 10
 // (op 1096) Deallocate 524
 // Freeing DataFlowTag524: 1 VGPR (Value: UInt32) (op 692): v21

 // Deleting tag 524
 // (op 1096) Deallocate 9
 // Deleting tag 9
 // (op 1096) Deallocate 49
 // Freeing : 1 VGPR (Value: E8M0) (op 341): v16

 // Deleting tag 49
 // (op 1096) Deallocate 434
 // Deleting tag 434
 // (op 1096) Deallocate 536
 // Freeing DataFlowTag536: 1 VGPR (Value: UInt32) (op 730): v20

 // Deleting tag 536
 // (op 1096) Deallocate 529
 // Deleting tag 529
 // (op 1096) Deallocate 527
 // Deleting tag 527
 // (op 1096) Deallocate 525
 // Deleting tag 525
 // (op 1096) Deallocate 552
 // Freeing DataFlowTag552: 1 SGPR (Value: UInt32) (op 783): s0

 // Deleting tag 552
 // (op 1096) Deallocate 614
 // Freeing DataFlowTag614: 1 VGPR (Value: UInt32) (op 985): v30

 // Deleting tag 614
 // (op 1096) Deallocate 487
 // Deleting tag 487
 // (op 1096) Deallocate 440
 // Deleting tag 440
 // (op 1096) Deallocate 508
 // Deleting tag 508
 // (op 1096) Deallocate 489
 // Deleting tag 489
 // (op 1096) Deallocate 505
 // Freeing DataFlowTag505: 1 VGPR (Value: UInt32) (op 628): v18

 // Deleting tag 505
 // (op 473) Deallocate{}(1096) END
 // (op 473) Barrier(1098) BEGIN
s_barrier  // (op 1098) 
 // (op 473) Barrier(1098) END
 // (op 473) end: generate({468})
 // (op 473) Unlock Scope 473
 // Unlock None
 // (op 314) Scope(473) END
 // (op 314) SetCoordinate(205) BEGIN
 // (op 205) generate({})
 // (op 205) end: generate({})
 // (op 205) generate({206})
 // (op 205) SetCoordinate(206) BEGIN
 // (op 206) generate({})
 // (op 206) end: generate({})
 // (op 206) generate({450})
 // (op 206) Scope(450) BEGIN
 // (op 450) Lock Scope 450
 // Lock Branch
 // (op 450) generate({444})
 // (op 450) NOP(444) BEGIN
 // (op 444) generate({})
 // (op 444) end: generate({})
 // (op 450) NOP(444) END
 // (op 450) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(455) BEGIN
 // (op 455) Assign dim(454) = Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32
 // tag 454: v**UNALLOCATED**
 // (op 455) Generate Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32 into DataFlowTag454: VGPR Value: UInt32 x 1: (unallocated)
 // (op 455) reg expression
 // (op 455) reg expression
 // (op 455) reg expression
 // (op 455) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
 // (op 455) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 455): v10
v_lshrrev_b32 v10, 3, v1 // (op 455) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 455): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 455) 
 // (op 455) BEGIN: Tile
 // (op 455) convert
// Allocated : 1 VGPR (Value: UInt32) (op 455): v11
v_lshl_add_u32 v11, s16, 4, v10 // (op 455) 
 // (op 455) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 455): v10

 // (op 455) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 455) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 455): s14
s_lshr_b32 s14, s0, 1 // (op 455) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 455): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 455): v10
v_and_b32 v10, -8, v1 // (op 455) 
 // (op 455) reg expression
 // (op 455) BEGIN: Flatten[1]
 // (op 455) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v10:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 455): v12
v_sub_u32 v12, v1, v10 // (op 455) 
 // (op 455) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 455): v10

 // (op 455) BEGIN: Tile
 // (op 455) {Tile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 455) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 455)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 455)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 455): v10
v_lshlrev_b32 v10, 4, v12 // (op 455) 
 // (op 455) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 455): v12

 // (op 455) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 455): v12
v_lshl_add_u32 v12, s14, 8, v10 // (op 455) 
 // (op 455) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 455): v10

 // Freeing : 1 SGPR (Value: UInt32) (op 455): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 455): v18, v19
v_mov_b32 v19, 0 // (op 455) convert
v_mov_b32 v18, v12 // (op 455) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 455): v12

 // (op 455) BEGIN: Split
 // (op 455) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 455)           lhs (Tile: VGPR Value: UInt32 x 1: v11) 
 // (op 455)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated : 2 VGPRs (Value: UInt64) (op 455): v12, v13
v_mul_lo_u32 v13, v11, s3 // (op 455) most significant: low of low * high
 // (op 455) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 455): v10
v_mul_hi_u32 v10, v11, s2 // (op 455) most significant: high of low * low
v_mul_lo_u32 v12, v11, s2 // (op 455) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 455) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 455): v10

 // (op 455) Add: dest (VGPR Value: UInt64 x 1: v[12:13]) = 
 // (op 455)      lhs (VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 455)      rhs (Tile: VGPR Value: UInt64 x 1: v[18:19])
// Allocated : 2 SGPRs (Value: Bool64) (op 455): s0, s1
v_add_co_u32 v12, s[0:1], v12, v18 // (op 455) least significant half
v_addc_co_u32 v13, s[0:1], v13, v19, s[0:1] // (op 455) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 455): s0, s1

 // (op 455) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 455): v18, v19

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 455): v11

 // (op 455) Multiply({Split: v[12:13]:U64}, 6:U32)U64
 // (op 455) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 455)           lhs (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 455)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 455) low of low * high omitted due to zero input.
// Allocated : 2 VGPRs (Value: UInt64) (op 455): v10, v11
v_mul_lo_u32 v11, v13, 6 // (op 455) most significant: low of high * low
// Allocated : 1 VGPR (Value: Int32) (op 455): v14
v_mul_hi_u32 v14, v12, 6 // (op 455) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 455) least significant: low of low * low
v_add_u32 v11, v11, v14 // (op 455) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 455): v14

 // Freeing Split: 2 VGPRs (Value: UInt64) (op 455): v12, v13

 // (op 455) LogicalShiftR(v[10:11]:U64, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 455): v12, v13
v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 455) 
 // Freeing : 2 VGPRs (Value: UInt64) (op 455): v10, v11

// Allocated DataFlowTag454: 1 VGPR (Value: UInt32) (op 455): v10
v_mov_b32 v10, v12 // (op 455) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 455): v12, v13

 // (op 450) Assign VGPR Convert(LogicalShiftR(Multiply({Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(397)U32, 4:U32, LogicalShiftR(DataFlowTag(74)U32, 3:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(74)U32, BitwiseAnd(DataFlowTag(74)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64)U64}, 6:U32)U64, 3:U32)U64)U32(455) END
 // (op 450) Assign VGPR 96:U64(456) BEGIN
 // (op 456) Assign dim(455) = 96:U64
 // (op 456) Assign stride expression
 // (op 450) Assign VGPR 96:U64(456) END
 // (op 450) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(457) BEGIN
 // (op 457) Assign dim(457) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 457) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // (op 457) reg expression
 // (op 457) reg expression
 // (op 457) Multiply({Tensor_0_size_0_1: s[6:7]:I64}, 6:U32)I64
 // (op 457) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 457)           lhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]) 
 // (op 457)           rhs (Literal Value: UInt32 x 0: 6)
 // (op 457) low of low * high omitted due to zero input.
// Allocated : 2 SGPRs (Value: Int64) (op 457): s0, s1
s_mul_i32 s1, s7, 6 // (op 457) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 457) least significant
// Allocated : 1 SGPR (Value: Int32) (op 457): s14
s_mul_hi_u32 s14, s6, 6 // (op 457) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 457) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 457): s14

 // (op 457) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, s[0:1]:I64)I64
 // (op 457) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 457)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 457)           rhs (SGPR Value: Int64 x 1: s[0:1])
// Allocated : 2 SGPRs (Value: Int64) (op 457): s14, s15
s_mul_i32 s15, s2, s1 // (op 457) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 457): s18
s_mul_i32 s18, s3, s0 // (op 457) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 457) least significant
// Allocated : 1 SGPR (Value: Int32) (op 457): s19
s_mul_hi_u32 s19, s2, s0 // (op 457) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 457) most significant: sum
s_add_u32 s15, s15, s19 // (op 457) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 457): s19

 // Freeing : 1 SGPR (Value: Int32) (op 457): s18

 // Freeing : 2 SGPRs (Value: Int64) (op 457): s0, s1

 // (op 457) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 457): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 457) 
 // Freeing : 2 SGPRs (Value: Int64) (op 457): s14, s15

 // (op 457) reg expression
 // (op 457) reg expression
s_mov_b32 s40, s4 // (op 457) reg expression
s_mov_b32 s41, s5 // (op 457) reg expression
s_mov_b32 s42, s0 // (op 457) reg expression
s_mov_b32 s43, 131072 // (op 457) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 457): s0, s1

 // (op 450) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(ArithmeticShiftR(Multiply(Tensor_0_stride_0_0:I64, Multiply(Tensor_0_size_0_1:I64, 6:U32)I64)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(457) END
 // (op 450) NOP(446) BEGIN
 // (op 446) generate({})
 // (op 446) end: generate({})
 // (op 450) NOP(446) END
 // (op 450) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(462) BEGIN
 // (op 462) Assign dim(459) = LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64
 // (op 462) Assign stride expression
 // (op 450) Assign VGPR LogicalShiftR(Multiply(Convert(Tensor_0_stride_0_0:I64)U64, 6:U32)U64, 3:U32)U64(462) END
 // (op 450) NOP(447) BEGIN
 // (op 447) generate({})
 // (op 447) end: generate({})
 // (op 450) NOP(447) END
 // (op 450) Assign VGPR 0:U64(465) BEGIN
 // (op 465) Assign dim(461) = 0:U64
 // (op 465) Assign stride expression
 // (op 450) Assign VGPR 0:U64(465) END
 // (op 450) NOP(513) BEGIN
 // (op 513) generate({})
 // (op 513) end: generate({})
 // (op 450) NOP(513) END
 // (op 450) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(522) BEGIN
 // (op 522) Assign dim(473) = {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 473: v**UNALLOCATED**
 // (op 522) Generate {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag473: VGPR Value: UInt64 x 1: (unallocated)
 // (op 522) reg expression
 // (op 522) reg expression
 // (op 522) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 522) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 522): v11
v_and_b32 v11, -8, v1 // (op 522) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 522): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 522) 
 // (op 522) BEGIN: Flatten[1]
 // (op 522) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
 // (op 522) convert
// Allocated : 1 VGPR (Value: UInt32) (op 522): v12
v_sub_u32 v12, v1, v11 // (op 522) 
 // (op 522) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 522): v11

 // (op 522) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 522) BEGIN: Tile
 // (op 522) {Tile: ShiftL({Flatten[1]: v12:U32}, 4:U32)U32}
 // (op 522) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 522)         value (Flatten[1]: VGPR Value: UInt32 x 1: v12) 
 // (op 522)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 SGPR (Value: UInt32) (op 522): s14
s_lshr_b32 s14, s0, 1 // (op 522) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 522): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 522): v11
v_lshlrev_b32 v11, 4, v12 // (op 522) 
 // (op 522) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 522): v12

 // (op 522) reg expression
 // (op 522) BEGIN: Tile
 // (op 522) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 522): v12
v_lshl_add_u32 v12, s14, 8, v11 // (op 522) 
 // (op 522) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 522): v11

 // Freeing : 1 SGPR (Value: UInt32) (op 522): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 522): v18, v19
v_mov_b32 v19, 0 // (op 522) convert
v_mov_b32 v18, v12 // (op 522) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 522): v12

 // (op 522) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 522): v11
v_lshrrev_b32 v11, 3, v1 // (op 522) 
 // (op 522) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 522): v12
v_lshl_add_u32 v12, s17, 5, v11 // (op 522) 
 // (op 522) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 522): v11

 // (op 522) Multiply({Tile: v12:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 522) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 522)           lhs (Tile: VGPR Value: UInt32 x 1: v12) 
 // (op 522)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 522): v20, v21
v_mul_lo_u32 v21, v12, s9 // (op 522) most significant: low of low * high
 // (op 522) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 522): v11
v_mul_hi_u32 v11, v12, s8 // (op 522) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 522) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 522) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 522): v11

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 522): v12

 // (op 522) BEGIN: Split
 // (op 522) {Split: Add({Tile: v[18:19]:U64}, v[20:21]:I64)U64}
 // (op 522) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 522)      lhs (Tile: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 522)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated DataFlowTag473: 2 VGPRs (Value: UInt64) (op 522): v22, v23
// Allocated : 2 SGPRs (Value: Bool64) (op 522): s0, s1
v_add_co_u32 v22, s[0:1], v18, v20 // (op 522) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 522) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 522): s0, s1

 // (op 522) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 522): v20, v21

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 522): v18, v19

 // (op 450) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(140)U32, BitwiseAnd(DataFlowTag(140)U32, -8:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(399)U32, 5:U32, LogicalShiftR(DataFlowTag(140)U32, 3:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(522) END
 // (op 450) Assign VGPR 128:U64(523) BEGIN
 // (op 523) Assign dim(474) = 128:U64
 // (op 523) Assign stride expression
 // (op 450) Assign VGPR 128:U64(523) END
 // (op 450) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(524) BEGIN
 // (op 524) Assign dim(476) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 524) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // (op 524) reg expression
 // (op 524) reg expression
 // (op 524) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 524) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 524)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 524)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 524): s0, s1
s_mul_i32 s1, s8, s13 // (op 524) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 524): s14
s_mul_i32 s14, s9, s12 // (op 524) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 524) least significant
// Allocated : 1 SGPR (Value: Int32) (op 524): s15
s_mul_hi_u32 s15, s8, s12 // (op 524) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 524) most significant: sum
s_add_u32 s1, s1, s15 // (op 524) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 524): s15

 // Freeing : 1 SGPR (Value: Int32) (op 524): s14

 // (op 524) reg expression
 // (op 524) reg expression
s_mov_b32 s44, s10 // (op 524) reg expression
s_mov_b32 s45, s11 // (op 524) reg expression
s_mov_b32 s46, s0 // (op 524) reg expression
s_mov_b32 s47, 131072 // (op 524) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 524): s0, s1

 // (op 450) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(524) END
 // (op 450) NOP(515) BEGIN
 // (op 515) generate({})
 // (op 515) end: generate({})
 // (op 450) NOP(515) END
 // (op 450) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(529) BEGIN
 // (op 529) Assign dim(478) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 529) Assign stride expression
 // (op 450) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(529) END
 // (op 450) NOP(516) BEGIN
 // (op 516) generate({})
 // (op 516) end: generate({})
 // (op 450) NOP(516) END
 // (op 450) Assign VGPR 1:U64(532) BEGIN
 // (op 532) Assign dim(480) = 1:U64
 // (op 532) Assign stride expression
 // (op 450) Assign VGPR 1:U64(532) END
 // (op 450) NOP(577) BEGIN
 // (op 577) generate({})
 // (op 577) end: generate({})
 // (op 450) NOP(577) END
 // (op 450) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32})U32}(586) BEGIN
 // (op 586) Assign dim(492) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32})U32}
 // tag 492: v**UNALLOCATED**
 // (op 586) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32})U32} into DataFlowTag492: VGPR Value: UInt32 x 1: (unallocated)
 // (op 586) reg expression
 // (op 586) reg expression
 // (op 586) reg expression
 // (op 586) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 586) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 586): v24
v_lshrrev_b32 v24, 7, v1 // (op 586) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 586): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 586) 
 // (op 586) BEGIN: Flatten[1]
 // (op 586) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
 // (op 586) convert
// Allocated : 1 VGPR (Value: UInt32) (op 586): v11
v_and_b32 v11, 15, v1 // (op 586) 
 // (op 586) END: Flatten[1]
 // (op 586) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 586) BEGIN: Tile
// Allocated : 1 SGPR (Value: UInt32) (op 586): s14
s_lshr_b32 s14, s0, 1 // (op 586) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 586): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 586): v12
v_lshl_add_u32 v12, v24, 4, v11 // (op 586) 
 // (op 586) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 586): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 586): v24

 // (op 586) reg expression
 // (op 586) BEGIN: Tile
 // (op 586) convert
// Allocated : 1 VGPR (Value: UInt32) (op 586): v11
v_lshl_add_u32 v11, s16, 4, v12 // (op 586) 
 // (op 586) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 586): v12

 // (op 586) BEGIN: Flatten[1]
 // (op 586) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 586): v24
v_and_b32 v24, 63, v1 // (op 586) 
 // (op 586) END: Flatten[1]
 // (op 586) LogicalShiftR({Flatten[1]: v24:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 586): v12
v_lshrrev_b32 v12, 4, v24 // (op 586) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 586): v24

 // (op 586) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 586): v24
v_lshl_add_u32 v24, s14, 3, v12 // (op 586) 
 // (op 586) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 586): v12

 // Freeing : 1 SGPR (Value: UInt32) (op 586): s14

 // (op 586) BEGIN: Split
// Allocated DataFlowTag492: 1 VGPR (Value: UInt32) (op 586): v12
v_mul_lo_u32 v12, v11, s20 // (op 586) 
v_add_u32 v12, v12, v24 // (op 586) 
 // (op 586) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 586): v24

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 586): v11

 // (op 450) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(401)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(202)U32, 7:U32)U32, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 15:U32)U32})U32})U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(202)U32, 63:U32)U32}, 4:U32)U32)U32})U32}(586) END
 // (op 450) Assign VGPR 4:U64(587) BEGIN
 // (op 587) Assign dim(493) = 4:U64
 // (op 587) Assign stride expression
 // (op 450) Assign VGPR 4:U64(587) END
 // (op 450) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(588) BEGIN
 // (op 588) Assign dim(495) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 588) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // (op 588) reg expression
 // (op 588) reg expression
 // (op 588) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 588) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 588)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 588)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 588): s0, s1
s_mul_i32 s1, s20, s7 // (op 588) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 588): s14
s_mul_i32 s14, s21, s6 // (op 588) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 588) least significant
// Allocated : 1 SGPR (Value: Int32) (op 588): s15
s_mul_hi_u32 s15, s20, s6 // (op 588) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 588) most significant: sum
s_add_u32 s1, s1, s15 // (op 588) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 588): s15

 // Freeing : 1 SGPR (Value: Int32) (op 588): s14

 // (op 588) reg expression
 // (op 588) reg expression
s_mov_b32 s48, s22 // (op 588) reg expression
s_mov_b32 s49, s23 // (op 588) reg expression
s_mov_b32 s50, s0 // (op 588) reg expression
s_mov_b32 s51, 131072 // (op 588) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 588): s0, s1

 // (op 450) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(588) END
 // (op 450) NOP(579) BEGIN
 // (op 579) generate({})
 // (op 579) end: generate({})
 // (op 450) NOP(579) END
 // (op 450) Assign VGPR 4:U64(593) BEGIN
 // (op 593) Assign dim(497) = 4:U64
 // (op 593) Assign stride expression
 // (op 450) Assign VGPR 4:U64(593) END
 // (op 450) NOP(580) BEGIN
 // (op 580) generate({})
 // (op 580) end: generate({})
 // (op 450) NOP(580) END
 // (op 450) Assign VGPR 1:U64(596) BEGIN
 // (op 596) Assign dim(499) = 1:U64
 // (op 596) Assign stride expression
 // (op 450) Assign VGPR 1:U64(596) END
 // (op 450) NOP(641) BEGIN
 // (op 641) generate({})
 // (op 641) end: generate({})
 // (op 450) NOP(641) END
 // (op 450) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(650) BEGIN
 // (op 650) Assign dim(511) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 511: v**UNALLOCATED**
 // (op 650) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag511: VGPR Value: UInt32 x 1: (unallocated)
 // (op 650) reg expression
 // (op 650) reg expression
 // (op 650) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
 // (op 650) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 650): v11
v_and_b32 v11, -4, v1 // (op 650) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 650): s0, s1
s_ashr_i64 s[0:1], s[28:29], 7 // (op 650) 
 // (op 650) BEGIN: Flatten[1]Tile
 // (op 650) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
 // (op 650) convert
// Allocated : 1 VGPR (Value: UInt32) (op 650): v24
v_sub_u32 v24, v1, v11 // (op 650) 
 // (op 650) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 650): v11

 // (op 650) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 650) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 650): s14
s_lshr_b32 s14, s0, 1 // (op 650) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 650): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 650): v11
v_lshrrev_b32 v11, 2, v1 // (op 650) 
 // (op 650) reg expression
 // (op 650) BEGIN: Tile
 // (op 650) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 650): v13
v_lshl_add_u32 v13, s14, 3, v24 // (op 650) 
 // (op 650) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 650): v24

 // Freeing : 1 SGPR (Value: UInt32) (op 650): s14

 // (op 650) convert
 // (op 650) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 650): v24
v_lshl_add_u32 v24, s17, 5, v11 // (op 650) 
 // (op 650) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 650): v11

 // (op 650) Multiply({Tile: v24:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 650): v11
v_mul_lo_u32 v11, v24, s24 // (op 650) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 650): v24

 // (op 650) BEGIN: Split
 // (op 650) {Split: Add({Tile: v13:U32}, v11:U32)U32}
// Allocated DataFlowTag511: 1 VGPR (Value: UInt32) (op 650): v24
v_add_u32 v24, v13, v11 // (op 650) 
 // (op 650) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 650): v11

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 650): v13

 // (op 450) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 1:U32)U32, 3:U32, {Flatten[1]Tile: Subtract(DataFlowTag(236)U32, BitwiseAnd(DataFlowTag(236)U32, -4:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(403)U32, 5:U32, LogicalShiftR(DataFlowTag(236)U32, 2:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(650) END
 // (op 450) Assign VGPR 4:U64(651) BEGIN
 // (op 651) Assign dim(512) = 4:U64
 // (op 651) Assign stride expression
 // (op 450) Assign VGPR 4:U64(651) END
 // (op 450) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(652) BEGIN
 // (op 652) Assign dim(514) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 652) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // (op 652) reg expression
 // (op 652) reg expression
 // (op 652) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 652) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 652)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 652)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 652): s0, s1
s_mul_i32 s1, s24, s13 // (op 652) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 652): s14
s_mul_i32 s14, s25, s12 // (op 652) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 652) least significant
// Allocated : 1 SGPR (Value: Int32) (op 652): s15
s_mul_hi_u32 s15, s24, s12 // (op 652) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 652) most significant: sum
s_add_u32 s1, s1, s15 // (op 652) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 652): s15

 // Freeing : 1 SGPR (Value: Int32) (op 652): s14

 // (op 652) reg expression
 // (op 652) reg expression
s_mov_b32 s52, s26 // (op 652) reg expression
s_mov_b32 s53, s27 // (op 652) reg expression
s_mov_b32 s54, s0 // (op 652) reg expression
s_mov_b32 s55, 131072 // (op 652) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 652): s0, s1

 // (op 450) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(652) END
 // (op 450) NOP(643) BEGIN
 // (op 643) generate({})
 // (op 643) end: generate({})
 // (op 450) NOP(643) END
 // (op 450) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(657) BEGIN
 // (op 657) Assign dim(516) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 657) Assign stride expression
 // (op 450) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(657) END
 // (op 450) NOP(644) BEGIN
 // (op 644) generate({})
 // (op 644) end: generate({})
 // (op 450) NOP(644) END
 // (op 450) Assign VGPR 1:U64(660) BEGIN
 // (op 660) Assign dim(518) = 1:U64
 // (op 660) Assign stride expression
 // (op 450) Assign VGPR 1:U64(660) END
 // (op 450) NOP(738) BEGIN
 // (op 738) generate({})
 // (op 738) end: generate({})
 // (op 450) NOP(738) END
 // (op 450) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(743) BEGIN
 // (op 743) Assign dim(540) = LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32
 // tag 540: v**UNALLOCATED**
 // (op 743) Generate LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32 into DataFlowTag540: VGPR Value: UInt32 x 1: (unallocated)
 // (op 743) reg expression
 // (op 743) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 743): v11
v_and_b32 v11, -8, v1 // (op 743) 
 // (op 743) BEGIN: Tile[1]
 // (op 743) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 743): v13
v_sub_u32 v13, v1, v11 // (op 743) 
 // (op 743) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 743): v11

 // (op 743) BEGIN: Flatten
 // (op 743) {Flatten: ShiftL({Tile[1]: v13:U32}, 4:U32)U32}
 // (op 743) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 743)         value (Tile[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 743)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 743): v11
v_lshlrev_b32 v11, 4, v13 // (op 743) 
 // (op 743) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 743): v13

 // (op 743) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 743): v13
v_lshrrev_b32 v13, 3, v1 // (op 743) 
 // (op 743) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 743): v14
v_lshl_add_u32 v14, v13, 7, v11 // (op 743) 
 // (op 743) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 743): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 743): v13

 // (op 743) Multiply({Flatten: v14:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 743): v11
v_mul_lo_u32 v11, v14, 6 // (op 743) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 743): v14

 // (op 743) LogicalShiftR(v11:U32, 3:U32)U32
// Allocated DataFlowTag540: 1 VGPR (Value: UInt32) (op 743): v13
v_lshrrev_b32 v13, 3, v11 // (op 743) 
 // Freeing : 1 VGPR (Value: UInt32) (op 743): v11

 // (op 450) Assign VGPR LogicalShiftR(Multiply({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(89)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(89)U32, BitwiseAnd(DataFlowTag(89)U32, -8:I)U32)U32}, 4:U32)U32})U32}, 6:U32)U32, 3:U32)U32(743) END
 // (op 450) Assign VGPR 96:U32(744) BEGIN
 // (op 744) Assign dim(541) = 96:U32
 // (op 744) Assign stride expression
 // (op 450) Assign VGPR 96:U32(744) END
 // (op 450) NOP(739) BEGIN
 // (op 739) generate({})
 // (op 739) end: generate({})
 // (op 450) NOP(739) END
 // (op 450) Assign VGPR 0:U32(748) BEGIN
 // (op 748) Assign dim(543) = 0:U32
 // (op 748) Assign stride expression
 // (op 450) Assign VGPR 0:U32(748) END
 // (op 450) NOP(791) BEGIN
 // (op 791) generate({})
 // (op 791) end: generate({})
 // (op 450) NOP(791) END
 // (op 450) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(796) BEGIN
 // (op 796) Assign dim(556) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 556: s**UNALLOCATED**
 // (op 796) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag556: SGPR Value: UInt32 x 1: (unallocated)
 // (op 796) reg expression
 // (op 796) BitwiseAnd({Workitem Index X: v1:U32}, -8:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 796): v11
v_and_b32 v11, -8, v1 // (op 796) 
 // (op 796) BEGIN: Tile[1]
 // (op 796) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 796): v14
v_sub_u32 v14, v1, v11 // (op 796) 
 // (op 796) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 796): v11

 // (op 796) BEGIN: Flatten
 // (op 796) {Flatten: ShiftL({Tile[1]: v14:U32}, 4:U32)U32}
 // (op 796) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 796)         value (Tile[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 796)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 796): v11
v_lshlrev_b32 v11, 4, v14 // (op 796) 
 // (op 796) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 796): v14

 // (op 796) LogicalShiftR({Workitem Index X: v1:U32}, 3:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 796): v14
v_lshrrev_b32 v14, 3, v1 // (op 796) 
 // (op 796) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 796): v15
v_lshl_add_u32 v15, v14, 7, v11 // (op 796) 
 // (op 796) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 796): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 796): v14

// Allocated DataFlowTag556: 1 SGPR (Value: UInt32) (op 796): s0
s_nop 0
v_readfirstlane_b32 s0, v15 // (op 796) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 796): v15

 // (op 450) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(155)U32, 3:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(155)U32, BitwiseAnd(DataFlowTag(155)U32, -8:I)U32)U32}, 4:U32)U32})U32})U32(796) END
 // (op 450) Assign VGPR 2048:U32(797) BEGIN
 // (op 797) Assign dim(557) = 2048:U32
 // (op 797) Assign stride expression
 // (op 450) Assign VGPR 2048:U32(797) END
 // (op 450) NOP(792) BEGIN
 // (op 792) generate({})
 // (op 792) end: generate({})
 // (op 450) NOP(792) END
 // (op 450) Assign VGPR 1:U32(801) BEGIN
 // (op 801) Assign dim(559) = 1:U32
 // (op 801) Assign stride expression
 // (op 450) Assign VGPR 1:U32(801) END
 // (op 450) NOP(843) BEGIN
 // (op 843) generate({})
 // (op 843) end: generate({})
 // (op 450) NOP(843) END
 // (op 450) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(848) BEGIN
 // (op 848) Assign dim(572) = {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}
 // tag 572: v**UNALLOCATED**
 // (op 848) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32} into DataFlowTag572: VGPR Value: UInt32 x 1: (unallocated)
 // (op 848) reg expression
 // (op 848) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 848): v11
v_and_b32 v11, -4, v1 // (op 848) 
 // (op 848) BEGIN: Tile[1]Flatten
 // (op 848) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v11:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 848): v14
v_sub_u32 v14, v1, v11 // (op 848) 
 // (op 848) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 848): v11

 // (op 848) BitwiseAnd({Workitem Index X: v1:U32}, 4294967292:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 848): v11
v_and_b32 v11, 4294967292, v1 // (op 848) 
 // (op 848) BEGIN: Flatten
 // (op 848) {Flatten: Add(v11:U32, {Tile[1]Flatten: v14:U32})U32}
// Allocated DataFlowTag572: 1 VGPR (Value: UInt32) (op 848): v15
v_add_u32 v15, v11, v14 // (op 848) 
 // (op 848) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 848): v14

 // Freeing : 1 VGPR (Value: UInt32) (op 848): v11

 // (op 450) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(251)U32, 4294967292:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(251)U32, BitwiseAnd(DataFlowTag(251)U32, -4:I)U32)U32})U32}(848) END
 // (op 450) Assign VGPR 4:U32(849) BEGIN
 // (op 849) Assign dim(573) = 4:U32
 // (op 849) Assign stride expression
 // (op 450) Assign VGPR 4:U32(849) END
 // (op 450) NOP(844) BEGIN
 // (op 844) generate({})
 // (op 844) end: generate({})
 // (op 450) NOP(844) END
 // (op 450) Assign VGPR 1:U32(853) BEGIN
 // (op 853) Assign dim(575) = 1:U32
 // (op 853) Assign stride expression
 // (op 450) Assign VGPR 1:U32(853) END
 // (op 450) ForLoopOp KLoopTail: LessThan(DataFlowTag(418)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(165) BEGIN
 // (op 165) Initialize For Loop
 // (op 165) generate({166})
 // (op 165) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32(166) BEGIN
 // (op 166) Assign dim(418) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32
 // tag 418: s**UNALLOCATED**
 // (op 166) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag418: SGPR Value: UInt32 x 1: (unallocated)
 // (op 166) reg expression
 // (op 166) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 166): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 166) 
 // (op 166) convert
 // (op 166) BitwiseAnd({ convertForwardSubset: s14:U32}, 4294967294:U32)U32
// Allocated DataFlowTag418: 1 SGPR (Value: UInt32) (op 166): s1
s_and_b32 s1, s14, 4294967294 // (op 166) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 166): s14, s15

 // (op 165) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32, 4294967294:U32)U32(166) END
 // (op 165) end: generate({166})
 // (op 165) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 165) 
 // (op 165) Generate LessThan(DataFlowTag(418)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 165) LessThan({DataFlowTag418: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32)BL
 // (op 165) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 165): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 165) 
// Allocated : 1 SGPR (Value: UInt32) (op 165): s18
s_mov_b32 s18, s14 // (op 165) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 165): s14, s15

s_cmp_lt_u32 s1, s18 // (op 165) 
 // Freeing : 1 SGPR (Value: UInt32) (op 165): s18

 // Freeing : 1 VGPR (Value: E8M0) (op 355): v25

 // Freeing : 1 VGPR (Value: E8M0) (op 362): v17

 // Freeing : 3 VGPRs (Value: BF6x16) (op 365): v26, v27, v28

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 165) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 165) 
s_cbranch_scc0 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 165) 
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165:
 // (op 165) 
 // (op 165) generate({167, 210})
 // (op 165) Assign SGPR Add(DataFlowTag(418)U32, 1:U32)U32(167) BEGIN
 // (op 167) Assign dim(418) = Add(DataFlowTag(418)U32, 1:U32)U32
 // (op 167) Generate Add(DataFlowTag(418)U32, 1:U32)U32 into DataFlowTag418: SGPR Value: UInt32 x 1: s1
 // (op 167) reg expression
 // (op 167) Add({DataFlowTag418: s1:U32}, 1:U32)U32
 // (op 165) SetCoordinate(210) BEGIN
 // (op 210) generate({})
 // (op 210) end: generate({})
 // (op 210) generate({170})
 // (op 210) LoadTiled Value: BF6x16(170) BEGIN
 // (op 170) GEN: loadMacroTileVGPRCI Value: BF6x16
 // (op 170) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 170) Generate 0:U64 into nullptr
 // tag 420: v**UNALLOCATED**
 // (op 170) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: (unallocated)
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 458: v**UNALLOCATED**
s_add_u32 s1, s1, 1 // (op 167) 
 // (op 165) Assign SGPR Add(DataFlowTag(418)U32, 1:U32)U32(167) END
// Allocated Offset458: 1 VGPR (Value: UInt32) (op 170): v11
v_mov_b32 v11, v10 // (op 170) 
 // (op 170) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset458: VGPR Value: UInt32 x 1: v11
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
 // 	info.bufDesc = DataFlowTag457: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 170) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 3 VGPRs (Value: BF6x16) (op 170): v16, v17, v18
buffer_load_dwordx3 v[16:18], v11, s[40:43], 0 offen // (op 170) Load value
// VMEM: Expected complete at 4326 (current 3846)
 // (op 210) LoadTiled Value: BF6x16(170) END
 // (op 210) Deallocate{}(1116) BEGIN
 // (op 1116) Deallocate 460
 // Deleting tag 460
 // (op 1116) Deallocate 458
 // Freeing Offset458: 1 VGPR (Value: UInt32) (op 170): v11

 // Deleting tag 458
 // (op 210) Deallocate{}(1116) END
 // (op 210) Assign VGPR Add(DataFlowTag(454)U32, Convert(DataFlowTag(455)U64)U32)U32(445) BEGIN
 // (op 445) Assign dim(454) = Add(DataFlowTag(454)U32, Convert(DataFlowTag(455)U64)U32)U32
 // (op 445) Generate Add(DataFlowTag(454)U32, Convert(DataFlowTag(455)U64)U32)U32 into DataFlowTag454: VGPR Value: UInt32 x 1: v10
 // (op 445) reg expression
 // (op 445) Add({DataFlowTag454: v10:U32}, 96:U32)U32
v_add_u32 v10, 96, v10 // (op 445) 
 // (op 210) Assign VGPR Add(DataFlowTag(454)U32, Convert(DataFlowTag(455)U64)U32)U32(445) END
 // (op 210) end: generate({170})
 // (op 165) SetCoordinate(210) END
 // (op 165) Barrier(1036) BEGIN
 // (op 165) Barrier(1049) BEGIN
s_barrier  // (op 1036) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 165) Barrier(1036) END
s_barrier  // (op 1049) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 165) Barrier(1049) END
 // (op 165) StoreLDSTile Value: BF6x16(171) BEGIN
 // (op 171) GEN: StoreLDSTile
 // (op 171) GEN: storeMacroTileLDS OP 171 LDS 16 MacroTile 420
 // (op 171) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 171) Generate 0:U32 into nullptr
 // (op 171) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[16:18]
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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

 // (op 171) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 1: v[16:18]
 // 	info.rowOffsetReg = DataFlowTag540: VGPR Value: UInt32 x 1: v13
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
 // 	info.offset = Literal Value: UInt32 x 0: 8320
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 171) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 12
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 165) SetCoordinate(218) BEGIN
 // (op 218) generate({})
 // (op 218) end: generate({})
 // (op 218) generate({425})
 // (op 218) LoadTileDirect2LDS Value: BF8x4(425) BEGIN
 // (op 425) GEN: LoadTileDirect2LDS
 // (op 425) GEN: loadMacroTileDirect2LDS OP 425 LDS 24 MacroTile 422
 // (op 425) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 425) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride478: VGPR Value: UInt64 x 1: (unallocated)
 // (op 425) reg expression
 // (op 425) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 425) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 425)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 425)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 425): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 425) 
// Allocated Stride478: 2 VGPRs (Value: UInt64) (op 425): v20, v21
v_mov_b64 v[20:21], s[14:15] // (op 425) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 425): s14, s15

 // (op 425) Generate 1:U64 into nullptr
 // (op 425) Generate 2048:U32 into nullptr
 // (op 425) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 2048
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride478: VGPR Value: UInt64 x 1: v[20:21]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(256:U32, CommandArgument(Tensor_2_stride_1)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	info.bufDesc = DataFlowTag476: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 477: v**UNALLOCATED**
// Allocated Offset477: 2 VGPRs (Value: UInt64) (op 425): v26, v27
v_mov_b64 v[26:27], v[22:23] // (op 425) 
// Allocated : 2 VGPRs (Value: UInt64) (op 425): v28, v29
v_mov_b64 v[28:29], v[26:27] // (op 425) 
 // (op 425) Generate Add(0:U32, {DataFlowTag556Flatten: s0:U32})U32 into nullptr
 // (op 425)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride478: v[20:21]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 425) Generate {DataFlowTag556Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 425) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 425) Load value direct to lds
// VMEM: Expected complete at 4335 (current 3855)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 425) Generate Add(v[28:29]:U64, v20:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 425) reg expression
 // (op 425) reg expression
 // (op 425) Add(v[28:29]:U64, v20:R)U64
 // (op 425) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 425)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 425)      rhs (VGPR Value: Raw32 x 1: v20)
// Allocated : 2 SGPRs (Value: Bool64) (op 425): s14, s15
v_add_co_u32 v28, s[14:15], v28, v20 // (op 425) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 425) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 425): s14, s15

 // (op 425) Generate Add(m0:U32, 2048:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 425) reg expression
 // (op 425) Add(m0:U32, 2048:U32)U32
 // (op 425) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 2048 // (op 425) 
 // (op 425) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 425) Load value direct to lds
// VMEM: Expected complete at 4339 (current 3859)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // Freeing Stride478: 2 VGPRs (Value: UInt64) (op 425): v20, v21

 // Freeing : 2 VGPRs (Value: UInt64) (op 425): v28, v29

 // (op 218) LoadTileDirect2LDS Value: BF8x4(425) END
 // (op 218) Deallocate{}(1178) BEGIN
 // (op 1178) Deallocate 479
 // Deleting tag 479
 // (op 1178) Deallocate 477
 // Freeing Offset477: 2 VGPRs (Value: UInt64) (op 425): v26, v27

 // Deleting tag 477
 // (op 1178) Deallocate 422
 // Deleting tag 422
 // (op 1178) Deallocate 558
 // Deleting tag 558
 // (op 218) Deallocate{}(1178) END
 // (op 218) Assign VGPR Add(DataFlowTag(473)U64, DataFlowTag(474)U64)U64(514) BEGIN
 // (op 514) Assign dim(473) = Add(DataFlowTag(473)U64, DataFlowTag(474)U64)U64
 // (op 514) Generate Add(DataFlowTag(473)U64, DataFlowTag(474)U64)U64 into DataFlowTag473: VGPR Value: UInt64 x 1: v[22:23]
 // (op 514) reg expression
 // (op 514) Add({DataFlowTag473: v[22:23]:U64}, 128:U64)U64
 // (op 514) Add: dest (VGPR Value: UInt64 x 1: v[22:23]) = 
 // (op 514)      lhs (DataFlowTag473: VGPR Value: UInt64 x 1: v[22:23]) 
 // (op 514)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v22, vcc, 128, v22 // (op 514) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 514) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 218) Assign VGPR Add(DataFlowTag(473)U64, DataFlowTag(474)U64)U64(514) END
 // (op 218) end: generate({425})
 // (op 165) SetCoordinate(218) END
s_waitcnt vmcnt(2)
ds_write_b96 v13, v[16:18] offset:8320 // (op 171) Store local data 
// Modelled addresses (normalized): [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600, 612, 624, 636, 648, 660, 672, 684, 696, 708, 720, 732, 744, 756, 768, 780, 792, 804, 816, 828, 840, 852, 864, 876, 888, 900, 912, 924, 936, 948, 960, 972, 984, 996, 1008, 1020, 1032, 1044, 1056, 1068, 1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1248, 1260, 1272, 1284, 1296, 1308, 1320, 1332, 1344, 1356, 1368, 1380, 1392, 1404, 1416, 1428, 1440, 1452, 1464, 1476, 1488, 1500, 1512, 1524]
// DSMEM: Expected complete at 3953 (current 3858)
// Extra dsts: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 165) StoreLDSTile Value: BF6x16(171) END
 // (op 165) SetCoordinate(223) BEGIN
 // (op 223) generate({})
 // (op 223) end: generate({})
 // (op 223) generate({184})
 // (op 223) LoadTiled Value: E8M0(184) BEGIN
 // (op 184) GEN: loadMacroTileVGPRCI Value: E8M0
 // (op 184) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 184) Generate 1:U64 into nullptr
 // tag 424: v**UNALLOCATED**
 // (op 184) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 515: v**UNALLOCATED**
 // (op 165) Deallocate{}(1118) BEGIN
 // (op 1118) Deallocate 542
 // Deleting tag 542
 // (op 1118) Deallocate 420
 // Freeing : 3 VGPRs (Value: BF6x16) (op 170): v16, v17, v18

 // Deleting tag 420
 // (op 165) Deallocate{}(1118) END
// Allocated Offset515: 1 VGPR (Value: UInt32) (op 184): v11
v_mov_b32 v11, v24 // (op 184) 
 // (op 184) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset515: VGPR Value: UInt32 x 1: v11
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag514: SGPR Buffer: None x 1: s[52:55]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 184) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 165) NOP(427) BEGIN
 // (op 427) generate({})
 // (op 427) end: generate({})
 // (op 165) NOP(427) END
// Allocated : 1 VGPR (Value: E8M0) (op 184): v14
buffer_load_ubyte v14, v11, s[52:55], 0 offen offset: 0 // (op 184) Load value
// VMEM: Expected complete at 4810 (current 4330)
 // (op 223) LoadTiled Value: E8M0(184) END
 // (op 223) Deallocate{}(1126) BEGIN
 // (op 1126) Deallocate 517
 // Deleting tag 517
 // (op 1126) Deallocate 515
 // Freeing Offset515: 1 VGPR (Value: UInt32) (op 184): v11

 // Deleting tag 515
 // (op 223) Deallocate{}(1126) END
 // (op 223) Assign VGPR Add(DataFlowTag(511)U32, Convert(DataFlowTag(512)U64)U32)U32(642) BEGIN
 // (op 642) Assign dim(511) = Add(DataFlowTag(511)U32, Convert(DataFlowTag(512)U64)U32)U32
 // (op 642) Generate Add(DataFlowTag(511)U32, Convert(DataFlowTag(512)U64)U32)U32 into DataFlowTag511: VGPR Value: UInt32 x 1: v24
 // (op 642) reg expression
 // (op 642) Add({DataFlowTag511: v24:U32}, 4:U32)U32
v_add_u32 v24, 4, v24 // (op 642) 
 // (op 223) Assign VGPR Add(DataFlowTag(511)U32, Convert(DataFlowTag(512)U64)U32)U32(642) END
 // (op 223) end: generate({184})
 // (op 165) SetCoordinate(223) END
 // (op 165) Barrier(1039) BEGIN
 // (op 165) Barrier(1062) BEGIN
s_barrier  // (op 1062) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 165) Barrier(1062) END
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1039) 
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 165) Barrier(1039) END
 // (op 165) StoreLDSTile Value: E8M0(182) BEGIN
 // (op 182) GEN: StoreLDSTile
 // (op 182) GEN: storeMacroTileLDS OP 182 LDS 32 MacroTile 424
 // (op 182) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 182) Generate 1:U32 into nullptr
 // (op 182) 	Dir = Store
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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

 // (op 182) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag572: VGPR Value: UInt32 x 1: v15
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 182) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 165) SetCoordinate(172) BEGIN
 // (op 172) generate({})
 // (op 172) end: generate({})
 // (op 172) generate({173})
 // (op 172) LoadLDSTile{Value: BF6x16}(173) BEGIN
 // (op 173) GEN: loadMacroTileWAVELDS OP 173 LDS 16 WaveTile 101
 // (op 173) 	varTypeInfo.elementBits = 96
 // 	varTypeInfo.packing = 16

 // (op 173) Generate 0:U32 into nullptr
 // tag 426: v**UNALLOCATED**
 // (op 173) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8320
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

 // (op 173) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 96
 // 	info.packedAmount = 16
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF6x16 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag548: VGPR Value: UInt32 x 1: v3
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
 // 	t.elementBlockStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(128:U32, 6:U32)U32, 8:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8320
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 173) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 24
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 6 VGPRs (Value: BF6x16) (op 173): v16, v17, v18, v19, v20, v21
ds_read_b128 v[16:19], v3 offset:8320 // (op 173) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4051 (current 3956)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
ds_read_b64 v[20:21], v3 offset:8336 // (op 173) Load local data 
// Modelled addresses (normalized): [0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512, 0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 24, 120, 216, 312, 408, 504, 600, 696, 792, 888, 984, 1080, 1176, 1272, 1368, 1464, 48, 144, 240, 336, 432, 528, 624, 720, 816, 912, 1008, 1104, 1200, 1296, 1392, 1488, 72, 168, 264, 360, 456, 552, 648, 744, 840, 936, 1032, 1128, 1224, 1320, 1416, 1512]
// DSMEM: Expected complete at 4052 (current 3957)
// Extra srcs: LDS Value: BF6x16 x 0: LDS:(Offset: 8320, Size: 1536)
 // (op 172) LoadLDSTile{Value: BF6x16}(173) END
 // (op 172) end: generate({173})
 // (op 165) SetCoordinate(172) END
s_waitcnt vmcnt(0)
ds_write_b8 v15, v14 offset:8192 // (op 182) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
// DSMEM: Expected complete at 4054 (current 3959)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 165) StoreLDSTile Value: E8M0(182) END
 // (op 165) Barrier(1052) BEGIN
 // (op 165) Deallocate{}(1124) BEGIN
 // (op 1124) Deallocate 574
 // Deleting tag 574
 // (op 1124) Deallocate 424
 // Freeing : 1 VGPR (Value: E8M0) (op 184): v14

 // Deleting tag 424
 // (op 165) Deallocate{}(1124) END
s_barrier  // (op 1052) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 165) Barrier(1052) END
 // (op 165) SetCoordinate(174) BEGIN
 // (op 174) generate({})
 // (op 174) end: generate({})
 // (op 174) generate({175})
 // (op 174) LoadLDSTile{Value: BF8x4}(175) BEGIN
 // (op 175) GEN: loadMacroTileWAVELDS OP 175 LDS 24 WaveTile 167
 // (op 175) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 175) Generate 1:U32 into nullptr
 // tag 428: v**UNALLOCATED**
 // (op 175) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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

 // (op 175) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag564: VGPR Value: UInt32 x 1: v4
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
 // 	t.elementBlockStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(64:U32, 1:U32)U32, 0:U32)U32
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
 // (op 175) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 64

// Allocated : 8 VGPRs (Value: BF8x4) (op 175): v26, v27, v28, v29, v30, v31, v32, v33
ds_read_b128 v[26:29], v4 // (op 175) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4147 (current 4052)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
ds_read_b128 v[30:33], v4 offset:64 // (op 175) Load local data 
// Modelled addresses (normalized): [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 16, 144, 272, 400, 528, 656, 784, 912, 1040, 1168, 1296, 1424, 1552, 1680, 1808, 1936, 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 48, 176, 304, 432, 560, 688, 816, 944, 1072, 1200, 1328, 1456, 1584, 1712, 1840, 1968, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 2064, 2192, 2320, 2448, 2576, 2704, 2832, 2960, 3088, 3216, 3344, 3472, 3600, 3728, 3856, 3984, 2080, 2208, 2336, 2464, 2592, 2720, 2848, 2976, 3104, 3232, 3360, 3488, 3616, 3744, 3872, 4000, 2096, 2224, 2352, 2480, 2608, 2736, 2864, 2992, 3120, 3248, 3376, 3504, 3632, 3760, 3888, 4016]
// DSMEM: Expected complete at 4148 (current 4053)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 0, Size: 4096)
 // (op 174) LoadLDSTile{Value: BF8x4}(175) END
 // (op 174) end: generate({175})
 // (op 165) SetCoordinate(174) END
 // (op 165) SetCoordinate(176) BEGIN
 // (op 176) generate({})
 // (op 176) end: generate({})
 // (op 176) generate({215})
 // (op 176) SetCoordinate(215) BEGIN
 // (op 215) generate({})
 // (op 215) end: generate({})
 // (op 215) generate({177})
 // (op 215) LoadTiled Value: E8M0(177) BEGIN
 // (op 177) GEN: loadMacroTileWAVE OP177 WaveTile 203
 // (op 177) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 177) Generate 1:U64 into nullptr
 // tag 430: v**UNALLOCATED**
 // (op 177) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // tag 496: v**UNALLOCATED**
// Allocated Offset496: 1 VGPR (Value: UInt32) (op 177): v11
v_mov_b32 v11, v12 // (op 177) 
 // (op 177) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset496: VGPR Value: UInt32 x 1: v11
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(16:U64, 1:U32)U64, 0:U32)U64
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = DataFlowTag495: SGPR Buffer: None x 1: s[48:51]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 177) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 177): v14
buffer_load_ubyte v14, v11, s[48:51], 0 offen offset: 0 // (op 177) Load value
// VMEM: Expected complete at 5388 (current 4908)
 // (op 215) LoadTiled Value: E8M0(177) END
 // (op 215) Deallocate{}(1120) BEGIN
 // (op 1120) Deallocate 496
 // Freeing Offset496: 1 VGPR (Value: UInt32) (op 177): v11

 // Deleting tag 496
 // (op 1120) Deallocate 498
 // Deleting tag 498
 // (op 215) Deallocate{}(1120) END
 // (op 215) Assign VGPR Add(DataFlowTag(492)U32, Convert(DataFlowTag(493)U64)U32)U32(578) BEGIN
 // (op 578) Assign dim(492) = Add(DataFlowTag(492)U32, Convert(DataFlowTag(493)U64)U32)U32
 // (op 578) Generate Add(DataFlowTag(492)U32, Convert(DataFlowTag(493)U64)U32)U32 into DataFlowTag492: VGPR Value: UInt32 x 1: v12
 // (op 578) reg expression
 // (op 578) Add({DataFlowTag492: v12:U32}, 4:U32)U32
v_add_u32 v12, 4, v12 // (op 578) 
 // (op 215) Assign VGPR Add(DataFlowTag(492)U32, Convert(DataFlowTag(493)U64)U32)U32(578) END
 // (op 215) end: generate({177})
 // (op 176) SetCoordinate(215) END
 // (op 176) end: generate({215})
 // (op 165) SetCoordinate(176) END
 // (op 165) Barrier(1066) BEGIN
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1066) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 165) Barrier(1066) END
 // (op 165) SetCoordinate(178) BEGIN
 // (op 178) generate({})
 // (op 178) end: generate({})
 // (op 178) generate({179})
 // (op 178) LoadLDSTile{Value: E8M0}(179) BEGIN
 // (op 179) GEN: loadMacroTileWAVELDS OP 179 LDS 32 WaveTile 263
 // (op 179) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 179) Generate 1:U32 into nullptr
 // tag 432: v**UNALLOCATED**
 // (op 179) 	Dir = Load
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
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag580: VGPR Value: UInt32 x 1: v0
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
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8192
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
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 179): v11
ds_read_u8 v11, v0 offset:8192 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127]
// DSMEM: Expected complete at 4154 (current 4059)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 128)
 // (op 178) LoadLDSTile{Value: E8M0}(179) END
 // (op 178) end: generate({179})
 // (op 165) SetCoordinate(178) END
 // (op 165) Multiply(180) BEGIN
 // (op 180) Generate ScaledMatrixMultiply(WaveTile16xBF6, WaveTile4xBF8, {DataFlowTag51: a[0:3]:S}, WaveTileE8M0, WaveTileE8M0)16xBF6 into DataFlowTag51: ACCVGPR Value: Float x 4: a[0:3]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[26:33], a[0:3], v14, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 180) 
 // (op 165) Multiply(180) END
 // (op 165) Deallocate{}(1122) BEGIN
 // (op 1122) Deallocate 426
 // Freeing : 6 VGPRs (Value: BF6x16) (op 173): v16, v17, v18, v19, v20, v21

 // Deleting tag 426
 // (op 1122) Deallocate 428
 // Freeing : 8 VGPRs (Value: BF8x4) (op 175): v26, v27, v28, v29, v30, v31, v32, v33

 // Deleting tag 428
 // (op 1122) Deallocate 430
 // Freeing : 1 VGPR (Value: E8M0) (op 177): v14

 // Deleting tag 430
 // (op 1122) Deallocate 432
 // Freeing : 1 VGPR (Value: E8M0) (op 179): v11

 // Deleting tag 432
 // (op 165) Deallocate{}(1122) END
 // (op 165) end: generate({167, 210})
 // (op 165) For Loop Increment
 // (op 165) generate({})
 // (op 165) end: generate({})
 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 if true)
 // (op 165) Generate LessThan(DataFlowTag(418)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 165) LessThan({DataFlowTag418: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64)U32)BL
 // (op 165) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 7:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 165): s14, s15
s_ashr_i64 s[14:15], s[28:29], 7 // (op 165) 
// Allocated : 1 SGPR (Value: UInt32) (op 165): s18
s_mov_b32 s18, s14 // (op 165) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 165): s14, s15

s_cmp_lt_u32 s1, s18 // (op 165) 
 // Freeing : 1 SGPR (Value: UInt32) (op 165): s18

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 165) 
s_cbranch_scc1 RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165:
 // (op 165) 
 // (op 165) Unlock For Loop
 // Unlock None
 // (op 450) ForLoopOp KLoopTail: LessThan(DataFlowTag(418)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 7:U32)I64)U32)BL(165) END
 // (op 450) Deallocate{}(1114) BEGIN
 // (op 1114) Deallocate 493
 // Deleting tag 493
 // (op 1114) Deallocate 459
 // Deleting tag 459
 // (op 1114) Deallocate 454
 // Freeing DataFlowTag454: 1 VGPR (Value: UInt32) (op 455): v10

 // Deleting tag 454
 // (op 1114) Deallocate 557
 // Deleting tag 557
 // (op 1114) Deallocate 572
 // Freeing DataFlowTag572: 1 VGPR (Value: UInt32) (op 848): v15

 // Deleting tag 572
 // (op 1114) Deallocate 511
 // Freeing DataFlowTag511: 1 VGPR (Value: UInt32) (op 650): v24

 // Deleting tag 511
 // (op 1114) Deallocate 474
 // Deleting tag 474
 // (op 1114) Deallocate 473
 // Freeing DataFlowTag473: 2 VGPRs (Value: UInt64) (op 522): v22, v23

 // Deleting tag 473
 // (op 1114) Deallocate 418
 // Freeing DataFlowTag418: 1 SGPR (Value: UInt32) (op 166): s1

 // Deleting tag 418
 // (op 1114) Deallocate 540
 // Freeing DataFlowTag540: 1 VGPR (Value: UInt32) (op 743): v13

 // Deleting tag 540
 // (op 1114) Deallocate 575
 // Deleting tag 575
 // (op 1114) Deallocate 516
 // Deleting tag 516
 // (op 1114) Deallocate 573
 // Deleting tag 573
 // (op 1114) Deallocate 518
 // Deleting tag 518
 // (op 1114) Deallocate 461
 // Deleting tag 461
 // (op 1114) Deallocate 478
 // Deleting tag 478
 // (op 1114) Deallocate 480
 // Deleting tag 480
 // (op 1114) Deallocate 492
 // Freeing DataFlowTag492: 1 VGPR (Value: UInt32) (op 586): v12

 // Deleting tag 492
 // (op 1114) Deallocate 499
 // Deleting tag 499
 // (op 1114) Deallocate 512
 // Deleting tag 512
 // (op 1114) Deallocate 541
 // Deleting tag 541
 // (op 1114) Deallocate 543
 // Deleting tag 543
 // (op 1114) Deallocate 497
 // Deleting tag 497
 // (op 1114) Deallocate 556
 // Freeing DataFlowTag556: 1 SGPR (Value: UInt32) (op 796): s0

 // Deleting tag 556
 // (op 1114) Deallocate 559
 // Deleting tag 559
 // (op 1114) Deallocate 455
 // Deleting tag 455
 // (op 450) Deallocate{}(1114) END
 // (op 450) end: generate({444})
 // (op 450) Unlock Scope 450
 // Unlock None
 // (op 206) Scope(450) END
 // (op 206) end: generate({450})
 // (op 205) SetCoordinate(206) END
 // (op 205) end: generate({206})
 // (op 314) SetCoordinate(205) END
 // (op 314) Deallocate{}(1128) BEGIN
 // (op 1128) Deallocate 89
 // Deleting tag 89
 // (op 1128) Deallocate 236
 // Deleting tag 236
 // (op 1128) Deallocate 476
 // Freeing DataFlowTag476: 4 SGPRs (Buffer: None) (op 547): s44, s45, s46, s47

 // Deleting tag 476
 // (op 1128) Deallocate 202
 // Deleting tag 202
 // (op 1128) Deallocate 401
 // Deleting tag 401
 // (op 1128) Deallocate 566
 // Deleting tag 566
 // (op 1128) Deallocate 399
 // Deleting tag 399
 // (op 1128) Deallocate 582
 // Deleting tag 582
 // (op 1128) Deallocate 251
 // Deleting tag 251
 // (op 1128) Deallocate 74
 // Deleting tag 74
 // (op 1128) Deallocate 495
 // Freeing DataFlowTag495: 4 SGPRs (Buffer: None) (op 611): s48, s49, s50, s51

 // Deleting tag 495
 // (op 1128) Deallocate 550
 // Deleting tag 550
 // (op 1128) Deallocate 32
 // Deleting tag 32
 // (op 1128) Deallocate 1
 // Deleting tag 1
 // (op 1128) Deallocate 4
 // Deleting tag 4
 // (op 1128) Deallocate 24
 // Deleting tag 24
 // (op 1128) Deallocate 155
 // Deleting tag 155
 // (op 1128) Deallocate 3
 // Deleting tag 3
 // (op 1128) Deallocate 514
 // Freeing DataFlowTag514: 4 SGPRs (Buffer: None) (op 675): s52, s53, s54, s55

 // Deleting tag 514
 // (op 1128) Deallocate 403
 // Deleting tag 403
 // (op 1128) Deallocate 16
 // Deleting tag 16
 // (op 1128) Deallocate 397
 // Deleting tag 397
 // (op 1128) Deallocate 140
 // Deleting tag 140
 // (op 1128) Deallocate 457
 // Freeing DataFlowTag457: 4 SGPRs (Buffer: None) (op 483): s40, s41, s42, s43

 // Deleting tag 457
 // (op 314) Deallocate{}(1128) END
 // (op 314) Deallocate{Tensor_2_stride_1_3, Tensor_4_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer}(1244) BEGIN
 // (op 1244) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1244) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 1244) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1244) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1244) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1244) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1244) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1244) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1244) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1244) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 314) Deallocate{Tensor_2_stride_1_3, Tensor_4_pointer, Tensor_0_pointer, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_2_pointer}(1244) END
 // (op 314) Barrier(1130) BEGIN
s_barrier  // (op 1130) 
 // (op 314) Barrier(1130) END
 // (op 314) end: generate({319})
 // (op 314) Unlock Scope 314
 // Unlock None
 // (op 1) Scope(314) END
 // (op 1) Deallocate{}(1160) BEGIN
 // (op 1160) Deallocate 610
 // Freeing DataFlowTag610: 1 VGPR (Value: UInt32) (op 973): v7

 // Deleting tag 610
 // (op 1160) Deallocate 580
 // Freeing DataFlowTag580: 1 VGPR (Value: UInt32) (op 876): v0

 // Deleting tag 580
 // (op 1160) Deallocate 599
 // Deleting tag 599
 // (op 1160) Deallocate 613
 // Deleting tag 613
 // (op 1160) Deallocate 618
 // Freeing DataFlowTag618: 1 VGPR (Value: UInt32) (op 997): v5

 // Deleting tag 618
 // (op 1160) Deallocate 581
 // Deleting tag 581
 // (op 1160) Deallocate 565
 // Deleting tag 565
 // (op 1160) Deallocate 549
 // Deleting tag 549
 // (op 1160) Deallocate 621
 // Deleting tag 621
 // (op 1160) Deallocate 567
 // Deleting tag 567
 // (op 1160) Deallocate 619
 // Deleting tag 619
 // (op 1160) Deallocate 611
 // Deleting tag 611
 // (op 1160) Deallocate 548
 // Freeing DataFlowTag548: 1 VGPR (Value: UInt32) (op 771): v3

 // Deleting tag 548
 // (op 1160) Deallocate 601
 // Deleting tag 601
 // (op 1160) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 823): v4

 // Deleting tag 564
 // (op 1160) Deallocate 551
 // Deleting tag 551
 // (op 1160) Deallocate 598
 // Freeing DataFlowTag598: 1 VGPR (Value: UInt32) (op 934): v6

 // Deleting tag 598
 // (op 1160) Deallocate 583
 // Deleting tag 583
 // (op 1) Deallocate{}(1160) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(35)S)BL(125) BEGIN
 // (op 125) Lock for Conditional Branch
 // Lock Branch
 // (op 125) Generate Equal(0.00000:S, DataFlowTag(35)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 125) reg expression
 // (op 125) Equal(0.00000:S, v8:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
 // (op 125) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 125) 
s_cbranch_vccz RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125
 // (op 125) generate({322})
 // (op 125) Scope(322) BEGIN
 // (op 322) Lock Scope 322
 // Lock Branch
 // (op 322) generate({325})
 // (op 322) NOP(325) BEGIN
 // (op 325) generate({})
 // (op 325) end: generate({})
 // (op 322) NOP(325) END
 // (op 322) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(46) BEGIN
 // (op 46) Assign dim(55) = Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA
 // (op 46) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA into nullptr
 // (op 46) reg expression
 // (op 46) reg expression
 // (op 46) Multiply(v9:S, {DataFlowTag51: a[0:3]:S})S
// Allocated : 4 VGPRs (Value: Float) (op 46): v4, v5, v6, v7
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
// Allocated : 4 VGPRs (Value: Float) (op 46): v10, v11, v12, v13
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 
 // Freeing : 4 VGPRs (Value: Float) (op 46): v4, v5, v6, v7

 // (op 322) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(46) END
 // (op 322) Assign VGPR Convert(DataFlowTag(55)NA)B(52) BEGIN
 // (op 52) Assign dim(57) = Convert(DataFlowTag(55)NA)B
 // (op 52) Generate Convert(DataFlowTag(55)NA)B into nullptr
 // (op 52) reg expression
// Allocated DataFlowTag55: 4 VGPRs (Value: BFloat16) (op 52): v4, v5, v6, v7
v_lshrrev_b32 v4, 16, v10 // (op 52) 
v_lshrrev_b32 v5, 16, v11 // (op 52) 
v_lshrrev_b32 v6, 16, v12 // (op 52) 
v_lshrrev_b32 v7, 16, v13 // (op 52) 
 // (op 322) Assign VGPR Convert(DataFlowTag(55)NA)B(52) END
 // (op 322) Deallocate{}(1092) BEGIN
 // (op 1092) Deallocate 55
 // Freeing DataFlowTag55: 4 VGPRs (Value: Float) (op 46): v10, v11, v12, v13

 // Deleting tag 55
 // (op 322) Deallocate{}(1092) END
 // (op 322) NOP(886) BEGIN
 // (op 886) generate({})
 // (op 886) end: generate({})
 // (op 322) NOP(886) END
 // (op 322) NOP(883) BEGIN
 // (op 883) generate({})
 // (op 883) end: generate({})
 // (op 322) NOP(883) END
 // (op 322) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(890) BEGIN
 // (op 890) Assign dim(584) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 584: v**UNALLOCATED**
 // (op 890) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag584: VGPR Value: UInt32 x 1: (unallocated)
 // (op 890) reg expression
 // (op 890) reg expression
 // (op 890) reg expression
 // (op 890) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 890) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 890): v0
v_lshrrev_b32 v0, 7, v1 // (op 890) 
 // (op 890) convert
 // (op 890) BEGIN: Tile[1]
 // (op 890) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 890): v3
v_and_b32 v3, 63, v1 // (op 890) 
 // (op 890) END: Tile[1]
 // (op 890) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 890): v10
v_lshrrev_b32 v10, 4, v3 // (op 890) 
 // (op 890) BEGIN: Flatten
 // (op 890) {Flatten: ShiftL(v10:U32, 2:U32)U32}
 // (op 890) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 890)         value (VGPR Value: UInt32 x 1: v10) 
 // (op 890)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 890): v11
v_lshlrev_b32 v11, 2, v10 // (op 890) 
 // (op 890) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 890): v10

 // (op 890) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 890): v10
v_lshl_add_u32 v10, v0, 4, v11 // (op 890) 
 // (op 890) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 890): v11

 // Freeing : 1 VGPR (Value: UInt32) (op 890): v0

 // (op 890) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 890): v0
v_lshl_add_u32 v0, s16, 4, v10 // (op 890) 
 // (op 890) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 890): v10

 // (op 890) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 890): v10
v_lshrrev_b32 v10, 2, v3 // (op 890) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 890): v3

 // (op 890) BEGIN: Tile[1]
 // (op 890) {Tile[1]: BitwiseAnd(v10:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 890): v3
v_and_b32 v3, 3, v10 // (op 890) 
 // (op 890) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 890): v10

 // (op 890) BEGIN: Tile[1]
 // (op 890) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 890): v10
v_and_b32 v10, 3, v1 // (op 890) 
 // (op 890) END: Tile[1]
 // (op 890) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 890): v11
v_lshl_add_u32 v11, v3, 2, v10 // (op 890) 
 // (op 890) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 890): v10

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 890): v3

 // (op 890) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 890): v3
v_lshrrev_b32 v3, 6, v1 // (op 890) 
 // (op 890) BEGIN: Tile[1]
 // (op 890) {Tile[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 890): v10
v_and_b32 v10, 1, v3 // (op 890) 
 // (op 890) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 890): v3

 // (op 890) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 890): v3
v_lshl_add_u32 v3, v10, 4, v11 // (op 890) 
 // (op 890) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 890): v11

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 890): v10

 // (op 890) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 890): v10
v_lshl_add_u32 v10, s17, 5, v3 // (op 890) 
 // (op 890) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 890): v3

 // (op 890) Multiply({Flatten: v10:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 890): v3
v_mul_lo_u32 v3, v10, s30 // (op 890) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 890): v10

 // (op 890) BEGIN: Join
// Allocated DataFlowTag584: 1 VGPR (Value: UInt32) (op 890): v10
v_add_lshl_u32 v10, v0, v3, 1 // (op 890) 
 // (op 890) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 890): v3

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 890): v0

 // (op 322) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(890) END
 // (op 322) Assign VGPR 32:U64(891) BEGIN
 // (op 891) Assign dim(585) = 32:U64
 // (op 891) Assign stride expression
 // (op 322) Assign VGPR 32:U64(891) END
 // (op 322) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(892) BEGIN
 // (op 892) Assign dim(587) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 587: s**UNALLOCATED**
 // (op 892) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag587: SGPR Buffer: None x 1: (unallocated)
 // (op 892) reg expression
 // (op 892) reg expression
 // (op 892) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 892) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 892)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 892)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 892): s14, s15
s_mul_i32 s15, s30, s13 // (op 892) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 892): s0
s_mul_i32 s0, s31, s12 // (op 892) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 892) least significant
// Allocated : 1 SGPR (Value: Int32) (op 892): s1
s_mul_hi_u32 s1, s30, s12 // (op 892) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 892) most significant: sum
s_add_u32 s15, s15, s1 // (op 892) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 892): s1

 // Freeing : 1 SGPR (Value: Int32) (op 892): s0

 // (op 892) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 892) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 892)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 892)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 892): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 892) 
 // Freeing : 2 SGPRs (Value: Int64) (op 892): s14, s15

 // (op 892) reg expression
 // (op 892) reg expression
// Allocated DataFlowTag587: 4 SGPRs (Buffer: None) (op 892): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 892) reg expression
s_mov_b32 s9, s33 // (op 892) reg expression
s_mov_b32 s10, s0 // (op 892) reg expression
s_mov_b32 s11, 131072 // (op 892) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 892): s0, s1

 // (op 322) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(892) END
 // (op 322) NOP(884) BEGIN
 // (op 884) generate({})
 // (op 884) end: generate({})
 // (op 322) NOP(884) END
 // (op 322) Assign VGPR 2:U64(897) BEGIN
 // (op 897) Assign dim(589) = 2:U64
 // (op 897) Assign stride expression
 // (op 322) Assign VGPR 2:U64(897) END
 // (op 322) StoreTiled Value: BFloat16x2(54) BEGIN
 // (op 54) GEN: storeMacroTileWAVE OP 54 MacroTile 57 WaveTile 333
 // (op 54) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 54) Generate 2:U64 into nullptr
 // (op 54) Convert in LSTGen
 // (op 54) Generate Convert({DataFlowTag55: v[4:7]:B})2xB into nullptr
 // (op 54) reg expression
// Allocated DataFlowTag55: 2 VGPRs (Value: BFloat16x2) (op 54): v12, v13
v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 
 // (op 54) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag55: VGPR Value: BFloat16x2 x 2: v[12:13]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
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
 // 	info.bufDesc = DataFlowTag587: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 54) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag55: VGPR Value: BFloat16x2 x 2: v[12:13]
 // 	info.rowOffsetReg = DataFlowTag584: VGPR Value: UInt32 x 1: v10
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
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
 // 	info.bufDesc = DataFlowTag587: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 54) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 0
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx2 v[12:13], v10, s[8:11], 0 offen // (op 54) Store value
// VMEM: Expected complete at 5923 (current 5443)
 // Freeing DataFlowTag55: 2 VGPRs (Value: BFloat16x2) (op 54): v12, v13

 // (op 322) StoreTiled Value: BFloat16x2(54) END
 // (op 322) Deallocate{}(1094) BEGIN
 // (op 1094) Deallocate 584
 // Freeing DataFlowTag584: 1 VGPR (Value: UInt32) (op 890): v10

 // Deleting tag 584
 // (op 1094) Deallocate 589
 // Deleting tag 589
 // (op 1094) Deallocate 585
 // Deleting tag 585
 // (op 1094) Deallocate 588
 // Deleting tag 588
 // (op 1094) Deallocate 57
 // Freeing DataFlowTag55: 4 VGPRs (Value: BFloat16) (op 52): v4, v5, v6, v7

 // Deleting tag 57
 // (op 322) Deallocate{}(1094) END
 // (op 322) end: generate({325})
 // (op 322) Unlock Scope 322
 // Unlock None
 // (op 125) Scope(322) END
 // (op 125) end: generate({322})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 125) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 125) 
s_branch RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125:
 // (op 125) 
 // (op 125) generate({328})
 // (op 125) Scope(328) BEGIN
 // (op 328) Lock Scope 328
 // Lock Branch
 // (op 328) generate({331})
 // (op 328) NOP(331) BEGIN
 // (op 331) generate({})
 // (op 331) end: generate({})
 // (op 328) NOP(331) END
 // (op 328) NOP(133) BEGIN
 // (op 133) generate({})
 // (op 133) end: generate({})
 // (op 328) NOP(133) END
 // (op 328) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(134) BEGIN
 // (op 134) Assign dim(391) = Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA
 // (op 134) Generate Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA into nullptr
 // (op 134) reg expression
 // (op 134) reg expression
 // (op 134) Multiply(v9:S, {DataFlowTag51: a[0:3]:S})S
 // (op 328) Scope(708) BEGIN
 // (op 708) Lock Scope 708
 // Lock Branch
 // (op 708) generate({705})
 // (op 708) NOP(705) BEGIN
 // (op 705) generate({})
 // (op 705) end: generate({})
 // (op 708) NOP(705) END
 // (op 708) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(292)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(292)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(713) BEGIN
 // (op 713) Assign dim(530) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(292)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(292)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}
 // tag 530: v**UNALLOCATED**
 // (op 713) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(292)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(292)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32} into DataFlowTag530: VGPR Value: UInt32 x 1: (unallocated)
 // (op 713) reg expression
 // (op 713) reg expression
 // (op 713) reg expression
 // (op 713) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 713) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 713): v0
v_lshrrev_b32 v0, 7, v1 // (op 713) 
 // (op 713) convert
 // (op 713) BEGIN: Flatten[1]
 // (op 713) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 713): v3
v_and_b32 v3, 63, v1 // (op 713) 
 // (op 713) END: Flatten[1]
 // (op 713) LogicalShiftR({Flatten[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 713): v4
v_lshrrev_b32 v4, 4, v3 // (op 713) 
 // (op 713) BEGIN: Tile
 // (op 713) {Tile: ShiftL(v4:U32, 2:U32)U32}
 // (op 713) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 713)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 713)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 713): v5
v_lshlrev_b32 v5, 2, v4 // (op 713) 
 // (op 713) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v4

 // (op 713) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 713): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 713) 
 // (op 713) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 713): v0

 // (op 713) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 713): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 713) 
 // (op 713) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v4

 // (op 713) LogicalShiftR({Flatten[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 713): v4
v_lshrrev_b32 v4, 2, v3 // (op 713) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 713): v3

 // (op 713) BEGIN: Flatten[1]
 // (op 713) {Flatten[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 713): v3
v_and_b32 v3, 3, v4 // (op 713) 
 // (op 713) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v4

 // (op 713) BEGIN: Flatten[1]
 // (op 713) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 713): v4
v_and_b32 v4, 3, v1 // (op 713) 
 // (op 713) END: Flatten[1]
 // (op 713) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 713): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 713) 
 // (op 713) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 713): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 713): v3

 // (op 713) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 713): v3
v_lshrrev_b32 v3, 6, v1 // (op 713) 
 // (op 713) BEGIN: Tile
 // (op 713) {Tile: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 713): v4
v_and_b32 v4, 1, v3 // (op 713) 
 // (op 713) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v3

 // (op 713) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 713): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 713) 
 // (op 713) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v4

 // (op 713) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 713): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 713) 
 // (op 713) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v3

 // (op 713) Multiply({Tile: v4:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 713): v3
v_mul_lo_u32 v3, v4, s34 // (op 713) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v4

 // (op 713) BEGIN: Split
// Allocated DataFlowTag530: 1 VGPR (Value: UInt32) (op 713): v4
v_add_lshl_u32 v4, v0, v3, 1 // (op 713) 
 // (op 713) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 713): v3

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 713): v0

 // (op 708) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(405)U32, 4:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(292)U32, 7:U32)U32, 4:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(407)U32, 5:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR(DataFlowTag(292)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(292)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(713) END
 // (op 708) Deallocate{}(1186) BEGIN
 // (op 1186) Deallocate 405
 // Deleting tag 405
 // (op 1186) Deallocate 407
 // Deleting tag 407
 // (op 1186) Deallocate 292
 // Deleting tag 292
 // (op 708) Deallocate{}(1186) END
 // (op 708) Assign VGPR 32:U64(714) BEGIN
 // (op 714) Assign dim(531) = 32:U64
 // (op 714) Assign stride expression
 // (op 708) Assign VGPR 32:U64(714) END
 // (op 708) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(715) BEGIN
 // (op 715) Assign dim(533) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 533: s**UNALLOCATED**
 // (op 715) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag533: SGPR Buffer: None x 1: (unallocated)
 // (op 715) reg expression
 // (op 715) reg expression
 // (op 715) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 715) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 715)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 715)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 715): s14, s15
s_mul_i32 s15, s34, s13 // (op 715) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 715): s0
s_mul_i32 s0, s35, s12 // (op 715) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 715) least significant
// Allocated : 1 SGPR (Value: Int32) (op 715): s1
s_mul_hi_u32 s1, s34, s12 // (op 715) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 715) most significant: sum
s_add_u32 s15, s15, s1 // (op 715) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 715): s1

 // Freeing : 1 SGPR (Value: Int32) (op 715): s0

 // (op 715) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 715) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 715)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 715)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 715): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 715) 
 // Freeing : 2 SGPRs (Value: Int64) (op 715): s14, s15

 // (op 715) reg expression
 // (op 715) reg expression
// Allocated DataFlowTag533: 4 SGPRs (Buffer: None) (op 715): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 715) reg expression
s_mov_b32 s5, s37 // (op 715) reg expression
s_mov_b32 s6, s0 // (op 715) reg expression
s_mov_b32 s7, 131072 // (op 715) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 715): s0, s1

 // (op 708) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(715) END
 // (op 708) Deallocate{Tensor_10_pointer}(1248) BEGIN
 // (op 1248) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 708) Deallocate{Tensor_10_pointer}(1248) END
 // (op 708) NOP(706) BEGIN
 // (op 706) generate({})
 // (op 706) end: generate({})
 // (op 708) NOP(706) END
 // (op 708) Assign VGPR 2:U64(720) BEGIN
 // (op 720) Assign dim(535) = 2:U64
 // (op 720) Assign stride expression
 // (op 708) Assign VGPR 2:U64(720) END
 // (op 708) LoadTiled Value: BFloat16x2(138) BEGIN
 // (op 138) GEN: loadMacroTileWAVECIACCUM OP 138 WaveTile 293
 // (op 138) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 138) Generate 2:U64 into nullptr
 // tag 387: v**UNALLOCATED**
 // (op 138) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BFloat16x2 x 2: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
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
 // 	info.bufDesc = DataFlowTag533: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 138) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BFloat16x2 x 2: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag530: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
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
 // 	info.bufDesc = DataFlowTag533: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 138) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 0
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 2 VGPRs (Value: BFloat16x2) (op 138): v6, v7
buffer_load_dwordx2 v[6:7], v4, s[4:7], 0 offen // (op 138) Load value
// VMEM: Expected complete at 6434 (current 5954)
 // (op 708) LoadTiled Value: BFloat16x2(138) END
 // (op 708) Deallocate{Tensor_10_stride_1_10}(1110) BEGIN
 // (op 1110) Deallocate 5
 // Deleting tag 5
 // (op 1110) Deallocate 533
 // Freeing DataFlowTag533: 4 SGPRs (Buffer: None) (op 715): s4, s5, s6, s7

 // Deleting tag 533
 // (op 1110) Deallocate 534
 // Deleting tag 534
 // (op 1110) Deallocate 531
 // Deleting tag 531
 // (op 1110) Deallocate 535
 // Deleting tag 535
 // (op 1110) Deallocate 530
 // Freeing DataFlowTag530: 1 VGPR (Value: UInt32) (op 713): v4

 // Deleting tag 530
 // (op 1110) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 708) Deallocate{Tensor_10_stride_1_10}(1110) END
 // (op 708) end: generate({705})
 // (op 708) Unlock Scope 708
 // Unlock None
 // (op 328) Scope(708) END
// Allocated : 4 VGPRs (Value: Float) (op 134): v10, v11, v12, v13
v_accvgpr_read v10, a0 // (op 134) 
 // (op 328) Assign VGPR Multiply(DataFlowTag(35)NA, DataFlowTag(387)NA)NA(139) BEGIN
 // (op 139) Assign dim(389) = Multiply(DataFlowTag(35)NA, DataFlowTag(387)NA)NA
 // (op 139) Generate Multiply(DataFlowTag(35)NA, DataFlowTag(387)NA)NA into nullptr
 // (op 139) reg expression
 // (op 139) reg expression
 // (op 139) Multiply(v8:S, v[6:7]:2xB)S
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
// Allocated : 4 VGPRs (Value: Float) (op 134): v14, v15, v16, v17
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 
 // Freeing : 4 VGPRs (Value: Float) (op 134): v10, v11, v12, v13

 // (op 328) Assign VGPR Multiply(DataFlowTag(44)NA, DataFlowTag(51)NA)NA(134) END
// Allocated : 2 VGPRs (Value: Float) (op 139): v4, v5
s_waitcnt vmcnt(0)
v_and_b32 v4, 65535, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
 // (op 139) ShiftL: dest (VGPR Value: Float x 1: v4) = 
 // (op 139)         value (VGPR Value: Float x 1: v4) 
 // (op 139)         shiftAmount (Literal Value: UInt32 x 0: 16)
v_lshlrev_b32 v4, 16, v4 // (op 139) 
 // (op 139) ShiftL: dest (VGPR Value: Float x 1: v5) = 
 // (op 139)         value (VGPR Value: Float x 1: v5) 
 // (op 139)         shiftAmount (Literal Value: UInt32 x 0: 16)
v_lshlrev_b32 v5, 16, v5 // (op 139) 
// Allocated : 4 VGPRs (Value: Float) (op 139): v10, v11, v12, v13
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_and_b32 v4, 65535, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
 // (op 139) ShiftL: dest (VGPR Value: Float x 1: v4) = 
 // (op 139)         value (VGPR Value: Float x 1: v4) 
 // (op 139)         shiftAmount (Literal Value: UInt32 x 0: 16)
v_lshlrev_b32 v4, 16, v4 // (op 139) 
 // (op 139) ShiftL: dest (VGPR Value: Float x 1: v5) = 
 // (op 139)         value (VGPR Value: Float x 1: v5) 
 // (op 139)         shiftAmount (Literal Value: UInt32 x 0: 16)
v_lshlrev_b32 v5, 16, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 
 // Freeing : 2 VGPRs (Value: Float) (op 139): v4, v5

 // (op 328) Assign VGPR Multiply(DataFlowTag(35)NA, DataFlowTag(387)NA)NA(139) END
 // (op 328) Deallocate{}(1112) BEGIN
 // (op 1112) Deallocate 387
 // Freeing : 2 VGPRs (Value: BFloat16x2) (op 138): v6, v7

 // Deleting tag 387
 // (op 328) Deallocate{}(1112) END
 // (op 328) Assign VGPR Add(DataFlowTag(389)NA, DataFlowTag(391)NA)NA(135) BEGIN
 // (op 135) Assign dim(393) = Add(DataFlowTag(389)NA, DataFlowTag(391)NA)NA
 // (op 135) Generate Add(DataFlowTag(389)NA, DataFlowTag(391)NA)NA into nullptr
 // (op 135) reg expression
 // (op 135) reg expression
 // (op 135) Add({DataFlowTag389: v[10:13]:S}, {DataFlowTag391: v[14:17]:S})S
// Allocated : 4 VGPRs (Value: Float) (op 135): v4, v5, v6, v7
v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 
 // (op 328) Assign VGPR Add(DataFlowTag(389)NA, DataFlowTag(391)NA)NA(135) END
 // (op 328) Deallocate{}(1104) BEGIN
 // (op 1104) Deallocate 391
 // Freeing DataFlowTag391: 4 VGPRs (Value: Float) (op 134): v14, v15, v16, v17

 // Deleting tag 391
 // (op 1104) Deallocate 389
 // Freeing DataFlowTag389: 4 VGPRs (Value: Float) (op 139): v10, v11, v12, v13

 // Deleting tag 389
 // (op 328) Deallocate{}(1104) END
 // (op 328) Assign VGPR Convert(DataFlowTag(393)NA)B(136) BEGIN
 // (op 136) Assign dim(395) = Convert(DataFlowTag(393)NA)B
 // (op 136) Generate Convert(DataFlowTag(393)NA)B into nullptr
 // (op 136) reg expression
// Allocated DataFlowTag393: 4 VGPRs (Value: BFloat16) (op 136): v10, v11, v12, v13
v_lshrrev_b32 v10, 16, v4 // (op 136) 
v_lshrrev_b32 v11, 16, v5 // (op 136) 
v_lshrrev_b32 v12, 16, v6 // (op 136) 
v_lshrrev_b32 v13, 16, v7 // (op 136) 
 // (op 328) Assign VGPR Convert(DataFlowTag(393)NA)B(136) END
 // (op 328) Deallocate{}(1106) BEGIN
 // (op 1106) Deallocate 393
 // Freeing DataFlowTag393: 4 VGPRs (Value: Float) (op 135): v4, v5, v6, v7

 // Deleting tag 393
 // (op 328) Deallocate{}(1106) END
 // (op 328) NOP(903) BEGIN
 // (op 903) generate({})
 // (op 903) end: generate({})
 // (op 328) NOP(903) END
 // (op 328) NOP(900) BEGIN
 // (op 900) generate({})
 // (op 900) end: generate({})
 // (op 328) NOP(900) END
 // (op 328) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(907) BEGIN
 // (op 907) Assign dim(590) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 590: v**UNALLOCATED**
 // (op 907) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag590: VGPR Value: UInt32 x 1: (unallocated)
 // (op 907) reg expression
 // (op 907) reg expression
 // (op 907) reg expression
 // (op 907) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 907) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 907): v0
v_lshrrev_b32 v0, 7, v1 // (op 907) 
 // (op 907) convert
 // (op 907) BEGIN: Tile[1]
 // (op 907) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 907): v3
v_and_b32 v3, 63, v1 // (op 907) 
 // (op 907) END: Tile[1]
 // (op 907) LogicalShiftR({Tile[1]: v3:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 907): v4
v_lshrrev_b32 v4, 4, v3 // (op 907) 
 // (op 907) BEGIN: Flatten
 // (op 907) {Flatten: ShiftL(v4:U32, 2:U32)U32}
 // (op 907) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 907)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 907)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 907): v5
v_lshlrev_b32 v5, 2, v4 // (op 907) 
 // (op 907) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 907): v4

 // (op 907) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 907): v4
v_lshl_add_u32 v4, v0, 4, v5 // (op 907) 
 // (op 907) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 907): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 907): v0

 // (op 907) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 907): v0
v_lshl_add_u32 v0, s16, 4, v4 // (op 907) 
 // (op 907) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 907): v4

 // (op 907) LogicalShiftR({Tile[1]: v3:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 907): v4
v_lshrrev_b32 v4, 2, v3 // (op 907) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 907): v3

 // (op 907) BEGIN: Tile[1]
 // (op 907) {Tile[1]: BitwiseAnd(v4:U32, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 907): v3
v_and_b32 v3, 3, v4 // (op 907) 
 // (op 907) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 907): v4

 // (op 907) BEGIN: Tile[1]
 // (op 907) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 907): v4
v_and_b32 v4, 3, v1 // (op 907) 
 // (op 907) END: Tile[1]
 // (op 907) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 907): v5
v_lshl_add_u32 v5, v3, 2, v4 // (op 907) 
 // (op 907) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 907): v4

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 907): v3

 // (op 907) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 907): v3
v_lshrrev_b32 v3, 6, v1 // (op 907) 
 // (op 907) BEGIN: Tile[1]
 // (op 907) {Tile[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 907): v4
v_and_b32 v4, 1, v3 // (op 907) 
 // (op 907) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 907): v3

 // (op 907) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 907): v3
v_lshl_add_u32 v3, v4, 4, v5 // (op 907) 
 // (op 907) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 907): v5

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 907): v4

 // (op 907) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 907): v4
v_lshl_add_u32 v4, s17, 5, v3 // (op 907) 
 // (op 907) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 907): v3

 // (op 907) Multiply({Flatten: v4:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 907): v3
v_mul_lo_u32 v3, v4, s30 // (op 907) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 907): v4

 // (op 907) BEGIN: Join
// Allocated DataFlowTag590: 1 VGPR (Value: UInt32) (op 907): v4
v_add_lshl_u32 v4, v0, v3, 1 // (op 907) 
 // (op 907) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 907): v3

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 907): v0

 // (op 328) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(409)U32, 4:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(332)U32, 7:U32)U32, 4:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 4:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(411)U32, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR(DataFlowTag(332)U32, 6:U32)U32, 1:U32)U32}, 4:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 63:U32)U32}, 2:U32)U32, 3:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(332)U32, 3:U32)U32})U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(907) END
 // (op 328) Assign VGPR 32:U64(908) BEGIN
 // (op 908) Assign dim(591) = 32:U64
 // (op 908) Assign stride expression
 // (op 328) Assign VGPR 32:U64(908) END
 // (op 328) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(909) BEGIN
 // (op 909) Assign dim(587) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 909) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag587: SGPR Buffer: None x 1: s[8:11]
 // (op 909) reg expression
 // (op 909) reg expression
 // (op 909) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 909) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 909)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 909)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 909): s14, s15
s_mul_i32 s15, s30, s13 // (op 909) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 909): s0
s_mul_i32 s0, s31, s12 // (op 909) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 909) least significant
// Allocated : 1 SGPR (Value: Int32) (op 909): s1
s_mul_hi_u32 s1, s30, s12 // (op 909) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 909) most significant: sum
s_add_u32 s15, s15, s1 // (op 909) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 909): s1

 // Freeing : 1 SGPR (Value: Int32) (op 909): s0

 // (op 909) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 909) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 909)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 909)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 909): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 909) 
 // Freeing : 2 SGPRs (Value: Int64) (op 909): s14, s15

 // (op 909) reg expression
 // (op 909) reg expression
s_mov_b32 s8, s32 // (op 909) reg expression
s_mov_b32 s9, s33 // (op 909) reg expression
s_mov_b32 s10, s0 // (op 909) reg expression
s_mov_b32 s11, 131072 // (op 909) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 909): s0, s1

 // (op 328) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(909) END
 // (op 328) NOP(901) BEGIN
 // (op 901) generate({})
 // (op 901) end: generate({})
 // (op 328) NOP(901) END
 // (op 328) Assign VGPR 2:U64(914) BEGIN
 // (op 914) Assign dim(593) = 2:U64
 // (op 914) Assign stride expression
 // (op 328) Assign VGPR 2:U64(914) END
 // (op 328) StoreTiled Value: BFloat16x2(137) BEGIN
 // (op 137) GEN: storeMacroTileWAVE OP 137 MacroTile 395 WaveTile 333
 // (op 137) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 2

 // (op 137) Generate 2:U64 into nullptr
 // (op 137) Convert in LSTGen
 // (op 137) Generate Convert({DataFlowTag393: v[10:13]:B})2xB into nullptr
 // (op 137) reg expression
// Allocated DataFlowTag393: 2 VGPRs (Value: BFloat16x2) (op 137): v6, v7
v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 
 // (op 137) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag393: VGPR Value: BFloat16x2 x 2: v[6:7]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
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
 // 	info.bufDesc = DataFlowTag587: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 137) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 2
 // 	info.elementBits = 32
 // 	info.packedAmount = 2
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag393: VGPR Value: BFloat16x2 x 2: v[6:7]
 // 	info.rowOffsetReg = DataFlowTag590: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
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
 // 	info.bufDesc = DataFlowTag587: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 137) 	info.m = 1
 // 	info.n = 2
 // 	elementsPerMove = 2
 // 	bytesPerMove = 8
 // 	rowStride = 0
 // 	colStride = 2
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_store_dwordx2 v[6:7], v4, s[8:11], 0 offen // (op 137) Store value
// VMEM: Expected complete at 6965 (current 6485)
 // Freeing DataFlowTag393: 2 VGPRs (Value: BFloat16x2) (op 137): v6, v7

 // (op 328) StoreTiled Value: BFloat16x2(137) END
 // (op 328) Deallocate{}(1108) BEGIN
 // (op 1108) Deallocate 395
 // Freeing DataFlowTag393: 4 VGPRs (Value: BFloat16) (op 136): v10, v11, v12, v13

 // Deleting tag 395
 // (op 1108) Deallocate 592
 // Deleting tag 592
 // (op 1108) Deallocate 593
 // Deleting tag 593
 // (op 1108) Deallocate 590
 // Freeing DataFlowTag590: 1 VGPR (Value: UInt32) (op 907): v4

 // Deleting tag 590
 // (op 1108) Deallocate 591
 // Deleting tag 591
 // (op 328) Deallocate{}(1108) END
 // (op 328) end: generate({331})
 // (op 328) Unlock Scope 328
 // Unlock None
 // (op 125) Scope(328) END
 // (op 125) Deallocate{Tensor_21_stride_1_16, Tensor_21_pointer}(1162) BEGIN
 // (op 1162) Deallocate 409
 // Deleting tag 409
 // (op 1162) Deallocate 411
 // Deleting tag 411
 // (op 1162) Deallocate 332
 // Deleting tag 332
 // (op 1162) Deallocate 587
 // Freeing DataFlowTag587: 4 SGPRs (Buffer: None) (op 892): s8, s9, s10, s11

 // Deleting tag 587
 // (op 1162) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 1162) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 125) Deallocate{Tensor_21_stride_1_16, Tensor_21_pointer}(1162) END
 // (op 125) end: generate({328})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 125) 
RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125:
 // (op 125) 
 // (op 125) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(35)S)BL(125) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1102) BEGIN
 // (op 1102) Deallocate 35
 // Freeing : 1 VGPR (Value: Float) (op 2): v8

 // Deleting tag 35
 // (op 1102) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 39): v9

 // Deleting tag 44
 // (op 1102) Deallocate 51
 // Freeing DataFlowTag51: 4 ACCVGPRs (Value: Float) (op 61): a0, a1, a2, a3

 // Deleting tag 51
 // (op 1102) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1102) END
 // (op 1) end: generate({767, 819, 872, 930, 969, 993})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 56
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 11520 // lds bytes
  .amdhsa_accum_offset 52
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B8
    DestDataType: b
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
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 11520
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 50
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

